# 06 — Thiết kế lưu trữ

Hệ thống có ba loại dữ liệu bền vững, tách bạch về mục đích:

| Loại | Nơi lưu | Tính chất | Ai ghi |
|---|---|---|---|
| Hiện vật mô hình | `models/*.joblib`, `models/*.json` | Chỉ đọc lúc chạy | Notebook 08 |
| Dữ liệu tham chiếu | `data/test_set.parquet`, `data/sample_pool.json` | Chỉ đọc lúc chạy | Notebook 08 |
| Trạng thái ứng dụng | **PostgreSQL 16** | Đọc/ghi lúc chạy | API |

Ranh giới này là hệ quả trực tiếp của AR-01: API không bao giờ ghi vào `models/`.

## 1. Chọn PostgreSQL

Cơ sở dữ liệu là **PostgreSQL 16**, chạy như một dịch vụ riêng trong Docker
Compose. Truy cập qua SQLAlchemy 2.0 với driver `psycopg` (phiên bản 3).

Điều này mang lại, so với một tệp nhúng:

| Lợi ích | Ý nghĩa cụ thể ở dự án này |
|---|---|
| Ghi đồng thời | Chế độ phát lại (SSE) ghi liên tục trong khi người dùng tải CSV và thẩm định — không có khóa ghi toàn cơ sở dữ liệu |
| Kiểu dữ liệu thật | `JSONB`, `TIMESTAMPTZ`, `NUMERIC` — không phải tự ép kiểu ở tầng ứng dụng |
| Ràng buộc toàn vẹn | `CHECK`, khóa ngoại, `UNIQUE` được thi hành thật sự |
| `COPY` | Nạp 10.000 dòng từ CSV trong khoảng một giây — điều kiện để đạt NFR-02 |
| Truy vấn phân tích | Hàm cửa sổ, `percentile_cont` cho phân vị `Amount` ở UI-02 |
| Gần với vận hành thật | Cùng hệ quản trị mà một hệ thống chống gian lận thật sẽ dùng |

Cái giá phải trả: thêm một dịch vụ trong Compose, thêm bước migration, và lần
khởi động đầu tiên lâu hơn (xem §7.2). Đây là đánh đổi chấp nhận được.

**Cấu hình kết nối** — đọc từ biến môi trường `DATABASE_URL`, không hard-code:

```
postgresql+psycopg://fraud:fraud@db:5432/fraud     # trong Docker Compose
postgresql+psycopg://fraud:fraud@localhost:5432/fraud   # khi chạy cục bộ
```

## 2. Lược đồ cơ sở dữ liệu

```sql
-- Bảng giao dịch đã chấm điểm
CREATE TABLE transactions (
    id              TEXT PRIMARY KEY,                    -- 'TX-8841'
    time_offset     DOUBLE PRECISION NOT NULL,           -- cột Time gốc, giây
    hour            SMALLINT         NOT NULL,           -- dẫn xuất, 0..23
    amount          NUMERIC(12, 2)   NOT NULL,
    features        JSONB            NOT NULL,           -- 28 giá trị V
    risk_score      DOUBLE PRECISION NOT NULL,
    model_version   TEXT             NOT NULL,
    true_label      SMALLINT,                            -- NULL nếu không có nhãn
    source          TEXT             NOT NULL,
    batch_id        TEXT,
    created_at      TIMESTAMPTZ      NOT NULL DEFAULT now(),

    CONSTRAINT ck_tx_time   CHECK (time_offset >= 0),
    CONSTRAINT ck_tx_hour   CHECK (hour BETWEEN 0 AND 23),
    CONSTRAINT ck_tx_amount CHECK (amount >= 0),
    CONSTRAINT ck_tx_score  CHECK (risk_score >= 0 AND risk_score <= 1),
    CONSTRAINT ck_tx_label  CHECK (true_label IS NULL OR true_label IN (0, 1)),
    CONSTRAINT ck_tx_source CHECK (source IN ('upload', 'sample', 'replay', 'manual'))
);

CREATE INDEX idx_tx_risk    ON transactions (risk_score DESC);
CREATE INDEX idx_tx_created ON transactions (created_at DESC);
CREATE INDEX idx_tx_batch   ON transactions (batch_id) WHERE batch_id IS NOT NULL;

-- Kết luận thẩm định của con người
CREATE TABLE reviews (
    id              BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    transaction_id  TEXT        NOT NULL UNIQUE
                                REFERENCES transactions(id) ON DELETE CASCADE,
    decision        TEXT        NOT NULL,
    threshold_used  DOUBLE PRECISION NOT NULL,
    reviewed_at     TIMESTAMPTZ NOT NULL DEFAULT now(),
    note            TEXT,

    CONSTRAINT ck_review_decision
        CHECK (decision IN ('confirmed_fraud', 'false_alarm')),
    CONSTRAINT ck_review_threshold
        CHECK (threshold_used > 0 AND threshold_used < 1)
);

-- Cấu hình thay đổi được lúc chạy
CREATE TABLE settings (
    key         TEXT        PRIMARY KEY,
    value       JSONB       NOT NULL,
    updated_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);
```

Khóa trong `settings`:

| Khóa | Kiểu JSON | Mặc định | Nguồn mặc định |
|---|---|---|---|
| `threshold` | object `{"value": number, "model_version": string}` | **không có dòng** — dùng τ\* của `threshold.json` | chỉ ghi khi người dùng gọi `PUT /threshold` |
| `cost_fn` | number | 122.21 | migration `0002`, rồi `threshold.json` lúc khởi động |
| `cost_fp` | number | 5.00 | như trên |
| `replay_speed` | number | 60 | Hằng số |

Lúc khởi động, API chèn khóa còn thiếu bằng `INSERT … ON CONFLICT (key) DO NOTHING`
với giá trị lấy từ `threshold.json`. Nhờ vậy khởi động lại không ghi đè lựa chọn mà
người dùng đã đặt.

**Khóa `threshold` khác các khóa còn lại (thay đổi ở giai đoạn 7).** Bản thiết kế đầu nạp sẵn
một con số vào đây. Làm vậy thì sau khi huấn luyện lại, bảng vẫn giữ τ\* của mô hình **cũ**
mà không ai biết — trong khi phân bố điểm của mô hình mới đã khác. Nay dòng này chỉ xuất
hiện khi người dùng tự đặt, và mang theo `model_version` lúc đặt. API chỉ dùng nó khi
`model_version` trùng mô hình đang chạy (`source = "user"`); ngược lại dùng τ\* của
`threshold.json` (`source = "artifact"`).

Mã giao dịch lấy từ dãy `transaction_id_seq` (migration `0001`): `TX-<số>` cho giao dịch
chấm qua API, lấy trước N mã rồi ghi bằng `COPY` (§4.3). Giao dịch phát lại dùng mã tất định
`RP-<dòng trong test_set.parquet>` và ghi bằng `ON CONFLICT DO NOTHING`, nên phát lại hai lần
không nhân đôi hàng đợi.

## 3. Quyết định thiết kế đáng chú ý

**ST-01 — Không lưu cột `decision` trong `transactions`.** Quyết định chặn/cho qua
là hàm của `risk_score` và ngưỡng hiện hành. Ngưỡng thay đổi được lúc chạy (AR-03),
nên nếu lưu `decision` thì mọi bản ghi cũ sẽ sai ngay khi người dùng kéo thanh
trượt. Tính lại lúc truy vấn rẻ hơn nhiều so với đồng bộ lại toàn bảng.

**ST-02 — Lưu `threshold_used` trong `reviews`.** Ngược lại với ST-01: quyết định
của con người được đưa ra dưới một chính sách cụ thể, và chính sách đó là dữ kiện
lịch sử. Cột này cho phép về sau trả lời "các cảnh báo phát sinh dưới ngưỡng 0,05
có tỷ lệ xác nhận gian lận bao nhiêu".

**ST-03 — 28 giá trị V lưu trong một cột `JSONB`.** Không tạo 28 cột riêng, vì ứng
dụng không bao giờ lọc theo từng giá trị V — nó chỉ đọc cả khối để đưa vào SHAP.
Một cột `JSONB` giữ lược đồ gọn và cho phép đổi số lượng đặc trưng mà không phải
migrate.

Dùng `JSONB` chứ không phải `JSON`: `JSONB` lưu ở dạng nhị phân đã phân tích, đọc
nhanh hơn và hỗ trợ chỉ mục GIN nếu về sau cần truy vấn theo từng khóa. Với 28 số,
một dòng `JSONB` chiếm khoảng 600 byte.

Lưu dạng đối tượng `{"V1": -1.3598, …}` chứ không phải mảng — tên khóa tốn thêm
khoảng 100 byte mỗi dòng nhưng loại bỏ hoàn toàn khả năng lệch thứ tự giữa lúc ghi
và lúc đọc. Ở dự án mà nhất quán đặc trưng là rủi ro lớn nhất (R-05), đánh đổi này
đáng giá.

**ST-04 — `UNIQUE` trên `reviews.transaction_id` và ghi bằng `UPSERT`.** Một giao
dịch chỉ có một kết luận. Gửi lại là ghi đè:

```sql
INSERT INTO reviews (transaction_id, decision, threshold_used, note)
VALUES (:tx_id, :decision, :threshold, :note)
ON CONFLICT (transaction_id) DO UPDATE
   SET decision       = EXCLUDED.decision,
       threshold_used = EXCLUDED.threshold_used,
       note           = EXCLUDED.note,
       reviewed_at    = now();
```

`ON CONFLICT` là thao tác nguyên tử, không cần đọc trước rồi mới quyết định chèn
hay cập nhật — tránh hẳn tình huống hai yêu cầu cùng lúc tạo ra hai bản ghi.

**ST-05 — `true_label` cho phép NULL.** Dữ liệu tải lên trong thực tế không có
nhãn. Khi có nhãn (như `test_set.parquet`), hệ thống dùng để tính chỉ số thực và
để so sánh quyết định của người thẩm định với sự thật (FR-44).

**ST-06 — `CHECK` thay cho kiểu `ENUM`.** `source` và `decision` ràng buộc bằng
`CHECK … IN (…)`. Kiểu `ENUM` của PostgreSQL cần `ALTER TYPE` trong migration mỗi
lần thêm giá trị, còn `CHECK` chỉ cần sửa ràng buộc. Với một dự án ba tuần còn
đang thay đổi, `CHECK` linh hoạt hơn mà vẫn chặn được dữ liệu rác.

**ST-07 — `amount` dùng `NUMERIC(12, 2)`, `risk_score` dùng `DOUBLE PRECISION`.**
Số tiền là đại lượng tiền tệ và tham gia vào phép cộng chi phí, nên cần số thập
phân chính xác; dữ liệu nguồn vốn đã có đúng 2 chữ số thập phân nên không mất mát
gì. Điểm rủi ro là số thực xấp xỉ, `DOUBLE PRECISION` vừa đúng và so sánh nhanh
hơn. Lưu ý ở tầng ứng dụng: `NUMERIC` trả về `decimal.Decimal` trong Python —
ép về `float` trước khi đưa vào `build_features()`.

**ST-08 — `TIMESTAMPTZ` ở mọi cột thời gian, máy chủ đặt `timezone = UTC`.**
API trả chuỗi ISO 8601 hậu tố `Z` theo quy ước ở [05 §5](05-thiet-ke-api.md). Dùng
`TIMESTAMP` không có múi giờ là nguồn lỗi kinh điển khi container và máy chủ khác
múi.

Lưu ý phân biệt: `hour` trong bảng `transactions` **không** liên quan tới
`created_at`. Nó là giờ trong ngày dẫn xuất từ cột `Time` của dữ liệu gốc
(DS-03), tức thuộc về nội dung giao dịch, không phải thời điểm hệ thống ghi nhận.

## 4. Truy cập dữ liệu

### 4.1 Engine và pool kết nối

```python
# api/db.py
engine = create_engine(
    os.environ["DATABASE_URL"],
    pool_size=5,            # đủ cho uvicorn một worker
    max_overflow=10,
    pool_pre_ping=True,     # tự phát hiện kết nối chết sau khi db khởi động lại
    pool_recycle=1800,
    future=True,
)
```

`pool_pre_ping` là bắt buộc trong môi trường Compose: khi container `db` khởi động
lại, các kết nối trong pool trở thành rác và yêu cầu đầu tiên sau đó sẽ lỗi nếu
không có nó.

### 4.2 Phạm vi giao dịch (transaction)

| Thao tác | Phạm vi |
|---|---|
| `GET` bất kỳ | Một giao dịch chỉ đọc, tự động commit |
| `POST /score` | Một giao dịch, commit sau khi ghi xong |
| `POST /score/upload` | **Một giao dịch cho cả lô** — hoặc toàn bộ dòng hợp lệ được ghi, hoặc không dòng nào |
| `POST /reviews` | Một giao dịch, dùng `UPSERT` ở ST-04 |
| `PUT /threshold` | Một giao dịch, `UPSERT` vào `settings` |
| SSE phát lại | Ghi theo từng mẻ 100 dòng, không giữ một giao dịch mở suốt phiên |

Điểm cuối cùng đáng lưu ý: giữ một giao dịch mở trong nhiều phút (suốt một phiên
phát lại) sẽ chặn `VACUUM` và làm phình bảng. Cắt thành từng mẻ nhỏ.

### 4.3 Nạp hàng loạt — điều kiện đạt NFR-02

Với `POST /score/upload` (tối đa 100 MB, khoảng 500.000 dòng), **không** dùng
`INSERT` từng dòng. Dùng `COPY` của PostgreSQL qua psycopg 3:

```python
with conn.cursor().copy(
    "COPY transactions (id, time_offset, hour, amount, features, "
    "risk_score, model_version, true_label, source, batch_id) FROM STDIN"
) as copy:
    for row in rows:
        copy.write_row(row)
```

Tham chiếu hiệu năng: `COPY` nạp khoảng 50.000 dòng mỗi giây; `INSERT` từng dòng
qua ORM chỉ đạt khoảng 1.000 dòng mỗi giây. Với yêu cầu 10.000 dòng dưới 30 giây,
phần lớn ngân sách thời gian nên dành cho `predict_proba`, không phải cho việc ghi.

### 4.4 Vài truy vấn chính

Hàng đợi thẩm định (API-06) — lọc theo ngưỡng, nối trạng thái thẩm định:

```sql
SELECT t.id, t.risk_score, t.amount, t.hour, t.true_label,
       r.decision, t.created_at
FROM   transactions t
LEFT   JOIN reviews r ON r.transaction_id = t.id
WHERE  t.risk_score >= :threshold
ORDER  BY t.risk_score DESC
LIMIT  :page_size OFFSET :offset;
```

Chỉ mục `idx_tx_risk` phục vụ trực tiếp cả `WHERE` lẫn `ORDER BY` ở đây.

Phân vị của `Amount` cho UI-02 (US-04) **không** tính bằng SQL. Bản thiết kế đầu dùng
`percent_rank() OVER (ORDER BY amount) … WHERE id = :tx_id`, nhưng `WHERE` lọc trước hàm
cửa sổ nên kết quả luôn là 0. Kể cả viết đúng thì bảng `transactions` chỉ chứa những gì
người dùng đã nạp, vài chục dòng thì phân vị vô nghĩa. API so với phân bố `Amount` của
tập kiểm thử, đã sắp sẵn trong bộ nhớ (`np.searchsorted`).

Đếm tổng cho phân trang: dùng `COUNT(*) OVER ()` trong cùng truy vấn thay vì gọi
`COUNT` riêng — tiết kiệm một lượt quét.

## 5. Migration

Dùng **Alembic**. Lược đồ không được tạo bằng `Base.metadata.create_all()` lúc
chạy, vì như vậy không có lịch sử thay đổi và không lặp lại được trên máy khác.

```
alembic.ini                         # ở gốc repo, để `alembic upgrade head` chạy từ gốc; chỉ dùng ASCII
api/migrations/
├── env.py                          # URL lấy từ DATABASE_URL (.env) hoặc do mã gọi truyền vào
└── versions/
    ├── 0001_initial_schema.py      # ba bảng, chỉ mục, ràng buộc, dãy transaction_id_seq
    └── 0002_seed_settings.py       # nạp cost_fn, cost_fp, replay_speed — không nạp threshold (§2)
```

`alembic.ini` chỉ được chứa ký tự ASCII: Alembic đọc nó bằng bảng mã mặc định của hệ điều
hành (cp1252 trên Windows), một chữ có dấu là sập. Hai migration viết tay, không autogenerate,
để khớp từng ràng buộc `CHECK` và chỉ mục một phần. `tests/test_db.py` kiểm
`upgrade head → downgrade base → upgrade head` cho lược đồ cuối giống lược đồ đầu (TC-47).

```bash
alembic upgrade head        # chạy tự động lúc container api khởi động
alembic revision --autogenerate -m "mô tả"
alembic downgrade -1
```

Container `api` chạy `alembic upgrade head` trong entrypoint trước khi khởi động
uvicorn. Lệnh này là idempotent — chạy lại khi lược đồ đã mới nhất thì không làm gì.

## 6. Cấu trúc hiện vật

Mọi tệp hiện vật do `notebooks/08_export_artifacts.ipynb` sinh ra. `src/artifacts.py` là bản thi hành
của mục này: các hàm `*_payload` dựng nội dung, các hàm `validate_*` kiểm tra ngược lại cấu trúc.
Notebook gọi chúng trước khi ghi, `tests/test_artifacts.py` gọi sau khi ghi, và API gọi lúc khởi
động (T-44). Ví dụ dưới đây là **số thật** của lần xuất `xgb_scaleposweight_v1` (2026-09-27).

Quy ước chung:

- JSON **chặt**: không có `NaN` hay `Infinity` (giá trị không xác định ghi thành `null`), vì
  `JSON.parse` của trình duyệt từ chối hai giá trị đó.
- Ghi **nguyên tử**: ghi ra `*.tmp` rồi đổi tên, nên API không bao giờ đọc phải tệp ghi dở.
- Số thực giữ nguyên độ chính xác float64. Việc làm tròn 4 chữ số là của tầng API khi trả phản hồi
  ([05 §5](05-thiet-ke-api.md)), không phải của hiện vật.

### 6.1 `models/threshold.json`

```json
{
  "default_threshold": 0.023172983899712563,
  "selection_method": "min_expected_cost",
  "selected_on": "out_of_fold",
  "cost_false_negative": 122.21,
  "cost_false_positive": 5.0,
  "alternatives": {
    "min_expected_cost": 0.023172983899712563,
    "max_f1": 0.720512330532074,
    "recall_at_least_90": 0.0005369935533963144,
    "budget_200_alerts": 0.9624789953231812,
    "default_naive": 0.5
  },
  "model_version": "xgb_scaleposweight_v1",
  "trained_at": "2026-09-27T08:14:32Z",
  "constraints": { "min_recall": 0.9, "alert_budget_per_day": 200.0 }
}
```

`default_threshold` luôn bằng `alternatives[selection_method]`. Mọi ngưỡng được chọn trên điểm
out-of-fold của tập huấn luyện (`selected_on`, ML-08), không trên tập kiểm thử.

### 6.2 `models/metrics.json`

Mười hai khóa bắt buộc, theo đúng thứ tự:

```json
{
  "model_version": "xgb_scaleposweight_v1",
  "trained_at": "2026-09-27T08:14:32Z",
  "dataset": {
    "n_total": 283726, "n_duplicates_removed": 1081,
    "n_train": 226980, "n_fraud_train": 378,
    "n_test": 56746,   "n_fraud_test": 95,
    "positive_rate": 0.001674, "test_fraction": 0.2000, "days": 2.0,
    "split": "phân tầng 80/20 theo Class, random_state=42, sau khi loại trùng lặp",
    "test_scores_order": "dòng của data/test_set.parquet (Time tăng dần)"
  },
  "headline": {
    "pr_auc":    { "value": 0.8252, "ci_low": 0.7470, "ci_high": 0.8960 },
    "roc_auc":   { "value": 0.9773, "ci_low": 0.9609, "ci_high": 0.9912 },
    "recall":    { "value": 0.8105, "ci_low": 0.7368, "ci_high": 0.8842 },
    "precision": { "value": 0.6696, "ci_low": 0.6000, "ci_high": 0.7522 },
    "f1":        { "value": 0.7333 },
    "baseline_pr_auc": 0.001674,
    "n_boot": 1000
  },
  "confusion_at_default": { "threshold": 0.023173, "tp": 77, "fp": 38, "fn": 18, "tn": 56613 },
  "pr_curve":   { "recall": [], "precision": [], "thresholds": [] },
  "roc_curve":  { "fpr": [], "tpr": [] },
  "cost_curve": [ { "threshold": 1.67e-08, "cost": 283255.0, "alerts": 56746, "alerts_per_day": 141863.0,
                    "tp": 95, "fp": 56651, "fn": 0, "precision": 0.0017, "recall": 1.0 } ],
  "test_scores": { "y_true": [], "y_score": [] },
  "grid_results": [
    { "model": "xgboost", "strategy": "class_weight",
      "pr_auc_mean": 0.8549, "pr_auc_std": 0.0301, "roc_auc": 0.9824, "roc_auc_std": 0.0081,
      "threshold": 0.0232, "recall": 0.852, "precision": 0.702, "f1": 0.769,
      "tp": 322, "fp": 137, "fn": 56, "alerts_per_day": 286.9, "expected_cost": 7528.76,
      "n_distinct_scores": 221674, "seconds_per_fold": 7.2, "fit_seconds": 36.1 }
  ],
  "shap_global": [
    { "feature": "V14", "mean_abs_shap": 2.679, "share": 0.158, "rank_shap": 1,
      "mean_abs_shap_fraud": 3.620, "mean_shap_fraud": 2.776,
      "rank_cohens_d": 2, "rank_cliffs_delta": 1, "rank_pearson": 2,
      "abs_cohens_d": 7.52, "abs_cliffs_delta": 0.894, "lr_std_coef": -1.49, "rank_lr": 3 }
  ],
  "split_comparison": {
    "random_stratified":     { "pr_auc": 0.8252, "pr_auc_ci_low": 0.7470, "pr_auc_ci_high": 0.8960, "fp": 38 },
    "random_downsized_day1": { "pr_auc": 0.8249, "fp": 16 },
    "temporal":              { "pr_auc": 0.7816, "pr_auc_ci_low": 0.7280, "pr_auc_ci_high": 0.8359, "fp": 338 }
  }
}
```

Ghi chú về từng khóa:

- `headline` — chỉ số trên tập kiểm thử tại `default_threshold`, bootstrap phân tầng 1.000 lần.
  Tính trên **thứ tự dòng của `split_data()`** với cùng hạt giống như notebook 05, nên cả giá trị lẫn
  hai đầu khoảng tin cậy trùng từng chữ số với `reports/final_test_metrics.csv` và với báo cáo.
- `confusion_at_default` mang theo ngưỡng mà nó được tính.
- `pr_curve`, `roc_curve` — tối đa 500 điểm, rải đều theo **chiều dài** đường cong
  (`src.evaluate.thin_curve`); `strategy_pr_curves` cũng vậy, 300 điểm mỗi đường. Bản trước lấy đều
  theo chỉ số ngưỡng: vì gần hết ngưỡng nằm ở vùng điểm thấp, cả vùng precision cao chỉ còn hai điểm
  và UI-04 vẽ đường PR thành một đoạn thẳng (phát hiện ở giai đoạn 8, kiểm bằng
  `test_t36_pr_curves_show_the_high_precision_region`).
- `cost_curve` — khoảng 200 điểm trên cùng lưới ngưỡng mà API-12 dùng (`threshold_grid`), tính trên
  tập kiểm thử với chi phí mặc định.
- `grid_results` — 20 dòng của `reports/grid_results.csv`, sắp theo `pr_auc_mean` giảm dần.
- `shap_global` — 31 đặc trưng, sắp theo `mean_abs_shap` giảm dần (log-odds, toàn tập kiểm thử), kèm
  thứ hạng thống kê của T-15 để UI-04 đặt hai bảng cạnh nhau. Hai đặc trưng không có trong T-15
  (`hour_sin`, `hour_cos`) có các cột thống kê bằng `null`. `Amount` ở đây là giá trị **sau**
  `RobustScaler`.
- `split_comparison` — ba cách chia của T-27, gồm dòng đối chứng `random_downsized_day1`.

Sáu khóa bổ sung, đứng sau các khóa bắt buộc, phục vụ UI-04 ([07 §6](07-thiet-ke-giao-dien.md)) và
tầng API:

| Khóa | Nội dung | Phục vụ |
|---|---|---|
| `strategy_pr_curves` | `{model, evaluated_on: "out_of_fold", baseline, curves: [{strategy, pr_auc_mean, pr_auc_std, recall[], precision[], thresholds[]}]}` — XGBoost dưới 5 chiến lược, 300 điểm mỗi đường. `null` nếu thiếu `reports/grid_results.npz` | UI-04 mục 2 |
| `baseline_comparison` | accuracy, PR-AUC kèm KTC, ROC-AUC, TP/FP/FN/TN của mô hình rỗng, hồi quy logistic, cây quyết định và mô hình xuất | UI-04 mục 4, G-2 |
| `threshold_options` | 5 phương án của `threshold.json` đo trên tập kiểm thử: TP/FP/FN/TN, precision, recall, chi phí, cảnh báo/ngày | UI-03, bảng T-30 |
| `training` | tham số XGBoost, thời gian huấn luyện, `feature_order` (31 cột vào pipeline), `model_feature_names` (thứ tự sau tiền xử lý — tên gán cho giá trị SHAP), `shap_base_value`, `shap_output: "log-odds"` | API `/explain` |
| `fingerprint` | `test_scores_sha256`, `oof_scores_sha256` — 16 ký tự đầu SHA-256 của mảng điểm | T-38: hai lần chạy cùng dấu vân tay nghĩa là điểm trùng từng bit |
| `environment` | phiên bản Python, `numpy`, `pandas`, `scikit-learn`, `imbalanced-learn`, `xgboost`, `shap`, `joblib`; số lõi CPU | nạp `.joblib` trên máy khác; XGBoost lệch nhẹ theo số luồng |

Trường `test_scores` là thứ làm API-11 và UI-D1 nhanh: hai mảng 56.746 phần tử **theo thứ tự dòng
của `data/test_set.parquet`**, nạp một lần vào bộ nhớ lúc khởi động. `y_score` giữ nguyên độ chính
xác float64: UI-D1 so từng điểm với ngưỡng ngay trong trình duyệt, làm tròn là lệch TP/FP/FN ở sát
ngưỡng (TC-12). Chúng **không** nằm trong cơ sở dữ liệu — đây là dữ liệu của mô hình, không phải
trạng thái ứng dụng (AR-01).

Kích thước đo được: **1,6 MB**, trong đó `test_scores` chiếm 1,46 MB. Còn xa mức cần tách
`test_scores` ra `models/test_scores.npz`, nên chưa tách.

### 6.3 `data/sample_pool.json`

```json
{
  "generated_at": "2026-09-27T08:14:32Z",
  "model_version": "xgb_scaleposweight_v1",
  "reference_threshold": 0.023172983899712563,
  "fraud_hard_cutoff": 0.5,
  "source": "data/test_set.parquet",
  "categories": {
    "fraud_easy": { "count": 29,  "rule": "gian lận, điểm ≥ 0,5" },
    "fraud_hard": { "count": 21,  "rule": "gian lận, điểm < 0,5 (ngưỡng mặc định bỏ lọt)" },
    "legit_easy": { "count": 112, "rule": "hợp lệ, điểm < τ* = 0,02317" },
    "legit_hard": { "count": 38,  "rule": "hợp lệ, điểm ≥ τ* = 0,02317 (cảnh báo giả)" }
  },
  "items": [
    {
      "id": "S-001",
      "category": "fraud_easy",
      "label": 1,
      "risk_score": 0.9999994039535522,
      "description": "Gian lận điển hình — điểm rủi ro rất cao",
      "test_row": 44742,
      "features": { "Time": 143456.0, "V1": -2.0066, "…": "…", "Amount": 1.0 }
    }
  ]
}
```

- **Nguồn là tập kiểm thử**, không phải tập huấn luyện: điểm của giao dịch mà mô hình đã học thuộc
  đẹp một cách giả tạo. `test_row` là vị trí dòng trong `test_set.parquet`.
- `features` có đúng 30 cột thô theo thứ tự `Time, V1…V28, Amount` — đầu vào của `POST /score`.
  Chấm lại các cột này bằng `model.joblib` ra đúng `risk_score` đã lưu.
- **Luật chia nhóm**: `fraud_hard` là gian lận có điểm dưới 0,5 (mô hình không tự tin, ngưỡng mặc
  định bỏ lọt); `legit_hard` là hợp lệ có điểm từ τ\* trở lên (cảnh báo giả ở chính sách đang chạy).
  Hai nhóm dùng hai mốc khác nhau vì chúng khó theo hai nghĩa khác nhau. Dùng τ\* cho cả hai thì
  `fraud_hard` chỉ còn 18 mẫu.
- Mỗi nhóm lấy **trải đều theo thứ hạng điểm**, không ngẫu nhiên: chạy lại cho đúng cùng thư viện.
  `legit_hard` lấy tối đa 40, hai nhóm "easy" lấp cho đủ 50 gian lận và 150 hợp lệ.
- `description` và `rule` viết số theo tiếng Việt (dấu phẩy thập phân), vì giao diện hiện nguyên văn.
- Mỗi nhóm "hard" phải có ít nhất 20 mẫu (T-37). Lần xuất này `fraud_hard` có 21 — tức **toàn bộ**
  gian lận dưới 0,5 của tập kiểm thử, biên rất mỏng. `validate_sample_pool` chặn việc xuất nếu một
  lần huấn luyện lại làm nhóm này tụt dưới 20.

### 6.4 `models/model.joblib`, `models/explainer.joblib`, `models/oof_scores.npz`, `data/test_set.parquet`

| Tệp | Nội dung | Cách dùng |
|---|---|---|
| `model.joblib` (1,2 MB) | `imblearn.pipeline.Pipeline` hai bước: `preprocess` (`ColumnTransformer`, `RobustScaler` cho `Amount`) → `clf` (`XGBClassifier`) | `model.predict_proba(build_features(df))[:, 1]`. `feature_names_in_` bằng `FEATURE_ORDER` |
| `explainer.joblib` (4,2 MB) | `shap.TreeExplainer` của **riêng** bước `clf` | đầu vào là `model[:-1].transform(build_features(df))`, **không** phải `build_features(df)`. Giá trị SHAP ở thang log-odds; `expected_value + Σ SHAP` bằng margin của booster |
| `oof_scores.npz` (0,8 MB) | `y_true` (int8), `y_score` (float32, giữ nguyên kiểu của `predict_proba`), `train_fraction`, `days` — 226.980 điểm out-of-fold của tập huấn luyện | `POST /threshold/optimize` chọn ngưỡng trên đây (ML-08); với chi phí mặc định ra đúng τ\*. Dấu vân tay khớp `metrics.json → fingerprint.oof_scores_sha256` (thêm ở giai đoạn 7) |
| `test_set.parquet` (15,9 MB) | 32 cột: 30 cột thô, `Class`, `risk_score` (do notebook 08 thêm) | phát lại (UI-05) theo thứ tự `Time`; `risk_score` trùng từng bit với `metrics.json → test_scores.y_score` |

Hai tệp `.joblib` là pickle: nạp được khi phiên bản `scikit-learn`, `imbalanced-learn`, `xgboost`,
`shap` khớp với `metrics.json → environment.packages`. Lệch phiên bản thì chạy lại notebook 08
thay vì cố nạp.

## 7. Vận hành cơ sở dữ liệu

### 7.1 Dịch vụ trong Docker Compose

```yaml
services:
  db:
    image: postgres:16-alpine
    environment:
      POSTGRES_USER:     fraud
      POSTGRES_PASSWORD: fraud
      POSTGRES_DB:       fraud
      TZ:                UTC
    volumes:
      - pgdata:/var/lib/postgresql/data
    ports:
      - "5432:5432"          # để nối bằng psql hoặc DBeaver khi phát triển
    healthcheck:
      test:     ["CMD-SHELL", "pg_isready -U fraud -d fraud"]
      interval: 5s
      timeout:  3s
      retries:  10

  api:
    build: ./api
    depends_on:
      db:
        condition: service_healthy
    environment:
      DATABASE_URL: postgresql+psycopg://fraud:fraud@db:5432/fraud
    volumes:
      - ./models:/app/models:ro
      - ./data:/app/data:ro
    ports:
      - "8000:8000"

volumes:
  pgdata:
```

Dữ liệu nằm trong **volume có tên** `pgdata`, không phải thư mục `data/` của repo —
nhờ vậy không lẫn với hiện vật mô hình và không bao giờ lọt vào git.

Mật khẩu `fraud/fraud` chấp nhận được vì đây là demo cục bộ không mở ra mạng
(xem phạm vi ở [00 §3.2](00-tong-quan.md)). Nếu về sau triển khai thật, chuyển sang
biến môi trường trong tệp `.env` không commit.

### 7.2 Thời gian khởi động

| Lần chạy | Thời gian tới khi `/health` trả 200 |
|---|---|
| Lần đầu (initdb + migration) | 20–40 giây |
| Các lần sau (volume đã có dữ liệu) | 8–12 giây |

NFR-04 (< 15 giây) áp dụng cho lần chạy thứ hai trở đi. Ghi rõ điều này khi
nghiệm thu AC-A8, và khi trình diễn thì khởi động trước một lần.

### 7.3 Đặt lại trạng thái demo

```bash
# Xóa sạch dữ liệu ứng dụng, giữ lược đồ
docker compose exec db psql -U fraud -d fraud \
  -c "TRUNCATE transactions, reviews RESTART IDENTITY CASCADE;"

# Xóa sạch mọi thứ kể cả lược đồ — lần up sau sẽ initdb và migrate lại
docker compose down -v
```

Lệnh thứ nhất giữ nguyên bảng `settings`, tức giữ ngưỡng người dùng đã chọn. Muốn
xóa cả ngưỡng thì thêm `settings` vào danh sách `TRUNCATE`.

### 7.4 Sao lưu và khôi phục

```bash
docker compose exec db pg_dump -U fraud -d fraud -Fc > backup/fraud.dump
docker compose exec -T db pg_restore -U fraud -d fraud --clean < backup/fraud.dump
```

Đáng làm một lần trước buổi bảo vệ: chuẩn bị sẵn một bản dump có dữ liệu demo đẹp
(đã tải CSV, đã thẩm định vài giao dịch) để khôi phục trong vài giây nếu có sự cố.

### 7.5 Bảo trì

Với quy mô dưới một triệu dòng, autovacuum mặc định là đủ, không cần chỉnh gì. Nếu
chạy phát lại nhiều lần rồi `TRUNCATE` liên tục, thỉnh thoảng chạy:

```sql
VACUUM ANALYZE transactions;
```

`ANALYZE` quan trọng hơn `VACUUM` ở đây: sau khi nạp lô lớn, thống kê của bộ tối ưu
truy vấn còn cũ và nó có thể bỏ qua `idx_tx_risk`, khiến hàng đợi tải chậm hẳn.

## 8. Vòng đời dữ liệu

| Sự kiện | Tác động |
|---|---|
| Chạy lại notebook 08 | Ghi đè `models/*`; cơ sở dữ liệu giữ nguyên; giao dịch cũ mang `model_version` cũ |
| Khởi động API | Chờ `db` khỏe → `alembic upgrade head` → nạp hiện vật vào bộ nhớ → chèn `settings` còn thiếu |
| Tải CSV lên | `COPY` vào `transactions` với `source='upload'` và cùng một `batch_id` |
| Đổi ngưỡng | `UPSERT` vào `settings`; không đụng tới `transactions` |
| Thẩm định | `UPSERT` một dòng `reviews` |
| `docker compose down` | Dữ liệu còn nguyên trong volume `pgdata` |
| `docker compose down -v` | Xóa volume — mất toàn bộ lịch sử, lần sau tạo lại từ đầu |

Giao dịch mang `model_version` khác phiên bản đang chạy vẫn hiển thị được, nhưng
giao diện gắn nhãn cảnh báo — điểm rủi ro của chúng không so sánh trực tiếp được
với điểm hiện hành.

## 9. Ước lượng kích thước

| Đối tượng | Ước lượng |
|---|---|
| Một dòng `transactions` | Khoảng 700 byte (chủ yếu là `JSONB` 28 số kèm tên khóa) |
| 57.000 giao dịch (cả tập kiểm thử) | Khoảng 40 MB dữ liệu + 6 MB chỉ mục |
| Bộ nhớ container `db` | 150–250 MB với cấu hình mặc định |
| `model.joblib` (XGBoost 500 cây) | **1,2 MB** (đo thực tế; ước lượng ban đầu 5–15 MB) |
| `explainer.joblib` | **4,2 MB** (đo thực tế) |
| `metrics.json` kèm `test_scores` | **1,6 MB** (đo thực tế; 1,46 MB trong đó là `test_scores`) |
| `data/sample_pool.json` | **0,2 MB** (200 mẫu × 30 đặc trưng) |
| `test_set.parquet` | **15,9 MB** sau khi notebook 08 thêm cột `risk_score` (15,4 MB khi notebook 03 ghi ra; V1–V28 là float64 sau PCA nên nén gần như không ăn thua — snappy 15,4 MB, brotli 14,1 MB) |

NFR-05 (< 1 GB) đặt cho tiến trình API. Container `db` được tính riêng và ở mức
khoảng 250 MB — nêu rõ khi báo cáo kết quả đo.

## 10. Sao lưu và git

| Đường dẫn | Trong git? | Lý do |
|---|---|---|
| `data/creditcard.csv` | Không | 144 MB, tải lại từ Kaggle được |
| Volume `pgdata` | Không | Nằm ngoài cây thư mục repo |
| `backup/*.dump` | Không | Có thể chứa dữ liệu lô lớn |
| `.env` | Không | Chứa thông tin kết nối |
| `data/test_set.parquet` | Không | Sinh lại từ notebook |
| `models/*.joblib`, `models/*.npz` | Không | Sinh lại từ notebook |
| `models/threshold.json` | Nên có | Dưới 1 KB, đọc được bằng mắt — lịch sử ngưỡng đã xuất theo commit |
| `models/metrics.json` | Có cân nhắc | 1,6 MB trên **một dòng** (JSON gọn), nên diff không đọc được; giữ lại thì có lịch sử kết quả theo commit |
| `api/migrations/versions/*.py` | **Có** | Lược đồ là mã nguồn — không có chúng thì không dựng lại được |
| `reports/figures/*.png` | Có | Cần cho báo cáo, dung lượng nhỏ |
| `docs/`, `src/`, `api/`, `web/`, `tests/` | Có | Mã nguồn và tài liệu |

Cần bổ sung vào `.gitignore`: `data/*.parquet`, `data/sample_pool.json`, `.env`,
`backup/`.
