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
| `threshold` | number | 0.0473 | `threshold.json` |
| `cost_fn` | number | 122.21 | `threshold.json` |
| `cost_fp` | number | 5.00 | `threshold.json` |
| `replay_speed` | number | 60 | Hằng số |

Lúc khởi động, API chèn khóa còn thiếu bằng `INSERT … ON CONFLICT (key) DO NOTHING`
với giá trị lấy từ `threshold.json`. Nhờ vậy huấn luyện lại mô hình không ghi đè
lựa chọn mà người dùng đã đặt.

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

Phân vị của `Amount` cho UI-02 (US-04):

```sql
SELECT percent_rank() OVER (ORDER BY amount) AS pct
FROM   transactions
WHERE  id = :tx_id;
```

Đếm tổng cho phân trang: dùng `COUNT(*) OVER ()` trong cùng truy vấn thay vì gọi
`COUNT` riêng — tiết kiệm một lượt quét.

## 5. Migration

Dùng **Alembic**. Lược đồ không được tạo bằng `Base.metadata.create_all()` lúc
chạy, vì như vậy không có lịch sử thay đổi và không lặp lại được trên máy khác.

```
api/migrations/
├── env.py
└── versions/
    ├── 0001_initial_schema.py      # ba bảng, chỉ mục, ràng buộc
    └── 0002_seed_settings.py       # nạp bốn khóa mặc định
```

```bash
alembic upgrade head        # chạy tự động lúc container api khởi động
alembic revision --autogenerate -m "mô tả"
alembic downgrade -1
```

Container `api` chạy `alembic upgrade head` trong entrypoint trước khi khởi động
uvicorn. Lệnh này là idempotent — chạy lại khi lược đồ đã mới nhất thì không làm gì.

## 6. Cấu trúc hiện vật

### 6.1 `models/threshold.json`

```json
{
  "default_threshold": 0.0473,
  "selection_method": "min_expected_cost",
  "cost_false_negative": 122.21,
  "cost_false_positive": 5.00,
  "alternatives": {
    "max_f1": 0.2140,
    "recall_at_least_90": 0.0089,
    "budget_200_alerts": 0.0612,
    "default_naive": 0.5000
  },
  "model_version": "xgb_scaleposweight_v3",
  "trained_at": "2026-09-15T10:22:41Z"
}
```

### 6.2 `models/metrics.json`

```json
{
  "model_version": "xgb_scaleposweight_v3",
  "trained_at": "2026-09-15T10:22:41Z",
  "dataset": {
    "n_total": 283726,
    "n_duplicates_removed": 1081,
    "n_train": 226980,
    "n_test": 56746,
    "n_fraud_test": 98,
    "positive_rate": 0.00173
  },
  "headline": {
    "pr_auc":    { "value": 0.8431, "ci_low": 0.7812, "ci_high": 0.8944 },
    "roc_auc":   { "value": 0.9782, "ci_low": 0.9611, "ci_high": 0.9893 },
    "recall":    { "value": 0.8265, "ci_low": 0.7449, "ci_high": 0.8980 },
    "precision": { "value": 0.3785, "ci_low": 0.3021, "ci_high": 0.4602 },
    "f1":        { "value": 0.5192 },
    "baseline_pr_auc": 0.00173
  },
  "confusion_at_default": { "tp": 81, "fp": 133, "fn": 17, "tn": 56515 },
  "pr_curve":   { "recall": [], "precision": [], "thresholds": [] },
  "roc_curve":  { "fpr": [], "tpr": [] },
  "cost_curve": [ { "threshold": 0.001, "cost": 4210.5, "alerts": 980 } ],
  "test_scores": { "y_true": [], "y_score": [] },
  "grid_results": [
    { "model": "XGBoost", "strategy": "scale_pos_weight",
      "pr_auc_mean": 0.8402, "pr_auc_std": 0.0211,
      "roc_auc": 0.9771, "recall": 0.83, "precision": 0.38,
      "fit_seconds": 41.2 }
  ],
  "shap_global": [ { "feature": "V14", "mean_abs_shap": 0.412 } ],
  "split_comparison": {
    "random_stratified": { "pr_auc": 0.8431 },
    "temporal":          { "pr_auc": 0.7914 }
  }
}
```

Trường `test_scores` là thứ làm API-11 và UI-D1 nhanh: hai mảng khoảng 57.000 phần
tử, chiếm vài MB, nạp một lần vào bộ nhớ lúc khởi động. Chúng **không** nằm trong
cơ sở dữ liệu — đây là dữ liệu của mô hình, không phải trạng thái ứng dụng (AR-01).

Nếu kích thước tệp thành vấn đề, tách `test_scores` ra `models/test_scores.npz` và
giữ `metrics.json` nhẹ để đọc bằng mắt.

### 6.3 `data/sample_pool.json`

```json
{
  "generated_at": "2026-09-15T10:25:00Z",
  "items": [
    {
      "id": "S-001",
      "label": 1,
      "risk_score": 0.9142,
      "category": "fraud_easy",
      "description": "Gian lận điển hình — điểm rủi ro rất cao",
      "features": { "Time": 41505.0, "V1": -3.04, "Amount": 1.0 }
    }
  ]
}
```

Bốn `category`: `fraud_easy`, `fraud_hard`, `legit_easy`, `legit_hard`. Tỷ lệ
khoảng 50 gian lận / 150 hợp lệ, trong đó mỗi nhóm "hard" chiếm ít nhất 20 mẫu.

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
| `model.joblib` (XGBoost 500 cây) | 5–15 MB |
| `explainer.joblib` | 1–5 MB |
| `metrics.json` kèm `test_scores` | 3–8 MB |
| `test_set.parquet` | **15,4 MB** (đo thực tế; V1–V28 là float64 sau PCA nên nén gần như không ăn thua — snappy 15,4 MB, brotli 14,1 MB) |

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
| `models/*.joblib` | Không | Sinh lại từ notebook |
| `models/metrics.json` | Có cân nhắc | Nhẹ nếu tách `test_scores`; giữ lại thì có lịch sử kết quả theo commit |
| `api/migrations/versions/*.py` | **Có** | Lược đồ là mã nguồn — không có chúng thì không dựng lại được |
| `reports/figures/*.png` | Có | Cần cho báo cáo, dung lượng nhỏ |
| `docs/`, `src/`, `api/`, `web/`, `tests/` | Có | Mã nguồn và tài liệu |

Cần bổ sung vào `.gitignore`: `data/*.parquet`, `data/sample_pool.json`, `.env`,
`backup/`.
