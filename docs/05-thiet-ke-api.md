# 05 — Thiết kế API

Máy chủ FastAPI, tiền tố `/api/v1`, trao đổi JSON. Tài liệu tự sinh tại `/docs`
(Swagger) và `/redoc`.

Tài liệu này là **hợp đồng** giữa backend và frontend. Khi có tranh cãi về hành vi,
`api/schemas.py` là bản thi hành của hợp đồng này.

## 1. Bảng endpoint

| Mã | Phương thức | Đường dẫn | Chức năng | Yêu cầu |
|---|---|---|---|---|
| API-01 | GET | `/health` | Tình trạng và phiên bản mô hình | NFR-04 |
| API-02 | POST | `/score` | Chấm điểm một giao dịch | FR-20 |
| API-03 | POST | `/score/batch` | Chấm điểm nhiều giao dịch | FR-21 |
| API-04 | POST | `/score/upload` | Nhận CSV, chấm điểm, lưu | FR-22 |
| API-05 | POST | `/explain` | Giá trị SHAP cho một giao dịch | FR-24 |
| API-06 | GET | `/transactions` | Danh sách có lọc và phân trang | FR-40 |
| API-07 | GET | `/transactions/{id}` | Chi tiết một giao dịch | FR-40 |
| API-08 | POST | `/reviews` | Ghi kết quả thẩm định | FR-41 |
| API-09 | GET | `/threshold` | Ngưỡng hiện hành và các phương án | FR-30, FR-34 |
| API-10 | PUT | `/threshold` | Đặt ngưỡng mới | FR-30 |
| API-11 | GET | `/threshold/preview` | Chỉ số ứng với một ngưỡng giả định | FR-31 |
| API-12 | POST | `/threshold/optimize` | Ngưỡng tối ưu theo tham số chi phí | FR-32, FR-33 |
| API-13 | GET | `/metrics` | Toàn bộ chỉ số đánh giá | FR-42 |
| API-14 | GET | `/samples` | Thư viện giao dịch mẫu | FR-25 |
| API-15 | GET | `/replay/stream` | Server-Sent Events cho phát lại | FR-43 |

## 2. Kiểu dữ liệu dùng chung

```
TransactionInput = {
  "Time": float, "V1": float, ..., "V28": float, "Amount": float
}                                     // 30 trường, xem DS-10..DS-14

RiskBand = "low" | "medium" | "high" | "critical"
Decision = "allow" | "review" | "block"
```

Quy tắc suy ra `risk_band` từ `risk_score`, ngưỡng hiện hành τ và **ngưỡng đề xuất chặn** τ_chặn
(`B = max(τ, τ_chặn)`):

| Dải | Điều kiện |
|---|---|
| `low` | `score < τ/2` |
| `medium` | `τ/2 ≤ score < τ` |
| `high` | `τ ≤ score < B` |
| `critical` | `score ≥ B` |

**τ_chặn** là ngưỡng nhỏ nhất mà **mọi** ngưỡng từ đó trở lên có precision ≥ 95% trên điểm
out-of-fold (`BLOCK_MIN_PRECISION` trong `api/config.py`, `pick_threshold(..., "min_precision")`).
API tính lúc nạp hiện vật; nó thuộc về mô hình, không đổi khi người dùng đặt τ. Mô hình hiện tại:
τ_chặn ≈ 0,9735. Lấy `max(τ, τ_chặn)` để khi người dùng nâng τ vượt mức chặn thì mọi cảnh báo đều là
`block`, không có dải `review` âm.

Thiết kế cũ đặt biên ở `3τ`. Bội số 3 không dựa trên phân tích nào, và đo trên tập kiểm thử tại
τ\* thì sai hẳn: dải `review` (τ…3τ) chứa 19 giao dịch, **không vụ gian lận nào**; dải `block` chặn
tự động 96 giao dịch, trong đó **19 khách hợp lệ** — một nửa số cảnh báo giả. Với τ_chặn theo
precision: `block` 70 giao dịch, 69 gian lận, 1 hợp lệ; `review` 45 giao dịch, trong đó 8 vụ gian
lận để người thẩm định bắt (notebook 06 §6.2, `reports/block_threshold.csv`).

Dải rủi ro là **dẫn xuất**, tính lại mỗi lần trả kết quả — vì τ thay đổi được
(AR-03). Không lưu vào cơ sở dữ liệu.

`decision` suy ra cùng cách, trùng với dải:

| `decision` | Điều kiện | Dải tương ứng |
|---|---|---|
| `allow` | `score < τ` | `low`, `medium` |
| `review` | `τ ≤ score < B` | `high` |
| `block` | `score ≥ B` | `critical` |

`GET /threshold` (API-09) trả `block_threshold` (τ_chặn) và `block_min_precision` để giao diện
giải thích được vì sao một giao dịch là "đề xuất chặn".

## 3. Chi tiết endpoint

### API-01 — `GET /api/v1/health`

```json
{
  "status": "ok",
  "model_version": "xgb_scaleposweight_v3",
  "model_loaded_at": "2026-09-20T09:00:03Z",
  "threshold": 0.0473,
  "db": "ok",
  "uptime_seconds": 1284
}
```

Trường `db` trả `ok` khi truy vấn thăm dò `SELECT 1` thành công, `down` khi không
kết nối được PostgreSQL.

Trả `503` nếu chưa nạp được mô hình **hoặc** không nối được cơ sở dữ liệu. Dùng
làm healthcheck cho dịch vụ `api` trong Docker Compose (dịch vụ `db` có
healthcheck riêng bằng `pg_isready`).

### API-02 — `POST /api/v1/score`

Yêu cầu:

```json
{
  "transaction": {
    "Time": 7834.0,
    "V1": -1.3598, "V2": -0.0728, "V3": 2.5363, "V4": 1.3782,
    "V5": -0.3383, "V6": 0.4624,  "V7": 0.2396, "V8": 0.0987,
    "V9": 0.3638,  "V10": 0.0908, "V11": -0.5516, "V12": -0.6178,
    "V13": -0.9914,"V14": -0.3112,"V15": 1.4682, "V16": -0.4704,
    "V17": 0.2080, "V18": 0.0258, "V19": 0.4040, "V20": 0.2514,
    "V21": -0.0183,"V22": 0.2778, "V23": -0.1105,"V24": 0.0669,
    "V25": 0.1285, "V26": -0.1891,"V27": 0.1336, "V28": -0.0211,
    "Amount": 149.62
  },
  "persist": true
}
```

`persist` mặc định `true`; đặt `false` cho lần thử nhanh không muốn ghi vào cơ sở
dữ liệu.

Phản hồi `200`:

```json
{
  "transaction_id": "TX-8841",
  "risk_score": 0.0037,
  "threshold": 0.0473,
  "decision": "allow",
  "risk_band": "low",
  "model_version": "xgb_scaleposweight_v3",
  "scored_at": "2026-09-20T14:03:11Z",
  "latency_ms": 4
}
```

### API-03 — `POST /api/v1/score/batch`

```json
{ "transactions": [ TransactionInput, ... ], "batch_id": "b-2026-09-20-01" }
```

Tối đa 50.000 phần tử (DS-15). Phản hồi:

```json
{
  "batch_id": "b-2026-09-20-01",
  "count": 10000,
  "alerts": 187,
  "threshold": 0.0473,
  "score_distribution": { "low": 9502, "medium": 311, "high": 151, "critical": 36 },
  "elapsed_ms": 2140,
  "results": [ { "transaction_id": "TX-1", "risk_score": 0.0012 }, ... ]
}
```

Hiệu năng: một lời gọi `predict_proba` duy nhất cho cả DataFrame, không lặp từng
dòng; ghi xuống cơ sở dữ liệu bằng `COPY` của PostgreSQL trong **một** giao dịch
([06 §4.3](06-thiet-ke-luu-tru.md)). Hai điều này cùng nhau là điều kiện để đạt
NFR-02.

### API-04 — `POST /api/v1/score/upload`

`multipart/form-data`, trường `file` là CSV có đủ 30 cột bắt buộc; cột `Class`
nếu có sẽ được lưu vào `true_label` và dùng để tính chỉ số thực tế.

Giới hạn 100 MB (NFR-06). Phản hồi giống API-03, bổ sung:

```json
{
  "rows_read": 10000,
  "rows_rejected": 3,
  "rejection_reasons": [
    { "row": 42, "reason": "Amount âm" },
    { "row": 77, "reason": "V13 không phải số" }
  ],
  "has_labels": true,
  "actual_metrics": { "precision": 0.41, "recall": 0.82, "pr_auc": 0.84 }
}
```

Dòng lỗi bị bỏ qua, không làm hỏng cả lô — nhưng phải báo cáo lại đầy đủ. Các
dòng hợp lệ được ghi trong một giao dịch duy nhất: hoặc vào hết, hoặc không dòng
nào ([06 §4.2](06-thiet-ke-luu-tru.md)).

### API-05 — `POST /api/v1/explain`

```json
{ "transaction_id": "TX-8841" }
```

hoặc gửi trực tiếp `{ "transaction": TransactionInput }` cho giao dịch chưa lưu.

Phản hồi:

```json
{
  "transaction_id": "TX-8841",
  "risk_score": 0.8312,
  "base_value": 0.0021,
  "top_positive": [
    { "feature": "V14", "value": -8.42, "shap": 0.312 },
    { "feature": "V4",  "value": 4.19,  "shap": 0.201 },
    { "feature": "V12", "value": -6.03, "shap": 0.157 },
    { "feature": "V10", "value": -5.11, "shap": 0.094 },
    { "feature": "amount_scaled", "value": 3.2, "shap": 0.041 }
  ],
  "top_negative": [
    { "feature": "V7",  "value": 0.88, "shap": -0.052 },
    { "feature": "hour_cos", "value": 0.71, "shap": -0.017 },
    { "feature": "V21", "value": -0.19, "shap": -0.008 }
  ],
  "elapsed_ms": 128
}
```

Tổng `base_value` cộng toàn bộ giá trị SHAP bằng log-odds của dự đoán — giao diện
dùng tính chất này để vẽ biểu đồ thác nước.

Endpoint tách riêng vì tốn 50–200 ms mỗi mẫu (NFR-01). Không bao giờ gọi cho cả
danh sách.

### API-06 — `GET /api/v1/transactions`

Tham số truy vấn:

| Tham số | Kiểu | Mặc định | Ý nghĩa |
|---|---|---|---|
| `min_score` | float | ngưỡng hiện hành | Lọc theo điểm rủi ro |
| `max_score` | float | 1.0 | |
| `band` | RiskBand | — | Lọc theo dải |
| `reviewed` | bool | — | Đã thẩm định hay chưa |
| `batch_id` | string | — | Lọc theo lô |
| `sort` | string | `-risk_score` | Trường sắp xếp, tiền tố `-` là giảm dần |
| `page` | int | 1 | |
| `page_size` | int | 50 | Tối đa 200 |

Phản hồi:

```json
{
  "items": [
    {
      "id": "TX-8841", "risk_score": 0.8312, "risk_band": "critical",
      "amount": 149.62, "hour": 2, "true_label": 1,
      "reviewed": false, "created_at": "2026-09-20T14:03:11Z"
    }
  ],
  "page": 1, "page_size": 50, "total": 187, "threshold": 0.0473
}
```

`total` là số bản ghi khớp bộ lọc, dùng cho phân trang ở giao diện.

### API-07 — `GET /api/v1/transactions/{id}`

Trả về đầy đủ 30 giá trị đặc trưng, điểm rủi ro, nhãn thật (nếu có), bản ghi thẩm
định (nếu có), và vị trí phân vị của `Amount` so với toàn tập — phục vụ US-04.

`404` nếu không tồn tại.

### API-08 — `POST /api/v1/reviews`

```json
{
  "transaction_id": "TX-8841",
  "decision": "confirmed_fraud",
  "note": "Khớp mẫu hình rút tiền nhỏ liên tiếp"
}
```

`decision` thuộc `confirmed_fraud` hoặc `false_alarm`. Máy chủ tự ghi
`threshold_used` bằng ngưỡng tại thời điểm gửi — chi tiết này cho phép về sau
phân tích các quyết định được đưa ra dưới chính sách nào.

Một giao dịch chỉ có một bản ghi thẩm định; gửi lại sẽ ghi đè (`UPSERT`).

### API-09 — `GET /api/v1/threshold`

```json
{
  "current": 0.0473,
  "source": "user",
  "cost_fn": 122.21,
  "cost_fp": 5.00,
  "alternatives": {
    "min_expected_cost": 0.0473,
    "max_f1": 0.2140,
    "recall_at_least_90": 0.0089,
    "budget_200_alerts": 0.0612,
    "default_naive": 0.5000
  },
  "model_version": "xgb_scaleposweight_v3",
  "block_threshold": 0.9735,
  "block_min_precision": 0.95
}
```

`source` là `artifact` (giá trị mặc định từ `threshold.json`) hoặc `user` (người
dùng đã đổi và giá trị đang nằm trong bảng `settings`). `block_threshold`: xem §2.

### API-10 — `PUT /api/v1/threshold`

```json
{ "value": 0.0612 }
```

`422` nếu ngoài khoảng (0, 1). Ghi vào bảng `settings`, có hiệu lực ngay cho mọi
truy vấn sau đó.

### API-11 — `GET /api/v1/threshold/preview?value=0.037`

Endpoint nóng nhất của giao diện — mỗi lần kéo thanh trượt gọi một lần
(có debounce 150 ms ở phía client).

```json
{
  "threshold": 0.037,
  "alerts": 214,
  "tp": 81, "fp": 133, "fn": 17, "tn": 56755,
  "precision": 0.3785, "recall": 0.8265, "f1": 0.5192,
  "expected_cost": 2742.57,
  "alerts_per_day": 107
}
```

Tính bằng phép so sánh vector hóa trên mảng điểm của tập kiểm thử đã nạp sẵn
trong bộ nhớ. Không chạy mô hình, không đụng ổ đĩa — khoảng 5–10 ms, đáp ứng
NFR-03.

### API-12 — `POST /api/v1/threshold/optimize`

```json
{
  "cost_fn": 200.0,
  "cost_fp": 3.0,
  "constraint": { "type": "max_alerts_per_day", "value": 200 }
}
```

`constraint` tùy chọn, các kiểu: `none`, `min_recall`, `max_alerts_per_day`.

Phản hồi:

```json
{
  "optimal_threshold": 0.0291,
  "constraint_binding": false,
  "metrics_at_optimal": { ... như API-11 ... },
  "curve": [ { "threshold": 0.001, "cost": 4210.5, "alerts": 980 }, ... ]
}
```

`curve` gồm khoảng 200 điểm để giao diện vẽ đường cong chi phí.
`constraint_binding` cho biết ràng buộc có thực sự chặn nghiệm tối ưu không —
thông tin này đáng hiển thị vì nó trả lời "tôi đang bị giới hạn bởi ngân sách
thẩm định hay bởi chính chi phí".

### API-13 — `GET /api/v1/metrics`

Trả nguyên nội dung `metrics.json`: chỉ số tổng hợp, khoảng tin cậy bootstrap,
điểm trên đường PR và ROC, đường cong chi phí, ma trận nhầm lẫn tại ngưỡng đề
xuất, bảng 20 tổ hợp, tầm quan trọng SHAP toàn cục. Cấu trúc chi tiết ở
[06 §3.2](06-thiet-ke-luu-tru.md).

Chấp nhận `?section=pr_curve` để lấy một phần, tránh truyền tệp lớn không cần thiết.

### API-14 — `GET /api/v1/samples`

```json
{
  "items": [
    { "id": "S-001", "label": 1, "risk_score": 0.91, "amount": 1.0,
      "description": "Gian lận điển hình — điểm rủi ro rất cao" },
    { "id": "S-002", "label": 0, "risk_score": 0.44, "amount": 89.5,
      "description": "Hợp lệ nhưng điểm cao — ca khó" }
  ]
}
```

Mẫu được chọn có chủ đích để trình diễn: gian lận dễ, gian lận khó (điểm thấp),
hợp lệ dễ, hợp lệ khó (điểm cao). Bốn nhóm này làm buổi bảo vệ thuyết phục hơn
nhiều so với mẫu ngẫu nhiên.

### API-15 — `GET /api/v1/replay/stream?speed=60&start=0`

Server-Sent Events. Mỗi sự kiện là một giao dịch từ `test_set.parquet`, phát theo
thứ tự `Time` với thời gian nén lại `speed` lần (mặc định 60 — một phút dữ liệu
mỗi giây).

```
event: transaction
data: {"id":"TX-5512","risk_score":0.0021,"amount":12.5,"sim_time":"00:04:21"}

event: alert
data: {"id":"TX-5518","risk_score":0.8812,"amount":1.0,"sim_time":"00:04:29"}

event: stats
data: {"processed":1240,"alerts":9,"elapsed_sim_seconds":260}
```

Client ngắt kết nối là dừng. Không giữ trạng thái phía máy chủ ngoài con trỏ của
chính kết nối đó.

## 4. Mô hình lỗi

Mọi lỗi trả về cùng một cấu trúc:

```json
{
  "error": {
    "code": "MISSING_FEATURES",
    "message": "Thiếu cột bắt buộc trong dữ liệu vào",
    "details": { "missing": ["V13", "V27"] }
  }
}
```

| HTTP | `code` | Khi nào |
|---|---|---|
| 400 | `INVALID_REQUEST` | Tham số truy vấn sai định dạng, hoặc `section` không có trong `metrics.json` |
| 405 | `METHOD_NOT_ALLOWED` | Sai phương thức HTTP cho một đường dẫn có thật |
| 422 | `INVALID_BODY` | Thân yêu cầu sai cấu trúc: thiếu trường không phải đặc trưng, sai kiểu JSON, thân rỗng (TC-41), tệp CSV không đọc được |
| 404 | `NOT_FOUND` | Không có giao dịch với id đã cho |
| 413 | `PAYLOAD_TOO_LARGE` | Lô vượt 50.000 dòng hoặc tệp vượt 100 MB |
| 422 | `MISSING_FEATURES` | Thiếu cột bắt buộc (DS-10) |
| 422 | `INVALID_FEATURE_VALUE` | Sai kiểu, NaN, Inf, `Amount` âm (DS-11…DS-13) |
| 422 | `INVALID_THRESHOLD` | Ngưỡng ngoài khoảng (0, 1) |
| 500 | `INTERNAL_ERROR` | Lỗi không lường trước; ghi log kèm mã truy vết |
| 503 | `DATABASE_UNAVAILABLE` | Mất kết nối PostgreSQL; `pool_pre_ping` đã thử kết nối lại mà không được |
| 503 | `MODEL_NOT_LOADED` | Chưa nạp xong hiện vật |
| 503 | `API_UNAVAILABLE` | **Do nginx trả, không phải API** (chỉ khi chạy bằng Docker Compose, qua cổng 3000): tiến trình API đang khởi động hoặc đã dừng (`deploy/nginx.conf`). Tệp vượt 110 MB thì nginx cũng tự trả `413 PAYLOAD_TOO_LARGE` cùng cấu trúc |

Nguyên tắc: **không bao giờ trả 200 kèm thông báo lỗi trong thân phản hồi**, và
không bao giờ để ngoại lệ chưa bắt làm sập tiến trình (AC-A10).

## 5. Quy ước chung

| Chủ đề | Quy ước |
|---|---|
| Thời gian | Chuỗi ISO 8601, múi UTC, hậu tố `Z` |
| Số thực | Điểm rủi ro trả **nguyên độ chính xác**, không làm tròn — xem §7 |
| Phân trang | `page` bắt đầu từ 1; luôn trả `total` |
| CORS | Cho phép `http://localhost:3000` khi phát triển |
| Phiên bản | Tiền tố `/api/v1`; thay đổi phá vỡ hợp đồng thì tăng lên `v2` |
| Xác thực | Không có — demo cục bộ (ngoài phạm vi, xem 00 §3.2) |
| Ghi log | Mỗi yêu cầu ghi: phương thức, đường dẫn, mã trạng thái, thời gian xử lý |

## 6. Quan hệ giữa endpoint và màn hình

| Màn hình | Endpoint sử dụng |
|---|---|
| UI-01 Hàng đợi | API-06, API-09, API-11 |
| UI-02 Chi tiết | API-07, API-05, API-08 |
| UI-03 Ngưỡng | API-09, API-10, API-11, API-12 |
| UI-04 Hiệu năng | API-13 |
| Phát lại | API-15, API-09 |
| Thư viện mẫu | API-14, API-02 |

## 7. Bản thi hành — những điểm cụ thể hơn hoặc khác hợp đồng ban đầu

`api/schemas.py` là bản thi hành của tài liệu này (giai đoạn 7, 2026-09-28). Các điểm dưới đây
là chỗ bản thi hành phải chọn một cách hiểu, hoặc phải đổi so với các ví dụ ở §3. Kiểm thử đi kèm:
`tests/test_api.py`, `tests/test_scoring.py`, `tests/test_db.py`.

| Endpoint | Điểm | Lý do |
|---|---|---|
| mọi phản hồi | `risk_score` **không** làm tròn 4 chữ số (khác §5 cũ) | Làm tròn ở máy chủ làm lệch so sánh với ngưỡng sát biên: điểm 0,023170 làm tròn thành 0,0232 thì trình duyệt thấy "vượt τ\* = 0,023173" trong khi máy chủ quyết định `allow`. Giao diện tự định dạng phần trăm |
| mọi phản hồi | thời gian dạng ISO 8601 hậu tố `Z`, có thể kèm phần lẻ giây | Pydantic v2 |
| API-01 | 503 trả mô hình lỗi chung; `details` chứa các trường của phản hồi 200 (`db`, `model_version`, `uptime_seconds`…) và `reason` khi thiếu hiện vật | TC-42 đòi `code = DATABASE_UNAVAILABLE`; hiện vật hỏng (pickle lỗi, lệch phiên bản) cũng thành 503 `MODEL_NOT_LOADED`, tiến trình không sập |
| API-02 | thêm `sample_id` (tùy chọn): giao dịch lưu với `source = "sample"` và nhãn thật của mẫu | Thư viện mẫu (API-14) có nhãn; giữ nhãn thì UI-02 so được quyết định của người thẩm định với sự thật (FR-44) |
| API-02 | `Amount` làm tròn tới xu **trước** khi chấm | Đúng giá trị nằm trong cột `NUMERIC(12, 2)` (ST-07), nên chấm lại từ cơ sở dữ liệu ra cùng điểm. Dữ liệu gốc đã có đúng 2 chữ số, không đổi gì |
| API-02, API-03 | giới hạn `Amount ≤ 9.999.999.999,99`, `Time ≥ 0` | Sức chứa của `NUMERIC(12, 2)` và ràng buộc `ck_tx_time` |
| API-03 | thêm `persist` (mặc định `true`) và `model_version`; mỗi phần tử `results` có `risk_band`, `decision` | TC-52; thử nhanh không ghi |
| API-03 | lô có một phần tử sai → 422 cả lô, `details.errors[].row` chỉ phần tử đó | JSON khác CSV: người gửi sửa được ngay |
| API-04 | `results` chỉ giữ 200 giao dịch điểm cao nhất (`results_truncated`); `rejection_reasons` tối đa 100 dòng (`rejection_reasons_truncated`) | Tệp 100 MB có khoảng 570.000 dòng — trả hết là 25 MB JSON. Hàng đợi đọc từ cơ sở dữ liệu, không từ phản hồi này |
| API-04 | `row` đếm từ 1, không tính dòng tiêu đề | |
| API-05 | `base_value` ở thang **log-odds** (3,28), không phải xác suất; thêm `margin` (= `base_value + Σ SHAP`), `remaining_shap`, `contributions` (cả 31 đặc trưng), `shap_output: "log-odds"` | Tính cộng chỉ đúng ở thang log-odds; thiếu `remaining_shap` thì biểu đồ thác nước không khép được |
| API-05 | tên đặc trưng như `FEATURE_ORDER` (`Amount`, không phải `amount_scaled`); `value` là giá trị **gốc** (Amount theo đơn vị tiền) | UI-02 hiện "Amount 1.809,68" |
| API-05 | `top_positive` là **tối đa** 5 đặc trưng có SHAP > 0, `top_negative` tối đa 3 đặc trưng có SHAP < 0 | Đo trên tập kiểm thử: 77% giao dịch có ít hơn 5 đặc trưng đẩy điểm lên (hợp lệ điểm thấp); trong hàng đợi (≥ τ\*) 114/115 có đủ 5; yếu tố âm luôn đủ 3. Cho đủ 5 bằng cách mượn một đóng góp âm là nói sai. AC-A4 hiểu là "đúng 5 và 3 khi mô hình có đủ" |
| API-06 | thêm bộ lọc `review_status` (`pending`/`confirmed_fraud`/`false_alarm`) và `source`; `sort` nhận `risk_score`, `amount`, `created_at`, `hour`, có hoặc không có `-` | Bộ lọc trạng thái của UI-01 |
| API-06 | lọc theo `band` mà không có `min_score` thì bỏ mặc định "≥ τ" | Nếu không, lọc `low`/`medium` luôn rỗng |
| API-06 | mỗi dòng thêm `decision`, `review_decision`, `source`, `batch_id`, `model_version` | |
| API-07 | `amount_percentile` so với phân bố `Amount` của **tập kiểm thử**, không với bảng `transactions` | Bảng chỉ có những gì người dùng đã nạp — vài chục dòng thì phân vị vô nghĩa. Câu SQL cũ ở 06 §4.4 còn sai: `WHERE` lọc trước hàm cửa sổ nên luôn ra 0 |
| API-07 | thêm `features` (30 cột thô), `decision`, `threshold`, `review`, `model_version_current` | 06 §8: gắn nhãn cảnh báo khi điểm do mô hình khác chấm |
| API-08 | phản hồi có `true_label` và `matches_label` | Nhãn chỉ lộ **sau** khi đã quyết định (UI-02) |
| API-05 | SHAP tính bằng `pred_contribs=True` của XGBoost (`api/serving.py`), không nạp `explainer.joblib` | Cùng thuật toán TreeSHAP, trùng **từng bit** với `explainer.joblib` (`tests/test_api.py::test_t48…`); ảnh `api` bỏ được shap + numba + llvmlite (~210 MB) |
| §2 | `critical`/`block` từ `max(τ, τ_chặn)`, τ_chặn chọn theo precision ≥ 95% trên out-of-fold — thay cho `3τ` | Luật 3τ chặn tự động 19/38 cảnh báo giả của tập kiểm thử và để dải `review` không có vụ gian lận nào (notebook 06 §6.2) |
| API-09 | thêm `default` (τ\* của `threshold.json`) | Nút "đặt lại" ở UI-03 |
| API-09 | thêm `block_threshold`, `block_min_precision` | §2 |
| API-09 | `source = "user"` chỉ khi ngưỡng trong `settings` được đặt cho **đúng** `model_version` đang chạy | Huấn luyện lại thì một con số đặt cho mô hình cũ không còn nghĩa (06 §2) |
| API-10 | nhận thêm `cost_fn`, `cost_fp` (tùy chọn) để lưu cùng ngưỡng | Nút "áp dụng" ở UI-03 áp cả tham số chi phí |
| API-11 | nhận thêm `cost_fn`, `cost_fp` (tùy chọn); mặc định lấy từ `settings` | |
| API-12 | ngưỡng **chọn trên out-of-fold** (`models/oof_scores.npz`), dò trên mọi điểm khác nhau; chỉ số tại ngưỡng đo trên tập kiểm thử (`metrics_at_optimal`) và kèm số out-of-fold (`metrics_at_optimal_oof`); `curve` cũng trên out-of-fold (`curve_source`) | ML-08: đây là một phép **chọn** ngưỡng. Với chi phí mặc định API trả đúng τ\* = 0,023173 của `threshold.json` |
| API-12 | có ràng buộc thì nghiệm là ngưỡng **chi phí thấp nhất trong vùng khả thi**; thêm `unconstrained_threshold`, `constraint_satisfied` | Khác phương án `budget_200_alerts` của `threshold.json` (τ **nhỏ nhất** trong ngân sách, tức recall cao nhất): với 200 cảnh báo/ngày, API chọn 0,9657 còn `threshold.json` ghi 0,9625 |
| API-12 | thân rỗng → 422 (TC-41); gửi `{}` để dùng mọi giá trị mặc định | |
| API-13 | `?section=<khóa>` trả `{"model_version", "<khóa>": …}`; khóa không có → 400 kèm danh sách khóa | |
| API-14 | mỗi mẫu có `category` và `features` (30 cột) | Giao diện gửi thẳng `features` vào API-02 kèm `sample_id` |
| API-15 | `start` là **giây mô phỏng tính từ đầu ngày 2** (0 ≤ start < 86.400), `speed` từ 1 tới 3.600, mặc định lấy `replay_speed` trong `settings` | Tạm dừng là ngắt kết nối; tiếp tục là mở lại với `start` = đồng hồ lúc dừng |
| API-15 | thêm sự kiện `start` (tổng số, ngưỡng, `batch_id`) và `end`; `stats` gửi mỗi giây, cũng là nhịp "còn sống"; mỗi giao dịch có `risk_band`, `sim_seconds` | Nhãn thật **không** gửi trong luồng |
| API-15 | mỗi giao dịch phát ra được ghi vào `transactions` (`source = "replay"`, mã `RP-<dòng>`), mẻ 100 dòng, `ON CONFLICT DO NOTHING` | Cảnh báo "rơi vào hàng đợi" (UI-05); phát lại lần hai không nhân đôi (TC-44) |
| API-15 | điểm rủi ro do **mô hình chấm lúc giao dịch tới giờ** trên đồng hồ mô phỏng (mẻ nhỏ, tối đa 500), cùng đường với `/score` — không đọc cột `risk_score` tính sẵn của `test_set.parquet` | Phát lại là trình diễn chấm điểm thật, không phải phát lại một bảng số |

Số đo trên máy phát triển (12 lõi, đang có tải nền khoảng 50% CPU):

| Chỉ tiêu | Kết quả |
|---|---|
| NFR-01 — `/score` một giao dịch, 1.000 lần liên tiếp | p95 **45,6 ms** phía máy chủ (`latency_ms`), 57,8 ms tính cả lớp HTTP của `TestClient` |
| NFR-02 / AC-A1 — `/score/upload` 10.000 dòng | **2,0 giây** gồm đọc CSV, chấm, `COPY` |
| TC-54 — ghi 10.000 dòng bằng `COPY` | 0,53 giây, nhanh gấp khoảng 160 lần `INSERT` từng dòng (119 dòng/giây) |
| T-47 — `/threshold/preview` | phần tính 0,12 ms; khoảng 15 ms tính cả HTTP và đọc `settings` |
| `/explain` | 60–100 ms cả lời gọi |
| AC-A6 — phát lại liên tục | 185 giây, 1.532 giao dịch, 0 lỗi |
| Khởi động tới khi `/health` trả 200 | khoảng 5 giây (nạp hiện vật và chấm lại 1/50 tập kiểm thử để kiểm) |

XGBoost chấm với **1 luồng** lúc phục vụ: dự đoán không phụ thuộc số luồng (trùng từng bit trên
56.746 giao dịch), còn 12 luồng cho một giao dịch làm p95 của `/score` gần gấp đôi vì chi phí đồng bộ.
