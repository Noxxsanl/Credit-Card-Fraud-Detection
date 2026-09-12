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

Quy tắc suy ra `risk_band` từ `risk_score` và ngưỡng hiện hành τ:

| Dải | Điều kiện |
|---|---|
| `low` | `score < τ/2` |
| `medium` | `τ/2 ≤ score < τ` |
| `high` | `τ ≤ score < 3τ` |
| `critical` | `score ≥ 3τ` |

Dải rủi ro là **dẫn xuất**, tính lại mỗi lần trả kết quả — vì τ thay đổi được
(AR-03). Không lưu vào cơ sở dữ liệu.

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
  "model_version": "xgb_scaleposweight_v3"
}
```

`source` là `artifact` (giá trị mặc định từ `threshold.json`) hoặc `user` (người
dùng đã đổi và giá trị đang nằm trong bảng `settings`).

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
| 400 | `INVALID_REQUEST` | Tham số truy vấn sai định dạng |
| 404 | `NOT_FOUND` | Không có giao dịch với id đã cho |
| 413 | `PAYLOAD_TOO_LARGE` | Lô vượt 50.000 dòng hoặc tệp vượt 100 MB |
| 422 | `MISSING_FEATURES` | Thiếu cột bắt buộc (DS-10) |
| 422 | `INVALID_FEATURE_VALUE` | Sai kiểu, NaN, Inf, `Amount` âm (DS-11…DS-13) |
| 422 | `INVALID_THRESHOLD` | Ngưỡng ngoài khoảng (0, 1) |
| 500 | `INTERNAL_ERROR` | Lỗi không lường trước; ghi log kèm mã truy vết |
| 503 | `DATABASE_UNAVAILABLE` | Mất kết nối PostgreSQL; `pool_pre_ping` đã thử kết nối lại mà không được |
| 503 | `MODEL_NOT_LOADED` | Chưa nạp xong hiện vật |

Nguyên tắc: **không bao giờ trả 200 kèm thông báo lỗi trong thân phản hồi**, và
không bao giờ để ngoại lệ chưa bắt làm sập tiến trình (AC-A10).

## 5. Quy ước chung

| Chủ đề | Quy ước |
|---|---|
| Thời gian | Chuỗi ISO 8601, múi UTC, hậu tố `Z` |
| Số thực | Điểm rủi ro làm tròn 4 chữ số thập phân trong phản hồi |
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
