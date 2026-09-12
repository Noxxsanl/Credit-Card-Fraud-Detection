# 03 — Thiết kế kiến trúc

## 1. Tổng thể

Hệ thống tách thành hai giai đoạn hoàn toàn độc lập, nối với nhau **chỉ qua các
tệp hiện vật**.

```
┌───────────────────────────────────────────────────────────────┐
│  GIAI ĐOẠN OFFLINE — huấn luyện (notebook, chạy tay)          │
│                                                               │
│  data/creditcard.csv                                          │
│        │                                                      │
│        ├─→ 01 EDA ──────────────→ reports/figures/            │
│        ├─→ 02 Kiểm định thống kê → reports/figures/           │
│        ├─→ 03..05 Huấn luyện ────→ chọn cấu hình tốt nhất     │
│        ├─→ 06 Ngưỡng & chi phí ──→ threshold.json             │
│        ├─→ 07 SHAP ──────────────→ explainer.joblib           │
│        └─→ 08 Xuất hiện vật ─────→ models/, data/             │
└───────────────────────────────────────────────────────────────┘
                              │
                              │  nạp một lần lúc khởi động
                              ▼
┌───────────────────────────────────────────────────────────────┐
│  GIAI ĐOẠN ONLINE — phục vụ                                   │
│                                                               │
│   ┌───────────────┐   HTTP/JSON    ┌───────────────────────┐  │
│   │  web/         │ ◄────────────► │  api/  (FastAPI)      │  │
│   │               │                │                       │  │
│   │  UI-01 Hàng   │   SSE          │  Tầng route           │  │
│   │  UI-02 Chi tiết│ ◄──────────── │  Tầng dịch vụ         │  │
│   │  UI-03 Ngưỡng │                │  Tầng truy cập dữ liệu│  │
│   │  UI-04 Hiệu năng│              └───────┬───────────────┘  │
│   └───────────────┘                        │                  │
│                            ┌───────────────┴──────────────┐   │
│                            ▼                              ▼   │
│                   ┌─────────────────┐          ┌──────────────┐│
│                   │ models/ (chỉ đọc)│          │ db           ││
│                   │ model.joblib     │          │ PostgreSQL 16││
│                   │ explainer.joblib │          │ transactions ││
│                   │ metrics.json     │          │ reviews      ││
│                   │ threshold.json   │          │ settings     ││
│                   └─────────────────┘          └──────┬───────┘│
│                     gắn volume chỉ đọc          volume │pgdata  │
└───────────────────────────────────────────────────────────────┘
```

## 2. Ba nguyên tắc kiến trúc

**AR-01 — Tách hoàn toàn huấn luyện khỏi phục vụ.** API không bao giờ huấn luyện,
không bao giờ đọc `creditcard.csv`. Nó chỉ nạp hiện vật. Nhờ vậy: phần nghiên cứu
và phần ứng dụng phát triển song song được, và con số hiện trong demo khớp chính
xác con số trong báo cáo.

**AR-02 — Mô hình lưu dạng pipeline hoàn chỉnh.** `model.joblib` chứa cả bước
chuẩn hóa lẫn bộ phân loại. Nếu chỉ lưu bộ phân loại, tầng phục vụ phải tự dựng
lại bước tiền xử lý và sớm muộn sẽ lệch. Đây là lỗi triển khai ML phổ biến nhất,
và nó im lặng — hệ thống vẫn chạy, chỉ là dự đoán sai.

**AR-03 — Ngưỡng là cấu hình, không phải một phần của mô hình.** Mô hình luôn trả
xác suất. Việc biến xác suất thành quyết định chặn/cho qua diễn ra ở tầng ứng
dụng, dựa trên ngưỡng đọc từ bảng `settings` lúc chạy. Nhờ vậy thanh trượt ngưỡng
phản hồi tức thì mà không phải tính lại điểm — đáp ứng NFR-03.

Hệ quả của AR-03 lên lưu trữ: cột `risk_score` trong bảng `transactions` là dữ
liệu **bất biến**, còn quyết định chặn/cho qua là **dẫn xuất**, tính lại mỗi lần
truy vấn. Không lưu cột `decision`.

## 3. Phân rã thành phần

### 3.1 Lớp dùng chung — `src/`

Được import bởi **cả** notebook và API. Đây là ranh giới chống lệch train/serve.

| Mô-đun | Trách nhiệm | Ai dùng |
|---|---|---|
| `src/features.py` | Sinh `hour_sin`, `hour_cos`, sắp xếp cột đúng thứ tự huấn luyện | Notebook + API |
| `src/data.py` | Nạp CSV, kiểm tra toàn vẹn, loại trùng lặp, chia tập | Notebook |
| `src/evaluate.py` | `compute_metrics`, `evaluate_model`, bootstrap khoảng tin cậy | Notebook + API |
| `src/threshold.py` | Quét ngưỡng, đường cong chi phí, chọn ngưỡng theo tiêu chí | Notebook + API |
| `src/plots.py` | Hàm vẽ dùng chung, lưu hình vào `reports/figures/` | Notebook |

Ràng buộc: `src/` **không được import** `fastapi`, `streamlit` hay bất kỳ thư viện
tầng ứng dụng nào. Nó là lớp thấp nhất.

### 3.2 Tầng API — `api/`

Ba tầng, phụ thuộc một chiều: route → dịch vụ → truy cập dữ liệu.

| Tệp | Trách nhiệm |
|---|---|
| `api/main.py` | Khởi tạo FastAPI, nạp hiện vật vào bộ nhớ lúc startup, cấu hình CORS |
| `api/schemas.py` | Toàn bộ mô hình Pydantic vào/ra — nguồn chân lý cho hợp đồng API |
| `api/deps.py` | Cung cấp mô hình, explainer, phiên cơ sở dữ liệu dạng dependency |
| `api/services/scoring.py` | Chấm điểm một mẫu và theo lô, gắn `model_version` |
| `api/services/explaining.py` | Gọi SHAP, sắp xếp và cắt lấy top yếu tố |
| `api/services/thresholding.py` | Tính lại chỉ số theo ngưỡng, tối ưu theo chi phí |
| `api/services/replay.py` | Sinh dòng sự kiện cho chế độ phát lại |
| `api/routes/*.py` | Chuyển đổi HTTP sang lời gọi dịch vụ, không chứa logic nghiệp vụ |
| `api/db.py` | Engine SQLAlchemy + pool kết nối PostgreSQL, định nghĩa bảng, quản lý phiên |
| `api/migrations/` | Alembic — lược đồ cơ sở dữ liệu dưới dạng mã, chạy `upgrade head` lúc khởi động |

**Vì sao tách services khỏi routes:** logic ngưỡng và chấm điểm cần kiểm thử được
mà không cần dựng máy chủ HTTP. Route chỉ còn là lớp vỏ mỏng.

### 3.3 Tầng giao diện — `web/`

Ứng dụng một trang, bốn màn hình. Chi tiết ở [07](07-thiet-ke-giao-dien.md).

### 3.4 Quản lý trạng thái lúc chạy

| Trạng thái | Nơi giữ | Vòng đời |
|---|---|---|
| Mô hình, explainer, metrics | Bộ nhớ tiến trình API, nạp lúc startup | Suốt vòng đời tiến trình |
| Ngưỡng hiện hành, tham số chi phí | Bảng `settings` trong PostgreSQL | Bền vững qua khởi động lại |
| Giao dịch đã chấm điểm | Bảng `transactions` | Bền vững |
| Quyết định thẩm định | Bảng `reviews` | Bền vững |
| Vị trí con trỏ phát lại | Bộ nhớ, theo từng kết nối SSE | Mất khi ngắt kết nối |

## 4. Ranh giới quan trọng nhất: nhất quán đặc trưng train/serve

Đây là điểm dễ hỏng nhất của mọi hệ thống ML và là lý do tồn tại của `src/features.py`.

```
notebook 05 ──┐
              ├─→ src/features.build_features(df) ──→ cùng một ma trận
api/services ─┘
```

Ba cơ chế canh giữ:

| Cơ chế | Nội dung |
|---|---|
| Một hàm duy nhất | `build_features()` là đường vào duy nhất; không nơi nào khác được tự tính `hour` |
| Danh sách cột cố định | `FEATURE_ORDER` là hằng số trong `features.py`; pipeline khẳng định thứ tự cột khi dự đoán |
| Kiểm thử vàng | `tests/test_features.py` giữ 20 dòng mẫu và giá trị đặc trưng kỳ vọng; chạy trong CI cục bộ |

Nếu ai đó thêm một đặc trưng mà quên cập nhật phía phục vụ, kiểm thử vàng sẽ đỏ
ngay. Không có cơ chế này thì lỗi chỉ lộ ra khi số liệu demo lệch báo cáo, và lúc
đó rất khó truy nguyên.

## 5. Luồng dữ liệu chính

### 5.1 Chấm điểm một giao dịch

```
Trình duyệt
   │ POST /api/v1/score {transaction}
   ▼
routes/scoring.py — xác thực bằng Pydantic (DS-10..DS-14)
   │
   ▼
services/scoring.py
   ├─ src.features.build_features()      ← dùng chung với huấn luyện
   ├─ model.predict_proba()[:, 1]        ← pipeline: scaler + classifier
   ├─ đọc ngưỡng hiện hành từ settings
   └─ suy ra decision và risk_band
   │
   ▼
db: ghi vào transactions (nếu source != 'adhoc')
   │
   ▼
Phản hồi JSON {risk_score, threshold, decision, risk_band, model_version, latency_ms}
```

### 5.2 Kéo thanh trượt ngưỡng

```
Trình duyệt (debounce 150 ms)
   │ GET /api/v1/threshold/preview?value=0.037
   ▼
services/thresholding.py
   ├─ nạp mảng (y_true, y_score) của tập kiểm thử từ metrics.json — đã ở bộ nhớ
   ├─ tính TP/FP/FN bằng phép so sánh vector hóa trên NumPy
   └─ tính precision, recall, f1, chi phí kỳ vọng
   │
   ▼
Phản hồi trong khoảng 5–10 ms — không đụng tới mô hình, không đụng tới ổ đĩa
```

Đây là lý do AR-03 tồn tại. Nếu ngưỡng nằm bên trong mô hình, mỗi lần kéo thanh
trượt phải chấm điểm lại 57.000 giao dịch.

### 5.3 Giải thích một giao dịch

```
POST /api/v1/explain {transaction_id}
   ▼
services/explaining.py
   ├─ đọc đặc trưng từ transactions
   ├─ explainer.shap_values(x)           ← 50–200 ms
   ├─ sắp xếp theo |giá trị SHAP|
   └─ lấy 5 yếu tố dương + 3 yếu tố âm, kèm base_value
```

Endpoint riêng vì chi phí tính toán cao (NFR-01). Giao diện chỉ gọi khi người dùng
mở chi tiết.

## 6. Lựa chọn công nghệ và lý do

### 6.1 Phần huấn luyện

| Thành phần | Chọn | Lý do |
|---|---|---|
| Ngôn ngữ | Python 3.11 | Hệ sinh thái ML |
| Xử lý dữ liệu | pandas, NumPy | Chuẩn ngành |
| Học máy | scikit-learn, XGBoost, LightGBM | Bao phủ cả tuyến tính lẫn ensemble |
| Mất cân bằng | imbalanced-learn | `Pipeline` của nó tự bỏ qua bước resample khi transform tập validation — điều `sklearn.pipeline` không làm được |
| Thống kê | SciPy, statsmodels | Kiểm định phi tham số, hiệu chỉnh FDR |
| Giải thích | SHAP | `TreeExplainer` cho mô hình cây là chính xác, không xấp xỉ |
| Lưu hiện vật | joblib | Chuẩn của scikit-learn, xử lý mảng NumPy tốt hơn pickle |

### 6.2 Phần ứng dụng

Hai phương án, chọn theo quỹ thời gian còn lại sau khi chốt mô hình.

**Phương án A — Streamlit đơn khối** (khoảng 3 ngày)

Toàn bộ trong `app.py`. Không có API riêng.

- Ưu: nhanh, ít mã, không cần biết frontend.
- Nhược: mỗi thao tác chạy lại toàn bộ script nên chế độ phát lại giật; không tách
  được tầng phục vụ để trình bày về kiến trúc; khó tùy biến giao diện.

**Phương án B — FastAPI + giao diện riêng** (khoảng 6 ngày) — **khuyến nghị và là
phương án mà các tài liệu 05, 06, 07 mô tả**

- Ưu: kiến trúc rõ ràng, tài liệu API tự sinh tại `/docs`, phát lại mượt qua SSE,
  giao diện tùy biến hoàn toàn.
- Nhược: nhiều việc hơn, cần biết frontend cơ bản.

Cho phương án B, frontend nên chọn **HTML + Alpine.js + Chart.js**: một tệp HTML,
không cần công cụ đóng gói, không có `node_modules`. Với bốn màn hình và trạng
thái đơn giản, nó cho kết quả tương đương Next.js mà tiết kiệm khoảng hai ngày
công. Chọn Next.js + React + Tailwind + Recharts chỉ khi đã thạo sẵn.

**Điểm quyết định:** chốt phương án chậm nhất vào cuối ngày 14. Nếu đến ngày 14
mô hình chưa xong, chuyển sang phương án A và giữ nguyên bốn màn hình.

`app.py` hiện có trong repo là bản Streamlit tối giản, đóng vai trò lưới an toàn
cho phương án A.

### 6.3 Đóng gói

`docker-compose.yml` với ba dịch vụ:

| Dịch vụ | Ảnh | Vai trò |
|---|---|---|
| `db` | `postgres:16-alpine` | Cơ sở dữ liệu, dữ liệu nằm trong volume có tên `pgdata` |
| `api` | Build từ `./api` | FastAPI; chờ `db` khỏe rồi chạy `alembic upgrade head` trước khi phục vụ |
| `web` | Build từ `./web` | Giao diện tĩnh |

Thư mục `models/` và `data/` gắn dạng volume chỉ đọc vào `api`, không nướng vào
image — nhờ vậy huấn luyện lại không phải build lại image. Dữ liệu PostgreSQL
nằm trong volume `pgdata`, tách khỏi cây thư mục repo.

`api` phụ thuộc `db` qua `depends_on: condition: service_healthy` với healthcheck
`pg_isready`. Không có điều kiện này, `api` sẽ khởi động trước khi PostgreSQL sẵn
sàng nhận kết nối và migration sẽ thất bại ngay lần chạy đầu.

Điều kiện nghiệm thu: chạy được toàn hệ thống bằng đúng một lệnh trên máy chỉ có
Docker. Chi tiết tệp Compose ở [06 §7.1](06-thiet-ke-luu-tru.md).

## 7. Cấu trúc mã nguồn mục tiêu

```
fraud-detection/
├── docs/                     # bộ tài liệu này
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_statistics.ipynb
│   ├── 03_baseline.ipynb
│   ├── 04_imbalance_strategies.ipynb
│   ├── 05_advanced_models.ipynb
│   ├── 06_threshold_and_cost.ipynb
│   ├── 07_explainability.ipynb
│   └── 08_export_artifacts.ipynb
│
├── src/                      # lớp dùng chung, không phụ thuộc tầng ứng dụng
│   ├── features.py           # ← ranh giới chống lệch train/serve
│   ├── data.py
│   ├── evaluate.py
│   ├── threshold.py
│   └── plots.py
│
├── api/
│   ├── main.py
│   ├── schemas.py
│   ├── deps.py
│   ├── db.py                 # engine + pool PostgreSQL
│   ├── models_orm.py         # bảng SQLAlchemy
│   ├── alembic.ini
│   ├── migrations/           # lược đồ dưới dạng mã — commit vào git
│   │   ├── env.py
│   │   └── versions/
│   ├── services/
│   │   ├── scoring.py
│   │   ├── explaining.py
│   │   ├── thresholding.py
│   │   └── replay.py
│   ├── routes/
│   │   ├── scoring.py
│   │   ├── transactions.py
│   │   ├── threshold.py
│   │   ├── metrics.py
│   │   └── replay.py
│   └── Dockerfile
│
├── web/
│   ├── index.html
│   ├── app.js
│   ├── styles.css
│   └── Dockerfile
│
├── tests/
│   ├── conftest.py           # fixture cơ sở dữ liệu kiểm thử
│   ├── test_features.py      # kiểm thử vàng, xem §4
│   ├── test_threshold.py
│   ├── test_scoring.py
│   └── test_api.py
│
├── models/                   # hiện vật — ngoài git
├── data/                     # dữ liệu — ngoài git
├── reports/
│   ├── bao-cao.md
│   └── figures/
├── app.py                    # lưới an toàn: demo Streamlit (phương án A)
├── docker-compose.yml        # db + api + web
├── .env.example              # DATABASE_URL mẫu, commit được
├── requirements.txt
└── README.md
```

Chênh lệch so với hiện trạng: cần bổ sung `src/features.py`, `src/threshold.py`,
toàn bộ `api/` (kèm `migrations/`), `web/`, `tests/`, `docker-compose.yml`, và các
notebook 02, 06, 07, 08. Kế hoạch bổ sung nằm ở [09](09-ke-hoach-trien-khai.md).

## 8. Đáp ứng yêu cầu phi chức năng

| NFR | Cơ chế đáp ứng |
|---|---|
| NFR-01 độ trễ | Mô hình nạp sẵn trong bộ nhớ; SHAP tách sang endpoint riêng |
| NFR-02 thông lượng | Chấm điểm theo lô vector hóa, một lời gọi `predict_proba` cho cả DataFrame; ghi xuống cơ sở dữ liệu bằng `COPY` thay vì `INSERT` từng dòng |
| NFR-03 thanh trượt | AR-03 — chỉ so sánh mảng điểm đã tính sẵn, không chạy lại mô hình |
| NFR-04 khởi động | Hiện vật gắn volume, không tải gì qua mạng; nạp joblib mất 1–3 giây. Áp dụng cho lần chạy thứ hai trở đi — lần đầu PostgreSQL phải `initdb` và chạy migration, xem [06 §7.2](06-thiet-ke-luu-tru.md) |
| NFR-05 bộ nhớ | Tiến trình API chỉ giữ mô hình, explainer và mảng điểm của tập kiểm thử; container `db` tính riêng, khoảng 250 MB |
| NFR-07 tái lập | `random_state=42` ở mọi bước ngẫu nhiên; ghim phiên bản thư viện |
| NFR-11 truy vết | `model_version` lưu trong `threshold.json`, gắn vào mọi phản hồi và mọi dòng `transactions` |
