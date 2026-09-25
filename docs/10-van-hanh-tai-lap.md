# 10 — Vận hành và tái lập

Tài liệu dành cho người chạy lại dự án từ đầu trên một máy sạch: người chấm, thành
viên mới, hoặc chính tác giả sau vài tháng.

## 1. Yêu cầu môi trường

| Thành phần | Phiên bản | Bắt buộc cho |
|---|---|---|
| Python | 3.11 hoặc 3.12 | Huấn luyện, API (môi trường hiện tại: 3.12.5) |
| RAM | ≥ 8 GB | SMOTE trên 227.000 dòng |
| Ổ đĩa trống | ≥ 2 GB | Dữ liệu và hiện vật |
| Docker Desktop | Bản mới | Chạy `db`, `api`, `web` bằng Compose |
| PostgreSQL | 16 | Chỉ khi muốn chạy cơ sở dữ liệu trực tiếp thay vì trong Docker |
| Tài khoản Kaggle | — | Tải dữ liệu |

Không cần GPU. Toàn bộ chạy trên CPU (C-05).

Cách đơn giản nhất là **không cài PostgreSQL lên máy**: chỉ cần Docker, rồi chạy
`docker compose up -d db` khi phát triển. Xem §2.3.

## 2. Chuẩn bị

### 2.1 Môi trường Python

```bash
py -3.12 -m venv .venv           # Windows
python3.12 -m venv .venv         # macOS / Linux

.venv\Scripts\activate           # Windows
source .venv/bin/activate        # macOS / Linux

pip install -r requirements.txt pytest
python -m ipykernel install --user --name fraud-detection
```

Kiểm tra nhanh môi trường:

```bash
python -m pytest        # phải xanh trước khi chạy notebook
```

### 2.2 Biến môi trường

Sao chép `.env.example` thành `.env` và chỉnh nếu cần:

```
DATABASE_URL=postgresql+psycopg://fraud:fraud@localhost:5432/fraud
TEST_DATABASE_URL=postgresql+psycopg://fraud:fraud@localhost:5432/fraud_test
```

Trong Docker Compose, dịch vụ `api` dùng host `db` thay cho `localhost` — giá trị
đó đã đặt sẵn trong `docker-compose.yml`, không cần sửa `.env`.

`.env` **không commit** vào git.

### 2.3 Cơ sở dữ liệu

```bash
docker compose up -d db                 # khởi động PostgreSQL 16
docker compose exec db pg_isready -U fraud    # kiểm tra đã sẵn sàng
alembic upgrade head                    # dựng lược đồ
```

Nối vào bằng psql khi cần xem dữ liệu:

```bash
docker compose exec db psql -U fraud -d fraud
# \dt            liệt kê bảng
# \d transactions   xem cấu trúc bảng
# SELECT count(*), max(risk_score) FROM transactions;
```

Nếu cổng 5432 đã bị một PostgreSQL khác trên máy chiếm, đổi ánh xạ cổng trong
`docker-compose.yml` thành `"5433:5432"` và sửa `DATABASE_URL` tương ứng.

### 2.4 Dữ liệu

Dùng `scripts/download_data.py`. Script tải, giải nén và **tự kiểm tra toàn vẹn**;
tệp sai sẽ bị báo ngay chứ không âm thầm lọt vào pipeline.

```bash
python scripts/download_data.py            # qua Kaggle API, cần token
python scripts/download_data.py --mirror   # bản sao công khai, không cần token
python scripts/download_data.py --check    # chỉ kiểm tra tệp đang có
python scripts/download_data.py --force    # tải lại
```

**Cách 1 — Kaggle API (nguồn chính thức).** Cần token: kaggle.com → Settings →
API → *Create New Token*, đặt `kaggle.json` vào `C:\Users\<tên>\.kaggle\`
(hoặc `~/.kaggle/`), hoặc đặt biến môi trường `KAGGLE_USERNAME` và `KAGGLE_KEY`.
Nếu chưa từng bấm Download trên trang dataset, phải bấm một lần để chấp nhận
điều khoản, nếu không API trả 403.

**Cách 2 — `--mirror`.** Tải bản sao công khai trên Hugging Face
(`David-Egea/Creditcard-fraud-detection`) khi không có tài khoản Kaggle. Đây là
nguồn thứ cấp do bên thứ ba đăng lại, nên phần kiểm tra toàn vẹn là bắt buộc —
script đối chiếu số dòng, số cột, số mẫu gian lận và SHA-256.

**Cách 3 — tải tay.** Vào https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud,
tải `creditcard.csv`, đặt vào `data/`, rồi chạy `--check`.

Tệp đúng phải cho kết quả sau:

```
Kích thước : 150.8 MB
SHA-256    : 76274b691b16a6c49d3f159c883398e03ccd6d1ee12d9d8ee38f4b4b98551a89
Số dòng    : 284.807   Số cột: 31   Ô thiếu: 0
Dòng trùng lặp: 1.081   Gian lận: 492 (0,173%)
```

Nếu các con số này không khớp, dừng lại — mọi kết quả về sau sẽ không so sánh
được với báo cáo.

## 3. Chạy lại toàn bộ pipeline

Chạy notebook theo đúng thứ tự, mỗi notebook trong kernel sạch:

| Thứ tự | Notebook | Thời gian ước tính | Sinh ra |
|---|---|---|---|
| 1 | `01_eda.ipynb` | 3–5 phút | `reports/figures/01_*.png` |
| 2 | `02_statistics.ipynb` | 2–4 phút | Bảng xếp hạng đặc trưng |
| 3 | `03_baseline.ipynb` | 5–10 phút | `data/test_set.parquet` |
| 4 | `04_imbalance_strategies.ipynb` | **1–3 giờ** | `reports/grid_results.csv` |
| 5 | `05_advanced_models.ipynb` | ~20 phút tìm kiếm (`scripts/run_search.py`) + ~20 phút notebook | `reports/search_results.csv`, `final_test_metrics.csv`, `split_comparison.csv` |
| 6 | `06_threshold_and_cost.ipynb` | 2–5 phút | Đường cong chi phí |
| 7 | `07_explainability.ipynb` | 10–20 phút | Biểu đồ SHAP |
| 8 | `08_export_artifacts.ipynb` | 2–5 phút | Toàn bộ `models/*`, `data/sample_pool.json` |

Tổng khoảng 2–4 giờ, phần lớn nằm ở notebook 04.

Notebook 05 đọc `reports/grid_results.npz` (điểm out-of-fold của notebook 04, không đưa vào git)
và `reports/search_results.csv`. Chạy trước hai lệnh:

```bash
python scripts/run_grid.py      # nếu chưa có reports/grid_results.npz
python scripts/run_search.py    # 30 lần thử × 5 fold, khoảng 20 phút trên 12 lõi
```

Chạy không cần giao diện:

```bash
jupyter nbconvert --to notebook --execute --inplace notebooks/01_eda.ipynb
# lặp lại cho 02 … 08, hoặc dùng vòng lặp
```

Kiểm tra hiện vật đã đủ:

```bash
ls models/
# model.joblib  explainer.joblib  metrics.json  threshold.json
```

## 4. Chạy ứng dụng

### 4.1 Docker Compose (khuyến nghị)

```bash
docker compose up
# Cơ sở dữ liệu → localhost:5432
# API           → http://localhost:8000/docs
# Giao diện     → http://localhost:3000
```

Thứ tự khởi động: `db` chạy trước, `api` chờ healthcheck `pg_isready` rồi tự chạy
`alembic upgrade head` trước khi phục vụ. Không cần làm gì thủ công.

Yêu cầu: `models/` và `data/` đã có hiện vật từ bước 3 — chúng được gắn dạng
volume chỉ đọc, không nướng vào image.

Lần chạy đầu tiên mất 20–40 giây vì PostgreSQL phải `initdb`; các lần sau còn
8–12 giây ([06 §7.2](06-thiet-ke-luu-tru.md)).

Dừng và đặt lại trạng thái:

```bash
docker compose down                 # dừng, GIỮ nguyên dữ liệu trong volume pgdata
docker compose down -v              # dừng và XÓA volume — mất toàn bộ dữ liệu ứng dụng

# Hoặc chỉ xóa dữ liệu, giữ lược đồ và ngưỡng đã chọn:
docker compose exec db psql -U fraud -d fraud \n  -c "TRUNCATE transactions, reviews RESTART IDENTITY CASCADE;"
```

### 4.2 Chạy trực tiếp khi phát triển

```bash
docker compose up -d db                        # chỉ cơ sở dữ liệu
alembic upgrade head                           # khi có migration mới
uvicorn api.main:app --reload --port 8000
python -m http.server 3000 --directory web     # cửa sổ terminal thứ hai
```

Quy trình khi đổi lược đồ:

```bash
alembic revision --autogenerate -m "them cot x"   # sinh migration từ models_orm.py
# đọc lại tệp trong migrations/versions/ — autogenerate không phải lúc nào cũng đúng
alembic upgrade head
alembic downgrade -1                              # lùi lại một bước nếu cần
```

### 4.3 Phương án A — Streamlit

```bash
docker compose up -d db
streamlit run app.py
```

## 5. Kiểm thử

```bash
docker compose up -d db             # kiểm thử tích hợp cần PostgreSQL
pytest                              # toàn bộ
pytest tests/test_features.py -v    # chỉ kiểm thử vàng, không cần cơ sở dữ liệu
pytest -k threshold                 # theo từ khóa
```

`tests/conftest.py` tự tạo cơ sở dữ liệu `fraud_test` và xóa sau khi chạy xong;
không bao giờ đụng tới cơ sở dữ liệu phát triển ([08 §1.1](08-ke-hoach-kiem-thu.md)).

Chạy `pytest` trước mỗi commit. Chi tiết các ca ở
[08 — Kế hoạch kiểm thử](08-ke-hoach-kiem-thu.md).

## 6. Bảo đảm tái lập (NFR-07)

| Yếu tố | Cách kiểm soát |
|---|---|
| Hạt giống ngẫu nhiên | `random_state=42` ở mọi bước có tham số này (ML-05) |
| Phiên bản thư viện | `requirements.txt` dùng `>=`; khi nộp bài, xuất bản ghim chính xác bằng `pip freeze > requirements.lock.txt` |
| Thứ tự dữ liệu | Không `shuffle` ngoài các chỗ đã ghim seed |
| Song song hóa | XGBoost `tree_method="hist"` cho điểm **lệch nhẹ theo số luồng** (thứ tự cộng dồn histogram thay đổi): cùng cấu hình chạy `n_jobs=1` và `n_jobs=-1` khác nhau ở chữ số thứ ba–tư của xác suất. Trên cùng một máy thì tái lập tuyệt đối. Khi kiểm tra tái lập (T-38) trên máy khác, ghi lại số lõi; nếu lệch vượt 0,001 thì ghim `n_jobs` cố định. Phát hiện ở giai đoạn 4, xem `tests/test_search.py` |
| Phiên bản mô hình | `model_version` ghi trong `threshold.json` và gắn vào mọi phản hồi API |

Quy trình kiểm tra tái lập (thực hiện ngày 20):

```bash
# 1. Ghi lại PR-AUC hiện tại
python -c "import json; print(json.load(open('models/metrics.json'))['headline']['pr_auc']['value'])"

# 2. Chạy lại toàn bộ notebook trong kernel sạch

# 3. So lại — chênh lệch phải < 0,001
```

Nếu lệch lớn hơn, tìm theo thứ tự: bước ngẫu nhiên chưa ghim seed → thao tác
không ổn định về thứ tự (`groupby`, `set`) → khác phiên bản thư viện.

## 7. Xử lý sự cố

| Triệu chứng | Nguyên nhân thường gặp | Cách xử lý |
|---|---|---|
| `/health` trả 503 | Chưa có hiện vật trong `models/` | Chạy notebook 08 |
| API khởi động chậm hơn 15 giây | Lần chạy đầu (PostgreSQL `initdb` + migration), hoặc `metrics.json` quá lớn do kèm `test_scores` | Đo lại ở lần chạy thứ hai; nếu vẫn chậm thì tách `test_scores` ra `.npz` ([06 §3.2](06-thiet-ke-luu-tru.md)) |
| Điểm rủi ro ở demo lệch số trong báo cáo | Đặc trưng lệch giữa train và serve | Chạy `pytest tests/test_features.py` — TC-01 sẽ chỉ ra |
| Kéo thanh trượt giật | Đang gọi API mỗi lần kéo thay vì tính tại máy khách | Kiểm tra UI-D1 ([07 §2](07-thiet-ke-giao-dien.md)) |
| Hết RAM khi chạy SMOTE | `sampling_strategy` đặt quá cao | Giảm về 0,1; giảm mẫu lớp âm trước |
| Notebook 04 chạy quá 3 giờ | `n_estimators` quá lớn ở giai đoạn so sánh | Hạ xuống 100, chỉ tăng lại khi tinh chỉnh |
| PR-AUC > 0,95 | Gần như chắc chắn rò rỉ dữ liệu | Rà theo [08 §3.1](08-ke-hoach-kiem-thu.md) |
| `docker compose up` báo thiếu tệp | Volume trỏ sai hoặc chưa có hiện vật | Kiểm tra đường dẫn trong `docker-compose.yml` |
| `connection refused` tới cổng 5432 | Container `db` chưa sẵn sàng, hoặc chạy API mà quên bật `db` | `docker compose up -d db`, đợi `pg_isready` trả `accepting connections` |
| `port is already allocated` cho 5432 | Máy đã có PostgreSQL khác | Đổi ánh xạ cổng thành `"5433:5432"` và sửa `DATABASE_URL` |
| API báo lỗi ở yêu cầu đầu tiên sau khi `db` khởi động lại | Kết nối cũ trong pool đã chết | Bật `pool_pre_ping=True` ([06 §4.1](06-thiet-ke-luu-tru.md)) |
| `alembic upgrade head` báo "target database is not up to date" | Lược đồ bị sửa tay không qua migration | `docker compose down -v` rồi dựng lại từ migration |
| Hàng đợi tải chậm sau khi nạp lô lớn | Thống kê của bộ tối ưu truy vấn còn cũ | `VACUUM ANALYZE transactions;` ([06 §7.5](06-thiet-ke-luu-tru.md)) |
| `TypeError: unsupported operand` với `amount` | `NUMERIC` trả về `Decimal`, không phải `float` | Ép `float(row.amount)` trước khi tính (ST-07) |

## 8. Quy ước làm việc với repo

| Chủ đề | Quy ước |
|---|---|
| Nhánh | Làm trên nhánh riêng, không commit thẳng vào nhánh mặc định |
| Trước khi commit | Chạy `pytest`; xóa đầu ra nặng trong notebook nếu vượt 5 MB |
| Tệp không commit | `data/*.csv`, `data/*.parquet`, `models/*.joblib`, `.env`, `backup/` — xem [06 §10](06-thiet-ke-luu-tru.md) |
| Migration | Mọi thay đổi lược đồ phải qua Alembic và **được commit**; không sửa bảng bằng tay trên cơ sở dữ liệu đang chạy |
| Sửa tài liệu | Sửa đúng tệp sở hữu quyết định đó; các tệp khác chỉ tham chiếu |
| Đổi đặc trưng | Bắt buộc cập nhật `src/features.py` **và** dữ liệu vàng của TC-01 cùng một lúc |

## 9. Danh sách kiểm cho người chấm

- [ ] Tạo môi trường theo §2.1, không báo lỗi.
- [ ] Tải dữ liệu theo §2.2, kiểm tra ra `(284807, 31) 492`.
- [ ] Chạy `pytest` — toàn bộ xanh.
- [ ] Chạy notebook 01 → 08, đối chiếu PR-AUC với báo cáo (lệch < 0,001).
- [ ] `docker compose up`, đợi `/health` trả 200 (lần đầu 20–40 giây), mở `http://localhost:3000`, duyệt bốn màn hình.
- [ ] Thẩm định một giao dịch, chạy `docker compose restart api`, tải lại trang — kết luận vẫn còn (AC-A7).
- [ ] Mở `http://localhost:8000/docs`, thử `POST /score` với một mẫu từ `/samples`.
- [ ] Ở màn hình ngưỡng, kéo từ 0,5 về ngưỡng đề xuất và quan sát các chỉ số đổi.
