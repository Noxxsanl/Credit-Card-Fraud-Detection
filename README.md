# Fraud Detection — Phát hiện gian lận thẻ tín dụng

Phân loại nhị phân cực kỳ mất cân bằng: phát hiện giao dịch thẻ tín dụng gian lận. Trọng tâm của đề
tài là kỹ thuật xử lý dữ liệu mất cân bằng, lựa chọn metric phù hợp và chọn ngưỡng theo chi phí.

## Đề tài

| | |
|---|---|
| **Dữ liệu** | [Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) — Machine Learning Group, ULB (Kaggle `mlg-ulb`), tệp `creditcard.csv`. Giao dịch của chủ thẻ châu Âu trong hai ngày của tháng 9/2013: **284.807 giao dịch, 492 gian lận (0,172%)**. 31 cột: `Time`, `V1`–`V28` (thành phần chính sau PCA, để ẩn danh), `Amount`, `Class` (0/1). Giấy phép Database Contents License (DbCL) v1.0 |
| **Công nghệ** | Python, pandas, NumPy; matplotlib, seaborn; scikit-learn; imbalanced-learn (SMOTE, undersampling); XGBoost; SHAP. Autoencoder (tùy chọn) dùng `MLPRegressor` vì môi trường không có PyTorch |
| **Pipeline** | (1) EDA + chuẩn hoá `Amount`, mã hoá giờ từ `Time`; (2) mô hình cơ sở Logistic Regression; (3) xử lý mất cân bằng: SMOTE, undersampling, `class_weight`; (4) Random Forest, XGBoost; (5) chọn ngưỡng theo đường PR và chi phí; (6) autoencoder (tùy chọn); (7) giải thích bằng SHAP |
| **Metric** | PR-AUC là chính; Recall, Precision, F1, ROC-AUC, ma trận nhầm lẫn. **Không** dùng accuracy: mô hình "luôn đoán hợp lệ" đã đạt 99,83% |
| **Sản phẩm** | Notebook phân tích, so sánh chiến lược mất cân bằng, mô hình đã lưu, báo cáo về đánh đổi Precision–Recall |
| **Chương trình học** | Ch.1; Ch.2 (PCA — chính `V1`–`V28` là thành phần chính); Ch.3 EDA (phân bố `Amount`/`Time`, tương quan); Ch.4 (xác suất, mất cân bằng cực đoan, ý nghĩa Precision/Recall/PR-AUC); Ch.5 (Logistic Regression, cây quyết định); Ch.6 (ensemble, SMOTE, chọn ngưỡng, học theo chi phí); Ch.8 (tùy chọn: autoencoder). Ch.7 không vận dụng |
| **Độ khó** | Dễ–Trung bình: dữ liệu đã sạch, PCA sẵn; thách thức nằm ở xử lý mất cân bằng và chọn metric |

## Cấu trúc

```
fraud-detection/
├── data/creditcard.csv          # không commit lên git (~144 MB)
├── notebooks/                   # 01 → 08, chạy theo thứ tự
├── src/                         # code dùng chung cho notebook và app
├── models/                      # hiện vật do notebook 08 sinh: model.joblib, explainer.joblib,
│                                #   metrics.json, threshold.json
├── reports/                     # figures/ và bao-cao.md
├── api/                         # FastAPI + PostgreSQL (phương án B, giai đoạn 7) — kèm Dockerfile
├── frontend/                    # giao diện 4 màn hình, Next.js + TypeScript + Tailwind + Recharts (bản đóng gói mặc định)
├── web/                         # cùng giao diện, bản HTML + Alpine.js + Chart.js (dự phòng)
├── deploy/nginx.conf            # nginx của dịch vụ web: tệp tĩnh + chuyển tiếp /api
├── docker-compose.yml           # db + api + web — cả hệ thống bằng một lệnh (giai đoạn 9)
├── scripts/                     # tải dữ liệu, lưới, tái lập, dữ liệu demo, bấm giờ, kiểm giao diện
├── tests/                       # pytest
├── app.py                       # demo Streamlit (phương án A, dự phòng)
└── requirements.txt
```

## Chạy lại từ đầu trên máy sạch

Cách nhanh nhất để xem hệ thống chạy: **chỉ cần Git và Docker**. Không cần Python, Node.js hay
PostgreSQL trên máy, không cần tệp `.env`.

### Bước 1 — Cài đặt

- [Git](https://git-scm.com/downloads).
- [Docker Desktop](https://www.docker.com/products/docker-desktop/) (Docker Engine 25 trở lên), **bật
  lên** trước khi chạy lệnh. Trên Windows, Docker Desktop cần WSL 2 — trình cài đặt tự hướng dẫn.
- Máy có ít nhất 4 GB RAM trống cho Docker và khoảng 3 GB ổ đĩa cho các ảnh.

### Bước 2 — Lấy mã nguồn

```bash
git clone https://github.com/Noxxsanl/Credit-Card-Fraud-Detection.git fraud-detection
cd fraud-detection
```

### Bước 3 — Đặt hiện vật mô hình vào chỗ

API không huấn luyện; nó nạp 6 tệp do notebook 08 xuất ra. Hai tệp `.json` đã có trong git, bốn tệp còn
lại không (dung lượng, và chúng sinh lại được). `explainer.joblib` chỉ notebook và kiểm thử cần — API
tính SHAP bằng chính XGBoost:

```
models/model.joblib   models/oof_scores.npz   models/explainer.joblib (không bắt buộc)
models/metrics.json   models/threshold.json            ← có sẵn trong git
data/test_set.parquet data/sample_pool.json
```

API tự kiểm hiện vật lúc khởi động — cùng một lần xuất, đúng cấu trúc, chấm lại tập kiểm thử ra đúng
điểm đã lưu. Sai thì `api` không lên `(healthy)` và `docker compose logs api` nói rõ lý do.

**Cách A — có gói hiện vật** (`hien-vat.tgz`, nộp kèm bài hoặc chép từ máy đã chạy notebook). Đặt tệp ở
gốc repo rồi giải nén — `tar` có sẵn trên Windows 10 trở lên, macOS và Linux:

```bash
tar -xzf hien-vat.tgz
```

Tạo gói này trên máy đã có hiện vật:

```bash
tar -czf hien-vat.tgz models/model.joblib models/explainer.joblib models/oof_scores.npz models/metrics.json models/threshold.json data/test_set.parquet data/sample_pool.json
```

**Cách B — tự tạo hiện vật** (cần Python 3.12, khoảng 20 phút cộng thời gian tải 150 MB dữ liệu).
`requirements.txt` ghim **đúng** phiên bản của lần xuất gốc — đừng nâng cấp gói, notebook 08 sẽ dừng vì
số lệch:

```bash
python -m venv .venv
.venv\Scripts\activate                          # Windows; macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
python scripts/download_data.py --mirror        # hoặc qua Kaggle API, xem docs/10 §2.4
jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=3600 notebooks/03_baseline.ipynb
jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=3600 notebooks/08_export_artifacts.ipynb
pytest tests/test_artifacts.py                  # kiểm chính các tệp vừa sinh
```

Notebook 08 đối chiếu từng con số với các bảng `reports/*.csv` của lần chạy gốc và dừng nếu lệch. Trên
máy có số lõi khác 12, XGBoost có thể cho điểm lệch nhẹ theo số luồng ([docs/10 §6](docs/10-van-hanh-tai-lap.md))
và notebook dừng ở bước đối chiếu — khi đó dùng cách A. Chạy lại toàn bộ 01 → 08: [docs/10 §3](docs/10-van-hanh-tai-lap.md).

### Bước 4 — Chạy

```bash
docker compose up --build -d
```

Lần đầu Docker tải ảnh nền và build hai ảnh, khoảng **4–5 phút** (cần Internet). Sau đó hệ thống tự khởi
động theo thứ tự: PostgreSQL → API dựng lược đồ (`alembic upgrade head`) và nạp mô hình → giao diện.
Khoảng 15 giây sau khi build xong, lệnh sau phải cho `api` ở trạng thái `(healthy)`:

```bash
docker compose ps
```

| Địa chỉ | Nội dung |
|---|---|
| http://localhost:3000 | Giao diện: hàng đợi, ngưỡng, hiệu năng mô hình, phát lại |
| http://localhost:8000/docs | Tài liệu API, thử từng endpoint |

Các cổng chỉ mở trên `127.0.0.1` (API không có đăng nhập). Cần mở giao diện từ máy khác, ví dụ máy chiếu
của phòng bảo vệ: đặt `BIND_ADDRESS=0.0.0.0` trong `.env` rồi `docker compose up -d` (xem `.env.example`).

Hàng đợi lúc đầu rỗng; giao diện hướng dẫn ba cách nạp dữ liệu (tải CSV, thư viện mẫu, phát lại ngày 2).
Có Python thì nạp sẵn một bộ dữ liệu demo bằng `python scripts/demo_db.py seed`
([docs/lenh-chay §9.5](docs/lenh-chay.md)).

### Dừng và chạy lại

```bash
docker compose down          # dừng; dữ liệu (giao dịch, kết luận thẩm định) vẫn giữ trong volume pgdata
docker compose up -d         # chạy lại, không build lại — khoảng 11 giây
docker compose down -v       # dừng và XÓA toàn bộ dữ liệu ứng dụng
docker compose restart api   # sau khi chạy lại notebook 08, để API nạp hiện vật mới
```

### Khi gặp lỗi

| Triệu chứng | Cách xử lý |
|---|---|
| `failed to connect to the docker API` | Docker Desktop chưa bật |
| `port is already allocated` (5432, 8000 hoặc 3000) | Máy đã có chương trình khác ở cổng đó. Tạo tệp `.env` ở gốc repo với `POSTGRES_HOST_PORT=5433` (hoặc `API_HOST_PORT=8001`, `WEB_HOST_PORT=3001`), chạy lại |
| `api` không lên `(healthy)`, giao diện báo "Chưa nạp được hiện vật mô hình" | Thiếu hoặc sai hiện vật ở bước 3: `docker compose logs api` ghi rõ tệp nào |
| Giao diện báo "API chưa sẵn sàng" trong vài giây đầu | Bình thường: API đang nạp mô hình. Giao diện tự thử lại mỗi 15 giây |
| Build lỗi ở `npm ci` (mạng chặn npm) | Dùng bản giao diện không cần build: `WEB_UI=web docker compose up --build -d` (PowerShell: `$env:WEB_UI="web"` trước lệnh) |

Đủ lệnh, số đo khởi động và kết quả nghiệm thu: [docs/lenh-chay §9](docs/lenh-chay.md).
Vận hành chi tiết: [docs/10 §4.1](docs/10-van-hanh-tai-lap.md).

## Cài đặt môi trường Python (notebook, kiểm thử, phát triển)

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
pip install -r requirements.txt
```

Tải dữ liệu:

```bash
python scripts/download_data.py            # qua Kaggle API (cần ~/.kaggle/kaggle.json)
python scripts/download_data.py --mirror   # bản sao công khai, không cần token
python scripts/download_data.py --check    # kiểm tra tệp đang có
```

Script tự kiểm tra toàn vẹn: phải ra đúng 284.807 dòng, 31 cột, 492 mẫu gian lận.

Chạy lưới 20 tổ hợp của notebook 04 (khoảng 20–40 phút, có điểm lưu nên ngắt được):

```bash
python scripts/run_grid.py --smoke    # thử nhanh trên 12.000 dòng
python scripts/run_grid.py            # chạy thật
python scripts/run_grid.py --status   # xem tiến độ
```

## Quy trình

| Notebook | Nội dung | Trạng thái |
|---|---|---|
| `01_eda.ipynb` | Toàn vẹn dữ liệu, 8 biểu đồ EDA bắt buộc → `reports/figures/01_*.png` | ✅ |
| `02_statistics.ipynb` | Mann–Whitney U, hiệu chỉnh BH, Cohen's d và Cliff's delta → `reports/feature_ranking.csv` | ✅ |
| `03_baseline.ipynb` | Mô hình rỗng + Logistic Regression + Decision Tree, bảng mốc cho G-2 → `data/test_set.parquet` | ✅ |
| `04_imbalance_strategies.ipynb` | Lưới 5 chiến lược × 4 mô hình → `reports/grid_results.csv` | ✅ |
| `05_advanced_models.ipynb` | Tinh chỉnh XGBoost, đánh giá trên tập test kèm khoảng tin cậy, chia theo thời gian, rà soát rò rỉ → `reports/final_test_metrics.csv` | ✅ |
| `06_threshold_and_cost.ipynb` | Chọn ngưỡng theo chi phí nghiệp vụ trên out-of-fold, độ nhạy theo tỷ lệ chi phí → `reports/threshold_comparison.csv`. Mục 6: chi phí theo số tiền, ngưỡng đề xuất chặn, độ tin cậy của điểm | ✅ |
| `06b_autoencoder.ipynb` | (Tùy chọn) autoencoder chỉ học lớp hợp lệ, so với mô hình có giám sát | ✅ |
| `07_explainability.ipynb` | SHAP toàn cục, đối chiếu xếp hạng thống kê, phân tích lỗi → `reports/shap_ranking.csv` | ✅ |
| `08_export_artifacts.ipynb` | Xuất hiện vật cho API → `models/*`, `data/sample_pool.json` | ✅ |

## Phương án ứng dụng — đã chốt: **B** (T-39)

Quyết định tại mốc cuối ngày 14 ([docs/09 §2](docs/09-ke-hoach-trien-khai.md)), ngày 2026-09-28:
làm **phương án B** — API FastAPI + PostgreSQL và giao diện web riêng (giai đoạn 7–9).

| Tiêu chí của mốc | Hiện trạng |
|---|---|
| Hiện vật đã đủ | Đủ: `model.joblib`, `explainer.joblib`, `metrics.json`, `threshold.json`, `oof_scores.npz`, `sample_pool.json` — kiểm bằng `tests/test_artifacts.py` (từ 2026-10-09 API không cần `explainer.joblib` nữa) |
| Không còn nợ việc ở phần mô hình | Giai đoạn 0–6 xong 38/38 việc, gồm cả việc tùy chọn T-34; AC-M1…AC-M8 đạt |
| Tái lập | T-38 đạt: chạy lại 01 → 08 trong kernel sạch, PR-AUC lệch 0 |

Lưới an toàn vẫn giữ: nếu giai đoạn 7–8 trễ tới mức đe doạ ngày báo cáo (rủi ro R-03, R-08), quay về
**phương án A** — Streamlit trong `app.py`, rút gọn giao diện và dồn thời gian cho báo cáo. `app.py` đã
nạp đúng `models/model.joblib` và τ\* của `threshold.json`, nên dùng được ngay làm phương án dự phòng
khi Docker không chạy được trong buổi bảo vệ.

## Chạy khi phát triển (không đóng gói)

Chạy API và giao diện trực tiếp trên máy để sửa mã và tải lại ngay; PostgreSQL vẫn trong Docker.

```bash
cp .env.example .env                # chỉnh POSTGRES_HOST_PORT nếu cổng 5432 đã bị chiếm
docker compose up -d db             # PostgreSQL 16
alembic upgrade head                # dựng lược đồ
uvicorn api.main:app --port 8000    # tài liệu API: http://localhost:8000/docs
python -m http.server 3000 --directory web    # cửa sổ thứ hai → giao diện: http://localhost:3000
```

Hoặc bản Next.js (cần Node.js 20.9+), cũng ở cổng 3000 — chạy một trong hai:

```bash
cd frontend && npm install && npm run dev
```

Cần có hiện vật trong `models/` (chạy notebook 08). Giao diện là tệp tĩnh (Alpine.js + Chart.js nằm sẵn
trong `web/vendor/`), không cần Node.js hay `npm install`. Chi tiết ở
[docs/10 §4.2](docs/10-van-hanh-tai-lap.md) và [docs/lenh-chay §8](docs/lenh-chay.md).

## Chạy demo Streamlit (phương án A, dự phòng)

```bash
streamlit run app.py
```

Chỉ cần môi trường Python và ba tệp `models/model.joblib`, `models/threshold.json`,
`data/sample_pool.json` — không cần Docker, PostgreSQL hay `creditcard.csv`. Có thư viện mẫu theo bốn
nhóm (gian lận dễ/khó, hợp lệ dễ/khó) và tab tải CSV; đặc trưng sinh bằng `build_features` như API.

## Công việc

Danh sách 67 việc để hoàn thành đề tài, tick được: [TASKS.md](TASKS.md).

## Tài liệu

Toàn bộ đặc tả và thiết kế nằm trong [docs/](docs/) — bắt đầu từ
[docs/README.md](docs/README.md).

| # | Tài liệu | Trả lời câu hỏi |
|---|---|---|
| 00 | [Tổng quan dự án](docs/00-tong-quan.md) | Làm gì, vì sao, ranh giới đến đâu |
| 01 | [Đặc tả yêu cầu](docs/01-dac-ta-yeu-cau.md) | Hệ thống phải làm được những gì |
| 02 | [Đặc tả dữ liệu](docs/02-dac-ta-du-lieu.md) | Dữ liệu có gì, hợp đồng dữ liệu ra sao |
| 03 | [Thiết kế kiến trúc](docs/03-thiet-ke-kien-truc.md) | Hệ thống gồm khối nào, ghép ra sao |
| 04 | [Thiết kế mô hình ML](docs/04-thiet-ke-mo-hinh-ml.md) | Huấn luyện, đánh giá, chọn ngưỡng thế nào |
| 05 | [Thiết kế API](docs/05-thiet-ke-api.md) | Hợp đồng giữa backend và frontend |
| 06 | [Thiết kế lưu trữ](docs/06-thiet-ke-luu-tru.md) | Cơ sở dữ liệu và hiện vật mô hình |
| 07 | [Thiết kế giao diện](docs/07-thiet-ke-giao-dien.md) | Bốn màn hình trông và hoạt động thế nào |
| 08 | [Kế hoạch kiểm thử](docs/08-ke-hoach-kiem-thu.md) | Chứng minh hệ thống đúng bằng cách nào |
| 09 | [Kế hoạch triển khai](docs/09-ke-hoach-trien-khai.md) | Ai làm gì, ngày nào, xong là thế nào |
| 10 | [Vận hành và tái lập](docs/10-van-hanh-tai-lap.md) | Chạy lại toàn bộ từ đầu bằng lệnh gì |

## Lưu ý về metric

Accuracy vô nghĩa ở tỉ lệ 0,17% — dự đoán "tất cả hợp lệ" đã đạt 99,83%.
Toàn bộ dự án đánh giá bằng **PR-AUC**, precision/recall tại ngưỡng đã chọn,
và chi phí nghiệp vụ (FN đắt hơn FP nhiều lần).

