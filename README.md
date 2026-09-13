# Fraud Detection — Phát hiện gian lận thẻ tín dụng

Đề tài  — Phát hiện gian lận thẻ tín dụng (Credit Card Fraud Detection)
Mô tả bài toán. Phân loại nhị phân cực kỳ mất cân bằng: phát hiện giao dịch gian lận. Trọng tâm là kỹ thuật xử lý dữ liệu mất cân bằng và lựa chọn metric phù hợp.
Dữ liệu (Kaggle). Dataset "Credit Card Fraud Detection" của owner Machine Learning Group - ULB (mlg-ulb) (kaggle.com/datasets/mlg-ulb/creditcardfraud), file creditcard.csv. Theo mô tả chính thức của mlg-ulb: "This dataset presents transactions that occurred in two days, where we have 492 frauds out of 284,807 transactions. The dataset is highly unbalanced, the positive class (frauds) account for 0.172% of all transactions." [Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud/activity)[Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud/metadata) → 284.807 giao dịch, 492 gian lận (0,172%), 31 cột: [uspto](https://image-ppubs.uspto.gov/dirsearch-public/print/downloadPdf/11416748) Time, V1–V28 (đã biến đổi PCA để ẩn danh), Amount, và Class (0/1). [arxiv](https://arxiv.org/pdf/2401.04139) Giao dịch của chủ thẻ châu Âu, tháng 9/2013. [Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud/activity)[Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud/metadata) License Database Contents License (DbCL) v1.0.
Công nghệ sử dụng. Python, pandas, NumPy; matplotlib, seaborn; scikit-learn (Logistic Regression, Random Forest, đánh giá); imbalanced-learn (SMOTE, undersampling); XGBoost/LightGBM; tùy chọn PyTorch (autoencoder cho anomaly detection); SHAP.
Ánh xạ với chương trình học. Ch.1; Ch.2 (PCA — chính các đặc trưng V1–V28 là thành phần chính); Ch.3 EDA (phân bố Amount/Time, tương quan); Ch.4 (xác suất, dữ liệu mất cân bằng cực đoan, ý nghĩa Precision/Recall/PR-AUC); Ch.5 (Logistic Regression, cây quyết định); Ch.6 (ensemble, SMOTE, tuning ngưỡng, cost-sensitive learning); Ch.8 (tùy chọn: autoencoder). Ch.7 không vận dụng.
Pipeline & mô hình. (1) EDA + scaling Amount/Time; (2) baseline: Logistic Regression; (3) xử lý mất cân bằng (SMOTE/undersampling/class_weight); (4) nâng cao: Random Forest/XGBoost; (5) tối ưu ngưỡng theo PR curve; (6) tùy chọn autoencoder; (7) giải thích SHAP.
Metrics. PR-AUC (quan trọng nhất do mất cân bằng), Recall, Precision, F1, ROC-AUC, confusion matrix. KHÔNG dùng accuracy làm metric chính (baseline "luôn đoán không gian lận" đã đạt 99,8% accuracy).
Deliverable. Notebook phân tích, so sánh chiến lược xử lý mất cân bằng, mô hình đã lưu, báo cáo về trade-off Precision–Recall.
Phạm vi & độ khó. Độ khó: Dễ–Trung bình (dữ liệu đã sạch, PCA sẵn), nhưng thách thức nằm ở xử lý mất cân bằng và chọn metric.

Dự án phân tích và xây dựng mô hình phát hiện giao dịch thẻ tín dụng gian lận
trên bộ dữ liệu [Credit Card Fraud Detection (Kaggle)](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
— 284.807 giao dịch, chỉ 492 gian lận (~0,17%), tức mất cân bằng cực nặng.

## Cấu trúc

```
fraud-detection/
├── data/creditcard.csv          # không commit lên git (~144 MB)
├── notebooks/                   # 01 → 07, chạy theo thứ tự
├── src/                         # code dùng chung cho notebook và app
├── models/best_model.pkl        # model tốt nhất sau notebook 05
├── reports/                     # figures/ và bao-cao.md
├── app.py                       # demo Streamlit
└── requirements.txt
```

## Cài đặt

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

## Quy trình

| Notebook | Nội dung | Trạng thái |
|---|---|---|
| `01_eda.ipynb` | Toàn vẹn dữ liệu, 8 biểu đồ EDA bắt buộc → `reports/figures/01_*.png` | ✅ |
| `02_statistics.ipynb` | Mann–Whitney U, hiệu chỉnh BH, Cohen's d và Cliff's delta → `reports/feature_ranking.csv` | ✅ |
| `03_baseline.ipynb` | Mô hình rỗng + Logistic Regression + Decision Tree, bảng mốc cho G-2 → `data/test_set.parquet` | ✅ |
| `04_imbalance_strategies.ipynb` | class_weight, undersampling, SMOTE | |
| `05_advanced_models.ipynb` | Random Forest, XGBoost/LightGBM, tuning → `best_model.pkl` | |
| `06_threshold_and_cost.ipynb` | Chọn ngưỡng theo chi phí nghiệp vụ | |
| `07_explainability.ipynb` | SHAP, feature importance, phân tích lỗi | |

## Chạy demo

```bash
streamlit run app.py
```

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

