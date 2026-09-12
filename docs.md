# ĐỒ ÁN CUỐI KHÓA: PHÁT HIỆN GIAN LẬN THẺ TÍN DỤNG
## Hướng dẫn triển khai chi tiết (3 tuần)
 
---
 
## PHẦN 0. ĐỊNH VỊ ĐỀ TÀI
 
### Điều gì làm nên một đồ án tốt ở đề tài này
 
Dataset `creditcard.csv` là bộ dữ liệu đã được xử lý sẵn: không có giá trị thiếu, V1–V28 đã qua PCA nên đã chuẩn hóa, không cần feature engineering phức tạp. Nghĩa là **phần "làm sạch dữ liệu" gần như bằng không** — bạn không thể ghi điểm ở đó.
 
Giá trị của đồ án nằm ở ba chỗ:
 
1. **Hiểu vì sao accuracy là metric vô dụng ở đây.** Mô hình "luôn dự đoán không gian lận" đạt 99,828% accuracy. Nếu báo cáo của bạn khoe accuracy 99,9%, hội đồng sẽ biết ngay bạn chưa hiểu bài toán.
2. **Xử lý mất cân bằng ĐÚNG CÁCH.** Đây là nơi 80% người làm đề tài này mắc lỗi nghiêm trọng (xem Phần 4).
3. **Chọn ngưỡng quyết định dựa trên chi phí kinh doanh**, không phải mặc định 0.5.
### Mục tiêu cụ thể cần đạt
 
| Hạng mục | Mức đạt | Mức tốt |
|---|---|---|
| PR-AUC (Average Precision) trên test | ≥ 0.75 | ≥ 0.82 |
| Recall tại ngưỡng đã tối ưu | ≥ 0.75 | ≥ 0.85 |
| Precision tương ứng | ≥ 0.60 | ≥ 0.80 |
| So sánh ≥ 4 chiến lược mất cân bằng | Bắt buộc | + phân tích chi phí |
| Không rò rỉ dữ liệu (data leakage) | Bắt buộc | + kiểm chứng bằng time-based split |
 
---
 
## PHẦN 1. HIỂU DỮ LIỆU
 
### 1.1 Cấu trúc
 
```
creditcard.csv — 284.807 dòng × 31 cột, ~144 MB
├── Time    : số GIÂY trôi qua kể từ giao dịch đầu tiên (0 → 172.792)
├── V1–V28  : 28 thành phần chính từ PCA (đã ẩn danh, đã chuẩn hóa)
├── Amount  : số tiền giao dịch (EUR), CHƯA chuẩn hóa
└── Class   : 0 = bình thường (284.315), 1 = gian lận (492)
```
 
Tỷ lệ gian lận: **0,172%** — cứ 578 giao dịch mới có 1 gian lận.
 
### 1.2 Ba đặc điểm quyết định cách bạn làm
 
**(a) `Time` không phải là timestamp thật.** Nó là số giây tương đối, trải dài đúng 2 ngày (172.792 giây ≈ 48 giờ). Dùng thô nó làm feature là sai — mô hình sẽ học "gian lận hay xảy ra ở giây thứ X", vô nghĩa khi triển khai. Có hai cách xử lý đúng:
- Biến đổi thành **giờ trong ngày**: `hour = (Time // 3600) % 24` → có ý nghĩa thực (gian lận tăng vào ban đêm).
- Hoặc **bỏ hẳn** `Time` khỏi feature, chỉ dùng nó để chia train/test theo thời gian.
**(b) `Amount` lệch cực mạnh.** Trung vị khoảng 22 EUR nhưng giá trị lớn nhất trên 25.000 EUR. Dùng `StandardScaler` sẽ bị outlier kéo lệch. Nên dùng **`RobustScaler`** (dựa trên trung vị và IQR) hoặc biến đổi `log1p(Amount)`.
 
**(c) V1–V28 đã là output của PCA** nên không diễn giải được ý nghĩa từng biến, và **không nên PCA lần nữa**. Nhưng đây là điểm hay để nói về Chương 2: bạn có thể giải thích tại sao PCA được dùng ở đây (ẩn danh dữ liệu nhạy cảm mà vẫn giữ được cấu trúc phương sai).
 
### 1.3 Hạn chế phải nêu trong báo cáo
 
Chỉ có 492 mẫu gian lận. Nếu tách 20% làm test, tập test chỉ có **~98 mẫu dương**. Nghĩa là mỗi mẫu dương bị phân loại sai làm Recall thay đổi ~1%. Kết quả của bạn có **phương sai lớn** — bắt buộc phải báo cáo qua cross-validation nhiều lần, không được báo cáo một con số duy nhất từ một lần chia.
 
---
 
## PHẦN 2. CẤU TRÚC PROJECT
 
```
fraud-detection/
├── data/
│   └── creditcard.csv          # không commit lên git (144 MB)
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_statistics.ipynb
│   ├── 03_baseline.ipynb
│   ├── 04_imbalance_strategies.ipynb
│   ├── 05_advanced_models.ipynb
│   ├── 06_threshold_and_cost.ipynb
│   └── 07_explainability.ipynb
├── src/
│   ├── data.py                 # load, split
│   ├── evaluate.py             # hàm đánh giá dùng chung
│   └── plots.py
├── models/
│   └── best_model.pkl
├── reports/
│   ├── figures/
│   └── bao-cao.pdf
├── app.py                      # demo Streamlit
├── requirements.txt
└── README.md
```
 
### Môi trường
 
```
python>=3.10
pandas, numpy, scipy
scikit-learn>=1.3
imbalanced-learn>=0.11
xgboost, lightgbm
matplotlib, seaborn
shap
streamlit
joblib
```
 
---
 
## PHẦN 3. LỘ TRÌNH 3 TUẦN
 
| Tuần | Nội dung | Chương học vận dụng |
|---|---|---|
| **Tuần 1** | EDA + thống kê + baseline + thiết lập khung đánh giá | Ch.2, 3, 4, 5 |
| **Tuần 2** | So sánh chiến lược mất cân bằng + mô hình nâng cao + tuning | Ch.5, 6 |
| **Tuần 3** | Tối ưu ngưỡng theo chi phí + SHAP + (autoencoder) + báo cáo + demo | Ch.6, 8 |
 
Chia nhỏ theo ngày:
 
- **Ngày 1–2:** Tải dữ liệu, EDA cơ bản, biểu đồ phân bố.
- **Ngày 3–4:** Kiểm định thống kê, so sánh phân bố hai lớp.
- **Ngày 5–7:** Thiết lập khung chia dữ liệu + hàm đánh giá + baseline Logistic Regression.
- **Ngày 8–11:** Chạy 5 chiến lược mất cân bằng, lập bảng so sánh.
- **Ngày 12–14:** XGBoost/LightGBM + tuning siêu tham số.
- **Ngày 15–17:** Phân tích ngưỡng, ma trận chi phí, chọn điểm vận hành.
- **Ngày 18–19:** SHAP + (tùy chọn) autoencoder.
- **Ngày 20–21:** Viết báo cáo, làm demo, chuẩn bị bảo vệ.
---
 
## PHẦN 4. ĐIỂM MẤU CHỐT — TRÁNH RÒ RỈ DỮ LIỆU
 
Đây là phần quan trọng nhất của cả đồ án. Đọc kỹ.
 
### 4.1 Lỗi kinh điển: SMOTE trước khi chia dữ liệu
 
```python
# ❌ SAI HOÀN TOÀN — đây là lỗi phổ biến nhất trên Kaggle
X_res, y_res = SMOTE().fit_resample(X, y)
X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.2)
# → Mẫu tổng hợp sinh ra từ mẫu test đã lọt vào train.
# → Kết quả: Recall 0.99, Precision 0.99. Đẹp giả tạo. Vô giá trị.
```
 
**Nguyên tắc bất di bất dịch: chỉ resample trên tập train, sau khi đã chia.** Và trong cross-validation, phải resample **bên trong mỗi fold**, không phải trước vòng lặp CV.
 
```python
# ✅ ĐÚNG — dùng Pipeline của imbalanced-learn
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import RobustScaler
from sklearn.linear_model import LogisticRegression
 
pipe = ImbPipeline([
    ('scaler', RobustScaler()),
    ('smote',  SMOTE(random_state=42)),   # chỉ áp dụng trên phần train của mỗi fold
    ('clf',    LogisticRegression(max_iter=1000))
])
# Khi đưa pipe này vào cross_val_score / GridSearchCV,
# imblearn tự động BỎ QUA bước SMOTE khi transform tập validation.
```
 
Lưu ý: `imblearn.pipeline.Pipeline` khác `sklearn.pipeline.Pipeline`. Chỉ bản của imblearn mới xử lý đúng bước resampling. Đây là chi tiết đáng đưa vào báo cáo.
 
### 4.2 Chia dữ liệu
 
Dùng **hai cách chia** và so sánh — đây là điểm cộng lớn:
 
```python
from sklearn.model_selection import train_test_split, StratifiedKFold
 
# Cách 1: chia ngẫu nhiên phân tầng (chuẩn cho báo cáo chính)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)
 
# Cách 2: chia theo thời gian (mô phỏng triển khai thực tế)
# Ngày 1 để train, ngày 2 để test
split_time = 86400  # 24 giờ
train_mask = df['Time'] < split_time
X_train_t, y_train_t = X[train_mask], y[train_mask]
X_test_t,  y_test_t  = X[~train_mask], y[~train_mask]
```
 
Cách 2 luôn cho kết quả thấp hơn. Đó không phải là mô hình tệ đi — đó là con số **trung thực hơn**, vì trong thực tế bạn luôn dự đoán tương lai bằng dữ liệu quá khứ. Đưa cả hai vào báo cáo và giải thích chênh lệch là một luận điểm rất mạnh.
 
### 4.3 Cross-validation
 
```python
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
```
 
Bắt buộc dùng **Stratified** — nếu không, một fold có thể không chứa mẫu gian lận nào.
 
---
 
## PHẦN 5. EDA (Chương 3)
 
### Danh sách biểu đồ cần có
 
| # | Biểu đồ | Insight kỳ vọng |
|---|---|---|
| 1 | Bar chart tỷ lệ hai lớp (thang log) | Trực quan hóa mức độ mất cân bằng 578:1 |
| 2 | Histogram `Amount` theo từng lớp (thang log) | Gian lận tập trung ở giá trị nhỏ–trung bình |
| 3 | Boxplot `Amount` hai lớp | Trung vị gian lận khác trung vị bình thường |
| 4 | Số giao dịch theo giờ trong ngày, tách hai lớp | Giao dịch bình thường theo nhịp sinh hoạt; gian lận phân bố khác |
| 5 | **Tỷ lệ** gian lận theo giờ (không phải số tuyệt đối) | Tỷ lệ gian lận cao rõ rệt vào khung 2–4 giờ sáng |
| 6 | Heatmap tương quan V1–V28 với `Class` | Một số biến (V14, V17, V12, V10, V4, V11) tương quan mạnh nhất |
| 7 | KDE plot 6 biến V tương quan mạnh nhất, tách hai lớp | Phân bố tách biệt rõ → tín hiệu phân loại tốt |
| 8 | Scatter 2D sau t-SNE/UMAP trên mẫu con | Cụm gian lận tách khỏi khối chính |
 
Biểu đồ số 5 quan trọng hơn số 4 nhiều. Số giao dịch gian lận tuyệt đối ban đêm thấp (vì tổng giao dịch thấp), nhưng **tỷ lệ** thì cao vọt. Đây là loại insight cho thấy bạn biết đọc dữ liệu mất cân bằng.
 
```python
df['hour'] = (df['Time'] // 3600) % 24
fraud_rate = df.groupby('hour')['Class'].agg(['mean', 'sum', 'count'])
fraud_rate['mean'] *= 100  # phần trăm
```
 
### Kiểm tra bắt buộc
 
```python
df.duplicated().sum()   # dataset này có ~1.081 dòng trùng lặp
```
 
Có dòng trùng lặp thật. Quyết định giữ hay bỏ, và **giải thích lý do** trong báo cáo — cả hai lựa chọn đều bảo vệ được, nhưng phải nhất quán.
 
---
 
## PHẦN 6. THỐNG KÊ (Chương 4)
 
Đây là phần nhiều người làm đề tài này bỏ qua, nên làm tốt sẽ nổi bật.
 
### 6.1 So sánh phân bố hai lớp
 
Vì hai nhóm có kích thước rất chênh lệch và phân bố không chuẩn, **không dùng t-test** mà dùng kiểm định phi tham số:
 
```python
from scipy.stats import mannwhitneyu, ks_2samp
 
results = []
for col in [f'V{i}' for i in range(1, 29)] + ['Amount']:
    a = df.loc[df.Class == 0, col]
    b = df.loc[df.Class == 1, col]
    u_stat, p_u = mannwhitneyu(a, b, alternative='two-sided')
    ks_stat, p_ks = ks_2samp(a, b)
    results.append({'feature': col, 'ks_stat': ks_stat, 'p_ks': p_ks, 'p_mwu': p_u})
```
 
### 6.2 Hiệu chỉnh đa kiểm định
 
Bạn vừa chạy 29 kiểm định cùng lúc. Nếu lấy ngưỡng p < 0,05 cho từng cái, xác suất có ít nhất một kết quả dương tính giả là rất cao. **Bắt buộc hiệu chỉnh:**
 
```python
from statsmodels.stats.multitest import multipletests
reject, p_adj, _, _ = multipletests(df_res['p_ks'], alpha=0.05, method='fdr_bh')
```
 
Nêu rõ trong báo cáo bạn dùng Benjamini–Hochberg (FDR) và vì sao. Đây chính xác là loại nội dung Chương 4 muốn thấy.
 
### 6.3 Effect size
 
p-value với n = 284.807 gần như luôn nhỏ hơn 0,001 — nó không nói lên điều gì về độ mạnh của tín hiệu. Cần báo cáo thêm **thống kê KS** hoặc **Cliff's delta** làm thước đo độ lớn hiệu ứng.
 
### 6.4 Khoảng tin cậy cho metric
 
Vì tập test chỉ có ~98 mẫu dương, mọi con số Recall đều có sai số lớn. Tính khoảng tin cậy bằng bootstrap:
 
```python
from sklearn.utils import resample
from sklearn.metrics import average_precision_score
import numpy as np
 
scores = []
for _ in range(1000):
    idx = resample(np.arange(len(y_test)), random_state=None)
    if y_test.iloc[idx].sum() < 5:      # bỏ qua mẫu bootstrap thiếu lớp dương
        continue
    scores.append(average_precision_score(y_test.iloc[idx], y_proba[idx]))
 
print(f"PR-AUC = {np.mean(scores):.3f}, 95% CI = [{np.percentile(scores, 2.5):.3f}, {np.percentile(scores, 97.5):.3f}]")
```
 
Báo cáo "PR-AUC = 0.82 (95% CI: 0.75–0.88)" thay vì "PR-AUC = 0.82" là khác biệt giữa đồ án khá và đồ án tốt.
 
---
 
## PHẦN 7. KHUNG ĐÁNH GIÁ
 
Viết một lần, dùng cho mọi mô hình. Đặt trong `src/evaluate.py`.
 
```python
import numpy as np
from sklearn.metrics import (average_precision_score, roc_auc_score,
                             precision_recall_curve, confusion_matrix,
                             classification_report)
 
def evaluate(y_true, y_proba, threshold=0.5, name=""):
    y_pred = (y_proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
 
    return {
        'model': name,
        'PR_AUC': average_precision_score(y_true, y_proba),   # metric CHÍNH
        'ROC_AUC': roc_auc_score(y_true, y_proba),
        'threshold': threshold,
        'precision': tp / (tp + fp) if (tp + fp) else 0,
        'recall': tp / (tp + fn) if (tp + fn) else 0,
        'f1': 2*tp / (2*tp + fp + fn) if tp else 0,
        'TP': tp, 'FP': fp, 'FN': fn, 'TN': tn,
        'alerts': tp + fp,                       # số giao dịch phải đưa đi thẩm định
        'precision_at_recall_80': precision_at_recall(y_true, y_proba, 0.80),
    }
 
def precision_at_recall(y_true, y_proba, target_recall):
    """Precision đạt được khi ép Recall >= target. Metric vận hành rất thực tế."""
    prec, rec, _ = precision_recall_curve(y_true, y_proba)
    mask = rec >= target_recall
    return prec[mask].max() if mask.any() else 0.0
```
 
### Vì sao PR-AUC chứ không phải ROC-AUC
 
ROC-AUC trên bộ này của hầu hết mô hình đều rơi vào 0,96–0,98 — nhìn thì đẹp nhưng gần như không phân biệt được mô hình tốt và mô hình tầm thường. Lý do: ROC dùng **tỷ lệ dương tính giả** (FP/tổng số âm), mà mẫu số ở đây là 284.315 nên FP tăng vài trăm cũng gần như không nhúc nhích đường cong.
 
PR-AUC dùng **Precision** (TP/(TP+FP)) — nhạy trực tiếp với số cảnh báo sai. Khi FP tăng từ 50 lên 500, ROC-AUC giảm 0,001 còn PR-AUC giảm 0,3. Đây chính là điều mà ngân hàng quan tâm.
 
Đường baseline của PR-AUC bằng tỷ lệ lớp dương = **0,00172**. Nghĩa là PR-AUC = 0,80 tương đương cải thiện gấp khoảng 465 lần so với đoán ngẫu nhiên. Nêu con số này trong báo cáo.
 
---
 
## PHẦN 8. BASELINE (Chương 5)
 
Chạy ba baseline theo thứ tự tăng dần, tất cả **không** xử lý mất cân bằng, để có mốc so sánh:
 
1. **Dummy classifier** (luôn đoán lớp 0) — để trưng ra accuracy 99,83% và chứng minh nó vô nghĩa.
2. **Logistic Regression** thuần.
3. **Decision Tree** đơn giản (max_depth=5) — dễ diễn giải.
```python
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
 
baselines = {
    'Dummy (always 0)': DummyClassifier(strategy='most_frequent'),
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
    'Decision Tree (d=5)': DecisionTreeClassifier(max_depth=5, random_state=42),
}
```
 
Bảng đầu tiên trong báo cáo nên trông như thế này — nó tự nói lên toàn bộ luận điểm về metric:
 
| Mô hình | Accuracy | ROC-AUC | PR-AUC | Recall | Precision |
|---|---|---|---|---|---|
| Dummy (always 0) | 0,9983 | 0,500 | 0,0017 | 0,000 | — |
| Logistic Regression | 0,9991 | ~0,97 | ~0,72 | ~0,62 | ~0,87 |
| Decision Tree | 0,9992 | ~0,90 | ~0,70 | ~0,74 | ~0,80 |
 
Dummy có accuracy 99,83% nhưng Recall bằng 0 — nó bỏ lọt **toàn bộ** gian lận.
 
---
 
## PHẦN 9. SO SÁNH CHIẾN LƯỢC MẤT CÂN BẰNG (Chương 6)
 
Đây là trọng tâm khoa học của đồ án. Chạy đủ 5 chiến lược, **cùng một mô hình cơ sở** (Logistic Regression) để so sánh công bằng, rồi lặp lại với XGBoost.
 
| # | Chiến lược | Cách làm | Ưu | Nhược |
|---|---|---|---|---|
| 1 | Không xử lý | Giữ nguyên | Không rủi ro | Recall thấp |
| 2 | `class_weight='balanced'` | Tăng trọng số lớp thiểu số trong hàm mất mát | Không sinh dữ liệu giả, rẻ, thường rất hiệu quả | Có thể tăng FP |
| 3 | Random Undersampling | Giảm mẫu lớp đa số | Train cực nhanh | Vứt bỏ ~99,8% dữ liệu, mất thông tin |
| 4 | SMOTE | Nội suy sinh mẫu tổng hợp lớp thiểu số | Giữ toàn bộ dữ liệu gốc | Sinh mẫu ở vùng chồng lấn → nhiều FP |
| 5 | SMOTE + Tomek / SMOTEENN | Sinh mẫu rồi dọn mẫu nhiễu ở biên | Biên quyết định sạch hơn | Chậm, phức tạp |
 
```python
from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.under_sampling import RandomUnderSampler
from imblearn.combine import SMOTETomek
 
strategies = {
    'none':          None,
    'class_weight':  'use_class_weight',        # xử lý riêng ở tham số model
    'undersample':   RandomUnderSampler(random_state=42),
    'smote':         SMOTE(random_state=42, k_neighbors=5),
    'smote_tomek':   SMOTETomek(random_state=42),
}
```
 
### Kết quả bạn nhiều khả năng sẽ thấy
 
`class_weight='balanced'` thường cho PR-AUC **cao hơn hoặc ngang** SMOTE, mà lại đơn giản và nhanh hơn nhiều. SMOTE thường đẩy Recall lên cao nhưng Precision tụt mạnh (nhiều cảnh báo sai).
 
Nếu bạn kết luận được điều này bằng số liệu của chính mình, kèm giải thích **tại sao** (SMOTE nội suy trong không gian đặc trưng đã qua PCA, tạo mẫu ở vùng hai lớp chồng lấn, làm biên quyết định lấn sang phía lớp đa số) — đó là một kết luận có giá trị, ngược với kỳ vọng thông thường rằng "SMOTE luôn tốt hơn".
 
### Trình bày kết quả
 
Vẽ **đường PR curve của cả 5 chiến lược trên cùng một biểu đồ**. Một hình này thay thế cả trang bảng số.
 
---
 
## PHẦN 10. MÔ HÌNH NÂNG CAO (Chương 6)
 
### 10.1 Cấu hình
 
```python
import xgboost as xgb
 
neg, pos = (y_train == 0).sum(), (y_train == 1).sum()
 
model = xgb.XGBClassifier(
    n_estimators=500,
    max_depth=4,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=neg/pos,          # ≈ 578 — cách xử lý mất cân bằng của XGBoost
    eval_metric='aucpr',               # tối ưu trực tiếp PR-AUC
    early_stopping_rounds=50,
    random_state=42,
    n_jobs=-1,
)
```
 
Ba tham số quan trọng nhất: `scale_pos_weight` (thay cho SMOTE), `eval_metric='aucpr'` (khớp với metric bạn quan tâm), và `max_depth` thấp (4–6; sâu hơn sẽ overfit vào 492 mẫu dương).
 
### 10.2 Tuning
 
Dùng `RandomizedSearchCV` hoặc Optuna với `scoring='average_precision'`. **Không dùng `scoring='accuracy'` hay `'f1'`** — nếu tuning theo accuracy, mô hình tốt nhất sẽ là mô hình đoán toàn 0.
 
```python
from sklearn.model_selection import RandomizedSearchCV
 
search = RandomizedSearchCV(
    model, param_distributions=param_grid,
    scoring='average_precision',       # = PR-AUC
    cv=StratifiedKFold(5, shuffle=True, random_state=42),
    n_iter=30, n_jobs=-1, random_state=42,
)
```
 
### 10.3 Mô hình nên chạy
 
- Random Forest (`class_weight='balanced_subsample'`)
- XGBoost (`scale_pos_weight`)
- LightGBM (`is_unbalance=True`)
- Ensemble bỏ phiếu mềm của 3 cái trên (tùy chọn)
---
 
## PHẦN 11. TỐI ƯU NGƯỠNG THEO CHI PHÍ
 
Đây là phần biến đồ án học thuật thành đồ án có tư duy ứng dụng. Rất ít sinh viên làm phần này.
 
### 11.1 Vấn đề với ngưỡng 0.5
 
Ngưỡng 0.5 là mặc định tùy tiện, không có cơ sở nào cho bài toán mất cân bằng. Ngưỡng đúng phải xuất phát từ **chi phí của hai loại lỗi**:
 
- **False Negative (bỏ lọt gian lận):** ngân hàng mất tiền — trung bình bằng `Amount` của giao dịch đó (~122 EUR với giao dịch gian lận trong bộ này).
- **False Positive (báo động giả):** chi phí thẩm định thủ công + phiền khách hàng — giả định ~5 EUR/vụ.
Tỷ lệ chi phí khoảng **24:1**, nghĩa là bỏ lọt một vụ gian lận đắt gấp 24 lần một cảnh báo sai. Ngưỡng tối ưu do đó phải **thấp hơn 0,5 rất nhiều**.
 
### 11.2 Tìm ngưỡng tối ưu
 
```python
import numpy as np
 
COST_FN = 122.0   # EUR mất trung bình khi bỏ lọt (dùng Amount trung bình của gian lận)
COST_FP = 5.0     # EUR chi phí thẩm định một cảnh báo sai
 
thresholds = np.linspace(0.001, 0.999, 500)
costs = []
for t in thresholds:
    y_pred = (y_proba >= t).astype(int)
    fn = ((y_pred == 0) & (y_test == 1)).sum()
    fp = ((y_pred == 1) & (y_test == 0)).sum()
    costs.append(fn * COST_FN + fp * COST_FP)
 
best_t = thresholds[int(np.argmin(costs))]
print(f"Ngưỡng tối ưu: {best_t:.4f}, tổng chi phí: {min(costs):,.0f} EUR")
```
 
Phiên bản chính xác hơn: dùng **`Amount` thực của từng giao dịch** thay vì giá trị trung bình, vì bỏ lọt một vụ 2.000 EUR tốn hơn nhiều so với vụ 20 EUR.
 
### 11.3 Ràng buộc vận hành
 
Cách tiếp cận thứ hai, thực tế trong ngành: **đội thẩm định chỉ xử lý được N vụ/ngày**. Giả sử 100 vụ/ngày trên ~140.000 giao dịch/ngày:
 
```python
# Lấy top-K giao dịch có điểm rủi ro cao nhất
K = 200                                    # ngân sách thẩm định cho 2 ngày test
top_k_idx = np.argsort(y_proba)[-K:]
recall_at_k = y_test.iloc[top_k_idx].sum() / y_test.sum()
print(f"Recall@{K} = {recall_at_k:.3f}")   # bắt được bao nhiêu % gian lận với K lần kiểm tra
```
 
Báo cáo **Recall@K** cho vài mức K (100, 200, 500, 1000) và vẽ thành đường cong. Đây là ngôn ngữ mà người vận hành hệ thống thật sự dùng.
 
### 11.4 Bảng cần có trong báo cáo
 
| Ngưỡng | Cơ sở chọn | Recall | Precision | TP | FP | FN | Chi phí (EUR) |
|---|---|---|---|---|---|---|---|
| 0,500 | Mặc định | | | | | | |
| ~0,05 | Tối ưu chi phí | | | | | | |
| — | F1 lớn nhất | | | | | | |
| — | Ràng buộc Recall ≥ 0,90 | | | | | | |
| — | Ngân sách 200 lượt thẩm định | | | | | | |
 
---
 
## PHẦN 12. AUTOENCODER (Chương 8, tùy chọn)
 
Cách tiếp cận khác về bản chất: coi gian lận là **bất thường** thay vì một lớp cần phân loại.
 
Ý tưởng: huấn luyện autoencoder **chỉ trên giao dịch bình thường**. Mô hình học cách nén và tái tạo giao dịch bình thường tốt. Khi gặp giao dịch gian lận, sai số tái tạo sẽ cao bất thường → dùng sai số làm điểm rủi ro.
 
```python
import torch, torch.nn as nn
 
class Autoencoder(nn.Module):
    def __init__(self, input_dim=29):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 20), nn.ReLU(),
            nn.Linear(20, 14), nn.ReLU(),
            nn.Linear(14, 7),  nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(7, 14),  nn.ReLU(),
            nn.Linear(14, 20), nn.ReLU(),
            nn.Linear(20, input_dim),
        )
    def forward(self, x):
        return self.decoder(self.encoder(x))
 
# Huấn luyện CHỈ trên X_train[y_train == 0]
# Điểm rủi ro = MSE tái tạo trên tập test
```
 
**Kỳ vọng thực tế:** autoencoder thường cho PR-AUC khoảng 0,3–0,5 trên bộ này, **thấp hơn rõ rệt** XGBoost. Đừng coi đó là thất bại — hãy giải thích trong báo cáo: khi đã có nhãn đầy đủ và tương đối nhiều (492 mẫu), học có giám sát tận dụng được thông tin nhãn tốt hơn nhiều so với phương pháp không giám sát. Autoencoder chỉ vượt trội khi **hoàn toàn không có nhãn** hoặc khi cần bắt kiểu gian lận mới chưa từng xuất hiện trong tập train. Kết luận có sắc thái như vậy đáng giá hơn nhiều so với việc chỉ khoe mô hình nào cao điểm nhất.
 
---
 
## PHẦN 13. GIẢI THÍCH MÔ HÌNH (SHAP)
 
```python
import shap
 
explainer = shap.TreeExplainer(best_model)
shap_values = explainer.shap_values(X_test_sample)     # lấy mẫu ~2000 dòng cho nhanh
 
shap.summary_plot(shap_values, X_test_sample)          # xếp hạng biến quan trọng
shap.summary_plot(shap_values, X_test_sample, plot_type='bar')
 
# Giải thích một giao dịch gian lận cụ thể
idx = np.where(y_test == 1)[0][0]
shap.force_plot(explainer.expected_value, shap_values[idx], X_test_sample.iloc[idx])
```
 
**Điểm cần bàn trong báo cáo:** V1–V28 là thành phần PCA nên không có ý nghĩa nghiệp vụ — bạn không thể nói "biến thu nhập quan trọng nhất". Đây là **đánh đổi giữa quyền riêng tư và khả năng diễn giải**: dữ liệu được ẩn danh để bảo vệ khách hàng, nhưng cái giá phải trả là mô hình trở thành hộp đen với người vận hành. Trong hệ thống thật, các quy định như GDPR yêu cầu giải thích được quyết định từ chối giao dịch — đây là một hạn chế nghiêm túc đáng nêu.
 
Điều bạn **vẫn** làm được: xác định V14, V17, V12, V10, V4 là nhóm biến chi phối, và kiểm tra tính nhất quán giữa xếp hạng của SHAP với xếp hạng từ kiểm định KS ở Phần 6. Nếu hai phương pháp độc lập cho cùng thứ tự, đó là bằng chứng tốt.
 
---
 
## PHẦN 14. DELIVERABLE
 
### Bắt buộc
1. **7 notebook** theo cấu trúc ở Phần 2, chạy được từ đầu đến cuối, có markdown giải thích giữa các bước.
2. **Báo cáo** 15–25 trang: đặt vấn đề → dữ liệu → EDA → thống kê → phương pháp → kết quả → phân tích chi phí → hạn chế → kết luận.
3. **Mô hình đã lưu:** `joblib.dump(pipeline, 'models/best_model.pkl')` — lưu cả pipeline (gồm scaler), không chỉ mô hình.
4. **Bảng so sánh tổng hợp** tất cả mô hình × chiến lược, cùng metric, cùng cách chia dữ liệu.
5. **File `requirements.txt`** và README nêu cách tái lập kết quả (kèm `random_state`).
### Điểm cộng
6. **Demo Streamlit:** nhập/tải lên giao dịch → trả về xác suất rủi ro, quyết định theo ngưỡng đã chọn, và biểu đồ SHAP cho giao dịch đó.
7. **Dockerfile** đóng gói app.
8. **Phân tích drift:** so sánh phân bố ngày 1 và ngày 2, bàn về việc mô hình cần huấn luyện lại theo chu kỳ.
---
 
## PHẦN 15. NHỮNG LỖI PHẢI TRÁNH
 
| Lỗi | Hậu quả | Cách tránh |
|---|---|---|
| SMOTE trước khi chia train/test | Kết quả ảo ~0,99, đồ án mất giá trị | Dùng `imblearn.pipeline.Pipeline` |
| Scaler `fit` trên toàn bộ dữ liệu | Rò rỉ thống kê từ test sang train | `fit` trên train, `transform` trên test |
| Báo cáo accuracy làm metric chính | Chứng tỏ chưa hiểu bài toán | Dùng PR-AUC + Recall + Precision |
| Dùng ngưỡng mặc định 0,5 | Bỏ lọt phần lớn gian lận | Tối ưu ngưỡng theo chi phí |
| Tuning theo `scoring='accuracy'` | Mô hình "tốt nhất" là mô hình đoán toàn 0 | `scoring='average_precision'` |
| Dùng `Time` thô làm feature | Học đặc điểm vô nghĩa của 2 ngày cụ thể | Chuyển thành giờ trong ngày, hoặc bỏ |
| `StandardScaler` cho `Amount` | Bị outlier 25.000 EUR kéo lệch | `RobustScaler` hoặc `log1p` |
| KFold thường thay vì Stratified | Có fold không chứa mẫu gian lận nào | `StratifiedKFold` |
| Báo cáo một con số từ một lần chia | Không phản ánh phương sai lớn (chỉ ~98 mẫu dương) | CV lặp lại + khoảng tin cậy bootstrap |
| Nói "mô hình đạt 99,9% nên rất tốt" | Sai lầm cốt lõi của cả đề tài | Luôn đối chiếu với baseline Dummy |
 
---
 
## PHẦN 16. CHUẨN BỊ BẢO VỆ
 
Những câu hội đồng nhiều khả năng sẽ hỏi:
 
1. **"Vì sao không dùng accuracy?"** → Trình bày ngay dòng Dummy classifier: 99,83% accuracy, Recall 0.
2. **"SMOTE hoạt động thế nào và vì sao nó rủi ro ở đây?"** → Nội suy giữa các mẫu thiểu số lân cận trong không gian đặc trưng; rủi ro vì sinh mẫu ở vùng chồng lấn hai lớp và vì rất dễ áp dụng sai thứ tự so với bước chia dữ liệu.
3. **"Vì sao PR-AUC chứ không phải ROC-AUC?"** → Mẫu số của FPR là 284.315 nên ROC không nhạy với số cảnh báo sai; Precision thì nhạy trực tiếp.
4. **"Ngưỡng của em chọn thế nào?"** → Cho xem đường cong chi phí và điểm cực tiểu, kèm giả định COST_FN/COST_FP.
5. **"V14 nghĩa là gì?"** → Không biết được, và đó là chủ ý — dữ liệu đã ẩn danh bằng PCA. Nêu luôn đánh đổi riêng tư ↔ diễn giải.
6. **"Mô hình này triển khai thật được không?"** → Nêu hạn chế: dữ liệu chỉ 2 ngày, năm 2013, một thị trường; concept drift; cần huấn luyện lại định kỳ; cần độ trễ dự đoán dưới ngưỡng cho phép trong xử lý giao dịch thời gian thực.
7. **"Nếu cho thêm thời gian em sẽ làm gì?"** → Chuẩn bị sẵn 2–3 hướng: dữ liệu chuỗi hành vi theo thẻ, học trực tuyến, mô hình cost-sensitive học trực tiếp ma trận chi phí.
---
 
## PHẦN 17. HƯỚNG MỞ RỘNG (nếu muốn điểm cao hơn)
 
1. **Đánh giá theo thời gian:** train ngày 1, test ngày 2 — báo cáo mức tụt so với chia ngẫu nhiên và giải thích.
2. **Cost-sensitive learning trực tiếp:** đưa ma trận chi phí vào hàm mất mát thay vì hậu xử lý ngưỡng.
3. **Hiệu chuẩn xác suất (calibration):** dùng `CalibratedClassifierCV` và vẽ reliability diagram — mô hình cây thường cho xác suất không hiệu chuẩn, mà phân tích chi phí lại phụ thuộc vào xác suất đúng.
4. **So sánh với Isolation Forest / One-Class SVM** để có thêm góc nhìn không giám sát bên cạnh autoencoder.
5. **Phân tích công bằng (fairness):** kiểm tra mô hình có tỷ lệ báo động sai cao bất thường ở nhóm giao dịch giá trị nhỏ hay không — liên quan trực tiếp đến trải nghiệm khách hàng.
---
 
## PHỤ LỤC: KHỞI ĐỘNG NHANH
 
```bash
mkdir -p fraud-detection/{data,notebooks,src,models,reports/figures}
cd fraud-detection
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install pandas numpy scipy scikit-learn imbalanced-learn xgboost lightgbm \
            matplotlib seaborn shap streamlit joblib statsmodels jupyter
```
 
Tải dữ liệu (cần Kaggle API token đặt tại `~/.kaggle/kaggle.json`):
 
```bash
pip install kaggle
kaggle datasets download -d mlg-ulb/creditcardfraud -p data/ --unzip
```
 
Kiểm tra nhanh:
 
```python
import pandas as pd
df = pd.read_csv('data/creditcard.csv')
print(df.shape)                                    # (284807, 31)
print(df['Class'].value_counts())                  # 284315 / 492
print(f"Tỷ lệ gian lận: {df['Class'].mean()*100:.4f}%")   # 0.1727%
print(f"Trùng lặp: {df.duplicated().sum()}")
print(f"Thiếu dữ liệu: {df.isnull().sum().sum()}")        # 0
```
 