# 04 — Thiết kế mô hình học máy

Tài liệu này mô tả toàn bộ giai đoạn offline: từ dữ liệu thô tới bộ hiện vật mà
API nạp. Lược đồ dữ liệu và quy tắc đặc trưng nằm ở [02](02-dac-ta-du-lieu.md).

## 1. Quy trình huấn luyện

```
B1.  Nạp dữ liệu, kiểm tra toàn vẹn, loại dòng trùng lặp        (src/data.py)
B2.  Sinh đặc trưng hour_sin, hour_cos                          (src/features.py)
B3.  Chia phân tầng 80/20, random_state=42
       → lưu tập kiểm thử ra data/test_set.parquet
B4.  Dựng pipeline imblearn: [scaler] → [resampler] → [classifier]
B5.  Chạy lưới 5 chiến lược × 4 mô hình
       StratifiedKFold(5), scoring = average_precision
B6.  Chọn cấu hình tốt nhất theo PR-AUC trung bình qua các fold
B7.  Tinh chỉnh siêu tham số: RandomizedSearchCV, 30 lần thử
B8.  Huấn luyện lại trên toàn bộ tập train, đánh giá trên tập test
B9.  Bootstrap khoảng tin cậy cho các chỉ số chính
B10. Dựng đường cong chi phí, xác định ngưỡng đề xuất
B11. Sinh SHAP explainer, tính tầm quan trọng toàn cục
B12. Xuất hiện vật ra models/ và data/
```

Ánh xạ sang notebook:

| Bước | Notebook |
|---|---|
| B1–B2 | `01_eda.ipynb` |
| — | `02_statistics.ipynb` (kiểm định, không đưa vào pipeline) |
| B3, baseline | `03_baseline.ipynb` |
| B4–B6 | `04_imbalance_strategies.ipynb` |
| B7–B9 | `05_advanced_models.ipynb` |
| B10 | `06_threshold_and_cost.ipynb` |
| B11 | `07_explainability.ipynb` |
| B12 | `08_export_artifacts.ipynb` |

## 2. Ràng buộc bắt buộc — chống rò rỉ dữ liệu

Đây là phần quyết định điểm số của đồ án. Vi phạm bất kỳ điều nào dưới đây làm
toàn bộ kết quả mất giá trị, dù các con số trông rất đẹp.

| Mã | Ràng buộc | Điều gì hỏng nếu vi phạm |
|---|---|---|
| ML-01 | Mọi bước lấy mẫu lại phải nằm **trong** `imblearn.pipeline.Pipeline` | Nếu resample trước khi chia, mẫu SMOTE nội suy từ mẫu test lọt vào train; mọi chỉ số vọt lên khoảng 0,99 và hoàn toàn vô nghĩa |
| ML-02 | Dùng `StratifiedKFold`, tuyệt đối không `KFold` | Với 0,172% lớp dương, một fold có thể không chứa mẫu gian lận nào; `average_precision` trên fold đó không xác định |
| ML-03 | `scoring='average_precision'` khi tinh chỉnh | Tối ưu theo accuracy sẽ chọn mô hình dự đoán toàn lớp 0 |
| ML-04 | Scaler `fit` chỉ trên tập huấn luyện | Trung vị và IQR tính từ cả tập là rò rỉ thống kê |
| ML-05 | `random_state=42` ở mọi bước ngẫu nhiên | NFR-07 tái lập |
| ML-06 | Loại dòng trùng lặp **trước** khi chia tập | Cùng một dòng nằm ở cả hai tập |
| ML-07 | Không dùng `Time` thô làm đặc trưng | Mô hình học một đặc điểm không tồn tại khi triển khai (DS-03) |
| ML-08 | Chọn ngưỡng trên dữ liệu huấn luyện hoặc dữ liệu out-of-fold, không trên tập kiểm thử | Nếu dò ngưỡng trên tập test rồi báo cáo chỉ số cũng trên tập test, chỉ số bị lạc quan |

### 2.1 Lỗi kinh điển và cách viết đúng

```python
# ❌ SAI — lỗi phổ biến nhất trong các notebook trên Kaggle
X_res, y_res = SMOTE().fit_resample(X, y)
X_train, X_test, y_train, y_test = train_test_split(X_res, y_res)
# Mẫu tổng hợp sinh từ mẫu test đã lọt vào train.
# Kết quả: Recall 0,99 — Precision 0,99. Đẹp giả tạo, vô giá trị.

# ✅ ĐÚNG — resampler nằm trong pipeline của imblearn
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import RobustScaler

pipe = ImbPipeline([
    ("scaler",     RobustScaler()),
    ("resampler",  SMOTE(random_state=42)),
    ("clf",        LogisticRegression(max_iter=1000, random_state=42)),
])
# Khi đưa pipe này vào cross_val_score / RandomizedSearchCV,
# imblearn TỰ ĐỘNG bỏ qua bước resampler khi transform tập validation.
# sklearn.pipeline.Pipeline KHÔNG làm được điều này.
```

**ML-01 được kiểm chứng thế nào:** rà soát mã trong buổi bảo vệ, và bằng
`tests/test_no_leakage.py` — kiểm tra rằng mọi pipeline trong lưới thí nghiệm là
instance của `imblearn.pipeline.Pipeline` khi có bước resample.

## 3. Lưới thí nghiệm

### 3.1 Chiến lược xử lý mất cân bằng

| # | Chiến lược | Cấu hình | Ý tưởng |
|---|---|---|---|
| S1 | Không xử lý | — | Mốc tham chiếu |
| S2 | Trọng số lớp | `class_weight='balanced'` hoặc `scale_pos_weight=578` | Phạt nặng lỗi trên lớp hiếm, không đổi dữ liệu |
| S3 | Giảm mẫu ngẫu nhiên | `RandomUnderSampler(sampling_strategy=0.1)` | Giảm lớp đa số, mất thông tin |
| S4 | SMOTE | `SMOTE(sampling_strategy=0.1, k_neighbors=5)` | Nội suy mẫu dương mới |
| S5 | SMOTE + Tomek | `SMOTETomek` | Sinh mẫu rồi dọn vùng biên chồng lấn |

Lưu ý về `sampling_strategy`: **không** đưa về 1:1. Cân bằng hoàn toàn làm méo
xác suất tiên nghiệm rất mạnh, khiến mô hình trả xác suất lệch cao và precision
sụp. Tỷ lệ 0,1 (khoảng 1:10) là điểm khởi đầu hợp lý; ghi rõ lựa chọn này trong
báo cáo.

### 3.2 Thuật toán

| # | Mô hình | Cấu hình khởi điểm | Vai trò |
|---|---|---|---|
| M1 | Logistic Regression | `max_iter=1000`, `solver='liblinear'` | Mốc tuyến tính, dễ diễn giải |
| M2 | Decision Tree | `max_depth=6` | Mốc phi tuyến đơn giản, minh họa Chương 5 |
| M3 | Random Forest | `n_estimators=300`, `n_jobs=-1` | Ensemble bagging |
| M4 | XGBoost / LightGBM | `n_estimators=500`, `learning_rate=0.05`, `max_depth=6` | Ensemble boosting, ứng viên vô địch |

5 × 4 = **20 tổ hợp**. Ghi toàn bộ vào một bảng duy nhất — đây là bảng trung tâm
của báo cáo, kể cả các dòng cho kết quả kém.

### 3.3 Định dạng bảng kết quả

| Mô hình | Chiến lược | PR-AUC (CV) | Độ lệch chuẩn | ROC-AUC | Recall@τ* | Precision@τ* | F1 | Thời gian huấn luyện |
|---|---|---|---|---|---|---|---|---|

τ* là ngưỡng tối ưu chi phí của chính cấu hình đó, xác định trên dữ liệu
out-of-fold (ML-08).

### 3.4 Kết quả nhiều khả năng sẽ thấy

Nêu trước để biết khi nào kết quả là bất thường:

- Boosting (M4) thắng ở PR-AUC, thường trong khoảng 0,80–0,87.
- `scale_pos_weight` (S2) ngang bằng hoặc **nhỉnh hơn** SMOTE với mô hình cây, mà
  rẻ hơn nhiều về tính toán. Đây là phát hiện đáng viết hẳn một đoạn thảo luận.
- SMOTE giúp rõ rệt cho Logistic Regression, giúp ít cho mô hình cây.
- Giảm mẫu (S3) cho recall cao nhưng precision thấp nhất — vứt đi quá nhiều thông tin.
- Nếu thấy PR-AUC > 0,95, gần như chắc chắn có rò rỉ. Kiểm tra lại ML-01 và ML-06.

### 3.5 Kết quả thực đo (notebook 04, giai đoạn 3)

Lưới chạy 45,7 phút trên 226.980 dòng. Đối chiếu với năm dự báo ở §3.4:

| Dự báo | Kết quả | |
|---|---|---|
| Boosting thắng, PR-AUC 0,80–0,87 | XGBoost 0,8549 | ✅ đúng |
| `scale_pos_weight` ≥ SMOTE với mô hình cây, rẻ hơn | **chỉ đúng với XGBoost** | ⚠️ một nửa |
| SMOTE giúp rõ rệt cho Logistic Regression | +0,0156 so với `class_weight`, và rẻ hơn | ✅ đúng |
| Giảm mẫu: recall cao, precision thấp nhất | recall cao ở cả 4; precision thấp nhất chỉ ở LR và DT | ⚠️ một nửa |
| PR-AUC > 0,95 là rò rỉ | cao nhất 0,8549 | ✅ không báo động |

**Đính chính dự báo thứ hai.** `scale_pos_weight` so với SMOTE, theo từng mô hình:

| Mô hình | `class_weight` | SMOTE | Chênh | SMOTE đắt hơn |
|---|---|---|---|---|
| XGBoost | **0,8549** | 0,8529 | +0,0020 (KTC chứa 0 → hoà) | 1,24× |
| Random Forest | 0,8404 | **0,8549** | −0,0145 (KTC không chứa 0) | 2,32× |
| Logistic Regression | 0,7416 | **0,7572** | −0,0156 | 0,61× (SMOTE rẻ hơn) |
| Decision Tree | 0,3586 | **0,6404** | −0,2818 | 1,11× |

Dự báo đúng ở chỗ quan trọng nhất (XGBoost — mô hình được chọn), nhưng sai với Random Forest
và sai hẳn với Decision Tree: `class_weight='balanced'` làm PR-AUC của cây sụp từ 0,6948 (không
xử lý gì) xuống 0,3586.

**S5 = S4.** SMOTE + Tomek cho mảng điểm out-of-fold **giống hệt SMOTE tới từng phần tử** ở cả
bốn mô hình; bước Tomek xoá đúng 0 cặp và tốn 1.089 giây, tức 40% thời gian lưới. Ở tỷ lệ
1:600, sau khi SMOTE nâng lớp dương lên 1:10 thì vùng biên vẫn quá thưa để hai điểm khác lớp
trở thành láng giềng gần nhất của nhau. Giữ S5 trong bảng báo cáo như một kết quả âm tính, nhưng
bỏ khỏi các lần chạy lại.

**Độ mịn điểm rủi ro — tiêu chí không có trong §3.4 nhưng quyết định lựa chọn.** Số điểm rủi ro
khác nhau trên 226.980 mẫu: XGBoost ~222.000, Logistic Regression ~226.300, **Random Forest chỉ
200–269**, Decision Tree 23–97. Random Forest 300 cây chỉ phát ra tối đa 301 giá trị `k/300`.
Vì §6 dựa toàn bộ vào việc dịch chuyển ngưỡng, mô hình chỉ có 200 bậc là không dùng được — đây
là lý do chọn XGBoost thay vì Random Forest dù hai bên hoà về PR-AUC.

## 4. Tinh chỉnh siêu tham số

`RandomizedSearchCV`, 30 lần thử, `cv=StratifiedKFold(5)`,
`scoring='average_precision'`, `n_jobs=-1`, `random_state=42`.

Không gian tìm kiếm cho XGBoost:

| Tham số | Miền |
|---|---|
| `n_estimators` | 200 … 800 |
| `max_depth` | 3 … 8 |
| `learning_rate` | log-uniform 0,01 … 0,3 |
| `subsample` | 0,6 … 1,0 |
| `colsample_bytree` | 0,6 … 1,0 |
| `scale_pos_weight` | 1, 10, 100, 578 |
| `min_child_weight` | 1 … 10 |

Không dùng `GridSearchCV` toàn diện: với 7 tham số, lưới đầy đủ mất hàng giờ và
không cho kết quả tốt hơn đáng kể so với 30 lần thử ngẫu nhiên.

### 4.1 Kết quả thực đo (giai đoạn 4, notebook 05)

**Hai điều chỉnh so với bảng trên.** (1) Mức cao nhất của `scale_pos_weight` là **599,5**, tỷ lệ
âm/dương thật của tập huấn luyện sau khi loại trùng lặp; 578 là tỷ lệ trên dữ liệu thô.
(2) Tìm kiếm chạy `n_jobs=-1` ở tầng ngoài còn XGBoost chạy một luồng, để 150 lần huấn luyện chia
đều cho các lõi. Mất 19,8 phút trên 12 lõi (`scripts/run_search.py`).

**Tinh chỉnh không thắng.** Cấu hình đầu bảng đạt PR-AUC CV 0,8538 ± 0,0328, **thấp hơn** cấu hình
khởi điểm ở §3.2 (0,8549). Năm cấu hình đầu cách nhau 0,0008, nhỏ hơn độ lệch chuẩn giữa các fold
khoảng 40 lần. `scale_pos_weight` gần như không ảnh hưởng tới PR-AUC (trung bình 0,846 / 0,845 /
0,844 / 0,841 cho 1 / 10 / 100 / 599,5), vì trọng số lớp dịch điểm rủi ro chứ không đổi thứ tự.

**Quy tắc chọn, đặt trước khi nhìn tập test:** cấu hình tinh chỉnh chỉ thay cấu hình khởi điểm
khi bootstrap hiệu PR-AUC theo cặp trên điểm out-of-fold (§5.4) có khoảng tin cậy không chứa 0.
Kết quả −0,0006 [−0,0087, +0,0072] → **giữ cấu hình khởi điểm** ở §3.2.

| Chỉ số trên tập kiểm thử (τ\* = 0,0232 chọn trên out-of-fold) | Giá trị [KTC 95%, bootstrap 1.000 lần] |
|---|---|
| PR-AUC | **0,825** [0,747 – 0,896] — AC-M1 đạt |
| ROC-AUC | 0,977 [0,961 – 0,991] |
| Recall@τ\* | **0,811** [0,737 – 0,884] — AC-M2 đạt (77/95) |
| Precision@τ\* | 0,670 [0,600 – 0,752] |

Cận dưới của PR-AUC và recall đều sát dưới 0,75: hai tiêu chí đạt theo ước lượng điểm, không đạt
"chắc chắn". Báo cáo phải nêu điều này.

**Chia theo thời gian (FR-10, AC-M7).** Ngày 1 → ngày 2 cho PR-AUC 0,782 [0,728 – 0,836]. Một dòng
đối chứng (chia ngẫu nhiên, thu tập train về đúng kích thước ngày 1) cho 0,825, nên −0,043 trong
tổng chênh −0,044 là do **độ lệch thời gian**, không phải do ít dữ liệu hơn. Tín hiệu rõ nhất là
precision tại ngưỡng cố định: 0,670 → 0,332, tỷ lệ báo động giả trên giao dịch hợp lệ tăng 3,6
lần. Chi tiết ở `reports/split_comparison.csv`.

## 5. Khung đánh giá

### 5.1 Metric và vai trò

| Metric | Vai trò | Lý do |
|---|---|---|
| **PR-AUC** (`average_precision_score`) | **Chính** | Chỉ quan tâm lớp dương, không bị lớp âm khổng lồ làm loãng |
| Recall | Chính | Tỷ lệ gian lận bắt được — mục tiêu nghiệp vụ |
| Precision | Chính | Tỷ lệ cảnh báo đúng — quyết định tải công việc thẩm định |
| F1 | Phụ | Tóm tắt tiện, nhưng giả định hai loại lỗi chi phí bằng nhau — giả định sai ở bài này |
| ROC-AUC | Phụ, để đối chiếu | Báo cáo kèm, kèm giải thích vì sao nó lạc quan |
| Chi phí kỳ vọng | Chính ở phần ngưỡng | Ngôn ngữ mà người ra quyết định thật sự dùng |
| Ma trận nhầm lẫn | Bắt buộc trình bày | Bốn con số tuyệt đối, không giấu được gì |
| Accuracy | **Không dùng** | Mô hình rỗng đạt 99,83% |

### 5.2 Vì sao PR-AUC chứ không phải ROC-AUC

ROC-AUC dùng tỷ lệ dương tính giả `FPR = FP / (FP + TN)`. Ở đây `TN` khoảng
284.000 nên mẫu số gần như không đổi. Tăng FP từ 100 lên 1.000 chỉ đẩy FPR từ
0,00035 lên 0,0035 — đồ thị ROC gần như không nhúc nhích, trong khi precision
sụp từ 0,50 xuống 0,09 và khối lượng thẩm định tăng gấp mười.

Đường PR dùng `Precision = TP / (TP + FP)`, phản ánh đúng thay đổi đó. Vì vậy hai
mô hình có ROC-AUC lần lượt 0,97 và 0,98 có thể có PR-AUC là 0,72 và 0,85 — khác
biệt rất lớn về mặt vận hành.

Đường cơ sở của PR-AUC bằng tỷ lệ lớp dương = **0,00172**. Báo cáo phải nêu con số
này để người đọc thấy PR-AUC 0,80 là gấp khoảng 465 lần đường cơ sở.

### 5.3 Khoảng tin cậy

Tập kiểm thử chỉ có khoảng 98 mẫu gian lận. Mọi chỉ số chính báo cáo kèm khoảng
tin cậy 95% bằng bootstrap phân tầng 1.000 lần lặp trên tập kiểm thử.

Cách trình bày: `PR-AUC = 0,843 [0,781 – 0,894]`.

Nếu khoảng tin cậy của hai mô hình chồng lấn nhiều, **không được kết luận mô hình
này tốt hơn mô hình kia**. Đây là điểm phân biệt báo cáo nghiêm túc với báo cáo
hời hợt.

### 5.4 Kiểm định thống kê (notebook 02)

| Việc | Phương pháp | Vì sao |
|---|---|---|
| So sánh phân bố từng đặc trưng giữa hai lớp | Mann–Whitney U | Không giả định phân phối chuẩn; hai nhóm chênh lệch kích thước cực lớn |
| Hiệu chỉnh đa kiểm định | Benjamini–Hochberg (FDR) | 30 kiểm định cùng lúc; không hiệu chỉnh thì kỳ vọng khoảng 1,5 kết quả dương giả |
| Độ lớn hiệu ứng | Cliff's delta hoặc Cohen's d | Với n lớn, p-value bé không đồng nghĩa khác biệt đáng kể |
| So sánh hai mô hình | Bootstrap hiệu PR-AUC, xét khoảng tin cậy có chứa 0 không | Kiểm định t trên các fold vi phạm giả định độc lập |

## 6. Tối ưu ngưỡng theo chi phí

### 6.1 Vấn đề với ngưỡng 0,5

Ngưỡng mặc định 0,5 giả định hai loại lỗi tốn kém như nhau và hai lớp cân bằng
nhau. Cả hai giả định đều sai ở đây. Với mô hình huấn luyện trên dữ liệu mất cân
bằng nặng, rất ít mẫu có xác suất vượt 0,5 — kết quả là recall thấp bất thường
trong khi mô hình thực ra xếp hạng rất tốt.

**Ngưỡng phải được chọn, không được mặc định.**

### 6.2 Hàm chi phí

```
Chi phí(τ) = cost_fn × FN(τ) + cost_fp × FP(τ)
```

Tham số mặc định, lấy từ chính dữ liệu:

| Tham số | Giá trị | Nguồn |
|---|---|---|
| `cost_fn` | 122,21 | Số tiền trung bình của giao dịch gian lận trong tập dữ liệu |
| `cost_fp` | 5,00 | Ước lượng chi phí nhân công thẩm định một cảnh báo |

Tỷ lệ khoảng 24:1. Cả hai giá trị đều là giả định (A-01, A-02) và **người dùng
thay đổi được lúc chạy** qua màn hình ngưỡng — đó chính là luận điểm mà demo muốn
làm nổi bật.

### 6.3 Các tiêu chí chọn ngưỡng

Hệ thống tính đồng thời bốn phương án và trình bày cạnh nhau:

| Tiêu chí | Định nghĩa | Dùng khi |
|---|---|---|
| `min_expected_cost` | τ tối thiểu hóa hàm chi phí | Mặc định của hệ thống |
| `max_f1` | τ tối đa hóa F1 | Khi không ước lượng được chi phí |
| `recall_at_least_90` | τ nhỏ nhất sao cho recall ≥ 0,90 | Khi có ràng buộc tuân thủ |
| `budget_k_alerts` | τ sao cho số cảnh báo mỗi ngày ≤ K | Khi năng lực thẩm định có giới hạn cứng |

Ràng buộc năng lực đáng chú ý vì nó là ràng buộc thật trong vận hành: đội thẩm
định xử lý được khoảng 200 giao dịch mỗi ngày thì ngưỡng phải chọn sao cho số
cảnh báo nằm dưới mức đó, bất kể chi phí lý thuyết nói gì.

### 6.4 Bảng bắt buộc trong báo cáo

| Ngưỡng | Cảnh báo/ngày | TP | FP | FN | Precision | Recall | Chi phí kỳ vọng |
|---|---|---|---|---|---|---|---|
| 0,5 (mặc định) | | | | | | | |
| τ* tối ưu chi phí | | | | | | | |
| τ cho recall ≥ 90% | | | | | | | |
| τ cho 200 cảnh báo/ngày | | | | | | | |

Kèm một đoạn diễn giải bằng lời: "Hạ ngưỡng từ 0,5 xuống 0,047 giúp bắt thêm N
giao dịch gian lận, đổi lại M cảnh báo giả mỗi ngày, tiết kiệm ròng X USD."

### 6.5 Phân tích độ nhạy

Vì `cost_fn` và `cost_fp` là giả định, báo cáo phải có một biểu đồ cho thấy ngưỡng
tối ưu dịch chuyển thế nào khi tỷ lệ chi phí thay đổi từ 5:1 tới 100:1. Nếu ngưỡng
tối ưu ổn định trong dải rộng thì kết luận vững; nếu nó nhảy mạnh thì phải nói rõ
rằng khuyến nghị phụ thuộc nhiều vào ước lượng chi phí.

## 7. Giải thích mô hình

| Mức | Công cụ | Đưa vào đâu |
|---|---|---|
| Toàn cục | `shap.summary_plot` trên mẫu 2.000 giao dịch | Báo cáo, màn hình UI-04 |
| Toàn cục | Xếp hạng tầm quan trọng trung bình `|SHAP|` | `metrics.json` |
| Cục bộ | `shap.TreeExplainer` cho một giao dịch | Endpoint `/explain`, màn hình UI-02 |
| Đối chiếu | So tầm quan trọng SHAP với hệ số hồi quy logistic | Báo cáo |

**Giới hạn bắt buộc nêu:** vì V1–V28 là thành phần chính sau PCA, kết luận chỉ
dừng ở mức "V14 đóng góp mạnh nhất vào điểm rủi ro". Không được diễn giải thành
nguyên nhân nghiệp vụ. Nói rõ điều này thể hiện hiểu vấn đề; lờ đi là điểm trừ.

Phần phân tích lỗi (notebook 07): lấy các FN có điểm rủi ro cao nhất và các FP có
điểm cao nhất, giải thích SHAP cho từng trường hợp, tìm mẫu hình chung.

## 8. Autoencoder — tùy chọn (FR-11)

Chỉ làm nếu còn thời gian sau khi hoàn thành mọi hạng mục bắt buộc.

| Mục | Thiết kế |
|---|---|
| Dữ liệu huấn luyện | **Chỉ** `X_train[y_train == 0]` — mô hình học "thế nào là bình thường" |
| Kiến trúc | 30 → 16 → 8 → 16 → 30, hàm kích hoạt ReLU, đầu ra tuyến tính |
| Hàm mất mát | MSE tái tạo |
| Điểm rủi ro | Sai số tái tạo trên tập kiểm thử |
| Đánh giá | Cùng khung đánh giá: PR-AUC trên sai số tái tạo |

Điều đáng nói trong báo cáo không phải là autoencoder thắng hay thua (nhiều khả
năng thua rõ rệt so với XGBoost), mà là **vì sao**: phương pháp không giám sát
không dùng tới 492 nhãn có sẵn. Nó chỉ đáng giá khi không có nhãn hoặc khi cần
bắt kiểu gian lận mới chưa từng xuất hiện.

## 9. Hiện vật đầu ra

| Tệp | Nội dung | Bên tiêu thụ |
|---|---|---|
| `models/model.joblib` | Pipeline hoàn chỉnh: scaler + classifier | API |
| `models/explainer.joblib` | SHAP `TreeExplainer` | API `/explain` |
| `models/threshold.json` | Ngưỡng mặc định, tham số chi phí, các phương án | API, UI-03 |
| `models/metrics.json` | Toàn bộ chỉ số, điểm tập kiểm thử, đường PR, đường cong chi phí, bảng 20 tổ hợp | API, UI-04 |
| `data/test_set.parquet` | Tập kiểm thử kèm nhãn thật | Phát lại, đánh giá |
| `data/sample_pool.json` | Khoảng 200 giao dịch mẫu | Thư viện mẫu |

Cấu trúc chi tiết của `threshold.json` và `metrics.json` nằm ở
[06 §3](06-thiet-ke-luu-tru.md).

## 10. Tiêu chí chấp nhận phần mô hình

| Mã | Tiêu chí | Cách kiểm chứng |
|---|---|---|
| AC-M1 | PR-AUC trên tập kiểm thử ≥ 0,75 | `metrics.json` |
| AC-M2 | Recall ≥ 0,75 tại ngưỡng đề xuất | Ma trận nhầm lẫn |
| AC-M3 | Đủ 20 tổ hợp trong bảng kết quả | Bảng trong báo cáo |
| AC-M4 | Không có rò rỉ dữ liệu | Rà soát mã ML-01…ML-08 + `tests/test_no_leakage.py` |
| AC-M5 | Tái lập được, PR-AUC lệch < 0,001 giữa hai lần chạy | Chạy lại notebook |
| AC-M6 | Có khoảng tin cậy bootstrap cho chỉ số chính | Báo cáo |
| AC-M7 | Có bảng đối chiếu hai cách chia tập | Báo cáo |
| AC-M8 | Có phân tích độ nhạy theo tỷ lệ chi phí | Báo cáo |
