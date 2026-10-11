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
       + lưới theo mức cân bằng r và CV lặp cho các cặp cần kết luận (§3.6, §3.7)
B6.  Chọn cấu hình cuối bằng quy tắc tie-break đặt trước (§3.7) → src/config.py
B7.  Tinh chỉnh siêu tham số: RandomizedSearchCV, 30 lần thử
B8.  Huấn luyện lại trên toàn bộ tập train, đánh giá trên tập test
B9.  Bootstrap khoảng tin cậy cho các chỉ số chính
B10. Điểm chọn ngưỡng (forward-chaining theo Time), đường cong chi phí,
       ngưỡng mặc định = chi phí nhỏ nhất trong ngân sách thẩm định (§6.3)
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
bốn mô hình; bước Tomek xoá đúng 0 cặp và tốn 1.089 giây, tức 40% thời gian lưới (lần chạy lại
2026-10-10, máy có tải nền: 1.689 giây trong 66,1 phút, 42,6%; điểm out-of-fold trùng từng bit với lần đầu).
Lý do — kiểm lại trên fold 1 của lưới (2026-10-09): dữ liệu gốc **có** cặp Tomek. Trước SMOTE,
`TomekLinks` tìm được 21 cặp (xoá 42 dòng, trong đó 21 vụ gian lận); sau SMOTE còn 0. Chính SMOTE
xoá chúng: mỗi vụ gian lận thật được dùng làm gốc để sinh khoảng 59 điểm tổng hợp trên các đoạn nối
nó với láng giềng cùng lớp, nên láng giềng gần nhất của nó trở thành một điểm gian lận tổng hợp —
không còn cặp khác lớp nào là láng giềng gần nhất **của nhau**. `SMOTETomek` chạy SMOTE trước, Tomek
sau, nên ở tỷ lệ này bước Tomek luôn vô tác dụng; muốn Tomek có tác dụng phải dọn **trước** khi sinh mẫu.
Giữ S5 trong bảng báo cáo như một kết quả âm tính, nhưng bỏ khỏi các lần chạy lại.

**Độ mịn điểm rủi ro — tiêu chí không có trong §3.4 nhưng quyết định lựa chọn.** Số điểm rủi ro
khác nhau trên 226.980 mẫu: XGBoost ~222.000, Logistic Regression ~226.300, **Random Forest chỉ
200–269**, Decision Tree 23–97. Random Forest 300 cây chỉ phát ra tối đa 301 giá trị `k/300`.
Vì §6 dựa toàn bộ vào việc dịch chuyển ngưỡng, mô hình chỉ có 200 bậc là không dùng được — đây
là lý do chọn XGBoost thay vì Random Forest dù hai bên hoà về PR-AUC.

### 3.6 Sau review 2026-10-10: so ở cùng mức cân bằng

Bảng §3.5 so `class_weight` **cân bằng hoàn toàn** (`"balanced"`, `scale_pos_weight` ≈ 599,5 — tương
đương 1:1) với SMOTE và undersample **đưa về 1:10**: hai chiến lược khác nhau ở cả loại kỹ thuật lẫn mức
cân bằng ([11](11-review-ml-va-ke-hoach-sua.md) M5). `scripts/run_ratio_grid.py` chạy ba chiến lược ở
**cùng** tỷ lệ dương:âm mục tiêu r ∈ {0,02; 0,1; 0,3; 1}: `class_weight` dùng trọng số lớp dương
r × N_âm/N_dương, SMOTE và undersample dùng `sampling_strategy = r`. 48 tổ hợp, cùng fold và cùng cách
chọn τ\* với lưới chuẩn; kết quả ở `reports/ratio_grid.csv`, trình bày ở notebook 04 §5.7.

PR-AUC out-of-fold (cột "không xử lý" lấy từ lưới chuẩn):

| Mô hình | Chiến lược | không xử lý | r = 0,02 | r = 0,1 | r = 0,3 | r = 1 |
|---|---|---|---|---|---|---|
| Decision Tree | class_weight | 0,6948 | 0,6498 | 0,6416 | 0,6104 | 0,3596 |
| | SMOTE | | 0,6797 | 0,6404 | 0,5923 | 0,4585 |
| | undersample | | 0,4740 | 0,2405 | 0,0907 | 0,0239 |
| Logistic Regression | class_weight | 0,7578 | **0,7627** | 0,7595 | 0,7440 | 0,7422 |
| | SMOTE | | 0,7605 | 0,7572 | 0,7439 | 0,7454 |
| | undersample | | 0,7524 | 0,7185 | 0,6572 | 0,5434 |
| Random Forest | class_weight | 0,8406 | 0,8428 | 0,8425 | 0,8416 | 0,8396 |
| | SMOTE | | 0,8526 | **0,8549** | 0,8518 | 0,8502 |
| | undersample | | 0,8033 | 0,7586 | 0,7577 | 0,7540 |
| XGBoost | class_weight | 0,8514 | 0,8540 | **0,8565** | 0,8556 | 0,8549 |
| | SMOTE | | 0,8541 | 0,8529 | 0,8532 | 0,8511 |
| | undersample | | 0,8333 | 0,7811 | 0,7790 | 0,7435 |

**Đính chính §3.5.** Hai kết luận ở đó là hiệu ứng của **mức** cân bằng, không phải của kỹ thuật:

- Decision Tree "sụp vì `class_weight`" (0,3586): ở cùng r = 0,1 hai kỹ thuật bằng nhau (0,6416 và
  0,6404), ở r = 1 cả hai cùng sụp (0,3596 và 0,4585). Với cây nông, mọi mức cân bằng đều kém không xử lý.
- "SMOTE giúp Logistic Regression hơn `class_weight`": ở r = 0,02 và 0,1 thì `class_weight` nhỉnh hơn.

Điều **đứng được** sau khi khử lẫn biến: undersample kém nhất ở mọi mô hình và càng kém khi r tăng; với
XGBoost mọi biến thể `class_weight`/SMOTE nằm trong 0,8511–0,8565, sát `none` (0,8514). Random Forest +
SMOTE hơn Random Forest + `class_weight` ở cả bốn mức trên CV một lần (+0,010 tới +0,012), nhưng CV lặp ở
§3.7 không xác nhận được. Brier tăng theo r ở gần như mọi tổ hợp — cân bằng càng mạnh, điểm càng xa xác suất
thật.

### 3.7 Sau review: CV lặp và quy tắc chọn cấu hình cuối

**CV lặp** (`scripts/run_repeated_cv.py`, [11](11-review-ml-va-ke-hoach-sua.md) m2). Bootstrap trên điểm
out-of-fold gộp chỉ lấy mẫu lại các dòng của **một** lần CV, bỏ qua dao động do huấn luyện. Các cặp cần kết
luận được chạy lại bằng `RepeatedStratifiedKFold` cùng `random_state` — mọi cấu hình gặp đúng cùng các fold
— rồi so theo cặp bằng **kiểm định t hiệu chỉnh Nadeau–Bengio** (`src/evaluate.py`): phương sai của hiệu nhân
`1/k + n_test/n_train` để bù cho việc các fold dùng chung dữ liệu huấn luyện. XGBoost 5 × 5 fold, Random
Forest 3 × 5 fold; tổng 81 phút. Kết quả ở `reports/repeated_cv.csv` và `reports/repeated_cv_tests.csv`.

| Khẳng định của bản trước | Bootstrap OOF gộp (cũ) | CV lặp, t hiệu chỉnh | |
|---|---|---|---|
| RF + SMOTE hơn RF + `class_weight` | +0,0146 [+0,0076; +0,0220] | +0,0106 [−0,0063; +0,0275] | không kết luận được |
| XGBoost + `class_weight` hơn RF + `class_weight` | +0,0152 [+0,0056; +0,0270] | +0,0143 [−0,0070; +0,0357] | không kết luận được |

**Quy tắc chọn cấu hình cuối**, đặt trước khi chạy CV lặp và lưới theo mức, áp lên họ XGBoost (Random Forest
bị loại vì chỉ có khoảng 200 mức điểm, §3.5):

1. ứng viên là mọi cấu hình XGBoost của hai lưới có PR-AUC CV trong vòng 0,01 của cấu hình cao nhất; cấu hình
   nào có hiệu so với cấu hình cao nhất theo CV lặp mà KTC 95% chứa 0 thì **hoà**;
2. trong nhóm hoà, chọn chiến lược đơn giản nhất: `none` < `class_weight` < `undersample` < `smote`;
3. rồi Brier thấp nhất trên OOF; 4. rồi ít cảnh báo/ngày nhất tại τ\* của chính nó.

**Kết quả:** chín cấu hình XGBoost ứng viên, mốc là `class_weight` r = 0,3 (0,8589 theo CV lặp). **Cả chín
hoà với mốc** — hiệu xa nhất −0,0077 [−0,0181; +0,0026] (SMOTE r = 1); `none` −0,0037 [−0,0119; +0,0046]. Bước 2
chọn **`xgboost` + `none`**, ghi vào `src/config.py` (`FINAL_MODEL`, `FINAL_STRATEGY`, `FINAL_RATIO`); notebook
05–08 dựng mô hình qua `final_pipeline()`. Lựa chọn cũ `class_weight` (`scale_pos_weight` ≈ 599,5) bị thay vì
không có số liệu nào cho thấy nó hơn `none`, trong khi `none` cho điểm gần xác suất hơn (Brier OOF 0,000378 so
với 0,000408) và ít cảnh báo giả hơn tại τ\* (precision OOF 0,866 so với 0,702). Lý do ghi ở bản trước —
"nhanh nhất trong nhóm, điểm mịn nhất" — sai ở cả hai vế: `none` nhanh hơn, `smote` có nhiều mức điểm hơn.

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
Sau review, cấu hình khởi điểm là `xgboost + none` (§3.7): hiệu tinh chỉnh − khởi điểm +0,0034
[−0,0028, +0,0097] → **giữ cấu hình khởi điểm**. (Bản trước, so với `class_weight`: −0,0006
[−0,0087, +0,0072], cùng kết luận.) `reports/final_choice.json` ghi lựa chọn; notebook 08 dừng nếu nó
khác `src/config.py`. Lần chạy lại 2026-10-11: tìm kiếm 13,2 phút, cùng 30 kết quả.

| Chỉ số trên tập kiểm thử (`xgb_none_v2`) | Giá trị [KTC 95%, bootstrap 1.000 lần] |
|---|---|
| PR-AUC | **0,828** [0,747 – 0,900] — AC-M1 đạt theo ước lượng điểm |
| ROC-AUC | 0,975 [0,958 – 0,991] |
| Recall tại ngưỡng mặc định 0,2675 (§6.6) | **0,758** [0,674 – 0,842] — AC-M2 đạt theo ước lượng điểm, sát nút (72/95) |
| Precision tại ngưỡng mặc định | 0,960 [0,915 – 1,000] |
| Recall / precision tại τ\* = 0,0468 chọn trên OOF ngẫu nhiên (cách chọn của bản trước, tham khảo) | 0,789 / 0,862 |

Cận dưới của PR-AUC (0,747) và recall (0,674) đều dưới 0,75: hai tiêu chí đạt theo ước lượng điểm,
không đạt "chắc chắn". Báo cáo phải nêu điều này.

**Chia theo thời gian (FR-10, AC-M7).** Ngày 1 → ngày 2 cho PR-AUC 0,773 [0,724 – 0,827]. Dòng
đối chứng (chia ngẫu nhiên, thu tập train về đúng kích thước ngày 1) cho 0,817, nên −0,044 trong
tổng chênh −0,055 là do **độ lệch thời gian**, −0,011 do ít dữ liệu hơn. Tín hiệu rõ nhất ở ngưỡng: τ\*
chọn trên CV ngẫu nhiên của ngày 1 (0,0156) cho precision ngày 2 chỉ 0,365 (so với 0,862 khi chia ngẫu
nhiên), tỷ lệ báo động giả trên giao dịch hợp lệ gấp khoảng 10 lần. Đây là lý do ngưỡng chọn trên điểm
forward-chaining ([11](11-review-ml-va-ke-hoach-sua.md) M2, §6.6). Chi tiết ở `reports/split_comparison.csv`.

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
| So sánh hai mô hình | Bootstrap hiệu PR-AUC, xét khoảng tin cậy có chứa 0 không | Kiểm định t thường trên các fold vi phạm giả định độc lập |
| So sánh hai cấu hình cần kết luận (sau review, [11](11-review-ml-va-ke-hoach-sua.md) m2) | CV lặp cùng các fold + kiểm định t hiệu chỉnh Nadeau–Bengio | Tính cả phương sai do huấn luyện mà bootstrap trên OOF gộp bỏ qua; hiệu chỉnh `1/k + n_test/n_train` bù cho các fold dùng chung dữ liệu |

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
| `cost_fn` | 115,94 | Số tiền trung bình của giao dịch gian lận **trong tập huấn luyện** (378 vụ). Trước review 2026-10-10 là 122,21 — trung bình trên dữ liệu thô, gồm cả tập kiểm thử ([11](11-review-ml-va-ke-hoach-sua.md) m3) |
| `cost_fp` | 5,00 | Ước lượng chi phí nhân công thẩm định một cảnh báo |

Tỷ lệ khoảng 23:1. Cả hai giá trị đều là giả định (A-01, A-02) và **người dùng
thay đổi được lúc chạy** qua màn hình ngưỡng — đó chính là luận điểm mà demo muốn
làm nổi bật.

### 6.3 Các tiêu chí chọn ngưỡng

Hệ thống tính đồng thời các phương án và trình bày cạnh nhau:

| Tiêu chí | Định nghĩa | Dùng khi |
|---|---|---|
| `min_cost_within_budget` | τ tối thiểu hóa hàm chi phí **trong** các ngưỡng có số cảnh báo/ngày ≤ ngân sách (200) | **Mặc định của hệ thống** từ review 2026-10-10 ([11](11-review-ml-va-ke-hoach-sua.md) M1) |
| `min_expected_cost` | τ tối thiểu hóa hàm chi phí, không ràng buộc | Mặc định trước review; giờ là phương án tham khảo |
| `max_f1` | τ tối đa hóa F1 | Khi không ước lượng được chi phí |
| `recall_at_least_90` | τ nhỏ nhất sao cho recall ≥ 0,90 | Khi có ràng buộc tuân thủ |
| `budget_k_alerts` | τ sao cho số cảnh báo mỗi ngày ≤ K | Khi năng lực thẩm định có giới hạn cứng |

Ràng buộc năng lực đáng chú ý vì nó là ràng buộc thật trong vận hành: đội thẩm
định xử lý được khoảng 200 giao dịch mỗi ngày thì ngưỡng phải chọn sao cho số
cảnh báo nằm dưới mức đó, bất kể chi phí lý thuyết nói gì. Vì vậy ngưỡng mặc định là
`min_cost_within_budget`: trong vùng đáy chi phí phẳng, chọn theo năng lực thẩm định.
`budget_k_alerts` khác ở chỗ lấy τ **nhỏ nhất** trong ngân sách (recall cao nhất, có thể đắt hơn).

**Chọn trên điểm nào.** Mọi tiêu chí chọn trên **điểm chọn ngưỡng**: điểm out-of-fold sinh bằng
forward-chaining theo `Time` trên tập huấn luyện (`src.modeling.selection_scores` — sắp theo thời gian, chia
5 khối liên tiếp, khối k chấm bằng mô hình học trên các khối trước nó; khối đầu không có điểm). Bản trước
dùng CV ngẫu nhiên, và đối chứng ngày 1 → ngày 2 cho thấy nó đánh giá thấp số cảnh báo trên một ngày mới
([11](11-review-ml-va-ke-hoach-sua.md) M2). Số cảnh báo/ngày quy theo đúng phần luồng (80%) và số ngày mà các cửa sổ kiểm định trải qua.

### 6.4 Bảng bắt buộc trong báo cáo

| Ngưỡng | Cảnh báo/ngày | TP | FP | FN | Precision | Recall | Chi phí kỳ vọng |
|---|---|---|---|---|---|---|---|
| 0,5 (mặc định) | | | | | | | |
| τ* tối ưu chi phí | | | | | | | |
| τ cho recall ≥ 90% | | | | | | | |
| τ cho 200 cảnh báo/ngày | | | | | | | |

Kèm một đoạn diễn giải bằng lời: "Hạ ngưỡng từ 0,5 xuống 0,047 giúp bắt thêm N
giao dịch gian lận, đổi lại M cảnh báo giả mỗi ngày, tiết kiệm ròng X EUR."

### 6.5 Phân tích độ nhạy

Vì `cost_fn` và `cost_fp` là giả định, báo cáo phải có một biểu đồ cho thấy ngưỡng
tối ưu dịch chuyển thế nào khi tỷ lệ chi phí thay đổi từ 5:1 tới 100:1. Nếu ngưỡng
tối ưu ổn định trong dải rộng thì kết luận vững; nếu nó nhảy mạnh thì phải nói rõ
rằng khuyến nghị phụ thuộc nhiều vào ước lượng chi phí.

### 6.6 Kết quả thực đo (notebook 06, chạy lại sau review 2026-10-11)

**Lệch khỏi đặc tả: ngưỡng ứng viên là mọi điểm khác nhau, không phải lưới 200 điểm.** Lưới 200 lượng tử
quá thô ở phần đuôi: với ràng buộc ngân sách nó chọn 0,4255 thay vì 0,2675, và lệch cả F1 lẫn recall ≥ 90%.
`pick_threshold`, `threshold_alternatives`, `policy_threshold` và `sensitivity_analysis` nhận tham số
`thresholds`; báo cáo và API truyền `np.unique(điểm)`.

**Điểm chọn ngưỡng** (forward-chaining, §6.3): 181.584 giao dịch, 249 gian lận, trải 1,45 ngày, PR-AUC 0,783.
Trên đó: τ\* không ràng buộc = **0,0191** (297 cảnh báo/ngày); **ngưỡng mặc định** = **0,2675** (198
cảnh báo/ngày, chi phí cao hơn cực tiểu 26%). Đáy chi phí phẳng — τ ∈ [0,0097; 0,0462] trong 5% cực tiểu —
nằm hoàn toàn bên kia ngân sách.

Bảng §6.4 trên tập kiểm thử (56.746 giao dịch, 95 gian lận), mọi ngưỡng chọn trên điểm chọn ngưỡng:

| Ngưỡng | Cảnh báo/ngày | TP | FP | FN | Precision | Recall | Chi phí kỳ vọng |
|---|---|---|---|---|---|---|---|
| 0,5 | 180 | 71 | 1 | 24 | 0,986 | 0,747 | 2.788 EUR |
| **mặc định 0,2675 (chi phí nhỏ nhất trong ngân sách)** | **187** | **72** | **3** | **23** | **0,960** | **0,758** | **2.682 EUR** |
| τ\* = 0,0191 (không ràng buộc) | 242 | 78 | 19 | 17 | 0,804 | 0,821 | 2.066 EUR |
| τ = 0,00011 cho recall ≥ 90% | 4.325 | 85 | 1.645 | 10 | 0,049 | 0,895 | 9.384 EUR |
| τ = 0,2664 nhỏ nhất trong 200 cảnh báo/ngày | 187 | 72 | 3 | 23 | 0,960 | 0,758 | 2.682 EUR |

"Hạ ngưỡng từ 0,5 xuống τ\* = 0,0191 bắt thêm **7** vụ gian lận, đổi lại **18** cảnh báo giả, tiết kiệm
ròng **722 EUR** [238; 1.347] trên tập kiểm thử" — khoảng tin cậy không chứa 0. Ngưỡng mặc định thì gần 0,5
hơn: chỉ hơn 0,5 một vụ (−106 [−343; +20] EUR, chứa 0) và kém τ\* sáu vụ (+616 EUR) — cái giá của ngân sách.

**AC-M2** tại ngưỡng mặc định: recall 0,758 [0,674; 0,842] = 72/95 — đạt theo ước lượng điểm, sát nút;
năng lực cần cho recall 0,75 ≈ 185 cảnh báo/ngày. Có điều kiện: dưới giả định chi phí và ngân sách, trên
tập kiểm thử chia ngẫu nhiên. Đối chứng ngày 1 → ngày 2 (chỉ dùng ngày 1 để chọn): ngưỡng trong ngân sách
cho recall ngày 2 khoảng 0,70; với τ\*, CV ngẫu nhiên ước tính 327 cảnh báo/ngày mà thật là 539 (×1,65),
forward-chaining ước tính 367, thật 536 (×1,46) — sát hơn, chưa đủ sát ([11](11-review-ml-va-ke-hoach-sua.md) M2).

**Giả định chi phí.** `c_FN` = 115,94 (chỉ tập huấn luyện); 122,21 hay 123,87 cho cùng hai ngưỡng. Tính chi
phí bỏ lọt bằng số tiền của chính vụ bị lọt: ngưỡng cực tiểu trùng τ\*, nhưng hiệu chi phí τ\* − 0,5 trên
test co từ −722 [−1.347; −238] xuống −169 [−673; +105] EUR — **chứa 0**. (Mô hình trước, có trọng số lớp:
τ\* **đắt hơn** 0,5, +160 [+103; +217] EUR.) Ở ngưỡng mặc định, 23 vụ lọt mất 4.052 EUR, ba vụ lớn nhất
chiếm 58%.

**Độ nhạy (§6.5, AC-M8).** τ\* đổi theo bậc: 0,127 (5:1), 0,046 (7,5:1), 0,038 (10:1), **0,0191
(15–75:1)**, 0,0035 (100:1); hối tiếc khi giữ τ\* mặc định ≤ 3,6% từ 7,5:1, tối đa 9,6% ở 5:1. Ngưỡng
trong ngân sách chỉ có hai giá trị trên cả dải (0,4986 ở 5:1, 0,2675 còn lại) — khi ngân sách là ràng
buộc chặt, năng lực thẩm định quyết định ngưỡng. Recall test của τ\* luôn trong 0,77–0,84; tỷ lệ chi phí
quyết định **khối lượng thẩm định** (218 → 513 cảnh báo/ngày). Kết quả ở `reports/threshold_comparison.csv`,
`reports/cost_sensitivity.csv`, `reports/acceptance.csv`, `reports/threshold_protocol.csv`.

**§6.1 dự báo đúng hơn với mô hình này.** Không có trọng số lớp, điểm của lớp dương không bị đẩy về sát 1:
ngưỡng 0,5 cho recall 0,747 (mô hình trước 0,779), và chọn ngưỡng tiết kiệm 26% chi phí trên tập kiểm thử
(2.066 so với 2.788 EUR) thay vì khoảng 9%.

**Hai hệ quả vận hành.** (1) Ràng buộc recall chọn vừa khít trên điểm chọn ngưỡng hơi hụt trên dữ liệu mới
(0,904 → 0,895): ràng buộc tuân thủ thật cần biên an toàn. (2) Ngưỡng chọn một lần rồi để yên sẽ trôi:
API-16 theo dõi tỷ lệ cảnh báo và precision thẩm định so với kỳ vọng.

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

### 7.1 Kết quả thực đo (giai đoạn 5, notebook 07)

**Lệch khỏi đặc tả:** SHAP được tính trên **toàn bộ** 56.746 giao dịch test (35 giây) để bảng
`mean|SHAP|` đại diện cho cả luồng. `summary_plot` dùng 2.000 mẫu như đặc tả, nhưng gồm cả 95 gian
lận cộng 1.905 giao dịch hợp lệ ngẫu nhiên, vì mẫu ngẫu nhiên thuần chỉ có khoảng 3 vụ gian lận.

- Kiểm tra tính cộng: `base + ΣSHAP` khớp logit của booster tới 2,4·10⁻⁵. Base value −6,85 (xác suất
  0,001) — mô hình `xgb_none_v2` không trọng số lớp (mô hình trước: +3,28).
- Top 5 theo `mean|SHAP|`: V4 (14,7%), V14, V12, V10, V11, cộng lại 49%. `Amount` hạng 8 (2,9%), giờ
  trong ngày chiếm 2,8%.
- So với T-15, Spearman bằng 0,40 với \|Cohen's d\|, 0,46 với \|Cliff's delta\| và 0,47 với hệ số
  chuẩn hoá của hồi quy logistic. V17 hạng 1 theo Cohen's d nhưng **hạng cuối (31/31)** theo SHAP, vì
  nó thừa thông tin khi đã có V14, V12, V10 (autoencoder ở §8.1 lại thấy V17 bất thường nhất).
- Phân tích lỗi tại ngưỡng mặc định: 13/23 FN có điểm dưới 0,001, và 61% FN có cả 5 đặc trưng chính
  nằm trong khoảng P1–P99 của lớp hợp lệ. Chỉ 3 FP, đều mang đủ "chữ ký" V14 / V10 / V12 của gian lận.
  Cả hai loại lỗi là giới hạn của đặc trưng, không phải của mô hình.

Chi tiết ở `reports/shap_ranking.csv` và `reports/error_analysis.csv`.

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

### 8.1 Kết quả thực đo (notebook 06b)

**Lệch khỏi đặc tả:** môi trường không có PyTorch, nên mạng dựng bằng `MLPRegressor` của
scikit-learn (`src/anomaly.py`), có 31 đầu vào vì `Time` đã được thay bằng `hour_sin`, `hour_cos`.
`Amount` lấy `log1p`, rồi mọi cột chuẩn hoá theo thống kê của lớp hợp lệ trong tập train.

PR-AUC **0,329** [0,246 – 0,438] so với 0,828 của XGBoost (`xgb_none_v2`), hiệu +0,499 [+0,393, +0,588].
ROC-AUC 0,949 so với 0,975 trông như ngang nhau. Autoencoder xếp 93/95 vụ gian lận trên trung vị, nên nó
không bỏ sót gian lận. Nó thua vì không phân biệt được giao dịch hợp lệ hiếm gặp với giao dịch
gian lận: trong 1% giao dịch bất thường nhất có 74 vụ gian lận và 494 giao dịch hợp lệ.

## 9. Hiện vật đầu ra

| Tệp | Nội dung | Bên tiêu thụ |
|---|---|---|
| `models/model.joblib` | Pipeline hoàn chỉnh: scaler + classifier | API |
| `models/explainer.joblib` | SHAP `TreeExplainer` | API `/explain` |
| `models/threshold.json` | Ngưỡng mặc định và chính sách chọn nó, tham số chi phí, ràng buộc, các phương án | API, UI-03 |
| `models/oof_scores.npz` | Điểm chọn ngưỡng (forward-chaining) và nhãn của tập huấn luyện | API-11, API-12, API-17, UI-D1 |
| `models/metrics.json` | Toàn bộ chỉ số, điểm tập kiểm thử, đường PR, đường cong chi phí, bảng 20 tổ hợp | API, UI-04 |
| `data/test_set.parquet` | Tập kiểm thử kèm nhãn thật | Phát lại, đánh giá |
| `data/sample_pool.json` | Khoảng 200 giao dịch mẫu | Thư viện mẫu |

Cấu trúc chi tiết của `threshold.json` và `metrics.json` nằm ở
[06 §3](06-thiet-ke-luu-tru.md).

## 10. Tiêu chí chấp nhận phần mô hình

| Mã | Tiêu chí | Cách kiểm chứng |
|---|---|---|
| AC-M1 | PR-AUC trên tập kiểm thử ≥ 0,75 | `metrics.json` |
| AC-M2 | Recall ≥ 0,75 tại ngưỡng mặc định (chi phí nhỏ nhất trong ngân sách 200 cảnh báo/ngày) — chỉ có nghĩa khi nghiệp vụ xác nhận hàm chi phí và ngân sách ([11](11-review-ml-va-ke-hoach-sua.md) M1) | `metrics.json → acceptance` |
| AC-M3 | Đủ 20 tổ hợp trong bảng kết quả | Bảng trong báo cáo |
| AC-M4 | Không có rò rỉ dữ liệu | Rà soát mã ML-01…ML-08 + `tests/test_no_leakage.py` |
| AC-M5 | Tái lập được, PR-AUC lệch < 0,001 giữa hai lần chạy | Chạy lại notebook |
| AC-M6 | Có khoảng tin cậy bootstrap cho chỉ số chính | Báo cáo |
| AC-M7 | Có bảng đối chiếu hai cách chia tập | Báo cáo |
| AC-M8 | Có phân tích độ nhạy theo tỷ lệ chi phí | Báo cáo |
