# Danh sách công việc — Đề tài phát hiện gian lận thẻ tín dụng

Bảng theo dõi để tick từng việc. Kế hoạch theo ngày và lý do đằng sau thứ tự này
nằm ở [docs/09 — Kế hoạch triển khai](docs/09-ke-hoach-trien-khai.md); tệp này là
bản thao tác hằng ngày.

**Quy ước:** `[ ]` chưa làm · `[~]` đang làm · `[x]` xong.
Mỗi việc có điều kiện *xong là khi* — chưa đạt điều kiện đó thì chưa được tick.

## Tiến độ

| Giai đoạn | Việc | Xong | Trạng thái |
|---|---|---|---|
| 0. Nền tảng mã nguồn | 10 | 10 | ✅ |
| 1. EDA và thống kê | 5 | 5 | ✅ |
| 2. Mô hình cơ sở | 4 | 4 | ✅ |
| 3. Chiến lược mất cân bằng | 4 | 4 | ✅ |
| 4. Tinh chỉnh và kiểm chứng | 5 | 5 | ✅ |
| 5. Ngưỡng, chi phí, SHAP | 6 | 6 | ✅ |
| 6. Xuất hiện vật | 4 | 4 | ✅ |
| 🚩 Mốc quyết định A/B | 1 | 1 | ✅ B |
| 7. API | 11 | 11 | ✅ |
| 8. Giao diện | 7 | 7 | ✅ |
| 9. Đóng gói | 5 | 5 | ✅ |
| 10. Báo cáo và bảo vệ | 5 | 3 | 🟡 còn T-66, T-67 |
| **Tổng** | **67** | **65** | **97%** |

---

## Giai đoạn 0 — Nền tảng mã nguồn ✅

Phần này đã hoàn thành. Giữ lại để biết cái gì đã có mà không phải viết lại.

- [x] **T-01** Tạo `.venv` (Python 3.12.5) và cài `requirements.txt` — *xong là khi:* `pytest` chạy được.
- [x] **T-02** `scripts/download_data.py` — tải từ Kaggle API hoặc `--mirror`, tự kiểm tra toàn vẹn — *xong là khi:* `--check` báo đúng 284.807 dòng / 492 gian lận.
- [x] **T-03** `src/config.py` — hằng số dùng chung, `RANDOM_STATE=42`, chi phí mặc định.
- [x] **T-04** `src/features.py` — `build_features()`, `validate_raw()`, `make_preprocessor()` — *xong là khi:* TC-01…TC-06 xanh.
- [x] **T-05** `src/threshold.py` — `sweep()`, `cost_curve()`, `pick_threshold()` 4 tiêu chí — *xong là khi:* TC-10…TC-15 xanh.
- [x] **T-06** `src/data.py` — `check_integrity()`, `drop_duplicates()`, `split_data()`, `temporal_split()`.
- [x] **T-07** `src/evaluate.py` — `bootstrap_ci()` phân tầng, `headline_metrics()`, `curve_points()`.
- [x] **T-08** `src/modeling.py` — `build_pipeline()`, `run_grid()`, `oof_scores()` — *xong là khi:* TC-20…TC-23 xanh.
- [x] **T-09** `src/plots.py` — 8 biểu đồ EDA + 6 biểu đồ đánh giá — *xong là khi:* cả 14 hàm vẽ được.
- [x] **T-10** Tải và kiểm chứng `data/creditcard.csv` — *xong là khi:* SHA-256 khớp, 0 ô thiếu.

---

## Giai đoạn 1 — EDA và thống kê (ngày 1–4) ✅

- [x] **T-11** `notebooks/01_eda.ipynb`: nạp qua `load_prepared()`, in báo cáo toàn vẹn, ghi nhận 1.081 dòng trùng lặp — *xong là khi:* notebook chạy từ đầu đến cuối trong kernel sạch.
- [x] **T-12** Sinh đủ 8 biểu đồ bắt buộc vào `reports/figures/` với tiền tố `01_` — *xong là khi:* đủ 8 tệp PNG, mỗi biểu đồ có tiêu đề và nhãn trục.
- [x] **T-13** Viết nhận xét EDA: mỗi biểu đồ một đoạn trả lời một câu hỏi cụ thể — *xong là khi:* không có biểu đồ nào bị bỏ trống nhận xét. Ba phát hiện đã biết cần nêu: giờ 2h có tỷ lệ gian lận 1,713% so với 0,048% lúc 10h; trung vị `Amount` của gian lận (9,25) thấp hơn hợp lệ (22,00) nhưng trung bình lại cao hơn; 27 giao dịch gian lận có `Amount = 0`.
- [x] **T-14** `notebooks/02_statistics.ipynb`: Mann–Whitney U cho 30 đặc trưng + hiệu chỉnh Benjamini–Hochberg + Cohen's d — *xong là khi:* có bảng xếp hạng và nêu rõ số kiểm định còn ý nghĩa sau hiệu chỉnh.
- [x] **T-15** Lưu bảng xếp hạng ra `reports/feature_ranking.csv` để notebook 07 đối chiếu với SHAP — *xong là khi:* tệp tồn tại, có cột `cohens_d` và `p_adjusted`.

> Điểm cần viết trong T-14: tương quan Pearson cao nhất chỉ −0,326 (V17) trong khi Cohen's d là −8,32. Với 0,17% mẫu dương, tương quan điểm-nhị phân bị nén xuống — cùng một cái bẫy như accuracy 99,83%.

### Ghi chú khi làm xong giai đoạn 1

**Các con số mốc ở T-13 và T-14 là số trên dữ liệu THÔ.** Pipeline chạy trên dữ liệu đã loại
1.081 dòng trùng lặp (DS-20), nên số thực tế lệch nhẹ. Notebook 01 §1.1 in bảng đối chiếu hai
cột; quy ước đã chốt: **báo cáo lấy cột "sau khi loại trùng lặp"**.

| Mốc | Dữ liệu thô | Sau khi loại trùng lặp |
|---|---|---|
| Tỷ lệ gian lận lúc 2h | 1,713% | **1,451%** (48/3.308) |
| Tỷ lệ gian lận lúc 10h | 0,048% | **0,048%** (8/16.548) |
| Trung vị `Amount` gian lận | 9,25 | **9,82** (hợp lệ 22,00) |
| Trung bình `Amount` gian lận | 122,21 | **123,87** (hợp lệ 88,41) |
| Gian lận có `Amount = 0` | 27 | **25** |
| Tương quan Pearson mạnh nhất (V17) | −0,326 | **−0,313** |
| Cohen's d của V17 | −8,32 | **−8,09** |

Hướng của cả ba phát hiện giữ nguyên, chỉ độ lớn nhích nhẹ. Xem thêm [docs/02 §6](docs/02-dac-ta-du-lieu.md).

**Phát sinh thêm trong giai đoạn này** (không có trong kế hoạch ban đầu, nhưng cần để T-14 và
T-15 tái lập được):

- `src/stats.py` — `feature_ranking()`, `benjamini_hochberg()`, `cohens_d()`,
  `cliffs_delta_from_u()`, `pearson_from_cohens_d()`.
- `tests/test_stats.py` — 24 ca, chốt thủ tục BH và công thức effect size bằng ví dụ tính tay được.
- `src/plots.py` — bổ sung nhãn trục còn thiếu ở EDA 4, 5, 6, 7, 8 để đạt điều kiện của T-12.
- Ba hình phụ tiền tố `02_` trong `reports/figures/` (không tính vào 8 hình bắt buộc của T-12).

**Kết quả chính:** 27/30 kiểm định còn ý nghĩa sau hiệu chỉnh BH (Bonferroni chỉ giữ 24);
17 đặc trưng có |Cohen's d| ≥ 0,8. Cohen's d và Cliff's delta bất đồng về thứ hạng
(Spearman 0,921): V17 hạng 1 theo d nhưng hạng 11 theo delta — để T-32 đối chiếu với SHAP.

---

## Giai đoạn 2 — Mô hình cơ sở (ngày 5–7) ✅

- [x] **T-16** `notebooks/03_baseline.ipynb`: chia phân tầng 80/20, lưu `data/test_set.parquet` — *xong là khi:* tập test có đúng 95 mẫu gian lận (sau khi loại trùng lặp).
- [x] **T-17** Mô hình rỗng luôn dự đoán 0 — *xong là khi:* có con số accuracy 99,83% đưa vào báo cáo làm bằng chứng cho G-2.
- [x] **T-18** Logistic Regression và Decision Tree không xử lý mất cân bằng — *xong là khi:* có bảng mốc so sánh.
- [x] **T-19** Bảng đối chiếu accuracy / PR-AUC / Recall / Precision cho ba mô hình — *xong là khi:* bảng cho thấy accuracy gần như không phân biệt được chúng còn PR-AUC thì có.

### Ghi chú khi làm xong giai đoạn 2

**Bảng mốc** (tập kiểm thử 56.746 dòng / 95 gian lận, τ = 0,5 chưa tối ưu):

| mô hình | accuracy | PR-AUC | ROC-AUC | recall | precision | bắt được |
|---|---|---|---|---|---|---|
| mô hình rỗng | 99,8326% | 0,00167 | 0,5000 | 0,000 | — | 0/95 |
| logistic regression | 99,9154% | 0,6961 [0,598–0,791] | 0,9577 | 0,600 | 0,851 | 57/95 |
| decision tree | 99,9383% | 0,6275 [0,522–0,737] | 0,8746 | 0,716 | 0,895 | 68/95 |

**Bằng chứng cho G-2 mạnh hơn dự kiến.** Accuracy không chỉ *không phân biệt* được ba mô hình
(chênh 0,11 điểm phần trăm) mà còn **xếp hạng ngược**: cây có accuracy cao hơn hồi quy
(99,9383% so với 99,9154%) nhưng PR-AUC lại thấp hơn (0,6275 so với 0,6961). PR-AUC chênh
**416 lần** giữa mô hình tệ nhất và tốt nhất.

**Chưa được tuyên bố mô hình nào thắng.** Bootstrap hiệu PR-AUC theo cặp (04 §5.4) cho
`+0,069 [−0,031, +0,158]` — khoảng tin cậy **chứa 0**. Hồi quy thắng ở 92% số lần lặp, đủ để
nói "có xu hướng nhỉnh hơn", không đủ để kết luận. Quy tắc này phải áp lại cho bảng 20 tổ hợp
ở T-20.

**Phát hiện đáng mang sang giai đoạn 5.** Decision Tree `max_depth=6` chỉ phát ra **8 điểm rủi
ro khác nhau** (hồi quy: 56.537), nên đường PR của nó là cầu thang 9 điểm. Đó là lý do cơ học
khiến nó thua ở PR-AUC dù thắng ở τ = 0,5 — và là cảnh báo rằng **không thể tối ưu ngưỡng trên
một mô hình chỉ có 8 bậc**. Mô hình cuối cùng bắt buộc phải thuộc họ ensemble.

**Phát sinh thêm:** `src/plots.py` thêm `plot_roc_curves()` (bản sinh đôi của `plot_pr_curves`,
để đặt ROC cạnh PR trên cùng bộ mô hình) và nhãn trục tiếng Việt cho `plot_confusion()`.
Hiện vật mới: `data/test_set.parquet` (15,4 MB), `reports/baseline_results.csv`, 4 hình `03_*`.

**Còn cách mục tiêu:** AC-M1 cần PR-AUC ≥ 0,75 (đang 0,696), AC-M2 cần recall ≥ 0,75
(đang 0,60 và 0,72).

---

## Giai đoạn 3 — Chiến lược mất cân bằng (ngày 8–10) ✅

- [x] **T-20** `notebooks/04_imbalance_strategies.ipynb`: chạy `run_grid()` đủ 20 tổ hợp — *xong là khi:* `reports/grid_results.csv` có 20 dòng. **Cảnh báo thời gian:** 1–3 giờ. Nếu vượt 3 giờ thì hạ `n_estimators` xuống 100 ở giai đoạn so sánh và ghi rõ trong báo cáo.
- [x] **T-21** Vẽ đường PR của 5 chiến lược chồng lên nhau cho mô hình tốt nhất — *xong là khi:* hình có đường cơ sở 0,00167.
- [x] **T-22** Viết thảo luận so sánh `scale_pos_weight` với SMOTE — *xong là khi:* nêu được cái nào thắng, thắng bao nhiêu, và rẻ hơn bao nhiêu về thời gian huấn luyện.
- [x] **T-23** Kiểm tra tính hợp lý: nếu PR-AUC > 0,95 thì dừng lại và rà rò rỉ — *xong là khi:* đã đối chiếu với dải kỳ vọng 0,80–0,87 ở [docs/04 §3.4](docs/04-thiet-ke-mo-hinh-ml.md).

### Cách chạy lưới

Lưới chạy ngoài notebook để kernel chết không mất trắng, và để theo dõi được tiến độ:

```powershell
.\.venv\Scripts\python.exe scripts\run_grid.py --smoke    # thử 12.000 dòng, ~1 phút
.\.venv\Scripts\python.exe scripts\run_grid.py            # thật, ~20-40 phút
.\.venv\Scripts\python.exe scripts\run_grid.py --status   # xem tiến độ, chạy ở cửa sổ khác
```

Ngắt giữa chừng thì chạy lại đúng lệnh đó, nó nối tiếp từ tổ hợp dang dở. Xong rồi mở
`notebooks/04_imbalance_strategies.ipynb` và Run All — notebook thấy điểm lưu nên bỏ qua
phần huấn luyện, chạy hết trong khoảng một phút.

### Ghi chú khi làm xong giai đoạn 3

Lưới chạy thật **45,7 phút** (ước lượng ban đầu 20–40 phút; phần vượt là do bước Tomek, xem
phát hiện 3). PR-AUC cao nhất **0,8549** — trong dải kỳ vọng 0,80–0,87, cách xa ngưỡng báo
động 0,95, **T-23 đạt, không có dấu hiệu rò rỉ**.

**Sáu dòng đầu bảng**

| mô hình | chiến lược | PR-AUC | Recall@τ* | Precision@τ* | điểm rủi ro khác nhau | giây |
|---|---|---|---|---|---|---|
| xgboost | class_weight | 0,8549 ± 0,0301 | 0,852 | 0,702 | 221.674 | **36** |
| random_forest | smote | 0,8549 ± 0,0311 | 0,855 | 0,666 | 219 | 368 |
| random_forest | smote_tomek | 0,8549 ± 0,0311 | 0,855 | 0,666 | 219 | 652 |
| xgboost | smote | 0,8529 ± 0,0297 | 0,852 | 0,638 | 222.236 | 45 |
| xgboost | smote_tomek | 0,8529 ± 0,0297 | 0,852 | 0,638 | 222.236 | 326 |
| xgboost | none | 0,8514 ± 0,0302 | 0,841 | **0,867** | 221.544 | 33 |

**1. Nhóm dẫn đầu hoà nhau, không được xếp hạng.** Dải PR-AUC của 6 dòng đầu là 0,0145, hẹp
hơn một nửa độ lệch chuẩn giữa các fold (0,030). Bootstrap hiệu theo cặp (04 §5.4): 3/4 cặp có
khoảng tin cậy **chứa 0**. Chỉ một khác biệt là thật — `xgboost+class_weight` hơn
`random_forest+class_weight` `+0,0152 [+0,0056, +0,0270]`.

**2. T-22 — dự báo của 04 §3.4 chỉ đúng với XGBoost.** Xem [docs/04 §3.4](docs/04-thiet-ke-mo-hinh-ml.md)
đã cập nhật số thực đo. Tóm tắt: `scale_pos_weight` hoà SMOTE ở XGBoost và rẻ hơn 1,24 lần
(đúng dự báo); nhưng **thua thật** ở Random Forest (−0,0145, KTC không chứa 0) và Logistic
Regression, còn ở Decision Tree thì `class_weight` làm PR-AUC **sụp từ 0,6948 xuống 0,3586**.
Không được dùng con số trung bình gộp cả 4 mô hình (−0,0775) vì nó bị decision_tree kéo lệch.

**3. SMOTE + Tomek KHÔNG khác SMOTE, mà ngốn 40% thời gian lưới.** Mảng điểm out-of-fold
**giống hệt nhau tới từng phần tử** ở cả 4 mô hình — bước Tomek xoá đúng 0 cặp. Chi phí
**1.089 giây = 18,2 phút** trên tổng 45,7 phút, đổi lại không một chữ số nào thay đổi.
Lý do — kiểm lại trên fold 1 của lưới (2026-10-09): dữ liệu gốc **có** cặp Tomek. Trước SMOTE,
`TomekLinks` tìm được 21 cặp (xoá 42 dòng, trong đó 21 vụ gian lận); sau SMOTE còn 0. Chính SMOTE
xoá chúng: mỗi vụ gian lận thật được dùng làm gốc để sinh khoảng 59 điểm tổng hợp trên các đoạn nối
nó với láng giềng cùng lớp, nên láng giềng gần nhất của nó trở thành một điểm gian lận tổng hợp —
không còn cặp khác lớp nào là láng giềng gần nhất **của nhau**. `SMOTETomek` chạy SMOTE trước, Tomek
sau, nên ở tỷ lệ này bước Tomek luôn vô tác dụng; muốn Tomek có tác dụng phải dọn **trước** khi sinh mẫu.
(Bản ghi chú đầu tiên giải thích là "vùng biên quá thưa" — sai: trước SMOTE có 21 cặp.)
**Khuyến nghị bỏ S5 khỏi các lần chạy lại về sau**, nhưng giữ trong bảng báo cáo vì kết quả âm
tính này tự nó là phát hiện.

**4. Random Forest hoà PR-AUC nhưng chỉ có 200–269 điểm rủi ro khác nhau** (XGBoost: ~222.000,
tức gấp **1.000 lần**). RF 300 cây chỉ phát ra tối đa 301 giá trị `k/300`. Đây là cùng vấn đề
notebook 03 đã chỉ ra ở Decision Tree, nhẹ hơn nhưng vẫn đủ để loại RF khỏi giai đoạn 5 — nơi
toàn bộ luận điểm nằm ở việc kéo ngưỡng.

**Chốt mô hình cho T-24: `xgboost` + `class_weight`.** Hoà ở nhóm PR-AUC cao nhất, nhanh hơn
10,2 lần so với `random_forest+smote` cùng điểm, và nhiều hơn 1.000 lần số điểm rủi ro khác
nhau.

**Phát sinh thêm:** `scripts/run_grid.py` (chạy lưới ngoài notebook, có điểm lưu và chạy nối
tiếp), `run_grid()` viết lại trong `src/modeling.py` (một lượt huấn luyện thay vì hai, bảng đủ
18 cột theo 04 §3.3, trả điểm out-of-fold), `tests/test_grid.py` 19 ca.

---

## Giai đoạn 4 — Tinh chỉnh và kiểm chứng (ngày 11–12) ✅

- [x] **T-24** `notebooks/05_advanced_models.ipynb`: `RandomizedSearchCV` 30 lần thử, `scoring='average_precision'` — *xong là khi:* có bảng 5 cấu hình tốt nhất.
- [x] **T-25** Huấn luyện lại trên toàn tập train, đánh giá trên tập test — *xong là khi:* AC-M1 (PR-AUC ≥ 0,75) và AC-M2 (Recall ≥ 0,75) đạt.
- [x] **T-26** Bootstrap 1.000 lần cho PR-AUC, Recall, Precision — *xong là khi:* mọi chỉ số chính trong báo cáo đều có khoảng tin cậy (AC-M6).
- [x] **T-27** Đánh giá theo cách chia thời gian (ngày 1 train, ngày 2 test) — *xong là khi:* có bảng đối chiếu hai cách chia và một đoạn giải thích vì sao cách theo thời gian thấp hơn (AC-M7).
- [x] **T-28** **Rà soát rò rỉ dữ liệu** theo danh sách kiểm [docs/08 §3.1](docs/08-ke-hoach-kiem-thu.md) — *xong là khi:* cả 7 ô trong danh sách được tick và `pytest tests/test_no_leakage.py` xanh (AC-M4). **Làm ở ngày 12, không để đến ngày 20.**

### Cách chạy

```powershell
.\.venv\Scripts\python.exe scripts\run_search.py --smoke   # thử 12.000 dòng, 4 lần thử, ~1 phút
.\.venv\Scripts\python.exe scripts\run_search.py           # thật: 30 lần thử × 5 fold, ~20 phút
.\.venv\Scripts\python.exe scripts\run_search.py --status  # xem kết quả đã lưu
```

Rồi mở `notebooks/05_advanced_models.ipynb` và Run All (~20 phút: điểm out-of-fold, bootstrap,
ba cách chia tập). Notebook cần `reports/grid_results.npz` của giai đoạn 3 — tệp này không nằm
trong git, thiếu thì chạy `scripts/run_grid.py` trước.

Đủ lệnh (PowerShell và Git Bash), hiện vật sinh ra, số đối chiếu và các bẫy đã gặp:
[docs/lenh-chay §4](docs/lenh-chay.md).

### Ghi chú khi làm xong giai đoạn 4

**1. T-24 — tinh chỉnh là kết quả âm tính.** 30 lần thử, 19,8 phút. Cấu hình đầu bảng
(`n_estimators=750, max_depth=6, learning_rate=0,090, scale_pos_weight=10`) đạt PR-AUC CV
0,8538 ± 0,0328, **thấp hơn** cấu hình mặc định của giai đoạn 3 (0,8549). Năm cấu hình đầu cách
nhau 0,0008 trong khi độ lệch chuẩn giữa các fold là 0,031. `scale_pos_weight` gần như không ảnh
hưởng PR-AUC: nó dịch điểm rủi ro chứ không đổi thứ tự.

**2. T-25 — chọn cấu hình MẶC ĐỊNH, và chọn trên tập train.** Quy tắc đặt trước: tinh chỉnh chỉ
thay mặc định nếu bootstrap hiệu PR-AUC theo cặp trên điểm out-of-fold không chứa 0. Kết quả
−0,0006 [−0,0087, +0,0072] → giữ mặc định. Trên tập test:

| chỉ số (τ\* = 0,0232, chọn trên OOF) | giá trị [KTC 95%] | tiêu chí |
|---|---|---|
| PR-AUC | **0,825** [0,747 – 0,896] | AC-M1 ≥ 0,75 ✅ |
| Recall@τ\* | **0,811** [0,737 – 0,884] — 77/95 vụ | AC-M2 ≥ 0,75 ✅ |
| Precision@τ\* | 0,670 [0,600 – 0,752] — 38 cảnh báo giả | |
| ROC-AUC | 0,977 [0,961 – 0,991] | |

**Cận dưới của PR-AUC và recall đều sát dưới 0,75** — đạt theo ước lượng điểm, không đạt "chắc
chắn". Không được viết khác đi trong báo cáo.

**3. T-27 — chia theo thời gian thấp hơn, do độ lệch thời gian chứ không do thiếu dữ liệu.**

| cách chia | PR-AUC test | Precision@τ\* | FP |
|---|---|---|---|
| ngẫu nhiên 80/20 | 0,825 | 0,670 | 38 |
| ngẫu nhiên, train thu nhỏ bằng ngày 1 (đối chứng) | 0,825 | 0,826 | 16 |
| ngày 1 → ngày 2 | **0,782** | **0,332** | 338 |

Dòng đối chứng tách được hai nguyên nhân: bớt dữ liệu chỉ mất −0,0004, còn −0,043 là do thời
gian. Khoảng tin cậy PR-AUC hai cách chia chồng lấn nhau, nhưng precision tại ngưỡng cố định thì
tách hẳn: cùng τ\*, ngày 2 sinh báo động giả gấp 3,6 lần. Tỷ lệ gian lận cũng giảm từ 0,189%
xuống 0,144%. Đây là đầu vào cho T-31 và phần hạn chế ở T-64.

**4. T-28 — 7/7 ô, thành kiểm thử tự động.** Mỗi ô ở [docs/08 §3.1](docs/08-ke-hoach-kiem-thu.md)
giờ là một kiểm thử trong `tests/test_no_leakage.py`, quét `src/`, `scripts/`, `app.py` và mọi
notebook. Lần quét tìm ra đúng một vi phạm: `app.py` dùng `random_state=0`, đã sửa.

**5. Phát hiện về tái lập — cần cho T-38.** XGBoost `hist` cho điểm **lệch nhẹ theo số luồng**:
cùng cấu hình chạy `n_jobs=1` và `n_jobs=-1` khác nhau ở chữ số thứ ba–tư. Cùng máy thì tái lập
tuyệt đối; khác số lõi thì có thể lệch. Đã ghi vào [docs/10 §6](docs/10-van-hanh-tai-lap.md).

**Phát sinh thêm:** `run_search()`, `search_space()`, `search_table()`, `best_params_from()` trong
`src/modeling.py`; `bootstrap_diff()` trong `src/evaluate.py` (bootstrap theo cặp, trước đây viết
lại trong từng notebook); `scripts/run_search.py`; `tests/test_search.py` 12 ca; 9 kiểm thử mới (30 ca sau khi tham số hoá) trong
`tests/test_no_leakage.py`. Hiện vật: `reports/search_results.csv`, `best_params.json`,
`final_test_metrics.csv`, `split_comparison.csv`, 4 hình `05_*`.

**Mang sang giai đoạn 5:** mô hình là `build_pipeline("xgboost", "class_weight", y=y_train)` với
cấu hình mặc định; τ\* chi phí trên out-of-fold = 0,0232.

---

## Giai đoạn 5 — Ngưỡng, chi phí, SHAP (ngày 13–14) ✅

- [x] **T-29** `notebooks/06_threshold_and_cost.ipynb`: đường cong chi phí + 4 phương án ngưỡng, chọn ngưỡng trên dữ liệu **out-of-fold** (`oof_scores`), không trên tập test — *xong là khi:* ML-08 được tôn trọng.
- [x] **T-30** Bảng bắt buộc trong báo cáo: 4 ngưỡng × (cảnh báo/ngày, TP, FP, FN, Precision, Recall, chi phí) — *xong là khi:* có thêm một đoạn diễn giải bằng lời dạng "hạ ngưỡng từ 0,5 xuống τ\* bắt thêm N vụ, đổi lại M cảnh báo giả, tiết kiệm ròng X".
- [x] **T-31** Phân tích độ nhạy theo tỷ lệ chi phí 5:1 → 100:1 bằng `sensitivity_analysis()` — *xong là khi:* có hình cho thấy ngưỡng tối ưu dịch chuyển thế nào (AC-M8).
- [x] **T-32** `notebooks/07_explainability.ipynb`: SHAP toàn cục (`summary_plot` trên 2.000 mẫu) + xếp hạng `mean|SHAP|` — *xong là khi:* có bảng so SHAP với xếp hạng thống kê ở T-15.
- [x] **T-33** Phân tích lỗi: giải thích SHAP cho các FN điểm cao nhất và FP điểm cao nhất — *xong là khi:* nêu được mẫu hình chung, kèm câu chốt rằng V1–V28 là PCA nên không diễn giải thành nguyên nhân nghiệp vụ.
- [x] **T-34** *(Tùy chọn)* Autoencoder huấn luyện chỉ trên lớp bình thường — *xong là khi:* so được PR-AUC với mô hình có giám sát và giải thích **vì sao** nó thua (không dùng tới 492 nhãn có sẵn).

### Cách chạy

```powershell
foreach ($nb in "06_threshold_and_cost", "07_explainability", "06b_autoencoder") {
    .\.venv\Scripts\python.exe -m jupyter nbconvert --to notebook --execute --inplace `
        --ExecutePreprocessor.timeout=3600 "notebooks\$nb.ipynb"
}
```

Tổng khoảng 10 phút. Cần `reports/grid_results.npz` (điểm out-of-fold của giai đoạn 3); thiếu thì
notebook tự tính lại. Đủ lệnh, hiện vật và số đối chiếu:
[docs/lenh-chay §5](docs/lenh-chay.md).

### Ghi chú khi làm xong giai đoạn 5

**1. T-29 — τ\* = 0,02317, dò trên mọi điểm out-of-fold chứ không trên lưới 200 điểm.** Với riêng
τ\* thì hai cách gần như trùng nhau (lưới cho 0,02321). Nhưng lưới 200 lượng tử làm hỏng tiêu chí
ngân sách: nó chỉ dùng 172,5 trong 200 cảnh báo/ngày và mất 8 điểm recall OOF. `pick_threshold`,
`threshold_alternatives` và `sensitivity_analysis` vì thế nhận thêm `thresholds=`. Đáy chi phí
phẳng: mọi τ ∈ [0,013; 0,140] đều nằm trong 5% cực tiểu.

**2. T-30 — bảng bắt buộc (tập test, ngưỡng chọn trên OOF):**

| Ngưỡng | Cảnh báo/ngày | TP | FP | FN | Precision | Recall | Chi phí |
|---|---|---|---|---|---|---|---|
| 0,5 (mặc định) | 195 | 74 | 4 | 21 | 0,949 | 0,779 | 2.586 EUR |
| **τ\* = 0,0232** | **287** | **77** | **38** | **18** | **0,670** | **0,811** | **2.390 EUR** |
| τ cho recall ≥ 90% | 2.007 | 83 | 720 | 12 | 0,103 | 0,874 | 5.067 EUR |
| τ cho 200 cảnh báo/ngày | 175 | 69 | 1 | 26 | 0,986 | 0,726 | 3.182 EUR |

*"Hạ ngưỡng từ 0,5 xuống τ\* bắt thêm 3 vụ, đổi lại 34 cảnh báo giả, tiết kiệm ròng 197 EUR"*
(khoảng 490 EUR/ngày trên toàn luồng). **Trên tập test, khoảng tin cậy của khoản tiết kiệm chứa 0**
(−197 [−676, +160]). Trên OOF với 378 gian lận thì không chứa 0 (−707 [−1.513, −24]). Báo cáo phải
viết đúng như vậy. Ngưỡng 0,5 không tệ như 04 §6.1 dự báo, vì `scale_pos_weight` đã đẩy điểm lên.
Có thêm `bootstrap_threshold_diff()` trong `src/evaluate.py` để có khoảng tin cậy cho câu diễn giải.

**3. T-31 — khuyến nghị vững.** τ\* không đổi trên dải 20:1 → 50:1. Nếu dùng τ\* mặc định khi tỷ lệ
thật khác, mức hối tiếc ≤ 5% trong dải 10:1 → 75:1. Recall test luôn nằm trong 0,77–0,85. Tỷ lệ
chi phí thật sự quyết định **khối lượng thẩm định** (190 → 770 cảnh báo/ngày), không quyết định recall.

**4. T-32 — V14, V4, V12, V11, V10 chiếm 46% mean|SHAP|.** Spearman với xếp hạng T-15 chỉ 0,42.
V17 hạng 1 theo Cohen's d nhưng hạng 24/31 theo SHAP, vì nó thừa thông tin khi mô hình đã có
V14/V12/V10. Đây chính là câu hỏi mà ghi chú giai đoạn 1 để lại cho T-32.

**5. T-33 — lỗi là giới hạn của đặc trưng.** 12/18 FN có điểm dưới 0,001, và 67% FN có cả 5 đặc
trưng chính nằm trong vùng của lớp hợp lệ. FP điểm cao thì mang đủ chữ ký V14/V10/V12. Đã có câu
chốt về PCA.

**6. T-34 — autoencoder PR-AUC 0,329 so với 0,825.** Nó thấy 93/95 vụ gian lận là bất thường,
nhưng không phân biệt được giao dịch hợp lệ hiếm gặp với gian lận. Môi trường không có PyTorch nên
dùng `MLPRegressor` (`src/anomaly.py`), notebook riêng `06b_autoencoder.ipynb`.

**Phát sinh thêm:** `src/anomaly.py`; `bootstrap_threshold_diff()`; tham số `thresholds=` và sửa
lỗi `sample_fraction` bị bỏ qua trong `sensitivity_analysis()` (phóng đại cảnh báo/ngày 4 lần khi
chạy trên OOF); `tests/test_anomaly.py`; 5 ca mới trong `test_threshold.py` và `test_search.py`;
notebook 06, 06b, 07 được thêm vào danh sách quét thứ tự của `test_no_leakage.py`.

**Mang sang giai đoạn 6:** `threshold.json` lấy τ mặc định = **0,02317** cùng 4 phương án ở bảng
trên. `explainer.joblib` bọc bước `clf` và nhận đầu vào sau `pipeline[:-1].transform`. Xếp hạng
mean|SHAP| lấy từ `reports/shap_ranking.csv`.

---

## Giai đoạn 6 — Xuất hiện vật (ngày 14) ✅

- [x] **T-35** `notebooks/08_export_artifacts.ipynb` sinh `models/model.joblib` (pipeline **hoàn chỉnh**, gồm cả scaler) — *xong là khi:* nạp lại tệp và `predict_proba` cho đúng kết quả như trong notebook (AR-02).
- [x] **T-36** Sinh `models/explainer.joblib`, `models/metrics.json`, `models/threshold.json` — *xong là khi:* cấu trúc khớp [docs/06 §6](docs/06-thiet-ke-luu-tru.md).
- [x] **T-37** Sinh `data/sample_pool.json` khoảng 200 giao dịch, đủ 4 nhóm `fraud_easy` / `fraud_hard` / `legit_easy` / `legit_hard` — *xong là khi:* mỗi nhóm "hard" có ít nhất 20 mẫu.
- [x] **T-38** **Kiểm tra tái lập**: chạy lại toàn bộ notebook trong kernel sạch — *xong là khi:* PR-AUC lệch dưới 0,001 (AC-M5, NFR-07).

### Cách chạy

```powershell
.\.venv\Scripts\python.exe -m jupyter nbconvert --to notebook --execute --inplace `
    --ExecutePreprocessor.timeout=3600 notebooks\08_export_artifacts.ipynb     # 5–10 phút
.\.venv\Scripts\python.exe -m pytest tests\test_artifacts.py                 # kiểm chính các tệp vừa sinh
.\.venv\Scripts\python.exe scripts\check_reproducibility.py                  # T-38: 01 → 08 trong kernel sạch, ~70 phút
```

Notebook 08 phải chạy **sau** notebook 03 (chạy lại 03 là mất cột `risk_score` của
`test_set.parquet`). Đủ lệnh, hiện vật, số đối chiếu và các bẫy đã gặp:
[docs/lenh-chay §6](docs/lenh-chay.md). Cấu trúc từng tệp:
[docs/06 §6](docs/06-thiet-ke-luu-tru.md), đã viết lại theo số thật.

### Ghi chú khi làm xong giai đoạn 6

**Không có con số mới nào ở giai đoạn này.** Notebook 08 tính lại mọi thứ từ đầu, so với bảng mà
notebook 03–07 đã ghi vào `reports/` ở những phiên chạy khác, và dừng ở `assert` nếu lệch.

| Đối chiếu | Lệch lớn nhất |
|---|---|
| điểm out-of-fold tính lại với `grid_results.npz` của giai đoạn 3 | **0** (trùng từng bit) |
| 5 ngưỡng với notebook 06 | 6·10⁻¹⁷ |
| chỉ số chính và **cả hai đầu** khoảng tin cậy với notebook 05 | 1·10⁻¹⁶ |
| `mean|SHAP|` với notebook 07 (tương đối) | 4·10⁻⁸ |

**1. T-35 — `model.joblib` 1,2 MB, trùng từng bit khi nạp lại.** Pipeline hai bước: `RobustScaler`
cho `Amount` (trung vị 22,08, IQR 72,16, học chỉ trên tập train) → `XGBClassifier`. Kiểm hai lần:
nạp lại trong notebook, và nạp lại trong **một tiến trình Python mới** đọc `test_set.parquet` qua
`build_features()` — đúng đường đi của API. Cả hai cho `predict_proba` trùng từng bit trên 56.746
giao dịch. `test_set.parquet` có thêm cột `risk_score` (32 cột, 15,9 MB).

**2. T-36 — ba tệp đúng cấu trúc 06 §6, trùng số với báo cáo.**

- `threshold.json`: τ\* = **0,023173** cùng 4 phương án, chọn trên out-of-fold.
- `explainer.joblib` 4,2 MB: tính cộng khớp margin tới 2,2·10⁻⁵. Giải thích **một** giao dịch mất trung
  vị 17–21 ms, chậm nhất 39–71 ms qua hai lần chạy, dưới ngân sách 50–200 ms của NFR-01.
- `metrics.json` 1,6 MB (90% là `test_scores`), dưới dự báo 3–8 MB nên chưa cần tách ra `.npz`.
  Ngoài 12 khóa bắt buộc có thêm 6 khóa cho UI-04 và API: `strategy_pr_curves`, `baseline_comparison`,
  `threshold_options`, `training`, `fingerprint`, `environment`.
- Khoảng tin cậy trùng báo cáo tới 10⁻¹⁶ là **có chủ đích**: bootstrap phụ thuộc thứ tự dòng, nên
  chỉ số tính trên thứ tự của `split_data()` như notebook 05, còn `test_scores` theo thứ tự của
  `test_set.parquet`. Tính trên thứ tự của tệp thì khoảng tin cậy lệch ở chữ số thứ ba.

**3. T-37 — 200 mẫu: `fraud_easy` 29, `fraud_hard` 21, `legit_easy` 112, `legit_hard` 38.**

- `fraud_hard` = **toàn bộ** gian lận có điểm dưới 0,5 của tập kiểm thử: 18 vụ bị bỏ lọt ở τ\* và 3 vụ
  chỉ bắt được khi hạ ngưỡng về τ\* — đúng 3 vụ "bắt thêm" trong câu diễn giải của T-30. Lấy mốc τ\*
  cho cả gian lận thì chỉ còn 18, không đạt điều kiện 20.
- `legit_hard` = toàn bộ 38 cảnh báo giả ở τ\*, trong đó 4 vụ vượt cả 0,5.
- **Biên của `fraud_hard` mỏng (21 so với 20)**; `validate_sample_pool` chặn việc xuất nếu huấn luyện
  lại làm nhóm này tụt dưới 20.
- Mô tả từng mẫu viết số theo tiếng Việt (dấu phẩy thập phân) vì giao diện hiện nguyên văn.

**4. T-38 — tái lập tuyệt đối trên cùng máy.** `scripts/check_reproducibility.py` chạy lại 9 notebook
01 → 08, mỗi notebook một kernel mới, tổng **71 phút** (notebook 05 chiếm 33).

| So trước và sau | Kết quả |
|---|---|
| PR-AUC | 0,8252465892 → 0,8252465892, **lệch 0** (dung sai 0,001) — AC-M5 đạt |
| 12 con số chính và τ\* | lệch 0 |
| Dấu vân tay điểm test và điểm out-of-fold | trùng |
| 11 bảng `reports/*.csv`, mọi hình PNG | trùng hoàn toàn (trừ cột đo thời gian chạy) |

Hai giới hạn phải nói khi báo cáo: script **không** chạy lại `run_grid.py` và `run_search.py` (notebook
04, 05 đọc điểm lưu; notebook 08 bù bằng cách tính lại điểm out-of-fold của mô hình được chọn), và mới
kiểm trên **một** máy 12 lõi. Kết quả từng dòng: `reports/reproducibility.csv`.

**Phát sinh thêm:** `src/artifacts.py` (dựng, ghi JSON chặt và nguyên tử, kiểm tra cấu trúc — API sẽ
dùng lại ở T-44); `tests/test_artifacts.py` (23 ca đơn vị + 7 ca tích hợp trên chính các tệp đã
xuất); `scripts/check_reproducibility.py`; notebook 08 được thêm vào danh sách quét của
`test_no_leakage.py`; hình `reports/figures/08_sample_pool.png`. Tài liệu: viết lại 06 §6 theo cấu
trúc thật, cập nhật 06 §2/§9/§10, 10 §3/§6, `models/README.md`, bảng notebook trong README.

**Mang sang giai đoạn 7:**

- `model_version = "xgb_scaleposweight_v1"`; τ mặc định cho bảng `settings` lấy từ `threshold.json`.
- `/explain` phải đưa `model[:-1].transform(build_features(x))` vào explainer, **không** đưa
  `build_features(x)`; tên đặc trưng theo `metrics.json → training.model_feature_names` (`Amount`
  đứng đầu). SHAP ở thang log-odds, `base_value` = `training.shap_base_value`.
- API gọi `validate_threshold`, `validate_metrics` lúc khởi động và trả 503 nếu hiện vật sai (T-44).
- `app.py` (phương án A) vẫn nạp `models/best_model.pkl` cũ — phải trỏ sang `model.joblib` nếu mốc
  T-39 chọn phương án A.

---

## 🚩 Mốc quyết định — cuối ngày 14 ✅

- [x] **T-39** Chốt phương án ứng dụng và ghi vào README — *xong là khi:* quyết định được viết ra, không để lửng.
  **→ Đã chốt phương án B** (2026-09-28, [README](README.md#phương-án-ứng-dụng--đã-chốt-b-t-39)): hiện vật đủ và đã kiểm, giai đoạn 0–6 xong 38/38 việc, T-38 tái lập lệch 0.
  - Hiện vật đã đủ, còn đủ 6 ngày → **phương án B** (FastAPI + web riêng), làm tiếp giai đoạn 7.
  - Còn nợ việc ở phần mô hình → **phương án A** (Streamlit trong `app.py`), bỏ giai đoạn 7, làm rút gọn giai đoạn 8, dồn thời gian cho báo cáo.

---

## Giai đoạn 7 — API (ngày 15–16, chỉ phương án B) ✅

- [x] **T-40** `docker-compose.yml` dịch vụ `db` (postgres:16-alpine) + volume `pgdata` + healthcheck `pg_isready` — *xong là khi:* `docker compose up -d db` và `pg_isready` trả `accepting connections`. **Làm đầu tiên trong giai đoạn này** (rủi ro R-09).
- [x] **T-41** `api/db.py` engine + pool (`pool_pre_ping=True`) và `api/models_orm.py` — *xong là khi:* kết nối được bằng `DATABASE_URL` từ `.env`.
- [x] **T-42** Alembic `0001_initial_schema` (3 bảng, chỉ mục, ràng buộc `CHECK`) và `0002_seed_settings` — *xong là khi:* `upgrade head` → `downgrade base` → `upgrade head` chạy trọn vẹn (TC-47).
- [x] **T-43** `api/schemas.py` — toàn bộ Pydantic vào/ra theo [docs/05](docs/05-thiet-ke-api.md) — *xong là khi:* thiếu cột trả 422 kèm danh sách cột thiếu (TC-32).
- [x] **T-44** `api/main.py` nạp hiện vật lúc startup + `/health` — *xong là khi:* `/health` trả 503 khi thiếu mô hình hoặc mất cơ sở dữ liệu (TC-42).
- [x] **T-45** API-02, API-03: `/score`, `/score/batch` — *xong là khi:* chấm một mẫu và chấm lô cho cùng kết quả (TC-50).
- [x] **T-46** API-04: `/score/upload` dùng `COPY` của PostgreSQL — *xong là khi:* 10.000 dòng chấm điểm và ghi xong dưới 30 giây (NFR-02, AC-A1, TC-54).
- [x] **T-47** API-09…API-12: nhóm ngưỡng — *xong là khi:* `/threshold/preview` trả trong khoảng 10 ms và `/threshold/optimize` báo đúng `constraint_binding`.
- [x] **T-48** API-05: `/explain` — *xong là khi:* trả đúng 5 yếu tố dương và 3 yếu tố âm, khớp giá trị SHAP tính trong notebook (AC-A4).
- [x] **T-49** API-06…API-08: `/transactions`, `/reviews` với `ON CONFLICT DO UPDATE` — *xong là khi:* gửi thẩm định hai lần không tạo dòng thứ hai (TC-38) và `threshold_used` được ghi đúng (TC-39).
- [x] **T-50** API-13…API-15: `/metrics`, `/samples`, `/replay/stream` (SSE) — *xong là khi:* phát lại chạy liên tục 3 phút không lỗi (AC-A6).

### Cách chạy

```powershell
Copy-Item .env.example .env                                    # máy đã có PostgreSQL ở 5432: đặt POSTGRES_HOST_PORT=5433
docker compose up -d db                                        # T-40
.\.venv\Scripts\python.exe -m alembic upgrade head             # T-42
.\.venv\Scripts\python.exe -m pytest tests\test_db.py tests\test_scoring.py tests\test_api.py   # ~3 phút
.\.venv\Scripts\python.exe -m uvicorn api.main:app --reload --port 8000                          # /docs
```

Đủ lệnh, số đối chiếu, cách kiểm AC-A6 trên uvicorn và các bẫy đã gặp:
[docs/lenh-chay §7](docs/lenh-chay.md). Những điểm bản thi hành cụ thể hơn
hợp đồng ban đầu: [docs/05 §7](docs/05-thiet-ke-api.md).

### Ghi chú khi làm xong giai đoạn 7

**Kết quả theo điều kiện "xong là khi":**

| Việc | Điều kiện | Kết quả |
|---|---|---|
| T-40 | `pg_isready` trả `accepting connections` | đạt; cổng máy chủ 5433 vì máy đã có PostgreSQL 17 ở 5432 |
| T-41 | kết nối bằng `DATABASE_URL` từ `.env` | đạt; `pool_pre_ping`, `connect_timeout=3` |
| T-42 | `upgrade → downgrade → upgrade` trọn vẹn | đạt, lược đồ cuối trùng lược đồ đầu (TC-47) |
| T-43 | thiếu cột → 422 kèm danh sách | đạt: `{"missing": ["V13"]}` (TC-32) |
| T-44 | `/health` 503 khi thiếu mô hình hoặc mất db | đạt, cả khi hiện vật hỏng; tiến trình không sập (TC-42) |
| T-45 | chấm một mẫu = chấm lô | đạt, sai số < 1e-12 trên 100 giao dịch (TC-50) |
| T-46 | 10.000 dòng dưới 30 giây | **2,0 giây**; `COPY` 10.000 dòng 0,53 giây, nhanh gấp 158 lần `INSERT` từng dòng (TC-54) |
| T-47 | preview khoảng 10 ms; optimize báo đúng `constraint_binding` | phần tính 0,12 ms (15 ms qua HTTP); 5 tình huống ràng buộc đều đúng |
| T-48 | 5 dương + 3 âm, khớp SHAP của notebook | đạt với mọi giao dịch vượt ngưỡng được kiểm, SHAP khớp explainer tới từng bit — xem điểm 2 |
| T-49 | thẩm định hai lần không tạo dòng thứ hai; `threshold_used` đúng | đạt (TC-38, TC-39) |
| T-50 | phát lại 3 phút không lỗi | 185 giây trên uvicorn thật, 1.532 giao dịch, 0 lỗi (AC-A6) |

Toàn bộ kiểm thử: **311 xanh, 1 bỏ qua** (TC-12, chờ UI-03). Máy không có PostgreSQL thì 88 ca cần
cơ sở dữ liệu tự bỏ qua. NFR-01: p95 của `/score` **45,6 ms** phía máy chủ (57,8 ms tính cả lớp HTTP
của `TestClient`), đo khi máy có tải nền khoảng 50% CPU — biên mỏng.

**1. `POST /threshold/optimize` chọn ngưỡng trên out-of-fold, không trên tập kiểm thử.** Đây là một phép
**chọn** ngưỡng nên ML-08 áp dụng. Notebook 08 xuất thêm `models/oof_scores.npz` (0,8 MB, float32 giữ
nguyên kiểu). Với chi phí mặc định, API trả **đúng** 0,023172983899712563 của `threshold.json`; chỉ số
tại ngưỡng vẫn đo trên tập kiểm thử. Có ràng buộc thì nghiệm là "chi phí thấp nhất trong vùng khả
thi", nên với 200 cảnh báo/ngày API chọn 0,9657. `threshold.json` ghi 0,9625 cho cùng ngân sách vì
đó là "recall cao nhất trong ngân sách". Hai câu hỏi khác nhau, hai con số đều đúng.

**2. AC-A4 không đạt được theo nghĩa đen với mọi giao dịch.** 77% giao dịch của tập kiểm thử có
**ít hơn 5** đặc trưng SHAP dương: với giao dịch hợp lệ điểm thấp, mô hình kéo gần hết đặc trưng về
phía an toàn. `/explain` trả tối đa 5 dương và 3 âm, không mượn một đóng góp âm để cho đủ 5 — làm vậy
là gọi một yếu tố kéo điểm xuống là "đẩy lên". Trong hàng đợi (điểm ≥ τ\*) 114/115 giao dịch có đủ 5;
yếu tố âm luôn đủ 3. Báo cáo và buổi bảo vệ nên nói đúng như vậy.

**3. Serve bằng một luồng XGBoost.** Chấm một giao dịch bằng 12 luồng cho p95 87 ms (trượt NFR-01).
Chấm bằng 1 luồng cho **cùng từng bit** trên 56.746 giao dịch — chỉ huấn luyện mới phụ thuộc số luồng
(10 §6) — và p95 còn 45,6 ms. Lô 10.000 dòng chậm đi (0,06 → 0,26 giây), vẫn xa ngân sách 30 giây.

**4. Ngưỡng người dùng đặt gắn với `model_version`.** Bảng `settings` không nạp sẵn khóa `threshold`
nữa. Dòng này chỉ có khi người dùng tự đặt, và chỉ có hiệu lực với đúng mô hình lúc đặt. Nạp sẵn
một con số như thiết kế cũ thì sau khi huấn luyện lại, bảng vẫn giữ τ\* của mô hình cũ mà không ai biết.

**5. Điểm rủi ro không làm tròn trong phản hồi** (khác 05 §5 cũ). Máy chủ làm tròn 0,023170 thành
0,0232 thì trình duyệt thấy "vượt τ\*" trong khi máy chủ quyết định `allow`.

**Phát sinh thêm:** `api/config.py`, `api/loader.py` (kiểm hiện vật lúc khởi động, chấm lại 1/50 tập
kiểm thử để bắt lệch phiên bản thư viện), `api/errors.py`, `api/services/runtime_settings.py`,
`api/services/transactions.py`; `src/artifacts.py` thêm `write/read/validate_oof_scores`; notebook 08
thêm mục 3.4; `.env.example`; `alembic.ini` ở gốc repo, **chỉ ASCII** (Alembic đọc bằng cp1252 trên
Windows); `httpx`, `pytest` vào `requirements.txt`; `models/*.npz` vào `.gitignore`. Tài liệu:
docs/05 §2 (bảng `decision`), §4 (mã `INVALID_BODY`, `METHOD_NOT_ALLOWED`), §7 mới; docs/06 §2,
§4.4, §5, §6.4; docs/03 §3.2, §7; docs/08 §2.6; docs/10 §2.2, §2.3, §5.

**Mang sang giai đoạn 8:**

- API chạy bằng `uvicorn api.main:app --port 8000`; CORS mở cho `http://localhost:3000`.
- UI-D1: tải `GET /api/v1/metrics?section=test_scores` **một lần**, tính TP/FP/FN trong trình duyệt,
  so bằng `score >= τ` trên số thực nguyên độ chính xác. TC-12 (T-54) đối chiếu với
  `src/threshold.confusion_counts`.
- UI-03: đường cong chi phí lấy từ `POST /threshold/optimize` (trên out-of-fold, `curve_source`),
  còn số ở thanh trượt tính trên tập kiểm thử — hai nguồn khác nhau, nhãn trên màn hình phải nói rõ.
- UI-02: `POST /explain {transaction_id}`; danh sách yếu tố có thể ngắn hơn 5 ở giao dịch điểm thấp;
  nhãn thật chỉ lấy từ phản hồi của `POST /reviews`.
- UI-05: SSE `GET /replay/stream?speed=…&start=…`; tạm dừng là đóng `EventSource`, tiếp tục là mở lại
  với `start` bằng `sim_seconds` cuối cùng đã nhận.

---

## Giai đoạn 8 — Giao diện (ngày 17–19) ✅

Thứ tự dưới đây đồng thời là thứ tự **giữ lại** nếu thiếu thời gian.

- [x] **T-51** Khung ứng dụng: thanh bên 4 mục, thanh trên luôn hiện ngưỡng và `model_version` — *xong là khi:* điều hướng được giữa các màn hình.
- [x] **T-52** UI-01 hàng đợi thẩm định — *xong là khi:* sắp đúng theo điểm giảm dần (AC-A2), có trạng thái rỗng hướng dẫn 3 cách nạp dữ liệu.
- [x] **T-53** UI-03 cấu hình ngưỡng, tính tại máy khách theo UI-D1 — *xong là khi:* kéo thanh trượt cập nhật dưới 200 ms (AC-A3) và đổi tham số chi phí làm dịch chuyển ngưỡng tối ưu (AC-A5).
- [x] **T-54** Viết `tests/test_threshold_parity.py` cho TC-12 và bỏ `@pytest.mark.skip` — *xong là khi:* hàm JavaScript và `src/threshold.py` cho cùng TP/FP/FN trên 20 ngưỡng mẫu.
- [x] **T-55** UI-02 chi tiết giao dịch + biểu đồ thác nước SHAP — *xong là khi:* chú thích PCA cố định hiện diện và nhãn thật chỉ hiện **sau** khi người dùng quyết định.
- [x] **T-56** UI-05 chế độ phát lại qua SSE — *xong là khi:* điều khiển phát / tạm dừng / đặt lại và đồng hồ mô phỏng hoạt động.
- [x] **T-57** UI-04 hiệu năng mô hình — *xong là khi:* có đủ 7 mục ở [docs/07 §6](docs/07-thiet-ke-giao-dien.md). **Được phép cắt đầu tiên**, thay tạm bằng ảnh PNG từ notebook.

### Cách chạy

```powershell
docker compose up -d db
.\.venv\Scripts\python.exe -m uvicorn api.main:app --port 8000        # cửa sổ 1
.\.venv\Scripts\python.exe -m http.server 3000 --directory web        # cửa sổ 2 → http://localhost:3000
.\.venv\Scripts\python.exe -m pytest tests\test_threshold_parity.py   # TC-12, cần Node.js
```

Đủ lệnh, cách nạp dữ liệu demo, số đối chiếu và các bẫy đã gặp:
[docs/lenh-chay §8](docs/lenh-chay.md). Những điểm bản thi hành cụ thể hơn
thiết kế: [docs/07 §12](docs/07-thiet-ke-giao-dien.md).

### Ghi chú khi làm xong giai đoạn 8

**Kết quả theo điều kiện "xong là khi"** — đo bằng Puppeteer điều khiển Chrome 153 thật, trên API và
PostgreSQL thật:

| Việc | Điều kiện | Kết quả |
|---|---|---|
| T-51 | điều hướng được giữa các màn hình | đạt: thanh bên 4 mục, địa chỉ `#/…` (tải lại giữ màn hình), thanh trên luôn hiện ngưỡng, nguồn của ngưỡng và trạng thái API/CSDL; chân thanh bên hiện `model_version` và ngày huấn luyện |
| T-52 | sắp đúng theo điểm giảm dần; trạng thái rỗng hướng dẫn 3 cách nạp | đạt: 75 điểm thật trên 3 trang đầu giảm dần (AC-A2); trạng thái rỗng tách "chưa có dữ liệu" và "lọc hết" |
| T-53 | kéo thanh trượt dưới 200 ms; đổi chi phí dịch ngưỡng tối ưu | lâu nhất **20,4 ms** trên 21 lần kéo 0,5 → τ\* (AC-A3); chi phí bỏ lọt 600 EUR dời ngưỡng tối ưu 0,02317 → 0,002262 (AC-A5) |
| T-54 | JS và Python cùng TP/FP/FN trên 20 ngưỡng | đạt, sai lệch 0 — và precision, recall, F1, chi phí trùng từng bit; `@pytest.mark.skip` đã bỏ |
| T-55 | chú thích PCA cố định; nhãn thật chỉ hiện sau khi quyết định | đạt; thác nước SHAP 5 dương + 3 âm khép từ giá trị cơ sở tới f(x); `F`/`A`/`Esc` hoạt động, trọng tâm trả về đúng dòng |
| T-56 | phát / tạm dừng / đặt lại và đồng hồ mô phỏng | đạt: 600× trong 6 giây = 1 giờ mô phỏng, 766 giao dịch; tạm dừng thì đứng yên, tiếp tục không đếm trùng |
| T-57 | đủ 7 mục của 07 §6 | đạt cả 7, thêm đường PR và ROC của mô hình xuất (FR-42) |

Toàn bộ kiểm thử: **323 xanh, 0 bỏ qua** (318 khi mới có bản `web/`; thêm 5 ca TC-12 cho bản Next.js). Tải CSV 10.000 dòng qua giao diện: 2,2 giây (AC-A1).
Console trình duyệt không lỗi trên cả bốn màn hình.

**1. Sửa một lỗi của hiện vật, phát hiện qua UI-04.** Đường PR trong `metrics.json` vẽ ra thành một
đoạn thẳng: `src.evaluate.curve_points` lấy điểm đều theo chỉ số ngưỡng, nên cả vùng precision cao
chỉ còn 2/500 điểm. Nay rải điểm theo chiều dài đường cong (`thin_curve`), có ca kiểm thử canh, và
notebook 08 đã chạy lại: `model.joblib`, OOF, tập kiểm thử trùng từng byte, SHAP trùng từng bit; chỉ
khác `trained_at`, `fit_seconds` và ba khóa đường cong. Hình `04_pr_curves_*.png` của notebook 04
không bị ảnh hưởng — chúng vẽ từ đường đầy đủ.

**2. Thanh trượt theo thang log, không tuyến tính.** Với mô hình thật τ\* = 0,0232 nằm ở 2% đầu một
thanh tuyến tính, và bước 0,001 lớn gấp đôi ngưỡng "recall ≥ 90%" (0,00054). Thang log trải đều
vùng 0,0005…0,97; ô nhập số cho giá trị chính xác. Phím: `←`/`→` ±2,3%, `PgUp`/`PgDn` ±26%.

**3. Mọi chi phí quy ra EUR/ngày.** Đường cong (out-of-fold, 80% dữ liệu) và các ô số (tập kiểm thử,
20%) cùng một thang; ô chi phí vẫn ghi con số trên tập kiểm thử (2.389,78 EUR tại τ\*) để khớp báo cáo.

**4. Kịch bản bảo vệ phải nói đúng số của mô hình thật.** Kéo 0,5 → τ\* chỉ tăng số bắt được từ
**74/95 lên 77/95** và giảm chi phí từ 6.466 xuống 5.974 EUR/ngày — không phải "60 lên 83" như ví dụ ở
07 §5 (viết trước khi có mô hình). Điểm đáng nói là cái giá: cảnh báo tăng từ 195 lên 287/ngày.
Đổi chi phí bỏ lọt lên 600 EUR là đoạn trình diễn ấn tượng hơn: ngưỡng tối ưu lùi 10 lần.

**5. Không cần mạng.** Alpine.js và Chart.js chép vào `web/vendor/` kèm mã băm, không nạp từ CDN (NFR-08).

**6. Thêm bản Next.js trong `frontend/`** (theo yêu cầu, sau khi xong bản `web/`): Next.js 16 + TypeScript +
Tailwind 4 + Recharts 3, xuất tĩnh ra `frontend/out/`. Cùng bốn màn hình, cùng số; kịch bản thử tự động
21/21 đạt, AC-A3 lâu nhất 39 ms, `tsc` và ESLint sạch. TC-12 nay đối chiếu **cả hai** tệp UI-D1 với Python
(`web/threshold.js` và `frontend/src/lib/threshold.mjs`). Bản `web/` giữ làm dự phòng. So sánh hai bản:
[docs/07 §13](docs/07-thiet-ke-giao-dien.md); lệnh chạy: [docs/lenh-chay §8.6](docs/lenh-chay.md).

**Phát sinh thêm:** `web/config.js` (địa chỉ API, để giai đoạn 9 thay khi đóng gói), hộp thoại thư
viện mẫu, `src.evaluate.thin_curve`, 2 ca trong `tests/test_artifacts.py`. Tài liệu: docs/07 §12 mới,
docs/06 §3 (cách rút mẫu đường cong), docs/08 §2.7, docs/10 §4.2, `docs/lenh-chay.md` §8.

**Mang sang giai đoạn 9:**

- Chọn bản giao diện để đóng gói. Bản Next.js: `web/Dockerfile` (hoặc `frontend/Dockerfile`) build hai
  tầng — `node:20` chạy `npm ci && npm run build`, rồi nginx phục vụ `out/` ở cổng 3000; build trên Linux
  nên không cần bước làm phẳng tên tệp. Bản Alpine: chỉ cần phục vụ tĩnh thư mục `web/`.
  `web/config.js` mặc định gọi `http://<cùng máy>:8000/api/v1` — đúng khi Compose mở cổng 8000 ra
  máy chủ; nếu đặt proxy `/api` trong nginx thì chỉ sửa `apiBase` trong tệp này.
- `CORS_ORIGINS` của dịch vụ `api` phải chứa origin mà trình duyệt thấy (`http://localhost:3000`).
- AC-A9 (T-61): đã thử sơ bộ — Tab đi qua thanh bên, các nút, bộ lọc, dòng hàng đợi, thanh trượt;
  `Enter` mở chi tiết. Cần một lượt đầy đủ bằng tay và Lighthouse.
- Chạy lại notebook 08 thì phải khởi động lại API để nạp `metrics.json` mới.

---

## Giai đoạn 9 — Đóng gói và nghiệm thu (ngày 20) ✅

- [x] **T-58** `api/Dockerfile`, `web/Dockerfile`, compose 3 dịch vụ với `depends_on: service_healthy`, entrypoint chạy `alembic upgrade head` — *xong là khi:* một lệnh `docker compose up` dựng cả hệ thống.
- [x] **T-59** Thử trên máy sạch hoặc máy ảo, bấm giờ cả lần đầu và lần sau — *xong là khi:* cold dưới 45 giây, warm dưới 15 giây (NFR-04, AC-A8).
- [x] **T-60** Tạo bản `pg_dump` dữ liệu demo đẹp để dự phòng cho buổi bảo vệ — *xong là khi:* khôi phục thử được trong vài giây.
- [x] **T-61** Duyệt toàn bộ AC-A1…AC-A10 và kiểm tra bàn phím (AC-A9) — *xong là khi:* mọi tiêu chí bắt buộc đạt hoặc được ghi rõ lý do không đạt.
- [x] **T-62** Cập nhật README: cách chạy lại từ đầu trên máy sạch — *xong là khi:* người khác làm theo được mà không cần hỏi thêm (AC-D3).

### Cách chạy

```powershell
docker compose up --build -d                                         # lần đầu ~4 phút build, rồi ~15 giây khởi động
docker compose ps                                                    # api (healthy) → http://localhost:3000
.\.venv\Scripts\python.exe scripts\time_startup.py --runs 3          # T-59 warm; thêm --cold --yes để đo cold (XÓA volume)
.\.venv\Scripts\python.exe scripts\demo_db.py seed                   # T-60: dữ liệu demo, rồi dump / restore
.\.venv\Scripts\python.exe -m pytest tests\test_packaging.py         # 12 ca, không cần Docker
cd scripts\ui; npm install; node flow.js; node keyboard.js           # T-61: Chrome thật, 21 + 24 bước
```

Đủ lệnh, số đo và các bẫy đã gặp: [docs/lenh-chay §9](docs/lenh-chay.md).
Hướng dẫn cho máy sạch: [README](README.md#chạy-lại-từ-đầu-trên-máy-sạch).

### Ghi chú khi làm xong giai đoạn 9

**Kết quả theo điều kiện "xong là khi":**

| Việc | Điều kiện | Kết quả |
|---|---|---|
| T-58 | một lệnh `docker compose up` dựng cả hệ thống | đạt: `db` → `api` (`depends_on: service_healthy`, entrypoint chạy `alembic upgrade head` rồi uvicorn) → `web` (nginx). Build sạch 244 giây |
| T-59 | cold < 45 s, warm < 15 s | cold **13,7–14,9 s**, warm **11,1–11,5 s** trên bản sao sạch của repo, 3 lần mỗi loại — xem điểm 4 về giới hạn của "máy sạch" |
| T-60 | khôi phục thử trong vài giây | `backup/fraud-demo.dump` 3,0 MB (10.179 giao dịch, 8 kết luận); khôi phục **2,3–2,4 s**, cả sau khi xóa dữ liệu lẫn sau khi mất hẳn volume |
| T-61 | mọi tiêu chí bắt buộc đạt hoặc ghi rõ lý do | AC-A1…AC-A10 đều đạt trên hệ thống đóng gói; AC-A9 24/24 và Lighthouse Accessibility 100 — bảng ở [08 §4.4](docs/08-ke-hoach-kiem-thu.md) |
| T-62 | người khác làm theo được không cần hỏi | README có mục "Chạy lại từ đầu trên máy sạch": 4 bước, hai cách có hiện vật, bảng lỗi thường gặp. **Chưa** có người thứ hai làm theo — xem "Mang sang giai đoạn 10" |

Toàn bộ kiểm thử: **335 xanh** (323 + 12 ca của `tests/test_packaging.py`).

**1. Trình duyệt chỉ nói chuyện với một cổng.** nginx của dịch vụ `web` phục vụ tệp tĩnh và chuyển
tiếp `/api/` sang `api:8000`; cả hai bản giao diện được build với địa chỉ API tương đối `/api/v1`. Không
còn phụ thuộc CORS, mở bằng IP hay đổi cổng vẫn chạy. Luồng SSE đi qua với `proxy_buffering off`: từng
sự kiện đến ngay, 185 giây không lỗi (AC-A6). Lỗi do chính nginx sinh ra cũng theo mô hình lỗi của
API: `503 API_UNAVAILABLE` khi `api` dừng, `413 PAYLOAD_TOO_LARGE` khi tệp vượt 110 MB. Cổng 8000 vẫn mở
cho `/docs`.

**2. Bản giao diện đóng gói mặc định là Next.js** (`frontend/Dockerfile`, hai tầng node → nginx, ảnh 95 MB).
`WEB_UI=web` đổi sang bản Alpine.js — build không cần npm, dùng khi mạng chặn npm. Cả hai dùng chung
`deploy/nginx.conf`.

**3. Ảnh `api` ghim đúng phiên bản lúc xuất hiện vật** (`api/requirements.txt`, có ca kiểm thử canh lệch
với `metrics.json`). Hai phát hiện khi chạy trong container:

- Điểm chấm trên Linux lệch **1 ulp float32** ở 35/56.746 giao dịch so với điểm notebook tính trên
  Windows; 0 quyết định bị lật. "Trùng từng bit" của T-35 chỉ đúng trên cùng hệ điều hành
  ([10 §6](docs/10-van-hanh-tai-lap.md)).
- Ảnh `python` chính thức xóa `.pyc` của thư viện chuẩn, làm warm start lên 14,4 s. Biên dịch sẵn lúc
  build đưa về 11–12 s.

**4. "Máy sạch" là bản sao sạch trên chính máy phát triển**, không phải máy thứ hai: chỉ tệp git theo
dõi cộng 7 hiện vật, không `.venv`/`node_modules`/`.env`, dự án Compose riêng nên ảnh và volume mới. Không
loại được bộ đệm ảnh nền của Docker, và phải đặt `POSTGRES_HOST_PORT=5434` vì PostgreSQL 17 của máy chiếm
5432. AC-A8 đạt theo nghĩa đó; lượt trên máy thật khác vẫn nên làm.

**5. AC-A9 hiểu là "Tab, Enter và phím mũi tên trong nhóm".** Bảng hàng đợi là một điểm dừng Tab
(roving tabindex, `↓`/`↑` giữa các dòng — 07 §10), thanh trượt và nhóm radio cũng vậy, đúng mẫu
WAI-ARIA. Mọi thao tác khác — bốn màn hình, thư viện mẫu, ngăn kéo, ghi kết luận, phát lại — làm được
bằng Tab và Enter; mọi điểm dừng có viền trọng tâm; trọng tâm không thoát khỏi hộp thoại; đóng hộp thoại
trả trọng tâm về đúng chỗ.

**6. Đường để người chấm có hiện vật là điểm yếu nhất của AC-D3.** Năm tệp hiện vật không nằm trong git.
Tự tạo bằng notebook 03 → 08 mất khoảng 20 phút nhưng có thể dừng ở bước đối chiếu trên máy khác số lõi
(XGBoost phụ thuộc số luồng khi huấn luyện). README vì vậy để **gói hiện vật** (`hien-vat.tgz`, 19 MB) là
cách A.

**Phát sinh thêm:** `scripts/time_startup.py`, `scripts/demo_db.py`, `scripts/ui/` (kịch bản Puppeteer
của giai đoạn 8 đưa vào repo, thêm `keyboard.js`), `tests/test_packaging.py`, `deploy/nginx.conf`,
`.dockerignore`; `pyarrow` vào `requirements.txt`. Tài liệu: README, docs/03 §6.3 và §7, docs/05 §4,
docs/06 §7, docs/08 §2.8, §4.4, §5, docs/10 §1, §2.2, §4.1, §6, §7, §9, `docs/lenh-chay.md` §9.

**Mang sang giai đoạn 10:**

- Quyết định cách nộp hiện vật: kèm `hien-vat.tgz` (19 MB) theo bài, hoặc đưa năm tệp vào git. Không có
  thì người chấm phải tự chạy notebook (cách B của README).
- Nhờ một người khác làm theo README trên máy của họ (AC-D3, AC-A8 đúng nghĩa).
- Kịch bản demo (T-66): khởi động trước, `python scripts/demo_db.py restore`, dùng số thật của
  [lenh-chay §8.3](docs/lenh-chay.md) (74 → 77/95, 6.466 → 5.974 EUR/ngày); đoạn
  "Gian lận khó + bấm A" vẫn để dành vì `seed` không nạp nhóm đó.
- Báo cáo (T-63, T-64) nên nói đúng hai điều: AC-A4 là "5 và 3 khi mô hình có đủ" (giai đoạn 7, điểm 2);
  điểm trong container trùng notebook tới 1 ulp, không phải từng bit.

---

## Rà soát sau giai đoạn 9 (2026-10-09)

Một lượt rà soát toàn dự án, ngoài kế hoạch 67 việc. Mọi điểm dưới đây đã sửa, trừ báo cáo
(giai đoạn 10).

| # | Vấn đề — bằng chứng lúc rà soát | Đã làm | Kết quả kiểm |
|---|---|---|---|
| 1 | Giải thích Tomek ở giai đoạn 3 sai ("vùng biên quá thưa") | Đo lại trên fold 1: trước SMOTE có 21 cặp Tomek, sau SMOTE còn 0 — chính SMOTE xoá chúng. Sửa ở ghi chú giai đoạn 3, 04 §3.4, lenh-chay §3, notebook 04 §8 | — |
| 2 | Chi phí bỏ lọt cố định 122,21 trong khi 36% gian lận ≤ 1 EUR; 122,21 tính trên dữ liệu thô | `fn_costs` (chi phí theo từng giao dịch) trong `src/threshold.py`; notebook 06 §6.1 | Theo số tiền, τ\* **đắt hơn** 0,5 trên tập kiểm thử: +160 [+103; +217] EUR. Con số chỉ-train 115,94 cho cùng τ\*. τ mặc định giữ nguyên — quyết định nghiệp vụ |
| 3 | Luật `block` ≥ 3τ: dải review không có vụ gian lận nào, chặn tự động 19 khách hợp lệ (tập kiểm thử) | τ_chặn chọn theo precision ≥ 95% trên OOF (`min_precision`), 05 §2, notebook 06 §6.2 | τ_chặn = 0,9735; tập kiểm thử: chặn 69 gian lận + 1 hợp lệ, review có 8 gian lận |
| 4 | NFR-01 trượt 2/3 lần đo (p95 60,6 / 51,3 / 49,9 ms) | `api/serving.py` (`FastScorer`, đối chiếu từng bit với pipeline lúc nạp); `build_features` dựng bằng numpy; ca đo mang dấu `perf` | p95 18,7 / 24,1 / 21,0 ms |
| 5 | `requirements.txt` chỉ ghi `>=` — cách B của README dễ gãy | Ghim `==` đúng `metrics.json → environment`; 2 ca kiểm thử canh | — |
| 6 | USD và EUR lẫn lộn giữa tài liệu và giao diện | Quy ước EUR (02 §2), thay ở docs, TASKS, notebook | — |
| 7 | Giả định `Time = 0` là 00:00 không ghi ở đâu | 02 DS-03, docstring `hour_of_day`, 00 §3.3 | Sáu giờ vắng nhất là 1h–6h |
| 8 | Ba cổng mở ra mọi địa chỉ, mật khẩu DB viết cứng | `BIND_ADDRESS` (mặc định 127.0.0.1), `POSTGRES_PASSWORD`; client Python dùng 127.0.0.1 | Quét IP LAN: 3000, 8000, 5433 đều đóng |
| 9 | Phát lại đọc điểm tính sẵn, không chạy mô hình | Chấm theo mẻ khi giao dịch tới giờ mô phỏng | — |
| 10 | `app.py` trỏ `best_model.pkl` không tồn tại; README dán kèm trích dẫn lạ; log và tệp `_smoke` trong git; đặc tả cũ ở gốc; 37 tệp CRLF; LightGBM không dùng | Viết lại `app.py` (AppTest chạy sạch); viết lại đầu README; `git rm --cached`; `docs/luu-tru/`; `.gitattributes`; bỏ LightGBM | — |
| 11 | Ảnh `api` 1,46 GB | SHAP bằng `pred_contribs` của XGBoost (trùng từng bit `explainer.joblib`), bỏ shap khỏi ảnh | 1,15 GB |
| 12 | Không có phân tích calibration | Notebook 06 §6.3 | Brier tốt hơn 4 lần; lệch ở hai đầu |

**Không làm:** giữ cả hai bản giao diện — `web/` là đường dự phòng khi mạng chặn npm (README, "Khi gặp
lỗi"); bỏ nó ngay trước buổi bảo vệ là tăng rủi ro.

**Mang sang giai đoạn 10 (bổ sung):**

- Báo cáo phải nêu: τ\* chỉ tối ưu dưới giả định chi phí cố định — tính theo số tiền thì khoản tiết
  kiệm đổi dấu (notebook 06 §6.1); quy ước EUR và giả định `Time = 0`; điểm rủi ro không phải xác suất
  ở hai đầu (06 §6.3).
- Kịch bản demo T-66: đoạn "kéo 0,5 → τ\*, chi phí giảm" chỉ đúng với chi phí cố định — nói rõ giả
  định khi trình diễn, hoặc trình diễn việc đổi chi phí bỏ lọt (AC-A5).
- Cách nộp hiện vật vẫn chưa chốt.
- Commit lần tới chạy `git add --renormalize .` để 37 tệp CRLF chuyển sang LF theo `.gitattributes`.

---

## Giai đoạn 10 — Báo cáo và bảo vệ (ngày 21)

- [x] **T-63** Viết `reports/bao-cao.md` theo dàn ý [docs/09 §5](docs/09-ke-hoach-trien-khai.md) — *xong là khi:* phần 6 (so sánh chiến lược) và phần 8 (ngưỡng và chi phí) là hai phần dài nhất.
- [x] **T-64** Viết phần hạn chế — *xong là khi:* nêu đủ ba điều: V1–V28 là PCA nên không diễn giải được, dữ liệu chỉ hai ngày tháng 9/2013, và 492 mẫu dương làm khoảng tin cậy rất rộng (AC-D2).
- [x] **T-65** Ghi nguồn dữ liệu và giấy phép DbCL v1.0 — *xong là khi:* có trích dẫn Kaggle mlg-ulb (AC-D5).
- [ ] **T-66** Chuẩn bị kịch bản demo 10 phút — *xong là khi:* đã diễn thử một lần trọn vẹn, gồm đoạn kéo thanh trượt ngưỡng từ 0,5 về τ\* để hội đồng thấy số vụ bắt được nhảy lên trong khi chi phí giảm.
- [ ] **T-67** Ôn ba câu hỏi chắc chắn bị hỏi — *xong là khi:* trả lời được trôi chảy không cần nhìn tài liệu:
  1. Vì sao PR-AUC chứ không phải ROC-AUC?
  2. SMOTE hoạt động thế nào và vì sao nó phải nằm trong pipeline?
  3. Ngưỡng này được chọn ra sao, sẽ đổi thế nào nếu chi phí thay đổi?


### Ghi chú khi viết báo cáo (2026-10-09)

`reports/bao-cao.md` theo dàn ý docs/09 §5, 12 mục + tài liệu tham khảo + phụ lục, 22 hình. Phần 8 (ngưỡng
và chi phí, ~14.500 ký tự) và phần 6 (so sánh chiến lược, ~10.200) là hai phần dài nhất — điều kiện của
T-63. T-64: §11.1 nêu đủ ba hạn chế bắt buộc (PCA, hai ngày 9/2013, 492 mẫu dương với độ rộng khoảng tin
cậy cụ thể). T-65: §1.2 và tài liệu tham khảo [1] (Kaggle mlg-ulb, Dal Pozzolo và cộng sự 2015, DbCL v1.0).

Còn phải tự điền: họ tên, MSSV, giảng viên hướng dẫn ở đầu báo cáo. Ảnh giao diện `reports/figures/10_ui_*.png`
chép từ `scripts/ui/shots/` (chụp 2026-10-02; các dòng trong ảnh đều có điểm > 99,9% nên vẫn đúng với luật
chặn mới). Cần bản PDF thì xuất từ Markdown (VS Code: Markdown PDF, hoặc `pandoc bao-cao.md -o bao-cao.pdf`).
---

## Thứ tự cắt giảm khi thiếu thời gian

Cắt từ dưới lên. Không bao giờ cắt những việc in đậm.

| Ưu tiên | Hạng mục | Cắt được? |
|---|---|---|
| 1 | **T-28 rà soát rò rỉ** | Không — mất cái này thì toàn bộ kết quả vô giá trị |
| 2 | **T-20 bảng 20 tổ hợp** | Không — đây là trọng tâm đề tài |
| 3 | **T-29…T-31 ngưỡng và chi phí** | Không — đây là luận điểm chính |
| 4 | **T-38 kiểm tra tái lập** | Không — AC-M5 |
| 5 | **T-63…T-65 báo cáo** | Không — ngày 21 bất khả xâm phạm |
| 6 | T-53 UI-03 ngưỡng | Giữ bằng mọi giá nếu có làm ứng dụng |
| 7 | T-52 UI-01 hàng đợi | Giữ |
| 8 | T-55 UI-02 chi tiết | Cắt được, thay bằng ảnh notebook |
| 9 | T-56 UI-05 phát lại | Cắt được |
| 10 | T-57 UI-04 hiệu năng | Cắt trước nhất |
| 11 | T-34 autoencoder | Tùy chọn từ đầu |
| 12 | T-27 chia theo thời gian | Nên có, không bắt buộc |

## Việc lặp lại hằng ngày

- [ ] Chạy `pytest` trước mỗi lần commit.
- [ ] Cập nhật `[ ]` → `[x]` trong tệp này ngay khi đạt điều kiện *xong là khi*.
- [ ] Ghi lại mọi quyết định lệch khỏi đặc tả vào tài liệu tương ứng trong `docs/`, không để trong đầu.
