# Lệnh chạy giai đoạn 6 — Xuất hiện vật (T-35…T-38)

Sổ lệnh để **chạy lại** giai đoạn 6 từ đầu: xuất mô hình, ngưỡng, explainer, chỉ số và thư viện
mẫu, rồi kiểm tra tái lập. Mọi lệnh chạy từ thư mục gốc của repo, gọi thẳng trình thông dịch trong
`.venv`. Cấu trúc từng tệp nằm ở [06 §6](06-thiet-ke-luu-tru.md); điều kiện nghiệm thu nằm ở
[TASKS.md](../TASKS.md).

> **Tiền đề:**
>
> - `data/creditcard.csv` đúng toàn vẹn, và `data/test_set.parquet` do notebook 03 ghi ra.
> - Các bảng `reports/*.csv` của notebook 04–07 (có trong git). Notebook 08 đối chiếu với chúng và
>   **dừng lại** nếu lệch.
> - Nên có `reports/grid_results.npz` (không nằm trong git). Thiếu thì notebook 08 vẫn chạy, nhưng
>   bỏ qua phần so điểm out-of-fold với giai đoạn 3 và ghi `strategy_pr_curves: null`.

## 1. Trình tự

### 1.1 PowerShell

```powershell
# 1. Kiểm môi trường
.\.venv\Scripts\python.exe -m pytest

# 2. Notebook 08 trong kernel sạch (5–10 phút)
.\.venv\Scripts\python.exe -m jupyter nbconvert --to notebook --execute --inplace `
    --ExecutePreprocessor.timeout=3600 notebooks\08_export_artifacts.ipynb

# 3. Kiểm chính các tệp vừa sinh — 7 ca tích hợp hết bị bỏ qua
.\.venv\Scripts\python.exe -m pytest tests\test_artifacts.py

# 4. T-38: chạy lại 01 → 08 trong kernel sạch rồi so số (~70 phút)
.\.venv\Scripts\python.exe scripts\check_reproducibility.py
```

### 1.2 Git Bash

```bash
./.venv/Scripts/python.exe -m pytest
./.venv/Scripts/python.exe -m jupyter nbconvert --to notebook --execute --inplace \
    --ExecutePreprocessor.timeout=3600 notebooks/08_export_artifacts.ipynb
./.venv/Scripts/python.exe -m pytest tests/test_artifacts.py
./.venv/Scripts/python.exe scripts/check_reproducibility.py
```

| Bước | Thời gian (12 lõi) | Phần nặng |
|---|---|---|
| notebook 08 | 5–10 phút tùy tải máy | điểm out-of-fold 1,5–4 phút, bootstrap 1,5–2 phút, SHAP toàn tập kiểm thử 30–50 giây |
| `check_reproducibility.py` | khoảng 70 phút | notebook 05 (33 phút: bootstrap, ba cách chia), 08 và 06b |
| `check_reproducibility.py --only 08` | 5–10 phút | kiểm nhanh sau khi sửa notebook 08 |

Hai lần chạy notebook 08 trong cùng ngày mất 4,6 và 9,4 phút: máy chậm hơn ở lần sau (huấn luyện
16 giây so với 55 giây), nhưng mọi con số trùng nhau.

**Notebook 08 phải chạy sau notebook 03.** Chạy lại 03 là ghi đè `test_set.parquet` và mất cột
`risk_score` mà 08 thêm vào.

## 2. Hiện vật sinh ra

| Tệp | Kích thước | Nội dung | Trong git? |
|---|---|---|---|
| `models/model.joblib` | 1,2 MB | pipeline hoàn chỉnh: `RobustScaler` cho `Amount` → `XGBClassifier` | Không |
| `models/explainer.joblib` | 4,2 MB | `shap.TreeExplainer` của bước `clf` | Không |
| `models/threshold.json` | < 1 KB | τ\* = 0,023173 và 4 phương án, chọn trên out-of-fold | Nên có |
| `models/metrics.json` | 1,6 MB | 12 khóa bắt buộc + 6 khóa bổ sung cho UI-04 | Có cân nhắc |
| `models/oof_scores.npz` | 0,8 MB | điểm out-of-fold cho `/threshold/optimize` — thêm ở giai đoạn 7 | Không (`.gitignore`) |
| `data/sample_pool.json` | 0,2 MB | 200 mẫu, bốn nhóm | Không (`.gitignore`) |
| `data/test_set.parquet` | 15,9 MB | thêm cột `risk_score` | Không (`.gitignore`) |
| `reports/figures/08_sample_pool.png` | | điểm rủi ro của 200 mẫu theo nhóm | Có |
| `reports/reproducibility.csv` | | kết quả T-38, từng dòng | Có |

## 3. Số đối chiếu (lần chạy 2026-09-27, 12 lõi)

Notebook 08 tính lại mọi con số từ đầu rồi so với bảng mà notebook 03–07 đã ghi ở những phiên chạy
khác. Ô nào lệch quá dung sai thì notebook dừng ở `assert`.

| Đối chiếu | Lệch lớn nhất | Dung sai |
|---|---|---|
| điểm out-of-fold với giai đoạn 3 (`grid_results.npz`) | 0 | 0 |
| 5 ngưỡng với notebook 06 | 6·10⁻¹⁷ | 10⁻¹² |
| chỉ số chính và khoảng tin cậy với notebook 05 | 1·10⁻¹⁶ | 10⁻¹² |
| PR-AUC với `split_comparison.csv` | 0 | 10⁻¹² |
| TP/FP/FN/chi phí của 5 phương án với notebook 06 | 5·10⁻¹³ | 10⁻⁹ |
| `mean|SHAP|` với notebook 07 (tương đối) | 4·10⁻⁸ | 10⁻⁴ |
| tính cộng của SHAP (`base + ΣSHAP − margin`) | 2,2·10⁻⁵ | 10⁻³ |

| Mốc | Giá trị |
|---|---|
| `model_version` | `xgb_scaleposweight_v1` |
| PR-AUC tập kiểm thử | 0,8252 [0,7470 – 0,8960] |
| τ\* | 0,023173 — TP 77, FP 38, FN 18, TN 56.613 |
| Giải thích một giao dịch | trung vị 17–21 ms, chậm nhất 39–71 ms (hai lần chạy) |
| Thư viện mẫu | `fraud_easy` 29, `fraud_hard` 21, `legit_easy` 112, `legit_hard` 38 |
| Dấu vân tay điểm test / OOF | `32e5d55eac11a289` / `0b7b0e99ca7b1c52` |

Chạy lại trên cùng máy thì hai dấu vân tay phải giữ nguyên. Khác thì mô hình đã cho điểm khác, dù
PR-AUC có thể vẫn nằm trong dung sai.

## 4. Kiểm tra tái lập — T-38

Chạy `scripts/check_reproducibility.py` ngày 2026-09-27, máy 12 lõi. Ảnh chụp TRƯỚC là hiện vật
của lần chạy notebook 08 đầu tiên; sau đó chạy lại cả 9 notebook, mỗi notebook một kernel mới.

| Notebook | Thời gian |
|---|---|
| 01_eda | 1,5 phút |
| 02_statistics | 1,1 phút |
| 03_baseline | 3,5 phút |
| 04_imbalance_strategies | 7,7 phút |
| 05_advanced_models | 33,1 phút |
| 06_threshold_and_cost | 3,9 phút |
| 06b_autoencoder | 7,8 phút |
| 07_explainability | 3,2 phút |
| 08_export_artifacts | 9,4 phút |
| **Tổng** | **71,2 phút** |

| So trước và sau | Kết quả |
|---|---|
| PR-AUC tập kiểm thử | 0,8252465892 → 0,8252465892, **lệch 0** (dung sai 0,001) — **T-38 / AC-M5 đạt** |
| 12 con số chính và τ\* | lệch 0 |
| Dấu vân tay `test_scores_sha256`, `oof_scores_sha256` | trùng |
| 11 bảng `reports/*.csv` | trùng hoàn toàn (bỏ qua 4 cột đo thời gian ở 3 bảng) |
| Hình `reports/figures/*.png` | không đổi byte nào (`git status` không báo sửa) |

Chi tiết từng dòng ở `reports/reproducibility.csv`. `git diff` sau lần chạy này chỉ còn đầu ra của
notebook (thời gian, số thứ tự thực thi) và cột `giây huấn luyện` của `baseline_results.csv`.

**Giới hạn.** Script không chạy lại `run_grid.py` và `run_search.py`; notebook 08 bù bằng cách tính
lại điểm out-of-fold của mô hình được chọn (trùng từng bit với điểm lưu). Chỉ kiểm trên một máy —
sang máy có số lõi khác thì dấu vân tay có thể đổi dù PR-AUC vẫn trong dung sai ([10 §6](10-van-hanh-tai-lap.md)).

## 5. Kiểm thử của giai đoạn này

```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_artifacts.py     # 23 ca đơn vị + 7 ca tích hợp
.\.venv\Scripts\python.exe -m pytest tests\test_no_leakage.py    # giờ quét cả notebook 08
```

`tests/test_artifacts.py` có hai phần. Phần **đơn vị** kiểm hàm dựng và hàm kiểm tra trong
`src/artifacts.py` trên dữ liệu tổng hợp: JSON chặt, giữ nguyên float64, luật chia bốn nhóm, tính
tất định, dấu phẩy thập phân trong mô tả. Phần **tích hợp** đọc chính các tệp trong `models/` và
`data/`, và tự bỏ qua trên máy chưa chạy notebook 08. Đây là bản tự động của các điều kiện "xong là
khi" ở T-35, T-36, T-37:

| Ca | Kiểm |
|---|---|
| `test_t35_model_is_a_complete_pipeline` | hai bước `preprocess` → `clf`, `RobustScaler` đã fit, `feature_names_in_` đúng `FEATURE_ORDER` |
| `test_t35_reloaded_model_reproduces_exported_scores` | chấm lại `test_set.parquet` trùng từng bit với `risk_score` và `test_scores` |
| `test_t36_json_artifacts_follow_the_documented_structure` | `validate_threshold`, `validate_metrics`; ba tệp cùng `model_version` |
| `test_t36_headline_matches_exported_scores` | PR-AUC và ma trận nhầm lẫn tính lại từ `test_scores`; AC-M1, AC-M2 |
| `test_t36_explainer_is_additive_on_the_exported_model` | `base + ΣSHAP` = margin trên 95 vụ gian lận + 200 giao dịch |
| `test_t37_sample_pool_meets_the_task_conditions` | `validate_sample_pool`: mỗi nhóm hard ≥ 20 |
| `test_t37_sample_pool_scores_are_what_the_model_returns` | chấm lại 30 cột thô của 200 mẫu ra đúng `risk_score` |

## 6. Những cái bẫy đã gặp

**Bootstrap phụ thuộc thứ tự dòng.** `test_set.parquet` sắp theo `Time`, còn notebook 05 tính khoảng
tin cậy trên thứ tự của `split_data()`. Cùng hạt giống nhưng khác thứ tự thì bootstrap lấy ra những
dòng khác, và khoảng tin cậy lệch ở chữ số thứ ba. Notebook 08 vì thế giữ **hai** thứ tự: chỉ số và
khoảng tin cậy tính trên thứ tự của `split_data()` (trùng báo cáo tới 10⁻¹⁶), còn `test_scores`,
`risk_score` và thư viện mẫu theo thứ tự của tệp.

**Kiểm tính cộng của SHAP bằng margin, không bằng logit của xác suất.** `predict_proba` trả float32.
Với điểm sát 1, `log(p / (1 − p))` mất gần hết độ chính xác và lệch hàng chục phần trăm, nên phép so
đỏ oan. So với `clf.predict(Z, output_margin=True)` thì khớp tới 2·10⁻⁵.

**`fraud_hard` chỉ có 18 mẫu nếu lấy mốc τ\*.** Tập kiểm thử chỉ có 95 vụ gian lận, và mô hình bỏ lọt
18 vụ ở τ\*. Lấy mốc 0,5 cho gian lận ("mô hình không tự tin") thì được 21. Biên vẫn mỏng, nên
`validate_sample_pool` chặn việc xuất nếu nhóm tụt dưới 20.

**`Path.write_text` trên Windows đổi `\n` thành `\r\n`.** Tệp JSON sinh ra trên Windows và Linux sẽ
khác nhau dù cùng nội dung. `write_json` ghi byte để tránh việc này.

**`DataFrame.iterrows()` gộp cả dòng về float64.** Số dòng và TP/FP/FN của `split_comparison` ra
`226980.0`, `77.0`. Notebook ép các cột đếm về số nguyên trước khi ghi.

**Chạy lại notebook 03 là mất `risk_score`.** Xem §1.
