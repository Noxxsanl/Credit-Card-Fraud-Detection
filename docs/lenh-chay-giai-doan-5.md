# Lệnh chạy giai đoạn 5 — Ngưỡng, chi phí, SHAP (T-29…T-34)

Sổ lệnh để **chạy lại** giai đoạn 5 từ đầu. Mọi lệnh chạy từ thư mục gốc của repo, gọi thẳng
trình thông dịch trong `.venv`. Phần **vì sao** nằm ở [04 §6.6, §7.1, §8.1](04-thiet-ke-mo-hinh-ml.md);
điều kiện nghiệm thu nằm ở [TASKS.md](../TASKS.md).

> **Tiền đề:** `data/creditcard.csv` đúng toàn vẹn, và nên có `reports/grid_results.npz` (điểm
> out-of-fold của giai đoạn 3, không nằm trong git). Thiếu tệp này thì notebook 06 và 07 tự tính
> lại điểm out-of-fold, mất thêm khoảng 2 phút mỗi notebook.

## 1. Trình tự

Ba notebook độc lập với nhau, chạy theo thứ tự nào cũng được. Không có script chạy ngoài: phần
nặng nhất chỉ mất vài phút.

### 1.1 PowerShell

```powershell
# 1. Kiểm môi trường
.\.venv\Scripts\python.exe -m pytest

# 2. Ba notebook trong kernel sạch
foreach ($nb in "06_threshold_and_cost", "07_explainability", "06b_autoencoder") {
    .\.venv\Scripts\python.exe -m jupyter nbconvert --to notebook --execute --inplace `
        --ExecutePreprocessor.timeout=3600 "notebooks\$nb.ipynb"
}
```

### 1.2 Git Bash

```bash
./.venv/Scripts/python.exe -m pytest
for nb in 06_threshold_and_cost 07_explainability 06b_autoencoder; do
    ./.venv/Scripts/python.exe -m jupyter nbconvert --to notebook --execute --inplace \
        --ExecutePreprocessor.timeout=3600 "notebooks/$nb.ipynb"
done
```

| Notebook | Việc | Thời gian (12 lõi) |
|---|---|---|
| `06_threshold_and_cost` | T-29, T-30, T-31 | khoảng 3,5 phút (phần lớn là bootstrap) |
| `07_explainability` | T-32, T-33 | khoảng 2 phút (SHAP trên toàn tập test: 35 giây) |
| `06b_autoencoder` | T-34 (tùy chọn) | khoảng 4 phút (autoencoder: 2,5 phút) |

## 2. Hiện vật sinh ra

| Tệp | Nguồn | Nội dung |
|---|---|---|
| `reports/threshold_comparison.csv` | 06 | bảng bắt buộc 04 §6.4: 5 phương án × chỉ số trên test, kèm KTC và số OOF |
| `reports/cost_sensitivity.csv` | 06 | 11 tỷ lệ chi phí 5:1 → 100:1: τ\*, chỉ số OOF và test, độ hối tiếc |
| `reports/shap_ranking.csv` | 07 | 31 đặc trưng: `mean|SHAP|`, các hạng thống kê của T-15, hệ số hồi quy logistic |
| `reports/error_analysis.csv` | 07 | 8 FN + 8 FP điểm cao nhất, top SHAP đẩy lên / kéo xuống |
| `reports/autoencoder_comparison.csv` | 06b | PR-AUC, ROC-AUC của autoencoder và XGBoost |
| `reports/figures/06_cost_curve.png` | 06 | đường cong chi phí OOF + đánh đổi recall / precision / cảnh báo theo τ |
| `reports/figures/06_threshold_options.png` | 06 | TP / FP / FN và chi phí trên test theo phương án |
| `reports/figures/06_cost_sensitivity.png` | 06 | 4 ô phân tích độ nhạy (AC-M8) |
| `reports/figures/07_shap_summary.png` | 07 | `summary_plot` trên 2.000 mẫu |
| `reports/figures/07_shap_vs_statistics.png` | 07 | mean\|SHAP\| và đối chiếu với T-15 |
| `reports/figures/07_error_shap.png` | 07 | heatmap SHAP của FN/FP + SHAP trung bình theo nhóm |
| `reports/figures/07_waterfall_fn_fp.png` | 07 | waterfall của FN và FP điểm cao nhất |
| `reports/figures/06b_autoencoder.png` | 06b | phân bố sai số tái tạo, đường PR, so thứ hạng hai mô hình |

Không notebook nào ghi vào `models/`. Việc đó là của notebook 08 (T-35, T-36).

## 3. Số đối chiếu (lần chạy 2026-09-25, 12 lõi)

| Mốc | Giá trị |
|---|---|
| τ\* chọn trên out-of-fold (dò trên mọi điểm) | **0,02317** (lưới 200 điểm: 0,02321) |
| Test tại τ\* | TP 77, FP 38, FN 18, chi phí 2.390 USD |
| Test tại 0,5 | TP 74, FP 4, FN 21, chi phí 2.586 USD |
| Hiệu chi phí τ\* − 0,5 | test −197 [−676, +160]; OOF −707 [−1.513, −24] |
| τ\* không đổi trên dải tỷ lệ | 20:1 → 50:1 |
| SHAP top 3 | V14, V4, V12 |
| Sai số cộng của SHAP | 2,2·10⁻⁵ |
| Autoencoder PR-AUC | 0,329 [0,246 – 0,438] |

## 4. Kiểm thử của giai đoạn này

```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_threshold.py   # thêm: ngưỡng chính xác, độ nhạy đơn điệu, sample_fraction
.\.venv\Scripts\python.exe -m pytest tests\test_search.py      # thêm: bootstrap_threshold_diff
.\.venv\Scripts\python.exe -m pytest tests\test_anomaly.py     # autoencoder: tái lập, chuẩn hoá chỉ từ tập train
.\.venv\Scripts\python.exe -m pytest tests\test_no_leakage.py  # giờ quét cả notebook 06, 06b, 07
```

`test_leak_5_…` đỏ ngay nếu notebook gọi `pick_threshold`, `threshold_alternatives` hay
`sensitivity_analysis` với đối số đầu tiên là nhãn test.

## 5. Những cái bẫy đã gặp

**Lưới ngưỡng 200 điểm quá thô cho tiêu chí có ràng buộc.** Xem 04 §6.6. Luôn truyền
`thresholds=np.unique(oof)` khi cần con số cho báo cáo.

**`sensitivity_analysis()` trước đây không nhận `sample_fraction`.** Khi chạy trên out-of-fold
(80% dữ liệu), cột `alerts_per_day` bị quy đổi theo 20%, tức phóng đại 4 lần. Đã sửa, và có kiểm
thử canh.

**SHAP cần đầu vào sau tiền xử lý.** `TreeExplainer` bọc bước `clf`, không bọc cả pipeline. Phải
đưa vào `pipeline[:-1].transform(X)`, trong đó `Amount` đã qua `RobustScaler`. Đưa `X` thô vào
thì SHAP của `Amount` sai mà không báo lỗi. Ô kiểm tra tính cộng ở notebook 07 bắt được lỗi này.
