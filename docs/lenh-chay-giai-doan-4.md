# Lệnh chạy giai đoạn 4 — Tinh chỉnh và kiểm chứng (T-24…T-28)

Sổ lệnh để **chạy lại** giai đoạn 4 từ đầu: tìm kiếm siêu tham số, đánh giá trên tập kiểm
thử, chia theo thời gian và rà soát rò rỉ. Mọi lệnh chạy từ thư mục gốc của repo và gọi
thẳng trình thông dịch trong `.venv` (không cần `activate`). Mục 1 có cả bản PowerShell
lẫn bản Git Bash; các mục sau viết theo PowerShell, quy đổi sang Git Bash theo bảng ở
[lệnh chạy giai đoạn 3 §1.3](lenh-chay-giai-doan-3.md).

Tài liệu này chỉ ghi *lệnh gì, ra cái gì, mất bao lâu*. Phần **vì sao** nằm ở
[04 §4.1](04-thiet-ke-mo-hinh-ml.md); điều kiện nghiệm thu từng việc nằm ở
[TASKS.md](../TASKS.md).

> **Tiền đề — hai tệp phải có sẵn:**
>
> - `data/creditcard.csv` đúng toàn vẹn (T-10). Máy sạch thì làm [10 §2](10-van-hanh-tai-lap.md) trước.
> - `reports/grid_results.npz` — điểm out-of-fold của giai đoạn 3. Tệp này **không nằm trong
>   git**; notebook 05 đọc nó để lấy điểm của cấu hình mặc định. Thiếu thì chạy
>   `scripts\run_grid.py` trước (~45 phút, xem [giai đoạn 3](lenh-chay-giai-doan-3.md)).

## 1. Trình tự đầy đủ

Năm lệnh, theo đúng thứ tự này.

### 1.1 PowerShell

```powershell
# 0. Có điểm out-of-fold của giai đoạn 3 chưa? Phải ra True
Test-Path reports\grid_results.npz

# 1. Kiểm môi trường — phải xanh trước khi bỏ ra 40 phút
.\.venv\Scripts\python.exe -m pytest -q

# 2. Chạy thử 12.000 dòng, 4 lần thử (~1 phút) — bắt lỗi cấu hình trước khi chạy thật
.\.venv\Scripts\python.exe scripts\run_search.py --smoke

# 3. Chạy thật: 30 lần thử × 5 fold = 150 lần huấn luyện (~20 phút), ghi cả log
.\.venv\Scripts\python.exe scripts\run_search.py 2>&1 | Tee-Object search_run.log

# 4. Chạy notebook 05 trong kernel sạch (~20 phút), không cần mở Jupyter
.\.venv\Scripts\python.exe -m jupyter nbconvert --to notebook --execute --inplace `
    --ExecutePreprocessor.timeout=3600 notebooks\05_advanced_models.ipynb
```

### 1.2 Git Bash

```bash
# 0. Có điểm out-of-fold của giai đoạn 3 chưa?
ls reports/grid_results.npz

# 1. Kiểm môi trường
./.venv/Scripts/python.exe -m pytest -q

# 2. Chạy thử
./.venv/Scripts/python.exe scripts/run_search.py --smoke

# 3. Chạy thật, ghi cả log
./.venv/Scripts/python.exe scripts/run_search.py 2>&1 | tee search_run.log

# 4. Chạy notebook 05 trong kernel sạch
./.venv/Scripts/python.exe -m jupyter nbconvert --to notebook --execute --inplace \
    --ExecutePreprocessor.timeout=3600 notebooks/05_advanced_models.ipynb
```

Bước 4 có thể thay bằng mở `notebooks/05_advanced_models.ipynb` trong VS Code / Jupyter rồi
**Run All**. Notebook thấy `reports/search_results.csv` nên **bỏ qua** bước tìm kiếm; nếu tệp
đó chưa có, nó tự chạy tìm kiếm ngay trong notebook (thêm ~20 phút).

`--ExecutePreprocessor.timeout=3600` là bắt buộc: mặc định `nbconvert` cắt mỗi ô sau 600 giây,
mà ô chia theo thời gian (§5 của notebook) chạy khoảng 9 phút trên 12 lõi — máy chậm hơn sẽ
vượt ngưỡng đó.

## 2. Bảng lệnh của `run_search.py`

Toàn bộ cờ của [scripts/run_search.py](../scripts/run_search.py):

| Lệnh | Tác dụng | Thời gian |
|---|---|---|
| `run_search.py` | 30 lần thử × 5 fold trên toàn tập huấn luyện | ~20 phút (12 lõi) |
| `run_search.py --smoke` | 12.000 dòng, 4 lần thử, ghi ra tệp `_smoke` riêng | ~1 phút |
| `run_search.py --status` | Chỉ in 5 dòng đầu của kết quả đã lưu rồi thoát | tức thì |

**Không có điểm lưu.** Khác `run_grid.py`, tìm kiếm là một lời gọi `RandomizedSearchCV` duy
nhất: bấm `Ctrl+C` giữa chừng là mất trắng, chạy lại thì bắt đầu từ đầu. Kết quả chỉ được ghi
khi xong cả 150 lần huấn luyện. Trong lúc chạy, log chỉ có một dòng
`Fitting 5 folds for each of 30 candidates, totalling 150 fits` — im lặng 20 phút là bình
thường, không phải treo.

Muốn chắc tiến trình còn sống, mở cửa sổ khác:

```powershell
Get-Process python* | Select-Object Id, CPU, StartTime
```

Cột `CPU` phải tăng đều giữa hai lần gọi.

**Song song hoá.** Tìm kiếm chạy `n_jobs=-1` ở tầng ngoài, XGBoost bên trong chạy một luồng —
150 lần huấn luyện chia đều cho mọi lõi. Máy 4 lõi sẽ mất khoảng gấp ba thời gian trên.

## 3. Hiện vật sinh ra

| Tệp | Nguồn | Ghi vào git? |
|---|---|---|
| `reports/search_results.csv` | bước 3 | Có — 30 dòng, bằng chứng của T-24 |
| `reports/best_params.json` | bước 3 | Có — tham số của dòng đầu bảng, thời gian chạy |
| `reports/search_results_smoke.csv`, `best_params_smoke.json` | bước 2 | Không cần, chỉ là kết quả chạy thử |
| `reports/final_test_metrics.csv` | bước 4 | Có — chỉ số trên tập test kèm khoảng tin cậy (T-26) |
| `reports/split_comparison.csv` | bước 4 | Có — ba cách chia tập (T-27) |
| `reports/figures/05_search_trials.png` | bước 4 | Có — PR-AUC của 30 lần thử theo từng tham số |
| `reports/figures/05_test_pr_and_confusion.png` | bước 4 | Có — đường PR và ma trận nhầm lẫn trên tập test |
| `reports/figures/05_bootstrap_ci.png` | bước 4 | Có — chỉ số chính kèm khoảng tin cậy |
| `reports/figures/05_temporal_split.png` | bước 4 | Có — đường PR theo cách chia, tỷ lệ gian lận theo giờ hai ngày |
| `search_run.log` | bước 3 | Không — nhật ký một lần chạy |

Notebook 05 **không** ghi vào `models/`. Hiện vật mô hình do notebook 08 sinh ra (T-35).

## 4. Kết quả của lần chạy thật (2026-09-23)

Để đối chiếu khi chạy lại — cùng `RANDOM_STATE=42` **và cùng số lõi CPU** thì các con số
dưới đây phải lặp lại tới chữ số cuối (xem §6):

| Mốc | Giá trị |
|---|---|
| Tập huấn luyện / kiểm thử | 226.980 dòng, 378 gian lận / 56.746 dòng, 95 gian lận |
| Máy chạy | 12 lõi logic |
| Thời gian tìm kiếm | **19,8 phút** |
| PR-AUC CV cao nhất | 0,8538 ± 0,0328 — **thấp hơn** cấu hình mặc định 0,8549 |
| Hiệu tinh chỉnh − mặc định trên OOF | −0,0006 [−0,0087, +0,0072] → **giữ cấu hình mặc định** |
| τ\* chọn trên out-of-fold | 0,0232 |
| PR-AUC trên tập test | **0,825** [0,747 – 0,896] — AC-M1 đạt |
| Recall@τ\* | **0,811** [0,737 – 0,884] — 77/95, AC-M2 đạt |
| Chia theo thời gian (ngày 1 → ngày 2) | PR-AUC 0,782, precision@τ\* 0,332 |
| Thời gian chạy notebook 05 | khoảng 21 phút |

Nếu dòng "Cấu hình được chọn" ở §3.1 của notebook ra **TINH CHỈNH** thay vì **MẶC ĐỊNH** thì
lần chạy đó đã khác lần này — kiểm tra lại số lõi và phiên bản `xgboost`.

### Điều kiện dừng

Notebook có `assert` ở những chỗ sau; ô nào đỏ thì **dừng lại**, không tick việc tương ứng:

| Assert | Nghĩa là |
|---|---|
| `tập kiểm thử phải trùng với notebook 03 (95 gian lận)` | cách chia đã đổi — mọi con số không còn so được với giai đoạn 2–3 |
| `cần 30 lần thử` | `search_results.csv` là bản chạy thử hoặc bị cắt |
| `PR-AUC > 0,95 — dừng lại rà rò rỉ` | gần như chắc chắn rò rỉ, xem [08 §3.1](08-ke-hoach-kiem-thu.md) |
| `T-25 chưa đạt — không tick` | AC-M1 hoặc AC-M2 trượt |
| ô `pytest tests/test_no_leakage.py` trả mã khác 0 | T-28 trượt — đọc đầu ra để biết ô nào |

`run_search.py` cũng tự in `!! BÁO ĐỘNG: PR-AUC > 0,95` ở cuối nếu gặp.

## 5. Kiểm thử của giai đoạn này

```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_search.py         # 12 ca: không gian tìm kiếm, bảng kết quả, bootstrap theo cặp
.\.venv\Scripts\python.exe -m pytest tests\test_no_leakage.py     # 57 ca, gồm 7 ô rà soát rò rỉ của T-28
.\.venv\Scripts\python.exe -m pytest                              # toàn bộ: 169 xanh, 1 skip
```

Không thêm `-q` vào hai lệnh đầu: `pytest.ini` đã có `addopts = -q`, thêm nữa thành `-qq` và
mất dòng tổng kết "N passed".

Riêng `test_no_leakage.py` quét mã nguồn trong `src/`, `scripts/`, `app.py` **và các ô code
của mọi notebook**. Sửa notebook mà phạm một ô của danh sách kiểm (ví dụ gõ `KFold(` thay vì
`StratifiedKFold(`, hay chọn ngưỡng bằng `pick_threshold(y_test, …)`) thì kiểm thử đỏ ngay.

## 6. Những cái bẫy đã gặp

**XGBoost cho điểm khác nhau theo số luồng.** Với `tree_method="hist"`, cùng một cấu hình chạy
`n_jobs=1` và `n_jobs=-1` cho xác suất lệch ở chữ số thứ ba–tư, vì thứ tự cộng dồn histogram
thay đổi. Hệ quả:

- Trên cùng một máy, kết quả tái lập tuyệt đối (đã kiểm: chạy notebook hai lần, mọi con số
  trùng nhau).
- Sang máy có số lõi khác, PR-AUC có thể lệch nhẹ. Khi làm T-38, ghi lại số lõi; nếu lệch vượt
  0,001 thì ghim `n_jobs` về một giá trị cố định. Chi tiết ở [10 §6](10-van-hanh-tai-lap.md).
- Điểm CV trong `search_results.csv` (1 luồng) và điểm của mô hình huấn luyện lại trong notebook
  (`n_jobs=-1`) vì thế không trùng khít — chênh nhỏ hơn nhiều so với độ lệch chuẩn giữa các fold.

**Console Windows cp1252.** Giống giai đoạn 3: `run_search.py` đã tự chuyển stdout sang UTF-8,
nên in tiếng Việt không làm sập lần chạy 20 phút. Tự viết script khác thì đặt
`$env:PYTHONIOENCODING = "utf-8"` trước.

**Máy bận làm mọi thứ chậm lại.** Tìm kiếm chiếm hết các lõi. Chạy `pytest` hay notebook khác
cùng lúc sẽ chậm gấp nhiều lần — ví dụ `tests/test_search.py` có một ca gọi `run_search` thật.
Đợi bước 3 xong rồi mới làm việc khác nặng CPU.
