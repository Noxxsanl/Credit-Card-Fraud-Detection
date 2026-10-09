# Lệnh chạy — giai đoạn 3 đến 9

Sổ lệnh để **chạy lại** từng giai đoạn của dự án, từ lưới chiến lược mất cân bằng tới hệ thống đóng
gói bằng Docker Compose. Mọi lệnh chạy từ thư mục gốc của repo; lệnh Python gọi thẳng trình thông dịch
trong `.venv` (không cần `activate`). Mục 1 của mỗi giai đoạn có cả bản PowerShell lẫn bản Git Bash;
các mục sau viết theo PowerShell, đổi sang Git Bash bằng bảng quy đổi ở §3.1.3.

Tài liệu này chỉ ghi *lệnh gì, ra cái gì, mất bao lâu*. Phần **vì sao** nằm ở tài liệu thiết kế dẫn ở
đầu mỗi giai đoạn; điều kiện nghiệm thu từng việc nằm ở [TASKS.md](../TASKS.md).

**Đánh số:** §N là giai đoạn N, §N.x là mục x của giai đoạn đó — ví dụ §8.1.3 là mục 1.3 của giai
đoạn 8. Số mục có tên tài liệu đứng trước (04 §3, 10 §6…) là của tài liệu thiết kế tương ứng.

| § | Giai đoạn | Chạy lại | Đối tượng |
|---|---|---|---|
| [3](#giai-đoạn-3--chiến-lược-mất-cân-bằng-t-20t-23) | Chiến lược mất cân bằng (T-20…T-23) | lưới 20 tổ hợp | Người làm mô hình |
| [4](#giai-đoạn-4--tinh-chỉnh-và-kiểm-chứng-t-24t-28) | Tinh chỉnh và kiểm chứng (T-24…T-28) | tìm kiếm siêu tham số và notebook 05 | Người làm mô hình |
| [5](#giai-đoạn-5--ngưỡng-chi-phí-shap-t-29t-34) | Ngưỡng, chi phí, SHAP (T-29…T-34) | notebook ngưỡng, SHAP và autoencoder | Người làm mô hình |
| [6](#giai-đoạn-6--xuất-hiện-vật-t-35t-38) | Xuất hiện vật (T-35…T-38) | xuất hiện vật và kiểm tra tái lập | Người làm mô hình |
| [7](#giai-đoạn-7--api-t-40t-50) | API (T-40…T-50) | dựng PostgreSQL, lược đồ và chạy API | Người lập trình |
| [8](#giai-đoạn-8--giao-diện-t-51t-57) | Giao diện (T-51…T-57) | chạy giao diện web, nạp dữ liệu demo, kiểm các tiêu chí giao diện | Người lập trình |
| [9](#giai-đoạn-9--đóng-gói-và-nghiệm-thu-t-58t-62) | Đóng gói và nghiệm thu (T-58…T-62) | dựng cả hệ thống bằng Docker Compose, sao lưu dữ liệu demo, nghiệm thu AC-A1…AC-A10 | Người chấm, người lập trình |

## Giai đoạn 3 — Chiến lược mất cân bằng (T-20…T-23)

Chạy lại lưới 20 tổ hợp của giai đoạn 3 từ đầu. Phần **vì sao** thiết kế lưới như vậy nằm ở
[04 §3](04-thiet-ke-mo-hinh-ml.md).

> **Dữ liệu tiền đề:** `data/creditcard.csv` phải có sẵn và đúng toàn vẹn (T-10).
> Nếu máy sạch, làm [10 §2](10-van-hanh-tai-lap.md) trước.

### 3.1 Trình tự đầy đủ

Bốn lệnh, theo đúng thứ tự này.

#### 3.1.1 PowerShell

```powershell
# 1. Kiểm môi trường — phải xanh trước khi bỏ ra 45 phút
.\.venv\Scripts\python.exe -m pytest -q

# 2. Chạy thử 12.000 dòng (~1 phút) — bắt lỗi cấu hình trước khi chạy thật
.\.venv\Scripts\python.exe scripts\run_grid.py --smoke

# 3. Chạy thật, ghi cả màn hình lẫn tệp log
.\.venv\Scripts\python.exe scripts\run_grid.py 2>&1 | Tee-Object grid_run.log

# 4. Mở notebook và Run All (~1 phút, vì nó đọc điểm lưu thay vì huấn luyện lại)
#    notebooks/04_imbalance_strategies.ipynb
```

#### 3.1.2 Git Bash

Cùng bốn bước đó, cùng một `python.exe`:

```bash
# 1. Kiểm môi trường — phải xanh trước khi bỏ ra 45 phút
./.venv/Scripts/python.exe -m pytest -q

# 2. Chạy thử 12.000 dòng (~1 phút) — bắt lỗi cấu hình trước khi chạy thật
./.venv/Scripts/python.exe scripts/run_grid.py --smoke

# 3. Chạy thật, ghi cả màn hình lẫn tệp log
./.venv/Scripts/python.exe scripts/run_grid.py 2>&1 | tee grid_run.log

# 4. Mở notebook và Run All (~1 phút, vì nó đọc điểm lưu thay vì huấn luyện lại)
#    notebooks/04_imbalance_strategies.ipynb
```

Vẫn là `.venv/Scripts/python.exe` chứ không phải `.venv/bin/python`: `.venv` này do
Python trên Windows tạo ra, Git Bash chỉ là trình bao khác chứ không đổi bố cục thư
mục. Cũng không cần `source .venv/Scripts/activate` — gọi thẳng đường dẫn là đủ.

Bước 2 không phải thủ tục thừa: nó chạy đúng đường mã của bước 3 trên 1/19 dữ liệu,
nên một lỗi nhập liệu hay lỗi tham số sẽ lộ ra sau một phút thay vì sau 40 phút.
Kết quả của nó ghi ra `grid_results_smoke.csv` riêng — **không dùng cho báo cáo**.

#### 3.1.3 Bảng quy đổi

Để tự chuyển các lệnh PowerShell ở những mục sau, kể cả của các giai đoạn sau, sang Git Bash:

| Việc | PowerShell | Git Bash |
|---|---|---|
| Gọi trình thông dịch | `.\.venv\Scripts\python.exe` | `./.venv/Scripts/python.exe` |
| Vừa xem vừa ghi log | `... 2>&1 \| Tee-Object grid_run.log` | `... 2>&1 \| tee grid_run.log` |
| Chỉ ghi log, không xem | `... > grid_run.log 2>&1` | `... > grid_run.log 2>&1` |
| Đặt biến môi trường | `$env:PYTHONIOENCODING = "utf-8"` | `export PYTHONIOENCODING=utf-8` |
| Dấu phân cách thư mục | `\` hoặc `/` đều được | chỉ `/` |

Hai cái bẫy riêng của Git Bash:

- **Đừng gõ `python` trống không** — trên nhiều máy nó bắt vào Python hệ thống hoặc
  mở Microsoft Store, chứ không phải `.venv`. Luôn viết đủ đường dẫn như trên.
- Git Bash tự dịch đường dẫn kiểu `/d/WorkSpace/...` sang `D:\WorkSpace\...` khi
  truyền cho chương trình Windows. Bọc trong nháy kép nếu đường dẫn có dấu cách.

### 3.2 Bảng lệnh của `run_grid.py`

Toàn bộ cờ của [scripts/run_grid.py](../scripts/run_grid.py):

| Lệnh | Tác dụng | Thời gian |
|---|---|---|
| `run_grid.py` | Chạy đủ 20 tổ hợp, nối tiếp điểm lưu nếu có | ~45 phút |
| `run_grid.py --smoke` | Chạy thử trên 12.000 dòng, ghi ra tệp `_smoke` riêng | ~1 phút |
| `run_grid.py --status` | Chỉ in tiến độ điểm lưu rồi thoát — **an toàn khi lưới đang chạy** | tức thì |
| `run_grid.py --fast` | Hạ `n_estimators` xuống 100 (phương án dự phòng của T-20) | ~15 phút |
| `run_grid.py --fresh` | Bỏ điểm lưu, huấn luyện lại toàn bộ từ đầu | ~45 phút |

#### Theo dõi tiến độ ở cửa sổ khác

Lưới ghi điểm lưu sau **mỗi** tổ hợp, nên mở một PowerShell thứ hai và hỏi bất cứ lúc nào:

```powershell
.\.venv\Scripts\python.exe scripts\run_grid.py --status
```

```
reports\grid_results.csv: 14/20 tổ hợp
  thời gian đã bỏ ra: 27.3 phút
  còn thiếu 6: random_forest+smote_tomek, xgboost+class_weight, …
  ước tính còn khoảng 12 phút
```

#### Ngắt giữa chừng

Bấm `Ctrl+C` rồi chạy lại **đúng lệnh cũ** — nó đọc `reports/grid_results.csv` và
bỏ qua các tổ hợp đã xong. Chỉ mất phần tổ hợp đang dở dang.

Muốn thật sự làm lại từ trắng thì mới dùng `--fresh`.

### 3.3 Hiện vật sinh ra

| Tệp | Nguồn | Ghi vào git? |
|---|---|---|
| `reports/grid_results.csv` | bước 3 | Có — 20 dòng × 19 cột, là bằng chứng của T-20 |
| `reports/grid_results.npz` | bước 3 | **Không** — ~14 MB điểm out-of-fold, đã có trong `.gitignore` |
| `reports/grid_results_smoke.csv` / `.npz` | bước 2 | Không cần, chỉ là kết quả chạy thử |
| `reports/figures/04_pr_curves_strategies.png` | bước 4 | Có — hình của T-21 |
| `reports/figures/04_pr_curves_all_models.png` | bước 4 | Có |
| `reports/figures/04_grid_heatmap.png` | bước 4 | Có |
| `reports/figures/04_cost_of_strategies.png` | bước 4 | Có |
| `grid_run.log` | bước 3 | Không — nhật ký một lần chạy |

`grid_results.npz` bị loại khỏi git vì nó tái tạo được: xoá đi rồi chạy lại bước 3
là có lại. Nhưng **notebook 04 cần nó** để vẽ đường PR, nên đừng xoá khi còn đang làm.

### 3.4 Kết quả của lần chạy thật (2026-09-16)

Để đối chiếu khi chạy lại — cùng `RANDOM_STATE=42` thì các con số dưới đây phải lặp lại:

| Mốc | Giá trị |
|---|---|
| Tập huấn luyện | 226.980 dòng, 378 gian lận (0,1665%) |
| Chế độ | đầy đủ theo [04 §3.2](04-thiet-ke-mo-hinh-ml.md) — không dùng `--fast` |
| Tổng thời gian | **45,4 phút** (tổng `total_seconds` của 20 dòng) |
| PR-AUC cao nhất | **0,8549** — `xgboost` + `class_weight` |
| Riêng 4 tổ hợp `smote_tomek` | 25,7 phút, so với 7,6 phút của `smote` → **đắt hơn 18,2 phút** mà kết quả giống hệt |

Dòng cuối là phát hiện 3 của giai đoạn 3: bước Tomek xoá đúng 0 cặp — vì SMOTE chạy trước đã xoá
hết 21 cặp Tomek có trong dữ liệu gốc ([04 §3.4](04-thiet-ke-mo-hinh-ml.md)).
Xem ghi chú giai đoạn 3 trong [TASKS.md](../TASKS.md).

#### Điều kiện dừng của T-23

Script tự kiểm tra và in một trong hai dòng ở cuối:

```
T-23: PR-AUC cao nhất 0.8549, dưới ngưỡng báo động 0,95. Dải kỳ vọng 04 §3.4 là 0,80–0,87.
```

Nếu thay vào đó là `!! BÁO ĐỘNG (T-23)` thì **dừng lại**, đừng chạy notebook, mở danh
sách rà rò rỉ ở [08 §3.1](08-ke-hoach-kiem-thu.md).

### 3.5 Kiểm thử của giai đoạn này

```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_grid.py -q     # 19 ca, chốt hợp đồng của run_grid
.\.venv\Scripts\python.exe -m pytest -q                        # toàn bộ: 127 xanh, 1 skip
```

Ca skip duy nhất là TC-12 (đối chiếu ngưỡng JavaScript ↔ Python), thuộc T-54 ở giai
đoạn 8 — đúng thiết kế, không phải lỗi.

### 3.6 Một cái bẫy đã gặp

Console Windows mặc định là cp1252: in tiếng Việt ra sẽ ném `UnicodeEncodeError` và
**giết cả lần chạy 40 phút ở giữa chừng**. `run_grid.py` đã tự `reconfigure` stdout sang
UTF-8 ngay đầu tệp nên không còn gặp. Nhưng nếu tự viết script khác gọi `run_grid`, nhớ
làm y như vậy, hoặc đặt `$env:PYTHONIOENCODING = "utf-8"` trước khi chạy.

Cùng lý do đó, `reconfigure` dùng `line_buffering=True` — nếu không, đầu ra bị chuyển
hướng vào `grid_run.log` sẽ bị đệm theo khối và người chạy không thấy gì suốt nhiều phút.

## Giai đoạn 4 — Tinh chỉnh và kiểm chứng (T-24…T-28)

Chạy lại giai đoạn 4 từ đầu: tìm kiếm siêu tham số, đánh giá trên tập kiểm thử, chia theo thời
gian và rà soát rò rỉ. Phần **vì sao** nằm ở [04 §4.1](04-thiet-ke-mo-hinh-ml.md).

> **Tiền đề — hai tệp phải có sẵn:**
>
> - `data/creditcard.csv` đúng toàn vẹn (T-10). Máy sạch thì làm [10 §2](10-van-hanh-tai-lap.md) trước.
> - `reports/grid_results.npz` — điểm out-of-fold của giai đoạn 3. Tệp này **không nằm trong
>   git**; notebook 05 đọc nó để lấy điểm của cấu hình mặc định. Thiếu thì chạy
>   `scripts\run_grid.py` trước (~45 phút, xem §3.1).

### 4.1 Trình tự đầy đủ

Năm lệnh, theo đúng thứ tự này.

#### 4.1.1 PowerShell

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

#### 4.1.2 Git Bash

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
mà ô chia theo thời gian (mục 5 của notebook) chạy khoảng 9 phút trên 12 lõi — máy chậm hơn sẽ
vượt ngưỡng đó.

### 4.2 Bảng lệnh của `run_search.py`

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

### 4.3 Hiện vật sinh ra

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

### 4.4 Kết quả của lần chạy thật (2026-09-23)

Để đối chiếu khi chạy lại — cùng `RANDOM_STATE=42` **và cùng số lõi CPU** thì các con số
dưới đây phải lặp lại tới chữ số cuối (xem §4.6):

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

Nếu dòng "Cấu hình được chọn" ở mục 3.1 của notebook ra **TINH CHỈNH** thay vì **MẶC ĐỊNH** thì
lần chạy đó đã khác lần này — kiểm tra lại số lõi và phiên bản `xgboost`.

#### Điều kiện dừng

Notebook có `assert` ở những chỗ sau; ô nào đỏ thì **dừng lại**, không tick việc tương ứng:

| Assert | Nghĩa là |
|---|---|
| `tập kiểm thử phải trùng với notebook 03 (95 gian lận)` | cách chia đã đổi — mọi con số không còn so được với giai đoạn 2–3 |
| `cần 30 lần thử` | `search_results.csv` là bản chạy thử hoặc bị cắt |
| `PR-AUC > 0,95 — dừng lại rà rò rỉ` | gần như chắc chắn rò rỉ, xem [08 §3.1](08-ke-hoach-kiem-thu.md) |
| `T-25 chưa đạt — không tick` | AC-M1 hoặc AC-M2 trượt |
| ô `pytest tests/test_no_leakage.py` trả mã khác 0 | T-28 trượt — đọc đầu ra để biết ô nào |

`run_search.py` cũng tự in `!! BÁO ĐỘNG: PR-AUC > 0,95` ở cuối nếu gặp.

### 4.5 Kiểm thử của giai đoạn này

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

### 4.6 Những cái bẫy đã gặp

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

## Giai đoạn 5 — Ngưỡng, chi phí, SHAP (T-29…T-34)

Chạy lại giai đoạn 5 từ đầu. Phần **vì sao** nằm ở [04 §6.6, §7.1, §8.1](04-thiet-ke-mo-hinh-ml.md).

> **Tiền đề:** `data/creditcard.csv` đúng toàn vẹn, và nên có `reports/grid_results.npz` (điểm
> out-of-fold của giai đoạn 3, không nằm trong git). Thiếu tệp này thì notebook 06 và 07 tự tính
> lại điểm out-of-fold, mất thêm khoảng 2 phút mỗi notebook.

### 5.1 Trình tự

Ba notebook độc lập với nhau, chạy theo thứ tự nào cũng được. Không có script chạy ngoài: phần
nặng nhất chỉ mất vài phút.

#### 5.1.1 PowerShell

```powershell
# 1. Kiểm môi trường
.\.venv\Scripts\python.exe -m pytest

# 2. Ba notebook trong kernel sạch
foreach ($nb in "06_threshold_and_cost", "07_explainability", "06b_autoencoder") {
    .\.venv\Scripts\python.exe -m jupyter nbconvert --to notebook --execute --inplace `
        --ExecutePreprocessor.timeout=3600 "notebooks\$nb.ipynb"
}
```

#### 5.1.2 Git Bash

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

### 5.2 Hiện vật sinh ra

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

### 5.3 Số đối chiếu (lần chạy 2026-09-25, 12 lõi)

| Mốc | Giá trị |
|---|---|
| τ\* chọn trên out-of-fold (dò trên mọi điểm) | **0,02317** (lưới 200 điểm: 0,02321) |
| Test tại τ\* | TP 77, FP 38, FN 18, chi phí 2.390 EUR |
| Test tại 0,5 | TP 74, FP 4, FN 21, chi phí 2.586 EUR |
| Hiệu chi phí τ\* − 0,5 | test −197 [−676, +160]; OOF −707 [−1.513, −24] |
| τ\* không đổi trên dải tỷ lệ | 20:1 → 50:1 |
| SHAP top 3 | V14, V4, V12 |
| Sai số cộng của SHAP | 2,2·10⁻⁵ |
| Autoencoder PR-AUC | 0,329 [0,246 – 0,438] |

### 5.4 Kiểm thử của giai đoạn này

```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_threshold.py   # thêm: ngưỡng chính xác, độ nhạy đơn điệu, sample_fraction
.\.venv\Scripts\python.exe -m pytest tests\test_search.py      # thêm: bootstrap_threshold_diff
.\.venv\Scripts\python.exe -m pytest tests\test_anomaly.py     # autoencoder: tái lập, chuẩn hoá chỉ từ tập train
.\.venv\Scripts\python.exe -m pytest tests\test_no_leakage.py  # giờ quét cả notebook 06, 06b, 07
```

`test_leak_5_…` đỏ ngay nếu notebook gọi `pick_threshold`, `threshold_alternatives` hay
`sensitivity_analysis` với đối số đầu tiên là nhãn test.

### 5.5 Những cái bẫy đã gặp

**Lưới ngưỡng 200 điểm quá thô cho tiêu chí có ràng buộc.** Xem 04 §6.6. Luôn truyền
`thresholds=np.unique(oof)` khi cần con số cho báo cáo.

**`sensitivity_analysis()` trước đây không nhận `sample_fraction`.** Khi chạy trên out-of-fold
(80% dữ liệu), cột `alerts_per_day` bị quy đổi theo 20%, tức phóng đại 4 lần. Đã sửa, và có kiểm
thử canh.

**SHAP cần đầu vào sau tiền xử lý.** `TreeExplainer` bọc bước `clf`, không bọc cả pipeline. Phải
đưa vào `pipeline[:-1].transform(X)`, trong đó `Amount` đã qua `RobustScaler`. Đưa `X` thô vào
thì SHAP của `Amount` sai mà không báo lỗi. Ô kiểm tra tính cộng ở notebook 07 bắt được lỗi này.

## Giai đoạn 6 — Xuất hiện vật (T-35…T-38)

Chạy lại giai đoạn 6 từ đầu: xuất mô hình, ngưỡng, explainer, chỉ số và thư viện mẫu, rồi kiểm
tra tái lập. Cấu trúc từng tệp nằm ở [06 §6](06-thiet-ke-luu-tru.md).

> **Tiền đề:**
>
> - `data/creditcard.csv` đúng toàn vẹn, và `data/test_set.parquet` do notebook 03 ghi ra.
> - Các bảng `reports/*.csv` của notebook 04–07 (có trong git). Notebook 08 đối chiếu với chúng và
>   **dừng lại** nếu lệch.
> - Nên có `reports/grid_results.npz` (không nằm trong git). Thiếu thì notebook 08 vẫn chạy, nhưng
>   bỏ qua phần so điểm out-of-fold với giai đoạn 3 và ghi `strategy_pr_curves: null`.

### 6.1 Trình tự

#### 6.1.1 PowerShell

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

#### 6.1.2 Git Bash

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

### 6.2 Hiện vật sinh ra

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

### 6.3 Số đối chiếu (lần chạy 2026-09-27, 12 lõi)

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

### 6.4 Kiểm tra tái lập — T-38

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

### 6.5 Kiểm thử của giai đoạn này

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

### 6.6 Những cái bẫy đã gặp

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

**Chạy lại notebook 03 là mất `risk_score`.** Xem §6.1.

## Giai đoạn 7 — API (T-40…T-50)

**Dựng và chạy lại** tầng API: PostgreSQL trong Docker, lược đồ bằng Alembic, máy chủ FastAPI và
bộ kiểm thử tích hợp. Hợp đồng API nằm ở [05](05-thiet-ke-api.md) (05 §7 ghi những điểm bản thi
hành cụ thể hơn hợp đồng ban đầu); lược đồ ở [06 §2](06-thiet-ke-luu-tru.md).

> **Tiền đề:**
>
> - Hiện vật của notebook 08 trong `models/` (gồm `oof_scores.npz`, thêm ở giai đoạn này) và
>   `data/sample_pool.json`, `data/test_set.parquet`. Thiếu thì API vẫn chạy nhưng `/health` trả 503.
> - Docker Desktop **đang chạy**.
> - `.env` sao từ `.env.example`. Máy đã có PostgreSQL chiếm cổng 5432 thì đặt `POSTGRES_HOST_PORT=5433`
>   và sửa hai URL tương ứng ([10 §2.3](10-van-hanh-tai-lap.md)).

### 7.1 Trình tự

#### 7.1.1 PowerShell

```powershell
# 0. Cấu hình (một lần)
Copy-Item .env.example .env          # rồi chỉnh cổng nếu 5432 đã bị chiếm

# 1. PostgreSQL 16 — T-40. Lần đầu tải image khoảng 30 giây
docker compose up -d db
docker compose exec db pg_isready -U fraud -d fraud          # phải ra "accepting connections"

# 2. Lược đồ — T-42
.\.venv\Scripts\python.exe -m alembic upgrade head

# 3. Kiểm thử: tự tạo rồi xoá cơ sở dữ liệu fraud_test (~3 phút)
.\.venv\Scripts\python.exe -m pytest tests\test_db.py tests\test_scoring.py tests\test_api.py

# 4. Máy chủ — tài liệu tự sinh ở http://localhost:8000/docs
.\.venv\Scripts\python.exe -m uvicorn api.main:app --reload --port 8000
```

#### 7.1.2 Git Bash

```bash
cp .env.example .env
docker compose up -d db
docker compose exec db pg_isready -U fraud -d fraud
./.venv/Scripts/python.exe -m alembic upgrade head
./.venv/Scripts/python.exe -m pytest tests/test_db.py tests/test_scoring.py tests/test_api.py
./.venv/Scripts/python.exe -m uvicorn api.main:app --reload --port 8000
```

#### 7.1.3 Thử nhanh bằng curl

```bash
curl localhost:8000/api/v1/health
curl "localhost:8000/api/v1/samples?category=fraud_hard"
curl "localhost:8000/api/v1/threshold/preview?value=0.5"
curl -X POST localhost:8000/api/v1/threshold/optimize -H "content-type: application/json" \
     -d '{"constraint": {"type": "max_alerts_per_day", "value": 200}}'
curl -F "file=@giao-dich.csv" localhost:8000/api/v1/score/upload
curl -N "localhost:8000/api/v1/replay/stream?speed=60"      # Ctrl+C để dừng
```

#### 7.1.4 Đặt lại dữ liệu demo

```bash
docker compose exec db psql -U fraud -d fraud \
  -c "TRUNCATE transactions, reviews RESTART IDENTITY CASCADE; DELETE FROM settings WHERE key = 'threshold';"
```

Lệnh này giữ `cost_fn`, `cost_fp`, `replay_speed`. Xoá khóa `threshold` là quay về τ\* của
`threshold.json`.

### 7.2 Tệp của giai đoạn này

| Tệp | Nội dung |
|---|---|
| `docker-compose.yml` | dịch vụ `db`: `postgres:16-alpine`, volume `pgdata`, healthcheck `pg_isready`, cổng `${POSTGRES_HOST_PORT:-5432}` |
| `.env.example` | mẫu biến môi trường, commit được |
| `alembic.ini` | ở gốc repo, **chỉ ASCII** |
| `api/migrations/versions/0001_initial_schema.py` | ba bảng, chỉ mục, `CHECK`, dãy `transaction_id_seq` |
| `api/migrations/versions/0002_seed_settings.py` | `cost_fn`, `cost_fp`, `replay_speed` |
| `api/config.py`, `db.py`, `models_orm.py` | cấu hình, engine (`pool_pre_ping`, `connect_timeout=3`), bảng ORM |
| `api/loader.py` | nạp và kiểm hiện vật; chấm lại 1/50 tập kiểm thử để bắt lệch phiên bản thư viện |
| `api/errors.py` | mô hình lỗi chung; `MISSING_FEATURES` kèm tên cột thiếu |
| `api/schemas.py` | hợp đồng Pydantic; `TransactionInput` dựng từ `RAW_REQUIRED_COLUMNS` |
| `api/services/*.py`, `api/routes/*.py` | dịch vụ và route mỏng cho 15 endpoint |
| `tests/conftest.py` | cơ sở dữ liệu `fraud_test` dùng một lần, máy chủ thật trên nó |
| `tests/test_db.py`, `test_scoring.py`, `test_api.py` | 18 + 20 + 62 ca |
| `models/oof_scores.npz` | điểm out-of-fold cho `/threshold/optimize` — notebook 08 xuất thêm |

### 7.3 Số đối chiếu (máy 12 lõi, 2026-09-28, có tải nền khoảng 50% CPU)

| Mốc | Giá trị | Yêu cầu |
|---|---|---|
| `pg_isready` | `accepting connections`, healthy sau khoảng 10 giây | T-40 |
| `upgrade head → downgrade base → upgrade head` | lược đồ cuối trùng lược đồ đầu | T-42, TC-47 |
| Khởi động tới `/health` 200 | khoảng 5 giây | NFR-04 |
| `/score` một giao dịch, 1.000 lần | p95 45,6 ms phía máy chủ | NFR-01 < 50 ms |
| `/score/upload` 10.000 dòng | 2,0 giây | T-46, AC-A1 < 30 giây |
| `COPY` 10.000 dòng | 0,53 giây; `INSERT` từng dòng 119 dòng/giây | TC-54 < 3 giây |
| `/threshold/preview` | phần tính 0,12 ms, khoảng 15 ms qua HTTP | T-47 khoảng 10 ms |
| `/threshold/optimize` chi phí mặc định | 0,023172983899712563 = τ\* của `threshold.json` | ML-08 |
| `/threshold/optimize`, tối đa 200 cảnh báo/ngày | 0,9657, `constraint_binding = true` | T-47 |
| `/explain` | 5 dương + 3 âm ở mọi mẫu vượt ngưỡng được kiểm; 114/115 giao dịch vượt ngưỡng của tập kiểm thử có đủ 5 dương; SHAP khớp explainer tới từng bit | T-48, AC-A4 |
| Phát lại liên tục | 185 giây, 1.532 giao dịch, 0 lỗi | T-50, AC-A6 |

### 7.4 Kiểm tra AC-A6 — phát lại 3 phút trên máy chủ thật

`TestClient` không stream thật, nên AC-A6 kiểm bằng uvicorn và `httpx`:

```powershell
.\.venv\Scripts\python.exe -m uvicorn api.main:app --port 8000      # cửa sổ 1
```

```python
# cửa sổ 2 — giữ kết nối 185 giây, đếm sự kiện và lỗi
import httpx, json, time
counts, started = {}, time.monotonic()
with httpx.stream("GET", "http://localhost:8000/api/v1/replay/stream", params={"speed": 60},
                  timeout=httpx.Timeout(10, read=30)) as r:
    name = None
    for line in r.iter_lines():
        if line.startswith("event: "):
            name = line[7:]
        elif line.startswith("data: "):
            json.loads(line[6:])
            counts[name] = counts.get(name, 0) + 1
        if time.monotonic() - started > 185:
            break
print(counts)   # lần chạy 2026-09-28: start 1, transaction 1518, alert 14, stats 158
```

Sau đó `SELECT count(*) FROM transactions WHERE source = 'replay'` phải bằng số giao dịch đã phát
(mẻ cuối được ghi cả khi client ngắt giữa chừng), và log uvicorn không có dòng `ERROR`.

### 7.5 Những cái bẫy đã gặp

**Cổng 5432 đã bị chiếm.** Máy phát triển có sẵn PostgreSQL 17. Cách xử lý là đặt
`POSTGRES_HOST_PORT=5433` trong `.env`, không sửa `docker-compose.yml`, để máy khác vẫn dùng
5432 mặc định.

**`alembic.ini` có chữ tiếng Việt thì Alembic sập** (`UnicodeDecodeError … cp1252`). Alembic đọc tệp
bằng bảng mã mặc định của hệ điều hành. Chú thích trong tệp này viết không dấu. Alembic 1.20 cũng
đổi tên khóa `version_path_separator` thành `path_separator`.

**"Đúng 5 yếu tố dương" không phải lúc nào cũng có.** 77% giao dịch của tập kiểm thử có ít hơn 5
đặc trưng SHAP dương: với giao dịch hợp lệ điểm thấp, mô hình kéo gần hết đặc trưng về phía an
toàn. `/explain` trả **tối đa** 5 và 3, không mượn đóng góp âm để cho đủ. Trong hàng đợi, 114/115
giao dịch có đủ 5.

**12 luồng XGBoost cho một giao dịch làm chậm gấp đôi.** Mô hình huấn luyện với `n_jobs=-1`. Chấm
một dòng bằng 12 luồng tốn chi phí đồng bộ và tranh CPU: p95 của `/score` là 87 ms. Chấm bằng
1 luồng cho **cùng từng bit** trên 56.746 giao dịch (chỉ huấn luyện mới phụ thuộc số luồng), và p95
còn 45,6 ms. `api/loader.py` đặt `n_jobs=1` sau khi nạp.

**Biên của dải `critical`/`block` không còn là 3τ** (đổi 2026-10-09). Nay là `max(τ, τ_chặn)`, với
τ_chặn ≈ 0,9735 chọn theo precision ≥ 95% trên out-of-fold — luật 3τ chặn tự động 19 khách hợp lệ của
tập kiểm thử ([05 §2](05-thiet-ke-api.md), notebook 06 §6.2).

**`/score` p95 còn khoảng 20 ms** (đổi 2026-10-09). Hai phần ba thời gian cũ là chi phí dựng DataFrame
trong `ColumnTransformer` và lớp sklearn của XGBoost. `api/serving.py` (`FastScorer`) làm cùng phép
tính trên numpy và `Booster.inplace_predict`; loader đối chiếu **từng bit** với pipeline lúc khởi động.
Ba lần đo: p95 phía máy chủ 18,7 / 24,1 / 21,0 ms (trước đó 60,6 / 51,3 / 49,9 ms — trượt NFR-01 hai
lần). Ca đo NFR-01 đánh dấu `perf`, không chạy trong lượt `pytest` mặc định: `pytest -m perf`.

**`TestClient` đọc hết thân phản hồi rồi mới trả.** Ca phát lại trong pytest phát giờ cuối của ngày
2 ở tốc độ 3.600. Ca 3 phút phải chạy trên uvicorn thật (§7.4).

**Trạng thái rò giữa các ca kiểm thử.** Một ca đổi `cost_fn` qua `PUT /threshold` làm đỏ ca xem
trước ngưỡng chạy sau nó. Fixture `clean_db` nay đưa cả bảng `settings` về trạng thái sau migration.

## Giai đoạn 8 — Giao diện (T-51…T-57)

**Chạy và kiểm lại** giao diện web: bốn màn hình trong `web/`, gọi API của giai đoạn 7. Thiết kế
ở [07](07-thiet-ke-giao-dien.md) (07 §12 ghi những điểm bản thi hành cụ thể hơn thiết kế ban đầu);
hợp đồng API ở [05](05-thiet-ke-api.md).

> **Tiền đề:**
>
> - API của giai đoạn 7 chạy được (§7.1): Docker
>   Desktop đang chạy, `.env` đã có, hiện vật của notebook 08 trong `models/` và `data/`.
> - Không cần Node.js, không cần `npm install`: giao diện là tệp tĩnh, Alpine.js và Chart.js nằm sẵn
>   trong `web/vendor/`. Node.js chỉ cần cho ca kiểm thử TC-12; máy không có Node thì ca đó tự bỏ qua.
> - Trang phải mở qua `http://localhost:3000` hoặc `http://127.0.0.1:3000` — hai origin CORS mà API
>   cho phép. Mở thẳng `web/index.html` (`file://`) sẽ bị trình duyệt chặn gọi API.

### 8.1 Trình tự

#### 8.1.1 PowerShell

```powershell
# 0. Cơ sở dữ liệu và lược đồ (bỏ qua nếu đang chạy)
docker compose up -d db
.\.venv\Scripts\python.exe -m alembic upgrade head

# 1. API — cửa sổ 1. /api/v1/health trả 200 sau khoảng 5 giây
.\.venv\Scripts\python.exe -m uvicorn api.main:app --port 8000

# 2. Giao diện — cửa sổ 2
.\.venv\Scripts\python.exe -m http.server 3000 --directory web

# 3. Mở http://localhost:3000

# 4. TC-12: hàm JavaScript của UI-D1 đối chiếu với src/threshold.py (vài giây)
.\.venv\Scripts\python.exe -m pytest tests\test_threshold_parity.py
```

#### 8.1.2 Git Bash

```bash
docker compose up -d db
./.venv/Scripts/python.exe -m alembic upgrade head
./.venv/Scripts/python.exe -m uvicorn api.main:app --port 8000          # cửa sổ 1
./.venv/Scripts/python.exe -m http.server 3000 --directory web          # cửa sổ 2
./.venv/Scripts/python.exe -m pytest tests/test_threshold_parity.py
```

#### 8.1.3 Nạp dữ liệu demo

Hàng đợi đọc từ cơ sở dữ liệu nên lúc đầu rỗng; màn hình hiện ba cách nạp:

| Cách | Trên giao diện | Ghi chú |
|---|---|---|
| Tải CSV | "Tải CSV" → chọn tệp đủ 30 cột, cột `Class` nếu có | Tạo tệp thử bên dưới; 10.000 dòng chấm xong khoảng 2 giây |
| Thư viện mẫu | "Chọn mẫu có sẵn" → chọn nhóm → "Nạp … mẫu còn lại của nhóm này" | Giữ nhãn thật, nên UI-02 đối chiếu được kết luận với sự thật |
| Phát lại | "▶ Phát lại ngày 2" → "Phát", đặt tốc độ 600× cho nhanh | Giao dịch vượt ngưỡng rơi vào hàng đợi theo mẻ 100 dòng |

Tệp CSV thử 10.000 dòng lấy từ tập kiểm thử (25 gian lận):

```powershell
.\.venv\Scripts\python.exe -c "import pandas as pd; from src.features import RAW_REQUIRED_COLUMNS as C; t = pd.read_parquet('data/test_set.parquet'); t[C + ['Class']].head(10000).to_csv('giao-dich-10000.csv', index=False)"
```

Đặt lại dữ liệu demo (giữ lược đồ, tham số chi phí và tốc độ phát lại; xóa ngưỡng người dùng đặt):

```bash
docker compose exec db psql -U fraud -d fraud \
  -c "TRUNCATE transactions, reviews RESTART IDENTITY CASCADE; DELETE FROM settings WHERE key = 'threshold';"
```

### 8.2 Tệp của giai đoạn này

| Tệp | Nội dung |
|---|---|
| `web/index.html` | khung ứng dụng (thanh bên, thanh trên luôn hiện ngưỡng và trạng thái API), bốn màn hình, ngăn kéo UI-02, hộp thoại thư viện mẫu |
| `web/app.js` | `Alpine.store("app")` và năm thành phần: `queueScreen`, `txDrawer`, `samplesDialog`, `thresholdScreen`, `performanceScreen`, `replayScreen` |
| `web/threshold.js` | UI-D1: `prepare()`, `counts()`, `metricsAt()` — bản JavaScript của `src/threshold.py`, chạy được cả trong Node |
| `web/charts.js` | Chart.js: đường cong chi phí trục log kèm vạch ngưỡng, các đường PR/ROC |
| `web/styles.css` | biến màu, bố cục, tương phản WCAG AA, viền trọng tâm |
| `web/config.js` | địa chỉ API (mặc định cùng máy, cổng 8000) — giai đoạn 9 chỉ cần thay tệp này |
| `web/vendor/` | Alpine.js 3.17.4, Chart.js 4.5.1 (MIT), kèm `README.md` ghi nguồn và SHA-256 |
| `tests/test_threshold_parity.py` | TC-12: chạy `web/threshold.js` bằng Node, so với Python trên 20 ngưỡng của tập kiểm thử thật và hai bộ dữ liệu tổng hợp |
| `src/evaluate.py` | `thin_curve()` — sửa cách rút mẫu đường PR/ROC (§8.5, bẫy đầu tiên) |
| `tests/test_artifacts.py` | 2 ca mới canh lỗi rút mẫu đường PR |

### 8.3 Số đối chiếu (Chrome 153 không giao diện, máy 12 lõi, 2026-09-28)

Đo bằng Puppeteer điều khiển Chrome thật, trên API và PostgreSQL thật.

| Mốc | Giá trị | Yêu cầu |
|---|---|---|
| Kéo thanh trượt 0,5 → τ\*, 21 vị trí | lâu nhất **20,4 ms** từ sự kiện `input` tới khi vẽ xong hai khung hình; phần tính UI-D1 dưới 0,1 ms | AC-A3, NFR-03 < 200 ms |
| TC-12 | TP/FP/FN/TN trùng tuyệt đối trên 20 ngưỡng; precision, recall, F1, chi phí, cảnh báo/ngày trùng từng bit | T-54 |
| Kịch bản bảo vệ 0,5 → τ\* | bắt 74/95 → **77/95**; chi phí 6.466 → **5.974 EUR/ngày**; cảnh báo 195 → 287/ngày | 07 §5 |
| Chi phí bỏ lọt 122,21 → 600 EUR | ngưỡng tối ưu 0,02317 → **0,002262** | AC-A5 |
| Thêm ràng buộc 150 cảnh báo/ngày | ngưỡng 0,99986 (hiện "0,9999"), báo "ràng buộc đang chặn nghiệm"; trên tập kiểm thử bắt 54/95, 137 cảnh báo/ngày | FR-33 |
| Tải CSV 10.000 dòng qua nút "Tải CSV" | **2,2 giây** tới lúc hiện kết quả (2,1 giây phía máy chủ) | AC-A1 < 30 giây |
| Hàng đợi | 75 điểm thật trên 3 trang đầu, giảm dần | AC-A2 |
| UI-02 | 5 yếu tố dương, 3 âm; chú thích PCA có mặt; nhãn thật ẩn tới khi bấm `F` | T-55, AC-A4 |
| Bàn phím | `↓`/`Enter` mở chi tiết, `F` ghi kết luận, `Esc` đóng và trả trọng tâm về đúng dòng; Tab đi qua thanh bên, nút, bộ lọc, tới thanh trượt | 07 §3, §4, §10 |
| Phát lại 600× trong 6 giây | 1 giờ mô phỏng, 766 giao dịch; tạm dừng thì bộ đếm đứng yên; tiếp tục đếm tiếp, không đếm trùng | T-56 |
| Console trình duyệt | không lỗi, không cảnh báo trên cả bốn màn hình | 08 §5 |

### 8.4 Kiểm tay trước buổi bảo vệ

- [ ] Mở cả bốn màn hình, console (F12) không có dòng đỏ.
- [ ] Màn hình Ngưỡng: chọn "Mặc định 0,5" rồi kéo thanh trượt về dấu ▲ τ\* — ô "Gian lận bắt được"
      tăng, ô "Chi phí mỗi ngày" giảm, dòng chênh lệch đổi màu.
- [ ] Gõ chi phí bỏ lọt 600 — vạch cam "tối ưu" dời sang trái, xuất hiện phương án "Tối ưu với tham số
      chi phí đang nhập". Đưa lại 122,21.
- [ ] Hàng đợi: nạp nhóm "Gian lận khó" từ thư viện mẫu, mở một dòng, bấm `A` (báo động sai) — nhãn
      thật hiện "Gian lận — ✗ khác kết luận của bạn". Đây là ví dụ FR-44 đáng trình diễn.
- [ ] Phát lại ở 600× khoảng một phút, bấm một cảnh báo để mở chi tiết.
- [ ] Rút dây mạng: giao diện vẫn chạy (không có tài nguyên nào tải từ Internet — NFR-08).

### 8.5 Những cái bẫy đã gặp

**Đường PR trong `metrics.json` bị rút mẫu sai.** UI-04 vẽ đường PR thành một đoạn thẳng từ (0; 1) tới
recall 0,85. `src.evaluate.curve_points` lấy 500 điểm đều theo **chỉ số ngưỡng**, mà với 0,17% lớp
dương gần như mọi ngưỡng nằm ở vùng precision ≈ 0 — cả vùng precision cao chỉ còn 2 điểm. Sửa bằng
`thin_curve()`: rải điểm đều theo chiều dài đường cong. Chạy lại notebook 08 để xuất lại
(`jupyter nbconvert --to notebook --execute --inplace notebooks/08_export_artifacts.ipynb`, khoảng
5 phút). Kết quả: `model.joblib`, `oof_scores.npz`, `test_set.parquet` trùng từng byte với bản cũ; SHAP
trùng từng bit; chỉ khác `trained_at`, `fit_seconds` và ba khóa đường cong. Đường PR mới có 150 điểm
precision ≥ 0,5 thay vì 2. Nhớ khởi động lại uvicorn để nạp `metrics.json` mới.

**Bộ theo dõi của Alpine huỷ chính yêu cầu vừa gửi.** Lúc khởi tạo màn hình ngưỡng, gán giá trị cho ô
chi phí làm `$watch` chạy sau một nhịp và gọi lại `/threshold/optimize`; lời gọi sau huỷ lời gọi
trước bằng `AbortController`, rồi bỏ qua vì thân yêu cầu trùng — đường cong không bao giờ hiện. Kiểm
trùng phải đứng **trước** khi huỷ.

**Chart.js không trả nguyên mảng trong tùy chọn plugin.** Bộ giải tùy chọn coi mảng là tùy chọn "theo
chỉ số". Danh sách vạch ngưỡng vì vậy gắn thẳng lên đối tượng biểu đồ (`chart.$markers`).

**`Space` mở ngăn kéo rồi đóng ngay.** Phím cách kích hoạt nút khi *nhả* phím; ngăn kéo mở lúc nhấn
và trọng tâm đã nhảy sang nút "Đóng", nên lúc nhả phím nút đó bị bấm. Dòng của hàng đợi chỉ mở bằng
`Enter`.

**`EventSource` tự nối lại với URL cũ.** Khi mất kết nối, trình duyệt tự mở lại đúng URL ban đầu, tức
phát lại từ `start` cũ và đếm trùng. Giao diện tự đóng rồi nối lại với `start` bằng giây cuối đã
nhận, và bỏ trùng theo mã giao dịch.

**Mảng điểm 56.746 phần tử không được nằm trong trạng thái Alpine.** Alpine bọc mọi thứ trong
`x-data` bằng Proxy: mỗi lần đọc một phần tử phải qua bẫy, và đối tượng Chart.js bị bọc thì hỏng.
Điểm của tập kiểm thử, biểu đồ và `EventSource` giữ ngoài trạng thái phản ứng.

**Next.js 16.3 xuất sai tên tệp tải trước trên Windows.** Bản `frontend/` sau `next build` báo 404 cho
`/threshold/__next.threshold.__PAGE__.txt` mỗi lần tải trang: `next/dist/export/index.js` đổi `/` thành `.`
trên một đường dẫn mang dấu `\` của Windows, nên tệp nằm ở thư mục con `__next.threshold/__PAGE__.txt`.
Điều hướng vẫn chạy (trình duyệt tải trang đầy đủ) nhưng console không sạch. `npm run build` chạy thêm
`scripts/fix-segment-files.mjs` để làm phẳng tên; build trên Linux không cần.

**React Compiler hiểu nhầm `threshold.current` là `ref.current`.** Trường `current` của `GET /threshold`
trùng tên với thuộc tính của ref, nên luật `react-hooks/preserve-manual-memoization` báo không giữ
được `useMemo`. Tách ra biến `const current = threshold?.current` trước khi dùng trong `useMemo`.

**Các luật mới của `eslint-config-next` 16.** `react-hooks/set-state-in-effect` cấm `setState` đồng bộ
trong effect, `react-hooks/purity` cấm `performance.now()` trong lúc render. Bản `frontend/` sửa theo đúng
khuyến nghị thay vì tắt luật: state dẫn xuất đặt trong lúc render ("điều chỉnh state khi prop đổi"),
effect chỉ lo DOM và gọi mạng, phép đo độ trễ nằm trong trình xử lý sự kiện.

**Thanh trượt của React không nhận `input.value = …`.** Kịch bản thử tự động phải gọi setter gốc
`Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, "value").set` rồi phát sự kiện `input`;
gán thẳng thì React không thấy thay đổi.

**In tiếng Việt ra console Windows.** Lệnh `python -c "print('…')"` trong Git Bash sập với
`UnicodeEncodeError … cp1252`. Đặt `PYTHONIOENCODING=utf-8` trước lệnh.

### 8.6 Bản Next.js (`frontend/`)

Cùng bốn màn hình, viết bằng Next.js 16 + TypeScript + Tailwind 4 + Recharts 3 (docs/07 §13). Cần
Node.js 20.9 trở lên. Dùng cổng 3000 như bản `web/` — không chạy hai bản cùng lúc.

```powershell
cd frontend
npm install                  # lần đầu, khoảng 2 phút
npm run dev                  # phát triển → http://localhost:3000

npm run build                # xuất tĩnh ra frontend/out/ (kèm bước làm phẳng tên tệp, §8.5)
npm run serve                # phục vụ out/ ở cổng 3000
npm run lint; npm run typecheck
```

Số đo trên Chrome 153, cùng kịch bản với §8.3 (21/21 đạt, console sạch cả khi chạy `npm run dev` lẫn bản
xuất tĩnh):

| Mốc | Giá trị |
|---|---|
| Kéo thanh trượt 0,5 → τ\*, 21 vị trí | lâu nhất **39 ms** (bản `web/`: 20 ms) — AC-A3 < 200 ms |
| TC-12 | `frontend/src/lib/threshold.mjs` trùng từng bit với Python trên 20 ngưỡng, như `web/threshold.js` |
| Kịch bản 0,5 → τ\*; chi phí 600 EUR; ràng buộc 150 cảnh báo/ngày | cùng số với §8.3 |
| Tải CSV 10.000 dòng | 2,1 giây — AC-A1 |
| UI-02 | 5 dương + 3 âm, nhãn thật ẩn tới khi bấm `F`, `Esc` trả trọng tâm về dòng cũ |
| Chuyển trang rồi quay lại | vị trí thanh trượt giữ nguyên; luồng phát lại chạy tiếp (767 → 956 giao dịch trong 2 giây ở Hàng đợi) |

## Giai đoạn 9 — Đóng gói và nghiệm thu (T-58…T-62)

**Dựng cả hệ thống bằng Docker Compose và kiểm lại** phần nghiệm thu. Kiến trúc đóng gói ở
[03 §6.3](03-thiet-ke-kien-truc.md); cơ sở dữ liệu, sao lưu ở [06 §7](06-thiet-ke-luu-tru.md); vận
hành ở [10 §4.1](10-van-hanh-tai-lap.md); điều kiện nghiệm thu ở [08 §4](08-ke-hoach-kiem-thu.md)
và [TASKS.md](../TASKS.md).

> **Tiền đề:**
>
> - Docker Desktop đang chạy (Docker Engine 25 trở lên, vì healthcheck dùng `start_interval`).
> - Hiện vật của notebook 08 có trong `models/` và `data/`: `model.joblib`, `explainer.joblib`,
>   `metrics.json`, `threshold.json`, `oof_scores.npz`, `test_set.parquet`, `sample_pool.json`.
>   Compose gắn hai thư mục này **chỉ đọc**; ảnh không chứa hiện vật.
> - Lần build đầu cần Internet (tải ảnh nền, `pip install`, `npm ci`). Sau đó chạy được ngoại tuyến.
> - Không cần Python, Node.js hay `.env` để **chạy** hệ thống. Python chỉ cần cho hai kịch bản tiện
>   ích ở §9.5 (`demo_db.py`, `time_startup.py`); Node.js chỉ cần cho kịch bản kiểm giao diện ở §9.4.

### 9.1 Trình tự

#### 9.1.1 PowerShell

```powershell
# Lần đầu: build hai ảnh (khoảng 4 phút) rồi khởi động
docker compose up --build -d
docker compose ps                     # api phải ở trạng thái (healthy), khoảng 15 giây sau lệnh trên

# Mở giao diện: http://localhost:3000      Tài liệu API: http://localhost:8000/docs

docker compose logs -f api            # xem API khởi động, Ctrl+C để thoát
docker compose down                   # dừng, GIỮ dữ liệu trong volume pgdata
docker compose up -d                  # các lần sau: không build lại, khoảng 11 giây

# Chạy lại notebook 08 → chỉ cần nạp lại hiện vật, không build lại ảnh
docker compose restart api

# Dùng bản giao diện HTML + Alpine.js thay cho bản Next.js (build không cần npm)
$env:WEB_UI = "web"; docker compose up --build -d; Remove-Item Env:WEB_UI
```

#### 9.1.2 Git Bash

```bash
docker compose up --build -d
docker compose ps
docker compose down
WEB_UI=web docker compose up --build -d
```

#### 9.1.3 Cổng đã bị chiếm

Ba cổng trên máy chủ đọc từ biến môi trường hoặc `.env` (mẫu ở `.env.example`):

| Biến | Mặc định | Dùng cho |
|---|---|---|
| `WEB_HOST_PORT` | 3000 | giao diện; nginx chuyển tiếp `/api/` sang dịch vụ `api` nên trình duyệt chỉ cần cổng này |
| `API_HOST_PORT` | 8000 | `/docs`, gọi API trực tiếp, kịch bản `demo_db.py` |
| `POSTGRES_HOST_PORT` | 5432 | psql, pytest và uvicorn chạy trên máy (docs/10 §4.2) |

Máy phát triển của dự án có PostgreSQL 17 chiếm 5432 nên `.env` đặt `POSTGRES_HOST_PORT=5433`.

### 9.2 Tệp của giai đoạn này

| Tệp | Nội dung |
|---|---|
| `docker-compose.yml` | 3 dịch vụ: `db` (healthcheck `pg_isready` qua TCP) → `api` (`depends_on: service_healthy`, gắn `models/` và `data/` chỉ đọc, healthcheck `/health`) → `web` |
| `api/Dockerfile` | `python:3.12-slim`, cài `api/requirements*.txt`, biên dịch sẵn `.pyc`, chạy bằng người dùng không phải root |
| `api/entrypoint.py` | `alembic upgrade head` (thử lại tối đa 30 lần khi db chưa nhận kết nối) rồi `uvicorn`, một worker |
| `api/requirements.txt`, `api/requirements-nodeps.txt` | thư viện lúc chạy, ghim **đúng** phiên bản lúc xuất hiện vật; xgboost cài `--no-deps` |
| `frontend/Dockerfile` | bản Next.js (mặc định): `node:22-alpine` build ra `out/`, rồi `nginx:stable-alpine` phục vụ |
| `web/Dockerfile` | bản HTML + Alpine.js (`WEB_UI=web`): chép thẳng `web/` vào nginx, thay `config.js` |
| `deploy/nginx.conf` | dùng chung cho hai bản: tệp tĩnh + chuyển tiếp `/api/` (SSE không đệm, lỗi của nginx trả JSON) |
| `.dockerignore` | danh sách trắng: chỉ `api/`, `src/`, `alembic.ini`, `deploy/`, `web/`, `frontend/` vào ngữ cảnh build |
| `scripts/time_startup.py` | bấm giờ NFR-04: cold (`down -v`) và warm (`down`), tới khi `/health` trả 200 |
| `scripts/demo_db.py` | `seed`, `dump`, `restore`, `reset`, `status` — dữ liệu demo cho buổi bảo vệ (T-60) |
| `scripts/ui/flow.js`, `scripts/ui/keyboard.js` | kiểm giao diện trên Chrome thật: luồng chính 21 bước, bàn phím 24 bước (§9.4) |
| `tests/test_packaging.py` | 12 ca: ghim phiên bản khớp `metrics.json`, entrypoint thử lại, cấu trúc compose, `.dockerignore` |
| `.env.example` | thêm `API_HOST_PORT`, `WEB_HOST_PORT`, `WEB_UI` |

### 9.3 Số đối chiếu (máy 12 lõi, Docker Desktop 29.5 trên WSL2, 2026-10-02)

#### 9.3.1 Khởi động — NFR-04, AC-A8 (T-59)

Đo trên **bản sao sạch** của repo: chỉ các tệp mà git theo dõi, cộng đúng 7 hiện vật; không `.venv`,
không `node_modules`, không `.env`; dự án Compose riêng nên ảnh và volume mới hoàn toàn. Đo bằng
`scripts/time_startup.py`, từ lúc gọi `docker compose up -d` tới khi `/api/v1/health` trả 200.

| Lần | db khỏe | giao diện 200 | `/health` 200 | Giới hạn |
|---|---|---|---|---|
| cold 1 (lần đầu tiên chạy ảnh mới) | 7,4 s | 7,4 s | **14,9 s** | 45 s |
| cold 2 / 3 | 6,1 / 7,6 s | | 13,7 / 14,7 s | 45 s |
| warm 1 / 2 / 3 | 4,8 / 4,6 / 4,1 s | | **11,5** / 11,2 / 11,1 s | 15 s |

Dòng thời gian một lần warm (11,3 s): Compose dựng mạng và container ~1,5 s → PostgreSQL sẵn sàng → chờ
healthcheck và tạo container `api` ~1,6 s → Python, alembic 1,7 s → import thư viện 3,7 s → nạp và kiểm
hiện vật 2,3 s. Cold thêm khoảng 3 giây cho `initdb` và hai migration.

| Mốc khác | Giá trị |
|---|---|
| Build sạch cả hai ảnh (`--no-cache --pull`, song song) | **244 s**; riêng `api` 234 s, riêng `web` 110 s |
| Kích thước ảnh | `api` 1,46 GB (scipy, numba, pyarrow, shap…), `web` 95 MB |
| `docker compose restart` → `/health` 200 | 10,0 s; chỉ `restart api`: 9,5 s |
| Điểm chấm trong container (Linux) so với điểm của notebook (Windows) | 35/56.746 giao dịch lệch **1 ulp float32** (lớn nhất 1,5·10⁻¹¹, ở vùng điểm ~10⁻⁶); **0** quyết định bị lật ở τ\* và ở 0,5 |
| `GET /metrics?section=test_scores` qua nginx | 1,50 MB → 0,63 MB nhờ gzip |

#### 9.3.2 Dữ liệu demo — T-60

| Bước | Kết quả |
|---|---|
| `demo_db.py seed` | tải lên 10.000 dòng của tập kiểm thử (2,1 giây), nạp 179 mẫu (`fraud_easy`, `legit_hard`, `legit_easy`), thẩm định 5 gian lận và 3 cảnh báo giả điểm cao nhất; hàng đợi 79 giao dịch |
| `demo_db.py dump` | `backup/fraud-demo.dump` 3,0 MB: 10.179 giao dịch, 8 thẩm định, 3 khóa `settings` |
| `restore` sau khi xóa dữ liệu | **2,4 s** |
| `restore` sau `down -v` (mất cả volume) | **2,3 s**; API phục vụ tiếp không cần khởi động lại; mã giao dịch mới nối tiếp (TX-10180), không trùng |

### 9.4 Nghiệm thu AC-A1…AC-A10 trên hệ thống đóng gói (T-61)

Mọi phép kiểm đi qua đúng đường của người dùng: Chrome 154 → nginx (cổng 3000) → `api` → PostgreSQL,
trên bản sao sạch ở §9.3.1. Bảng đầy đủ ở [08 §4.4](08-ke-hoach-kiem-thu.md).

| Mã | Kết quả |
|---|---|
| AC-A1 | tải CSV 10.000 dòng qua nút "Tải CSV": **3,1 giây** (< 30) |
| AC-A2 | 25 dòng trang đầu giảm dần |
| AC-A3 | lâu nhất **43–131 ms** qua 4 lượt, mỗi lượt 21 lần kéo 0,5 → τ\* (< 200); 131 ms ở lượt đầu, trùng lúc đang khôi phục dữ liệu |
| AC-A4 | thác nước 5 dương + 3 âm; chú thích PCA có mặt; nhãn thật ẩn tới khi quyết định |
| AC-A5 | chi phí bỏ lọt 600 EUR: ngưỡng tối ưu 0,02317 → 0,002262 |
| AC-A6 | luồng SSE qua nginx **185 giây**: 1.605 giao dịch, 14 cảnh báo, 0 lỗi, khoảng lặng dài nhất 1,0 s |
| AC-A7 | 2 kết luận còn nguyên sau `docker compose restart` (so cả qua API lẫn `SELECT … FROM reviews`) |
| AC-A8 | bản sao sạch, một lệnh `docker compose up --build` — §9.3.1. Giới hạn: cùng một máy vật lý, xem §9.6 |
| AC-A9 | `keyboard.js` **24/24**; Lighthouse Accessibility **100** và Best Practices **100** trên cả 4 màn hình |
| AC-A10 | 13 đầu vào sai qua nginx đều trả đúng mã và mã lỗi JSON (422, 400, 404, 405, 413); container không khởi động lại |

Kiểm lại trước buổi bảo vệ:

```bash
cd scripts/ui && npm install                 # lần đầu; cần Node.js và Chrome
node flow.js                                 # 21 bước: AC-A1…A3, A5, UI-02, UI-05, UI-04 — cần data/giao-dich-10000.csv
node keyboard.js                             # 24 bước: AC-A9
npx lighthouse http://localhost:3000/ --only-categories=accessibility --chrome-flags="--headless=new" --view
```

Hai kịch bản **ghi** vào cơ sở dữ liệu (nạp mẫu, tải CSV, thẩm định). Chạy trên dữ liệu demo rồi
`python scripts/demo_db.py restore`, hoặc chạy sau `reset`. Tệp CSV tạo theo
§8.1.3, ghi vào `data/giao-dich-10000.csv`. Chrome ở
chỗ khác thì đặt `CHROME_PATH`; hệ thống ở cổng khác thì đặt `BASE`.

### 9.5 Dữ liệu demo và sao lưu (T-60)

```bash
python scripts/demo_db.py seed       # cơ sở dữ liệu phải rỗng; --rows để đổi số dòng tải lên
python scripts/demo_db.py dump       # → backup/fraud-demo.dump (backup/ không vào git)
python scripts/demo_db.py restore    # khôi phục, vài giây; tải lại trang là thấy
python scripts/demo_db.py reset      # xóa giao dịch, thẩm định, ngưỡng người dùng đặt
python scripts/demo_db.py status     # đếm dòng
```

`seed` gọi API ở `http://localhost:8000/api/v1` (đổi bằng `--api`) và **để dành** nhóm "Gian lận khó"
cho đoạn trình diễn trực tiếp (§8.4): nạp nhóm đó, mở
một dòng, bấm `A`, giao diện báo kết luận sai. 5 gian lận và 3 cảnh báo giả điểm cao nhất của hàng
đợi đã thẩm định sẵn theo nhãn thật, có ghi chú, để bộ lọc "Đã thẩm định" có nội dung.

Máy không có Python — chạy tay đúng các lệnh mà kịch bản gọi (in ra khi thêm `--dry-run`), được cả trong
PowerShell lẫn Git Bash:

```bash
# Sao lưu
docker compose exec -T db pg_dump -U fraud -d fraud -Fc -f /tmp/fraud-demo.dump
docker compose cp db:/tmp/fraud-demo.dump backup/fraud-demo.dump

# Khôi phục
docker compose cp backup/fraud-demo.dump db:/tmp/fraud-demo.dump
docker compose exec -T db pg_restore -U fraud -d fraud --clean --if-exists --single-transaction --no-owner /tmp/fraud-demo.dump
```

Bấm giờ khởi động trên máy khác (thời gian build không tính):

```bash
python scripts/time_startup.py --runs 3              # warm: GIỮ dữ liệu
python scripts/time_startup.py --cold --yes          # cold: XÓA volume pgdata — sao lưu trước
```

### 9.6 Những cái bẫy đã gặp hoặc đã chặn trước

**Ảnh `python` chính thức không có `.pyc` của thư viện chuẩn** *(đã gặp)*. Ảnh xóa chúng cho nhẹ, còn
`PYTHONDONTWRITEBYTECODE=1` cấm ghi lại, nên mỗi lần container khởi động, `argparse`, `tarfile`, `pydoc`…
đều biên dịch lại từ mã nguồn: `import api.main` mất 6,0 s, warm start 14,4 s — sát giới hạn 15 s.
`python -m compileall` lúc build đưa import về 4,5 s và warm start về 11,1–12,3 s.

**Điểm chấm trên Linux lệch 1 ulp so với Windows** *(đã gặp)*. Cùng `model.joblib`, cùng xgboost 3.4.1,
35/56.746 giao dịch của tập kiểm thử cho điểm khác ở bit cuối của float32. Không quyết định nào đổi,
`api/loader.py` vẫn qua bước chấm lại (dung sai 10⁻⁶). Hệ quả: "trùng từng bit" của T-35 chỉ đúng trên
cùng hệ điều hành; giữa Windows và container chỉ "trùng tới 1 ulp" ([10 §6](10-van-hanh-tai-lap.md)).

**`requirements.txt` gốc không khai báo `pyarrow`** *(đã gặp, qua `tests/test_packaging.py`)*. Notebook
03, 08 và API đọc/ghi parquet nhưng `pyarrow` chỉ có mặt nhờ `streamlit` kéo theo. Đã thêm tường minh.

**`pg_isready` qua socket Unix báo khỏe quá sớm** *(chặn trước)*. Lúc `initdb`, ảnh postgres chạy một
máy chủ tạm chỉ nghe socket Unix, chạy xong thì tắt rồi khởi động lại thật. Kiểm qua socket thì báo
khỏe đúng lúc máy chủ sắp tắt và migration của `api` hỏng ở lần chạy đầu. Healthcheck dùng
`-h 127.0.0.1` (TCP); `entrypoint.py` vẫn thử lại thêm cho chắc.

**xgboost trên Linux kéo theo thư viện GPU** *(chặn trước)*. Gói `xgboost` khai báo `nvidia-nccl-cu12`
(vài trăm MB) trên Linux x86_64. Cài bằng `--no-deps`; không dùng `xgboost-cpu` vì nó mang tên phân
phối khác và `api/loader.py` sẽ cảnh báo lệch phiên bản.

**`>` của PowerShell 5.1 làm hỏng bản dump** *(chặn trước)*. Lệnh cũ ở 06 §7.4
(`pg_dump … > backup/fraud.dump`) chạy được trong Git Bash, nhưng PowerShell 5.1 coi đầu ra của chương
trình là văn bản và mã hóa lại. `pg_dump -f` ghi tệp **trong** container rồi `docker compose cp` chép ra.

**Chuyển hướng của nginx làm rơi cổng 3000** *(chặn trước)*. Mở `/threshold` không có dấu `/` cuối,
nginx chuyển tới `/threshold/` bằng địa chỉ tuyệt đối dựng theo cổng nó nghe trong container (80), tức
`http://localhost/threshold/`, và trình duyệt rời cổng 3000. `absolute_redirect off` cho
`Location: /threshold/` (đã kiểm bằng `curl -I`).

**`error_page` của nginx không kế thừa khi `location` tự khai báo** *(gặp khi viết cấu hình)*. Đặt
`error_page 413` ở mức `server` thì `location /api/` — nơi có `error_page 502 504` riêng — không nhận
nó, tệp quá lớn trả trang HTML. Hai dòng `error_page` phải nằm cùng trong `location /api/`.

**Kiểm bàn phím: hàng đợi chỉ là MỘT điểm dừng Tab** *(đã gặp khi viết kịch bản)*. Bảng dùng roving
tabindex: dòng đang chọn có `tabindex=0`, các dòng khác `-1`, `↓`/`↑` để đi (07 §10, mẫu lưới của
WAI-ARIA). Chỉ Tab và Enter thì mở được đúng một dòng. AC-A9 vì vậy hiểu là "Tab, Enter, và phím mũi
tên trong các nhóm mà WAI-ARIA quy định" — bảng, thanh trượt, nhóm radio. Hai lỗi kịch bản khác: nút
mang `<kbd>A</kbd>` nên chữ là "A Đánh dấu báo động sai", không bắt đầu bằng "Báo động sai"; và chữ
của dòng bị cắt 50 ký tự làm mất cột trạng thái ở cuối.

**`flow.js` trượt một bước ngay sau `seed`** *(đã gặp)*. Bước "nhãn thật ẩn trước khi quyết định" mở
dòng thứ 3 của hàng đợi; sau `seed` dòng đó đã được thẩm định nên ngăn kéo hiện kết luận cũ kèm nhãn
thật — đúng thiết kế. Chạy `flow.js` sau `reset` thì 21/21.

**Thử "máy sạch" trên chính máy phát triển** *(giới hạn của T-59)*. Bản sao sạch loại được `.venv`,
`node_modules`, `.env`, ảnh và volume cũ, nhưng không loại được: ảnh nền (`python`, `node`, `nginx`,
`postgres`) đã có trong bộ đệm của Docker, và PostgreSQL 17 của máy chiếm 5432 nên lần thử đặt
`POSTGRES_HOST_PORT=5434`. Thư viện Python và gói npm thì có tải lại từ đầu (`--no-cache`), nên 244
giây build đã gồm phần đó; trên máy mới chỉ cộng thêm thời gian tải bốn ảnh nền. Lượt thử trên một máy
thật khác vẫn nên làm trước buổi bảo vệ.

### 9.7 Kiểm tay trước buổi bảo vệ

- [ ] Khởi động một lần **trước** buổi bảo vệ (`docker compose up -d`), để lần mở trước hội đồng là warm.
- [ ] `python scripts/demo_db.py restore` — hàng đợi 79 giao dịch, 8 đã thẩm định.
- [ ] Mở `http://localhost:3000`, đi qua bốn màn hình, F12 không có dòng đỏ.
- [ ] Rút dây mạng / tắt Wi-Fi: giao diện vẫn chạy (NFR-08).
- [ ] Mang theo `backup/fraud-demo.dump` và một bản hiện vật (`models/`, `data/`) trên USB.
