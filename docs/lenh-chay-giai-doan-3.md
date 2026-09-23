# Lệnh chạy giai đoạn 3 — Chiến lược mất cân bằng (T-20…T-23)

Sổ lệnh để **chạy lại** lưới 20 tổ hợp của giai đoạn 3 từ đầu. Mọi lệnh chạy từ thư
mục gốc của repo và gọi thẳng trình thông dịch trong `.venv` (không cần `activate`).
Mục 1 có cả bản PowerShell lẫn bản Git Bash; các mục sau viết theo PowerShell, đổi
sang Git Bash bằng bảng quy đổi ở §1.3.

Tài liệu này chỉ ghi *lệnh gì, ra cái gì, mất bao lâu*. Phần **vì sao** thiết kế lưới
như vậy nằm ở [04 §3](04-thiet-ke-mo-hinh-ml.md); phần điều kiện nghiệm thu từng việc
nằm ở [TASKS.md](../TASKS.md).

> **Dữ liệu tiền đề:** `data/creditcard.csv` phải có sẵn và đúng toàn vẹn (T-10).
> Nếu máy sạch, làm [10 §2](10-van-hanh-tai-lap.md) trước.

## 1. Trình tự đầy đủ

Bốn lệnh, theo đúng thứ tự này.

### 1.1 PowerShell

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

### 1.2 Git Bash

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

### 1.3 Bảng quy đổi

Để tự chuyển các lệnh ở những mục sau sang Git Bash:

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

## 2. Bảng lệnh của `run_grid.py`

Toàn bộ cờ của [scripts/run_grid.py](../scripts/run_grid.py):

| Lệnh | Tác dụng | Thời gian |
|---|---|---|
| `run_grid.py` | Chạy đủ 20 tổ hợp, nối tiếp điểm lưu nếu có | ~45 phút |
| `run_grid.py --smoke` | Chạy thử trên 12.000 dòng, ghi ra tệp `_smoke` riêng | ~1 phút |
| `run_grid.py --status` | Chỉ in tiến độ điểm lưu rồi thoát — **an toàn khi lưới đang chạy** | tức thì |
| `run_grid.py --fast` | Hạ `n_estimators` xuống 100 (phương án dự phòng của T-20) | ~15 phút |
| `run_grid.py --fresh` | Bỏ điểm lưu, huấn luyện lại toàn bộ từ đầu | ~45 phút |

### Theo dõi tiến độ ở cửa sổ khác

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

### Ngắt giữa chừng

Bấm `Ctrl+C` rồi chạy lại **đúng lệnh cũ** — nó đọc `reports/grid_results.csv` và
bỏ qua các tổ hợp đã xong. Chỉ mất phần tổ hợp đang dở dang.

Muốn thật sự làm lại từ trắng thì mới dùng `--fresh`.

## 3. Hiện vật sinh ra

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

## 4. Kết quả của lần chạy thật (2026-09-16)

Để đối chiếu khi chạy lại — cùng `RANDOM_STATE=42` thì các con số dưới đây phải lặp lại:

| Mốc | Giá trị |
|---|---|
| Tập huấn luyện | 226.980 dòng, 378 gian lận (0,1665%) |
| Chế độ | đầy đủ theo [04 §3.2](04-thiet-ke-mo-hinh-ml.md) — không dùng `--fast` |
| Tổng thời gian | **45,4 phút** (tổng `total_seconds` của 20 dòng) |
| PR-AUC cao nhất | **0,8549** — `xgboost` + `class_weight` |
| Riêng 4 tổ hợp `smote_tomek` | 25,7 phút, so với 7,6 phút của `smote` → **đắt hơn 18,2 phút** mà kết quả giống hệt |

Dòng cuối là phát hiện 3 của giai đoạn 3: bước Tomek xoá đúng 0 cặp ở tỷ lệ 1:600.
Xem ghi chú giai đoạn 3 trong [TASKS.md](../TASKS.md).

### Điều kiện dừng của T-23

Script tự kiểm tra và in một trong hai dòng ở cuối:

```
T-23: PR-AUC cao nhất 0.8549, dưới ngưỡng báo động 0,95. Dải kỳ vọng 04 §3.4 là 0,80–0,87.
```

Nếu thay vào đó là `!! BÁO ĐỘNG (T-23)` thì **dừng lại**, đừng chạy notebook, mở danh
sách rà rò rỉ ở [08 §3.1](08-ke-hoach-kiem-thu.md).

## 5. Kiểm thử của giai đoạn này

```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_grid.py -q     # 19 ca, chốt hợp đồng của run_grid
.\.venv\Scripts\python.exe -m pytest -q                        # toàn bộ: 127 xanh, 1 skip
```

Ca skip duy nhất là TC-12 (đối chiếu ngưỡng JavaScript ↔ Python), thuộc T-54 ở giai
đoạn 8 — đúng thiết kế, không phải lỗi.

## 6. Một cái bẫy đã gặp

Console Windows mặc định là cp1252: in tiếng Việt ra sẽ ném `UnicodeEncodeError` và
**giết cả lần chạy 40 phút ở giữa chừng**. `run_grid.py` đã tự `reconfigure` stdout sang
UTF-8 ngay đầu tệp nên không còn gặp. Nhưng nếu tự viết script khác gọi `run_grid`, nhớ
làm y như vậy, hoặc đặt `$env:PYTHONIOENCODING = "utf-8"` trước khi chạy.

Cùng lý do đó, `reconfigure` dùng `line_buffering=True` — nếu không, đầu ra bị chuyển
hướng vào `grid_run.log` sẽ bị đệm theo khối và người chạy không thấy gì suốt nhiều phút.
