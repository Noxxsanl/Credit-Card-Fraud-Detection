# Lệnh chạy giai đoạn 8 — Giao diện (T-51…T-57)

Sổ lệnh để **chạy và kiểm lại** giao diện web: bốn màn hình trong `web/`, gọi API của giai đoạn 7.
Mọi lệnh chạy từ thư mục gốc của repo. Thiết kế ở [07](07-thiet-ke-giao-dien.md) (§12 ghi những
điểm bản thi hành cụ thể hơn thiết kế ban đầu); hợp đồng API ở [05](05-thiet-ke-api.md); điều kiện
nghiệm thu ở [TASKS.md](../TASKS.md).

> **Tiền đề:**
>
> - API của giai đoạn 7 chạy được ([lenh-chay-giai-doan-7.md](lenh-chay-giai-doan-7.md)): Docker
>   Desktop đang chạy, `.env` đã có, hiện vật của notebook 08 trong `models/` và `data/`.
> - Không cần Node.js, không cần `npm install`: giao diện là tệp tĩnh, Alpine.js và Chart.js nằm sẵn
>   trong `web/vendor/`. Node.js chỉ cần cho ca kiểm thử TC-12; máy không có Node thì ca đó tự bỏ qua.
> - Trang phải mở qua `http://localhost:3000` hoặc `http://127.0.0.1:3000` — hai origin CORS mà API
>   cho phép. Mở thẳng `web/index.html` (`file://`) sẽ bị trình duyệt chặn gọi API.

## 1. Trình tự

### 1.1 PowerShell

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

### 1.2 Git Bash

```bash
docker compose up -d db
./.venv/Scripts/python.exe -m alembic upgrade head
./.venv/Scripts/python.exe -m uvicorn api.main:app --port 8000          # cửa sổ 1
./.venv/Scripts/python.exe -m http.server 3000 --directory web          # cửa sổ 2
./.venv/Scripts/python.exe -m pytest tests/test_threshold_parity.py
```

### 1.3 Nạp dữ liệu demo

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

## 2. Tệp của giai đoạn này

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
| `src/evaluate.py` | `thin_curve()` — sửa cách rút mẫu đường PR/ROC (§5, bẫy đầu tiên) |
| `tests/test_artifacts.py` | 2 ca mới canh lỗi rút mẫu đường PR |

## 3. Số đối chiếu (Chrome 153 không giao diện, máy 12 lõi, 2026-09-28)

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

## 4. Kiểm tay trước buổi bảo vệ

- [ ] Mở cả bốn màn hình, console (F12) không có dòng đỏ.
- [ ] Màn hình Ngưỡng: chọn "Mặc định 0,5" rồi kéo thanh trượt về dấu ▲ τ\* — ô "Gian lận bắt được"
      tăng, ô "Chi phí mỗi ngày" giảm, dòng chênh lệch đổi màu.
- [ ] Gõ chi phí bỏ lọt 600 — vạch cam "tối ưu" dời sang trái, xuất hiện phương án "Tối ưu với tham số
      chi phí đang nhập". Đưa lại 122,21.
- [ ] Hàng đợi: nạp nhóm "Gian lận khó" từ thư viện mẫu, mở một dòng, bấm `A` (báo động sai) — nhãn
      thật hiện "Gian lận — ✗ khác kết luận của bạn". Đây là ví dụ FR-44 đáng trình diễn.
- [ ] Phát lại ở 600× khoảng một phút, bấm một cảnh báo để mở chi tiết.
- [ ] Rút dây mạng: giao diện vẫn chạy (không có tài nguyên nào tải từ Internet — NFR-08).

## 5. Những cái bẫy đã gặp

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

## 6. Bản Next.js (`frontend/`)

Cùng bốn màn hình, viết bằng Next.js 16 + TypeScript + Tailwind 4 + Recharts 3 (docs/07 §13). Cần
Node.js 20.9 trở lên. Dùng cổng 3000 như bản `web/` — không chạy hai bản cùng lúc.

```powershell
cd frontend
npm install                  # lần đầu, khoảng 2 phút
npm run dev                  # phát triển → http://localhost:3000

npm run build                # xuất tĩnh ra frontend/out/ (kèm bước làm phẳng tên tệp, §5)
npm run serve                # phục vụ out/ ở cổng 3000
npm run lint; npm run typecheck
```

Số đo trên Chrome 153, cùng kịch bản với §3 (21/21 đạt, console sạch cả khi chạy `npm run dev` lẫn bản
xuất tĩnh):

| Mốc | Giá trị |
|---|---|
| Kéo thanh trượt 0,5 → τ\*, 21 vị trí | lâu nhất **39 ms** (bản `web/`: 20 ms) — AC-A3 < 200 ms |
| TC-12 | `frontend/src/lib/threshold.mjs` trùng từng bit với Python trên 20 ngưỡng, như `web/threshold.js` |
| Kịch bản 0,5 → τ\*; chi phí 600 EUR; ràng buộc 150 cảnh báo/ngày | cùng số với §3 |
| Tải CSV 10.000 dòng | 2,1 giây — AC-A1 |
| UI-02 | 5 dương + 3 âm, nhãn thật ẩn tới khi bấm `F`, `Esc` trả trọng tâm về dòng cũ |
| Chuyển trang rồi quay lại | vị trí thanh trượt giữ nguyên; luồng phát lại chạy tiếp (767 → 956 giao dịch trong 2 giây ở Hàng đợi) |
