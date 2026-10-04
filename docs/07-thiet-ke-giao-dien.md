# 07 — Thiết kế giao diện

**Tên sản phẩm:** Bảng điều khiển thẩm định gian lận (Fraud Review Console)

Ứng dụng web một trang, bốn màn hình, điều hướng bằng thanh bên. Không đăng nhập —
demo cục bộ một người dùng.

## 1. Bản đồ màn hình

| Mã | Màn hình | Vai chính | Yêu cầu | Ưu tiên cắt giảm |
|---|---|---|---|---|
| UI-01 | Hàng đợi thẩm định | U-ANALYST | FR-40 | 1 — giữ bằng mọi giá |
| UI-03 | Cấu hình ngưỡng | U-MANAGER | FR-30…FR-34 | 2 — giữ bằng mọi giá |
| UI-02 | Chi tiết giao dịch | U-ANALYST | FR-24, FR-41 | 3 |
| UI-05 | Chế độ phát lại | U-MANAGER | FR-43 | 4 |
| UI-04 | Hiệu năng mô hình | U-GRADER | FR-42 | 5 — có thể thay tạm bằng ảnh từ notebook |

Thứ tự ở cột cuối là thứ tự **giữ lại** khi thiếu thời gian. UI-03 phải giữ vì đó
là nơi thể hiện luận điểm chính của đồ án.

## 2. Quyết định thiết kế xuyên suốt

**UI-D1 — Tính chỉ số ngưỡng ở phía máy khách.** Lúc tải trang, giao diện gọi
`GET /api/v1/metrics?section=test_scores` một lần để lấy hai mảng `y_true`,
`y_score` (khoảng 57.000 phần tử, vài MB). Mọi lần kéo thanh trượt sau đó tính
lại TP/FP/FN ngay trong trình duyệt, không gọi mạng. Đây là cách chắc chắn đạt
NFR-03 (< 200 ms) kể cả khi máy chủ chậm.

`API-11 /threshold/preview` vẫn tồn tại: nó là bản cài đặt chuẩn ở phía máy chủ,
dùng cho kiểm thử đối chiếu (TC-12 — hai bên phải cho cùng kết quả) và cho phương
án A (Streamlit) vốn không có logic phía máy khách.

**UI-D2 — Không bịa ý nghĩa nghiệp vụ cho V1–V28.** Mọi nơi hiển thị tên đặc
trưng đều kèm chú thích cố định: đây là thành phần chính sau PCA, không diễn giải
được. Gán nhãn kiểu "V14 = giao dịch ở nước ngoài" là sai sự thật và sẽ bị trừ
điểm.

**UI-D3 — Điểm rủi ro hiển thị dạng phần trăm, quyết định hiển thị bằng chữ.**
Người dùng không đọc `0.0473`; họ đọc "4,7% — vượt ngưỡng, đề xuất chặn".

**UI-D4 — Ngưỡng luôn hiển thị ở thanh trên cùng của mọi màn hình.** Vì mọi con
số trong ứng dụng đều phụ thuộc nó, giấu nó đi sẽ khiến người xem hiểu nhầm.

## 3. UI-01 — Hàng đợi thẩm định

Màn hình mặc định khi mở ứng dụng.

```
┌────────────────────────────────────────────────────────────────┐
│  Hàng đợi thẩm định                    Ngưỡng: 0,047  [Đổi]    │
│  ────────────────────────────────────────────────────────────  │
│  187 cảnh báo · ước tính bắt được 84% gian lận                 │
│                                                                 │
│  [Tải CSV]  [Chọn mẫu có sẵn]  [▶ Phát lại ngày 2]             │
│  ────────────────────────────────────────────────────────────  │
│   Rủi ro │ Mã GD   │ Số tiền   │ Giờ   │ Trạng thái  │        │
│  ─────────────────────────────────────────────────────────     │
│   ●99,7% │ TX-8841 │  1.809,68 │ 02:14 │ Chờ xử lý   │  ▸     │
│   ●98,2% │ TX-1207 │      0,76 │ 03:41 │ Chờ xử lý   │  ▸     │
│   ●94,1% │ TX-5563 │    529,00 │ 02:58 │ Đã chặn     │  ▸     │
│   ○71,3% │ TX-9930 │     14,20 │ 11:02 │ Báo động sai│  ▸     │
│   ○68,8% │ TX-4415 │    212,50 │ 22:37 │ Chờ xử lý   │  ▸     │
│  ─────────────────────────────────────────────────────────     │
│                                              ‹ 1 2 3 … 8 ›     │
└────────────────────────────────────────────────────────────────┘
```

| Hạng mục | Đặc tả |
|---|---|
| Nguồn dữ liệu | `GET /transactions?min_score={τ}&sort=-risk_score&page=…` |
| Mã màu | Đỏ ≥ 90%, cam 60–90%, vàng < 60% — kèm ký hiệu hình dạng cho người mù màu |
| Bộ lọc | Trạng thái: chờ xử lý / đã chặn / báo động sai / tất cả |
| Sắp xếp | Theo điểm rủi ro hoặc số tiền, hai chiều |
| Phân trang | 25 dòng mỗi trang |
| Đổi ngưỡng | Cập nhật danh sách ngay, không tải lại trang |
| Trạng thái rỗng | Khi chưa có dữ liệu: hướng dẫn ba cách nạp (tải CSV, chọn mẫu, phát lại) |
| Trạng thái đang tải | Khung xương (skeleton) 5 dòng, không dùng spinner toàn trang |
| Bàn phím | `↑`/`↓` di chuyển giữa các dòng, `Enter` mở chi tiết, `Esc` đóng |

Dòng tóm tắt "187 cảnh báo · ước tính bắt được 84% gian lận" lấy từ phép tính
UI-D1; nó là thứ khiến người xem hiểu ngay ngưỡng đang đánh đổi cái gì.

## 4. UI-02 — Chi tiết giao dịch

Ngăn kéo trượt từ bên phải, mở khi bấm một dòng ở UI-01.

```
┌────────────────────────────────────────────────────────────────┐
│  TX-8841                                              [Đóng]   │
│  ════════════════════════════════════════════════════════════  │
│      Điểm rủi ro                                                │
│         99,7%          Vượt ngưỡng 0,047 → đề xuất CHẶN        │
│      ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓░               │
│                                                                 │
│      Số tiền   1.809,68        Giờ giao dịch   02:14           │
│      (cao hơn 97% giao dịch)                                    │
│  ────────────────────────────────────────────────────────────  │
│  Yếu tố đẩy điểm rủi ro lên                                    │
│    V14   -8,42  ████████████████████████  +0,31                │
│    V17   -6,17  ██████████████████        +0,24                │
│    V12   -5,88  ███████████████           +0,19                │
│    V10   -4,03  ██████████                +0,12                │
│    Amount 1.809,68 ████                   +0,04                │
│  ────────────────────────────────────────────────────────────  │
│  Yếu tố kéo điểm rủi ro xuống                                  │
│    V4     1,12          ████                      -0,05        │
│  ────────────────────────────────────────────────────────────  │
│  ⓘ V1–V28 là thành phần chính sau PCA. Chúng cho biết          │
│    đặc trưng nào ảnh hưởng tới dự đoán, nhưng không            │
│    diễn giải được thành nguyên nhân nghiệp vụ.                 │
│                                                                 │
│   [Xác nhận gian lận]        [Đánh dấu báo động sai]           │
└────────────────────────────────────────────────────────────────┘
```

| Hạng mục | Đặc tả |
|---|---|
| Nguồn dữ liệu | `GET /transactions/{id}` rồi `POST /explain` |
| Số yếu tố | 5 dương, 3 âm — theo `|SHAP|` giảm dần (FR-24) |
| Chú thích PCA | Cố định, không thu gọn được (UI-D2) |
| So sánh phân bố | Phân vị của `Amount` so với toàn tập (US-04) |
| Hai nút hành động | `POST /reviews`, cập nhật trạng thái ở UI-01 ngay |
| Nhãn thật | Nếu có `true_label`, **chỉ hiện sau khi** người dùng đã bấm quyết định — để tự đối chiếu mà không bị mồi |
| Đang tải SHAP | Hiện khung xương ở khu vực yếu tố; phần điểm rủi ro hiện ngay vì không phải chờ |
| Bàn phím | `F` xác nhận gian lận, `A` báo động sai, `Esc` đóng |

Việc ẩn nhãn thật cho tới khi người dùng quyết định là chi tiết nhỏ nhưng biến
demo thành một thí nghiệm thật: người thẩm định có làm tốt hơn mô hình không
(FR-44).

## 5. UI-03 — Cấu hình ngưỡng

Màn hình có giá trị học thuật cao nhất.

```
┌────────────────────────────────────────────────────────────────┐
│  Cấu hình ngưỡng quyết định                                    │
│  ════════════════════════════════════════════════════════════  │
│  Ngưỡng   ├────────●──────────────────────────────┤   0,047    │
│         0,001                                    0,999          │
│                                                                 │
│  ┌──────────────┬──────────────┬──────────────┬─────────────┐  │
│  │ Cảnh báo/ngày│ Bắt được     │ Precision    │ Chi phí/ngày│  │
│  │      94      │  83 / 98     │    0,44      │  3.312 EUR  │  │
│  └──────────────┴──────────────┴──────────────┴─────────────┘  │
│  ────────────────────────────────────────────────────────────  │
│  Chi phí bỏ lọt 1 gian lận  [ 122,21 ] EUR                     │
│  Chi phí 1 lần thẩm định    [   5,00 ] EUR                     │
│                                                                 │
│  Đường cong tổng chi phí theo ngưỡng                            │
│    ▲ chi phí                                                    │
│    │╲                                          ╱                │
│    │ ╲                                    ╱                     │
│    │  ╲__________            ______╱                            │
│    │             ╲______╱                                       │
│    │              ▲ 0,047 — cực tiểu                            │
│    └──────────────────────────────────────────► ngưỡng          │
│                                                                 │
│  Ngưỡng gợi ý sẵn:                                              │
│   ( ) 0,500  Mặc định — không có cơ sở cho dữ liệu mất cân bằng │
│   (•) 0,047  Cực tiểu chi phí kỳ vọng                           │
│   ( ) 0,214  F1 lớn nhất                                        │
│   ( ) 0,009  Ràng buộc Recall ≥ 90%                             │
│   ( ) 0,061  Ngân sách 200 lượt thẩm định                       │
│                                                                 │
│                                     [Áp dụng ngưỡng này]        │
└────────────────────────────────────────────────────────────────┘
```

| Hạng mục | Đặc tả |
|---|---|
| Phản hồi thanh trượt | < 200 ms, tính tại máy khách theo UI-D1 |
| Đổi tham số chi phí | Vẽ lại đường cong và dịch vị trí cực tiểu; gọi `POST /threshold/optimize` với debounce 300 ms |
| Ngưỡng gợi ý sẵn | Từ `GET /threshold` → trường `alternatives` |
| Nút áp dụng | `PUT /threshold` — ảnh hưởng toàn ứng dụng và bền vững qua khởi động lại |
| Đường cong | Trục hoành thang log; đánh dấu ngưỡng hiện tại và cực tiểu bằng hai đường dọc khác màu |
| Bàn phím | Thanh trượt nhận `←`/`→` bước 0,001, `PgUp`/`PgDn` bước 0,01 |

**Kịch bản trình bày khi bảo vệ:** kéo thanh trượt từ 0,5 về 0,047 và để hội đồng
nhìn ô "Bắt được" nhảy từ khoảng 60/98 lên hơn 83/98 trong khi ô "Chi phí/ngày"
giảm. Ba mươi giây thao tác này thuyết phục hơn ba trang giải thích.

## 6. UI-04 — Hiệu năng mô hình

Màn hình đọc `GET /metrics`, gần như tĩnh.

Nội dung bắt buộc:

1. Bảng 20 tổ hợp mô hình × chiến lược, sắp xếp theo PR-AUC, có cột độ lệch chuẩn giữa các fold.
2. Đường Precision–Recall của 5 chiến lược vẽ chồng, kèm đường cơ sở 0,00172.
3. Ma trận nhầm lẫn tại ngưỡng hiện hành — cập nhật khi ngưỡng đổi.
4. Bảng đối chiếu cho thấy accuracy gần như không phân biệt được ba mô hình còn PR-AUC thì có. Đây là bằng chứng trực quan cho G-2.
5. Xếp hạng đặc trưng theo SHAP đặt cạnh xếp hạng theo kiểm định thống kê (notebook 02) để đối chiếu.
6. Bảng đối chiếu hai cách chia tập (ngẫu nhiên phân tầng và theo thời gian).
7. Khoảng tin cậy bootstrap hiển thị dạng thanh sai số, không chỉ con số trần.

Nếu hết thời gian, màn hình này thay tạm bằng các ảnh PNG xuất từ notebook — đây
là hạng mục được phép cắt đầu tiên.

## 7. UI-05 — Chế độ phát lại

Truy cập từ nút trên UI-01.

| Hạng mục | Đặc tả |
|---|---|
| Nguồn dữ liệu | `GET /replay/stream` (SSE) đọc ngày thứ hai từ `test_set.parquet` theo `Time` tăng dần |
| Tốc độ mặc định | Nén 60 lần — 1 giờ dữ liệu ≈ 1 phút thực; chỉnh được bằng thanh trượt |
| Hiển thị | Dòng giao dịch chảy từ trên xuống; dưới ngưỡng thì xám nhạt trôi qua, vượt ngưỡng thì chuyển đỏ, dừng một nhịp rồi rơi vào hàng đợi |
| Điều khiển | Phát / tạm dừng / đặt lại, thanh tốc độ, đồng hồ thời gian mô phỏng |
| Bộ đếm | Đã xử lý, số cảnh báo, tỷ lệ cảnh báo trên giờ |
| Kỹ thuật | Server-Sent Events — một chiều nên đơn giản hơn WebSocket |
| Phương án A | Nếu dùng Streamlit: vòng lặp với `st.empty()` và `time.sleep()` |

Chi phí khoảng một ngày công, nhưng đây là thứ hội đồng sẽ nhớ. Nếu phải cắt, cắt
UI-04 trước.

## 8. Khung giao diện chung

```
┌──────────┬─────────────────────────────────────────────────────┐
│          │  [Tên màn hình]            Ngưỡng: 0,047  [Đổi]     │
│  Fraud   ├─────────────────────────────────────────────────────┤
│  Console │                                                      │
│          │                                                      │
│ ▸ Hàng đợi│               Nội dung màn hình                     │
│ ▸ Ngưỡng │                                                      │
│ ▸ Hiệu năng                                                     │
│ ▸ Phát lại│                                                     │
│          │                                                      │
│ ─────────│                                                      │
│ v3 · 09-15│  Mô hình: xgb_scaleposweight_v3                     │
└──────────┴─────────────────────────────────────────────────────┘
```

Chân thanh bên luôn hiện `model_version` và ngày huấn luyện — đáp ứng NFR-11 và
tránh tình huống trình bày nhầm kết quả của một mô hình cũ.

## 9. Trạng thái phía máy khách

| Trạng thái | Phạm vi | Nguồn |
|---|---|---|
| `threshold` | Toàn cục | `GET /threshold`, cập nhật khi `PUT` thành công |
| `costFn`, `costFp` | Toàn cục | Như trên |
| `testScores` | Toàn cục, nạp một lần | `GET /metrics?section=test_scores` |
| `metrics` | Toàn cục, nạp một lần | `GET /metrics` |
| `queue` | UI-01 | `GET /transactions`, nạp lại khi ngưỡng hoặc bộ lọc đổi |
| `selectedTx`, `explanation` | UI-02 | `GET /transactions/{id}`, `POST /explain` |
| `replayState` | UI-05 | Kết nối SSE |

Không dùng thư viện quản lý trạng thái. Với bảy biến và một nguồn chân lý duy
nhất ở máy chủ, `Alpine.store` hoặc một đối tượng JavaScript thường là đủ.

## 10. Khả năng tiếp cận (NFR-10)

| Yêu cầu | Cách làm |
|---|---|
| Điều hướng bằng bàn phím | Mọi hành động đến được bằng `Tab` + `Enter`; có phím tắt ở §4 |
| Không chỉ dựa vào màu | Dải rủi ro kèm ký hiệu hình dạng và nhãn chữ, không chỉ chấm màu |
| Tương phản | Đạt WCAG AA (tỷ lệ ≥ 4,5:1 cho chữ thường) |
| Trọng tâm hiển thị | Viền trọng tâm rõ ràng, không tắt `outline` |
| Ngăn kéo chi tiết | Bẫy trọng tâm khi mở, trả trọng tâm về dòng cũ khi đóng |
| Nhãn cho biểu đồ | Mỗi biểu đồ kèm bảng số liệu tương ứng hoặc `aria-label` mô tả |
| Ngôn ngữ | `lang="vi"` trên thẻ gốc |

## 11. Tiêu chí chấp nhận phần giao diện

Xem [08 — Kế hoạch kiểm thử §4](08-ke-hoach-kiem-thu.md) cho danh sách AC-A1…AC-A10
và cách kiểm chứng.

## 12. Bản thi hành — những điểm cụ thể hơn hoặc khác thiết kế ban đầu

Giai đoạn 8 (2026-09-28). Mã nằm trong `web/`: `index.html` (khung và bốn màn hình), `app.js`
(trạng thái Alpine, gọi API), `threshold.js` (UI-D1), `charts.js` (Chart.js), `styles.css`,
`config.js` (địa chỉ API). Alpine.js 3.17.4 và Chart.js 4.5.1 **chép vào `web/vendor/`**, không nạp
từ CDN: buổi bảo vệ không được phụ thuộc mạng (NFR-08). Lệnh chạy:
[lenh-chay §8](lenh-chay.md).

| Màn hình | Điểm | Lý do |
|---|---|---|
| Chung | Điều hướng bằng địa chỉ `#/queue`, `#/threshold`, `#/performance`, `#/replay` | Tải lại trang giữ nguyên màn hình; nút Back của trình duyệt dùng được |
| Chung | Thanh trên hiện cả trạng thái API/CSDL (thăm `/health` mỗi 15 giây), cạnh ngưỡng | Mất CSDL hay thiếu hiện vật thì hiện dải thông báo kèm đúng lệnh cần chạy. Thanh trượt ngưỡng và UI-04 không cần CSDL nên vẫn dùng được |
| UI-01 | Màu theo điểm tuyệt đối như §3, **thêm** cột "Đề xuất" (`decision` của API: chặn khi ≥ 3τ, cần thẩm định khi τ…3τ) | Mức 60%/90% không nói gì về hành động; dải tương đối với τ mới là thứ quyết định chặn hay xem xét |
| UI-01 | Bộ lọc mặc định "Tất cả", không phải "Chờ xử lý" | Dòng vừa thẩm định đứng tại chỗ và đổi trạng thái ngay, thay vì biến mất khỏi danh sách |
| UI-01 | Dòng tóm tắt nói rõ "trên N giao dịch kiểm thử" | Số cảnh báo của UI-D1 (tập kiểm thử) khác số dòng trong hàng đợi (dữ liệu người dùng nạp); để chung một câu là gây nhầm |
| UI-01 | Trạng thái rỗng tách hai trường hợp: bảng `transactions` trống (ba cách nạp) và bộ lọc không khớp | Hỏi thêm `GET /transactions?min_score=0&page_size=1` khi danh sách rỗng |
| UI-01 | Tải lại giữ bảng cũ mờ đi; khung xương chỉ ở lần tải đầu | Không nháy bố cục mỗi lần đổi trang hay đổi ngưỡng |
| UI-01 | "Chọn mẫu có sẵn" là hộp thoại: chấm từng mẫu hoặc cả nhóm; mỗi mẫu một lời gọi `POST /score` kèm `sample_id` | `/score/batch` không giữ được nhãn thật của mẫu, mà UI-02 cần nhãn để đối chiếu (FR-44) |
| UI-02 | Ngăn kéo là `<dialog>` mở bằng `showModal()` | Trình duyệt tự bẫy trọng tâm và bắt `Esc`; mã chỉ còn phải trả trọng tâm về dòng cũ |
| UI-02 | Thác nước SHAP ở **thang log-odds**, dựng bằng bảng HTML; có dòng "N đặc trưng còn lại" (`remaining_shap`) và vạch ngưỡng quy về log-odds | Chỉ ở thang log-odds các đóng góp mới cộng được (05 §7). Bảng thật đọc được bằng trình đọc màn hình, không cần bảng thay thế |
| UI-02 | Danh sách yếu tố dương có thể ngắn hơn 5, kèm một câu giải thích | 77% giao dịch kiểm thử có ít hơn 5 đặc trưng SHAP dương (05 §7) |
| UI-02 | Giao dịch đã có kết luận từ trước thì nhãn thật hiện ngay khi mở | Người dùng đã quyết định rồi; kèm "khớp / khác kết luận của bạn" (FR-44) |
| UI-02 | `F`/`A` không có tác dụng khi đang gõ ghi chú hay khi giữ Ctrl/Alt | `Ctrl+F` vẫn là tìm kiếm của trình duyệt |
| UI-03 | Thanh trượt theo **thang log10**, 0,0001 → 0,999. `←`/`→` ±0,01 đơn vị log (≈ 2,3%), `PgUp`/`PgDn` ±0,1 (≈ 26%), `Home`/`End` hai đầu; thêm ô nhập số chính xác | Với mô hình thật, τ\* = 0,0232 nằm ở 2% đầu một thanh tuyến tính, và bước 0,001 lớn gấp đôi ngưỡng "recall ≥ 90%" (0,00054). Thang log trải đều vùng quyết định 0,0005…0,97 và khớp trục của đường cong chi phí |
| UI-03 | Mọi chi phí quy ra **EUR/ngày trên toàn luồng** (chia tỷ lệ mẫu và số ngày, như `alerts_per_day`); ô chi phí ghi thêm con số trên tập kiểm thử | Đường cong đo trên out-of-fold (80% dữ liệu), ô số đo trên tập kiểm thử (20%): quy về mỗi ngày thì hai nguồn cùng thang. Con số tập kiểm thử (2.389,78 EUR tại τ\*) là con số của báo cáo |
| UI-03 | Trục tung đường cong cắt ở khoảng 2,5 lần cực tiểu | Hai đầu vọt tới khoảng 30.000 EUR/ngày và đè bẹp vùng đáy chữ U — chỗ duy nhất có quyết định |
| UI-03 | Mỗi ô số kèm chênh lệch so với ngưỡng đang áp dụng | Kịch bản bảo vệ 0,5 → τ\* đọc được ngay trên màn hình |
| UI-03 | Ràng buộc vận hành (FR-33) nhập ở đây; `constraint_binding` hiện bằng câu chữ | Trả lời "bị giới hạn bởi ngân sách thẩm định hay bởi chính chi phí" (05 §3, API-12) |
| UI-03 | Màn hình tự ghi thời gian tính và thời gian tới khi vẽ xong khung hình | Bằng chứng AC-A3 ngay trên màn hình, không cần mở DevTools |
| UI-04 | Thêm đường PR và ROC của mô hình xuất trên tập kiểm thử | FR-42 đòi đường ROC; §6 không liệt kê |
| UI-04 | Khoảng tin cậy dạng thanh sai số HTML: bốn chỉ số chính, và PR-AUC của bốn mô hình ở bảng đối chiếu | Thấy ngay khoảng của hồi quy logistic và cây quyết định chồng lên nhau |
| UI-05 | Trình duyệt tự đếm (bỏ trùng theo mã giao dịch); tiếp tục phát với `start` = `sim_seconds` cuối cùng đã nhận | `stats` của máy chủ đếm lại từ 0 mỗi kết nối; giây cuối được phát lại khi tiếp tục nên phải bỏ trùng |
| UI-05 | Mất kết nối thì đóng `EventSource` và tự nối lại sau 3 giây từ chỗ dừng | Để mặc định, `EventSource` tự nối lại với URL cũ, tức phát lại từ `start` ban đầu |
| UI-05 | Sự kiện gộp theo khung hình (`requestAnimationFrame`); tốc độ 9 nấc từ 1 tới 3.600 lần | Ở 3.600 lần có hơn nghìn giao dịch mỗi giây — vẽ từng sự kiện là treo trang |
| UI-05 | Bấm một cảnh báo để mở chi tiết; mã `RP-…` chưa ghi xong thì ngăn kéo báo và cho thử lại | Phát lại ghi theo mẻ 100 dòng (05 §7), nên giao dịch vừa phát có thể chưa có trong CSDL vài giây |

## 13. Bản Next.js (`frontend/`)

Viết sau bản `web/`, theo nhánh "Next.js + React + Tailwind + Recharts" mà [03 §6.2](03-thiet-ke-kien-truc.md)
để ngỏ. Hai bản cùng gọi một API, cùng quy tắc định dạng số, cùng các quyết định ở §12; bản `web/`
giữ lại làm phương án dự phòng cho buổi bảo vệ. Cách chạy: [frontend/README.md](../frontend/README.md)
và [lenh-chay §8.6](lenh-chay.md).

| Điểm | Bản `web/` (Alpine.js) | Bản `frontend/` (Next.js) |
|---|---|---|
| Điều hướng | `#/queue`, `#/threshold`… trong một trang | route thật `/`, `/threshold/`, `/performance/`, `/replay/`, xuất tĩnh ra `out/` |
| Giữ trạng thái khi đổi màn hình | cả 4 màn hình luôn nằm trong DOM (`x-show`) | trạng thái cần giữ nằm ở context của layout gốc: `ThresholdDraftContext` (vị trí thanh trượt, tham số chi phí), `ReplayContext` (luồng SSE chạy tiếp khi sang Hàng đợi) |
| UI-D1 | `web/threshold.js` (UMD) | `frontend/src/lib/threshold.mjs` (ES module). TC-12 đối chiếu **cả hai** với `src/threshold.py` |
| Mảng điểm tập kiểm thử | ngoài trạng thái Alpine (Proxy làm chậm) | trong state React — so sánh theo tham chiếu nên kích thước không ảnh hưởng |
| Biểu đồ | Chart.js (canvas) | Recharts (SVG); PR/ROC không có tooltip, số liệu nằm ở bảng ngay dưới |
| Biểu đồ vẽ lại khi kéo | trong `requestAnimationFrame` | ở mức ưu tiên thấp (`useDeferredValue`): ô số cập nhật trước, biểu đồ theo sau |
| AC-A3 (lâu nhất trên 21 lần kéo) | 20 ms | 39 ms |
| Kiểm tra tĩnh | — | `tsc --noEmit`, ESLint gồm các luật React Compiler, không tắt luật nào |

Bẫy riêng của bản này (Next.js 16.3 trên Windows ghi sai tên tệp tải trước khi xuất tĩnh; React
Compiler hiểu nhầm trường `threshold.current` là `ref.current`): [lenh-chay §8.5](lenh-chay.md).

**Đóng gói (giai đoạn 9):** `docker compose up` đóng gói bản Next.js; `WEB_UI=web` đổi sang bản `web/`,
build không cần npm. Trong Compose cả hai gọi API qua đường dẫn tương đối `/api/v1` mà nginx chuyển
tiếp ([03 §6.3](03-thiet-ke-kien-truc.md)). Trên bản đóng gói, kịch bản bàn phím `scripts/ui/keyboard.js`
đạt 24/24 và Lighthouse cho Accessibility 100 ở cả bốn màn hình (AC-A9, [08 §4.4](08-ke-hoach-kiem-thu.md)).
