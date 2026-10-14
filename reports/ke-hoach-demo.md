# Kế hoạch demo hệ thống

Demo chạy sau phần thuyết trình, dài khoảng **10 phút** (bản 5 phút ở cuối file). Toàn bộ hệ thống chạy
bằng Docker; demo trên giao diện ở http://localhost:3000. Không cần Python hay Node.js.

Demo cho hội đồng thấy ba điều đã nói trong bài:

1. Mô hình chạy thật từ đầu đến cuối: tải dữ liệu → chấm điểm → hàng đợi → thẩm định.
2. Ngưỡng là **cấu hình**: đổi ngưỡng thì thấy ngay cái giá (cảnh báo/ngày, số vụ bắt được, chi phí).
3. Hệ thống **theo dõi** ngưỡng sau khi áp dụng, vì ngày mới khác ngày cũ.

---

## Phần A — Dựng hệ thống và mở giao diện

Mọi lệnh chạy trong PowerShell, tại thư mục gốc repo (`D:\WorkSpace\AI_Project\fraud-detection`).

### A1. Lần đầu tiên (làm trước buổi bảo vệ ít nhất một ngày)

Máy hiện chưa có image của dự án, nên lần đầu phải build. Cần Internet, mất khoảng 4 phút.

```powershell
# Docker Desktop phải đang chạy (biểu tượng cá voi ở khay hệ thống)
docker compose up --build -d
docker compose ps
```

Chờ tới khi `docker compose ps` cho thấy `api` ở trạng thái **(healthy)** — khoảng 15 giây sau khi build
xong. Cơ sở dữ liệu lúc này còn trống; nạp dữ liệu demo ở bước A3.

### A2. Các lần sau và ngày bảo vệ

```powershell
docker compose up -d          # không build lại, khoảng 11 giây
docker compose ps             # api phải (healthy)
```

Mở trình duyệt vào **http://localhost:3000**. Tài liệu API ở http://localhost:8000/docs nếu hội đồng muốn xem.

Bật hệ thống **trước** giờ bảo vệ khoảng 30 phút, đừng bật lần đầu trước mặt hội đồng.

### A3. Nạp dữ liệu demo

Bản sao lưu `backup/fraud-demo.dump` chứa sẵn khoảng 10.000 giao dịch đã chấm, các mẫu từ thư viện
(trừ nhóm "Gian lận khó", để dành cho demo trực tiếp) và 8 giao dịch đã thẩm định có ghi chú.

```powershell
docker compose cp backup/fraud-demo.dump db:/tmp/fraud-demo.dump
docker compose exec -T db pg_restore -U fraud -d fraud --clean --if-exists --single-transaction --no-owner /tmp/fraud-demo.dump
```

Tải lại trang là thấy dữ liệu, không cần khởi động lại API. Chạy lại hai lệnh này bất cứ lúc nào cần đưa
dữ liệu về trạng thái chuẩn: sau khi tập dượt, và **sau buổi demo**.

### A4. Kiểm tra trước khi lên

- [ ] Thanh trên hiện "API và CSDL hoạt động" và ngưỡng **0,268** (tức 0,2675).
- [ ] Góc dưới thanh bên: "Mô hình đang phục vụ" là `xgb_none_v2`.
- [ ] Hàng đợi có dữ liệu; bộ lọc "Xác nhận gian lận" và "Báo động sai" có dòng.
- [ ] Đi qua 4 màn hình, F12 không có dòng đỏ.
- [ ] Có sẵn tệp `data/giao-dich-demo.csv` (10.000 giao dịch chưa nạp, dùng ở bước 2).
- [ ] Phóng to trình duyệt (Ctrl + +); tắt thông báo của Windows.
- [ ] Mở sẵn trong cửa sổ khác: `reports/figures/10_ui_queue.png`, `10_ui_drawer.png`,
      `10_ui_threshold.png` — dùng khi demo trục trặc.

### A5. Tắt hệ thống

```powershell
docker compose down           # dừng, GIỮ dữ liệu
```

Không dùng `docker compose down -v`: lệnh đó xoá luôn dữ liệu demo.

---

## Phần B — Kịch bản demo

| # | Bước | Màn hình | Thời gian |
|---|---|---|---|
| 1 | Giới thiệu hệ thống | Hàng đợi | 0,5 phút |
| 2 | Tải CSV | Hàng đợi | 1,5 phút |
| 3 | Chi tiết giao dịch và thẩm định | Ngăn kéo chi tiết | 1,5 phút |
| 4 | Gian lận khó | Thư viện mẫu | 1,5 phút |
| 5 | Cấu hình ngưỡng | Ngưỡng | 2,5 phút |
| 6 | Phát lại ngày 2 | Phát lại | 1 phút |
| 7 | Hiệu năng và theo dõi vận hành | Hiệu năng | 1 phút |
| 8 | Kết thúc | — | 0,5 phút |

Mỗi bước gồm **Làm** (thao tác), **Thấy** (con số nên hiện) và **Nói** (lời với hội đồng). Nếu số trên màn
hình hơi khác thì cứ nói tiếp, đừng dừng lại giải thích.

### Bước 1 — Giới thiệu hệ thống (0,5 phút)

**Làm:** đang ở màn **Hàng đợi**. Chỉ chuột lần lượt vào thanh bên, góc dưới thanh bên, thanh trên.

**Nói:**

> Sau đây em xin demo hệ thống. Toàn bộ chạy trong Docker gồm ba phần: giao diện web, API FastAPI và cơ
> sở dữ liệu PostgreSQL, dựng bằng một lệnh `docker compose up`.
>
> Bên trái là bốn màn hình. Góc dưới là mô hình đang phục vụ — XGBoost không xử lý mất cân bằng, đúng
> mô hình cuối trong báo cáo. Thanh trên luôn hiện ngưỡng đang áp dụng và trạng thái của API.

### Bước 2 — Tải CSV (1,5 phút)

**Làm:**

1. Bấm **Tải CSV** → chọn `data/giao-dich-demo.csv`.
2. Chỉ vào bảng kết quả: "Dòng đọc được", "Vượt ngưỡng", "Thời gian phía máy chủ", "Trên nhãn của tệp".
3. Bấm **Chỉ xem lô này**, rồi **Bỏ lọc theo lô**.

**Thấy:** 10.000 dòng đọc được, **6 vượt ngưỡng**, cả 6 đều là gian lận theo nhãn của tệp, khoảng 2 giây.

**Nói:**

> Em tải lên 10.000 giao dịch mà hệ thống chưa thấy. Hệ thống chấm xong trong khoảng 2 giây, có 6 giao
> dịch vượt ngưỡng. Tệp này có nhãn thật nên em đối chiếu được: cả 6 đều là gian lận thật, không có cảnh
> báo giả nào.
>
> Hàng đợi sắp theo điểm rủi ro giảm dần, kèm đề xuất "Cần thẩm định" hoặc "Đề xuất chặn". Chỉ những
> vụ gần như chắc chắn — điểm trên 99,9% — mới được đề xuất chặn; còn lại qua tay người thẩm định.

### Bước 3 — Chi tiết giao dịch và thẩm định (1,5 phút)

**Làm:**

1. Bấm vào một dòng đầu hàng đợi có trạng thái "Chờ xử lý" (hoặc phím `↓` rồi `Enter`).
2. Chỉ vào điểm rủi ro so với ngưỡng, rồi biểu đồ thác nước SHAP và chú thích về PCA.
3. Bấm phím **`F`** (Xác nhận gian lận) → nhãn thật hiện "✓ khớp kết luận của bạn".
4. Bấm `Esc` để đóng.

**Nói:**

> Đây là chi tiết một giao dịch. Biểu đồ SHAP cho biết đặc trưng nào đẩy điểm rủi ro lên, đặc trưng nào
> kéo xuống. V1 đến V28 là thành phần PCA đã ẩn danh, nên hệ thống chỉ nói được "V14 đẩy điểm lên",
> không nói được nguyên nhân nghiệp vụ — chú thích này luôn hiện trên màn hình.
>
> Nhãn thật bị ẩn cho tới khi người thẩm định kết luận, để họ không bị dẫn dắt. Em bấm F… và nhãn thật
> hiện ra, khớp với kết luận. Kết luận được lưu vào cơ sở dữ liệu, khởi động lại vẫn còn.

### Bước 4 — Gian lận khó (1,5 phút)

Đoạn đáng diễn nhất: nó cho thấy trực tiếp kết luận của phần phân tích lỗi.

**Làm:**

1. Bấm **Chọn mẫu có sẵn** → chọn nhóm **Gian lận khó (24)**. Đọc dòng mô tả: "gian lận, điểm < 0,5
   (ngưỡng 0,5 bỏ lọt)".
2. Bấm **Nạp 24 mẫu còn lại của nhóm này**.
3. Chỉ vào thông báo: *Đã nạp 24 mẫu nhóm "Gian lận khó", 1 vào hàng đợi*. Lướt cột **Kết quả**: hầu hết
   là "Cho qua". Ghi lại mã GD của dòng duy nhất "Cần thẩm định".
4. Đóng hộp thoại (`Esc`). Trên hàng đợi, bấm tiêu đề cột **Rủi ro** để sắp tăng dần, lọc **Chờ xử lý**,
   tìm dòng có mã vừa ghi và mở ra.
5. Bấm phím **`A`** (Đánh dấu báo động sai) → nhãn thật hiện "Gian lận — ✗ khác kết luận của bạn".
6. Bấm lại tiêu đề **Rủi ro** để trả về sắp giảm dần.

**Thấy:** 1/24 vào hàng đợi (điểm 43,5%). Một mẫu điểm 25,1%, sát dưới ngưỡng 26,75%. Một nửa số mẫu
(12/24) hiện "< 0,1%".

**Nói:**

> Đây là 24 vụ gian lận thật mà ngưỡng 0,5 của thư viện bỏ lọt. Em nạp cả nhóm vào hệ thống.
>
> Ở ngưỡng 0,2675, chỉ một vụ vào hàng đợi. 23 vụ còn lại mô hình cho điểm rất thấp, một nửa dưới 0,1%.
> Đúng như phần phân tích lỗi: phần lớn vụ bỏ lọt trông giống hệt giao dịch hợp lệ trên các đặc trưng
> quan trọng nhất. Đó là giới hạn của đặc trưng, không phải của ngưỡng.
>
> Giả sử người thẩm định nhìn vụ này và cho là báo động sai… Hệ thống hiện nhãn thật: đây là gian lận.
> Trong vận hành thật, những trường hợp như vậy là dữ liệu quý để huấn luyện lại.

### Bước 5 — Cấu hình ngưỡng (2,5 phút)

**Làm:**

1. Sang màn **Ngưỡng**. Chỉ vào đường cong chi phí và dòng chú thích dưới nó.
2. Trong **Ngưỡng gợi ý sẵn**, chọn **"0,5 — mặc định của thư viện"**. Chỉ vào các ô "Cảnh báo mỗi ngày",
   "Gian lận bắt được", "Chi phí mỗi ngày".
3. Chọn **"Cực tiểu chi phí, không ràng buộc"** → các ô đổi theo.
4. Chọn lại **"Chi phí nhỏ nhất trong ngân sách thẩm định (mặc định)"**. Kéo thanh trượt qua lại vài lần.
5. Chỉ vào dòng "Ngưỡng đang áp dụng, đo trên tập kiểm thử (chỉ để báo cáo)".
6. Ở **Tham số chi phí**, gõ "Chi phí bỏ lọt 1 gian lận" = **600** → ngưỡng tối ưu dời sang trái.
7. Ở **Ràng buộc vận hành**, chọn **Số cảnh báo tối đa mỗi ngày** = **150** → hiện "Ràng buộc đang chặn
   nghiệm".
8. **Trả lại như cũ:** chi phí bỏ lọt về **115,94**, ràng buộc về **Không ràng buộc**.

**Không bấm "Áp dụng ngưỡng này"** trong cả bước này — chỉ xem, không đổi ngưỡng của hệ thống.

**Thấy** (trên điểm chọn ngưỡng):

| Ngưỡng gợi ý | τ | Cảnh báo/ngày |
|---|---|---|
| 0,5 — mặc định của thư viện | 0,5 | khoảng 184 |
| Chi phí nhỏ nhất trong ngân sách (mặc định) | 0,2675 | khoảng 198 |
| Cực tiểu chi phí, không ràng buộc | 0,0191 | khoảng 297 |

Dòng tập kiểm thử: bắt **72/95** gian lận. Chi phí bỏ lọt 600 EUR: ngưỡng tối ưu dời từ 0,0191 xuống
**0,0035**.

**Nói:**

> Đây là màn hình của người quản lý rủi ro. Mọi con số ở đây tính trên điểm forward-chaining của tập huấn
> luyện, không trên tập kiểm thử — vì kéo thanh trượt chính là đang chọn ngưỡng.
>
> Ở ngưỡng 0,5, hệ thống sinh khoảng 184 cảnh báo mỗi ngày. Ngưỡng cực tiểu chi phí, 0,0191, bắt nhiều
> vụ hơn và tổng chi phí thấp hơn — nhưng cảnh báo lên gần 300 mỗi ngày, vượt ngân sách 200 của đội thẩm
> định. Vì vậy ngưỡng mặc định là 0,2675: rẻ nhất trong những ngưỡng còn nằm trong ngân sách. Kéo thanh
> trượt, mọi chỉ số tính lại ngay trong trình duyệt, dưới 70 mili giây.
>
> Giả sử mỗi vụ bỏ lọt thật ra tốn 600 euro. Ngưỡng tối ưu tự dời xuống 0,0035. Nhưng nếu đội thẩm định
> chỉ xử lý được 150 cảnh báo mỗi ngày, hệ thống báo ngay: ràng buộc đang chặn nghiệm. Đúng như kết luận
> trong báo cáo: năng lực thẩm định, chứ không phải hàm chi phí, quyết định ngưỡng.

### Bước 6 — Phát lại ngày 2 (1 phút)

**Làm:**

1. Sang màn **Phát lại**. Kéo **Tốc độ** lên **600×**.
2. Bấm **▶ Phát**, để chạy khoảng 30 giây.
3. Chỉ vào dòng giao dịch (xám là dưới ngưỡng, đỏ là vượt ngưỡng), các bộ đếm và ô "Ngưỡng máy chủ đang dùng".
4. Bấm **❚❚ Tạm dừng**.

**Nói:**

> Chế độ này phát lại giao dịch của ngày 2 theo đồng hồ mô phỏng, ở đây nén 600 lần. Giao dịch dưới
> ngưỡng trôi qua màu xám; vượt ngưỡng chuyển đỏ rồi rơi vào hàng đợi. Hệ thống đọc lại ngưỡng mỗi 5 giây,
> nên đổi ngưỡng giữa chừng thì có tác dụng ngay, không cần khởi động lại.

### Bước 7 — Hiệu năng và theo dõi vận hành (1 phút)

**Làm:**

1. Sang màn **Hiệu năng**. Chỉ vào thẻ **Chỉ số kèm khoảng tin cậy**.
2. Cuộn qua bảng **20 tổ hợp mô hình × chiến lược** (dòng tô đậm là mô hình đang phục vụ) và
   **Đối chiếu cách chia tập**.
3. Cuộn tới thẻ **Theo dõi vận hành**.

**Thấy:** PR-AUC kiểm thử **0,828** (0,747–0,900); chia theo thời gian 0,773.

**Nói:**

> Màn hình này tóm tắt những gì em đã trình bày: chỉ số kèm khoảng tin cậy, bảng 20 tổ hợp, và đối chiếu
> chia ngẫu nhiên với chia theo thời gian.
>
> Quan trọng nhất là thẻ theo dõi vận hành. Cứ mỗi khoảng 6 giờ, hệ thống so tỷ lệ cảnh báo, precision
> theo kết luận thẩm định và recall với kỳ vọng. Khoảng nào lệch quá dao động mẫu sẽ bị đánh dấu — tín
> hiệu để chọn lại ngưỡng. Vì trên một ngày mới, ngưỡng đặt sẵn không còn cho kết quả như cũ.

### Bước 8 — Kết thúc (0,5 phút)

**Nói:**

> Tóm lại, hệ thống cho thấy ba điều: mô hình chạy được từ dữ liệu thô tới quyết định; ngưỡng tách khỏi
> mô hình thành cấu hình, người dùng thấy ngay cái giá của mỗi lựa chọn; và ngưỡng được theo dõi sau khi
> áp dụng. Hệ thống có 406 ca kiểm thử tự động, gồm cả kiểm thử chống rò rỉ dữ liệu. Em xin cảm ơn, và
> rất mong nhận được câu hỏi của thầy cô.

**Sau buổi:** chạy lại hai lệnh ở A3 để đưa dữ liệu về như cũ.

---

## Khi demo trục trặc

| Tình huống | Xử lý | Câu nói lấp chỗ trống |
|---|---|---|
| Thanh trên báo "Đang kết nối API…" hoặc "Mất kết nối CSDL" | `docker compose restart api` (khoảng 10 giây), tải lại trang | "API đang khởi động lại, mất khoảng 10 giây." |
| `docker compose up` báo cổng bị chiếm | Đặt cổng khác trong `.env` (`WEB_HOST_PORT=3001`), chạy lại, mở cổng mới | — |
| Hàng đợi trống hoặc lộn xộn | Chạy hai lệnh ở A3, tải lại trang | "Em khôi phục dữ liệu demo từ bản sao lưu." |
| Lỡ bấm "Áp dụng ngưỡng này" | Màn Ngưỡng → chọn "Chi phí nhỏ nhất trong ngân sách thẩm định (mặc định)" → Áp dụng; kiểm thanh trên về 0,268 | — |
| Tải CSV báo lỗi | Bỏ bước 2, dùng dữ liệu đã nạp sẵn | "Dữ liệu này em đã tải sẵn bằng đúng chức năng đó." |
| Docker không lên | Chuyển sang ảnh `10_ui_*.png`, nói theo lời của từng bước | "Em xin trình bày qua ảnh chụp hệ thống." |

---

## Bản rút gọn 5 phút

Dùng khi hội đồng hết thời gian.

| # | Bước | Thời gian | Làm gì |
|---|---|---|---|
| 1 | Giới thiệu | 0,5 phút | Như bước 1 |
| 2 | Chi tiết và thẩm định | 1 phút | Mở dòng đầu hàng đợi, chỉ SHAP, bấm `F` |
| 3 | Gian lận khó | 1,5 phút | Như bước 4 |
| 4 | Cấu hình ngưỡng | 1,5 phút | Chỉ so 0,5 / mặc định / cực tiểu chi phí, bỏ đoạn chi phí 600 |
| 5 | Theo dõi vận hành | 0,5 phút | Mở thẻ Theo dõi vận hành, nói một câu |
