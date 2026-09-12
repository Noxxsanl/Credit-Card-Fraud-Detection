# 01 — Đặc tả yêu cầu

Tài liệu này định nghĩa **hệ thống phải làm được gì**, không nói làm bằng cách
nào. Cách làm thuộc về các tài liệu thiết kế 03–07.

## 1. Vai người dùng

| Vai | Ký hiệu | Mô tả | Câu hỏi họ đặt ra |
|---|---|---|---|
| Người thẩm định | U-ANALYST | Xử lý từng giao dịch bị gắn cờ | "Giao dịch này có đáng ngờ thật không, vì sao?" |
| Quản lý rủi ro | U-MANAGER | Đặt chính sách chặn cho toàn hệ thống | "Nếu siết ngưỡng, tôi bắt thêm bao nhiêu gian lận và tốn thêm bao nhiêu công thẩm định?" |
| Người chấm đồ án | U-GRADER | Đánh giá tính đúng đắn phương pháp | "Có rò rỉ dữ liệu không? Kết quả có tái lập được không?" |

Hai vai đầu là vai giả lập trong demo; không có đăng nhập, không phân quyền.

## 2. User story

### U-ANALYST

- **US-01** — Là người thẩm định, tôi muốn thấy danh sách giao dịch rủi ro cao
  xếp theo điểm giảm dần, để xử lý việc quan trọng trước.
- **US-02** — Là người thẩm định, tôi muốn mở một giao dịch và thấy yếu tố nào
  đẩy điểm rủi ro lên hay xuống, để không phải tin mô hình một cách mù quáng.
- **US-03** — Là người thẩm định, tôi muốn đánh dấu kết luận của mình
  (gian lận thật / báo động giả) và thấy nó được giữ lại sau khi tải lại trang.
- **US-04** — Là người thẩm định, tôi muốn so sánh giao dịch đang xem với phân bố
  chung, để biết nó bất thường tới mức nào.

### U-MANAGER

- **US-05** — Là quản lý rủi ro, tôi muốn kéo một thanh trượt ngưỡng và thấy ngay
  số cảnh báo, recall, precision và chi phí kỳ vọng thay đổi ra sao.
- **US-06** — Là quản lý rủi ro, tôi muốn nhập chi phí bỏ lọt và chi phí báo động
  giả của riêng tổ chức mình, và được hệ thống đề xuất ngưỡng tối ưu tương ứng.
- **US-07** — Là quản lý rủi ro, tôi muốn đặt ràng buộc "mỗi ngày chỉ thẩm định
  được 200 giao dịch" và biết ngưỡng nào đáp ứng được ràng buộc đó.
- **US-08** — Là quản lý rủi ro, tôi muốn xem hệ thống chạy trên dòng giao dịch
  mô phỏng theo thời gian, để hình dung tải công việc thực tế.

### U-GRADER

- **US-09** — Là người chấm, tôi muốn chạy lại toàn bộ pipeline bằng một chuỗi
  lệnh ngắn và thu được cùng kết quả.
- **US-10** — Là người chấm, tôi muốn thấy bảng so sánh đầy đủ các chiến lược xử
  lý mất cân bằng, không chỉ mô hình thắng cuộc.

## 3. Yêu cầu chức năng

### 3.1 Nhóm phân tích và huấn luyện (offline)

| Mã | Yêu cầu | Ưu tiên |
|---|---|---|
| FR-01 | Nạp `creditcard.csv`, kiểm tra toàn vẹn: đủ 31 cột, không thiếu giá trị, báo cáo số dòng trùng lặp | Bắt buộc |
| FR-02 | Sinh tối thiểu 8 biểu đồ EDA: phân bố nhãn, `Amount` theo lớp, `hour` theo lớp, ma trận tương quan, phân bố các đặc trưng V phân tách mạnh nhất | Bắt buộc |
| FR-03 | Kiểm định thống kê so sánh phân bố hai lớp cho 30 đặc trưng, có hiệu chỉnh đa kiểm định (Benjamini–Hochberg) và effect size | Bắt buộc |
| FR-04 | Chia tập phân tầng 80/20 với `random_state` cố định; lưu tập kiểm thử ra tệp để ứng dụng dùng lại | Bắt buộc |
| FR-05 | Huấn luyện đủ lưới 5 chiến lược mất cân bằng × 4 thuật toán = 20 tổ hợp, đánh giá bằng `StratifiedKFold(5)` với scoring `average_precision` | Bắt buộc |
| FR-06 | Tinh chỉnh siêu tham số cho cấu hình tốt nhất bằng `RandomizedSearchCV` ≥ 30 lần thử | Bắt buộc |
| FR-07 | Tính khoảng tin cậy bootstrap cho PR-AUC, Recall, Precision của mô hình cuối | Bắt buộc |
| FR-08 | Dựng đường cong chi phí theo ngưỡng và xác định ngưỡng tối thiểu hóa chi phí kỳ vọng | Bắt buộc |
| FR-09 | Sinh giải thích SHAP: tầm quan trọng toàn cục và giải thích cho từng giao dịch | Bắt buộc |
| FR-10 | Đánh giá bổ sung theo cách chia theo thời gian (ngày 1 huấn luyện, ngày 2 kiểm thử) và đối chiếu với chia ngẫu nhiên | Nên có |
| FR-11 | Autoencoder phát hiện bất thường, huấn luyện chỉ trên lớp bình thường, so sánh với mô hình có giám sát | Tùy chọn |
| FR-12 | Xuất toàn bộ hiện vật ra `models/` bằng một notebook duy nhất | Bắt buộc |

### 3.2 Nhóm chấm điểm (online)

| Mã | Yêu cầu | Ưu tiên |
|---|---|---|
| FR-20 | Chấm điểm một giao dịch từ 30 giá trị đặc trưng, trả về điểm rủi ro, ngưỡng hiện hành, quyết định và dải rủi ro | Bắt buộc |
| FR-21 | Chấm điểm hàng loạt tối đa 50.000 giao dịch trong một yêu cầu | Bắt buộc |
| FR-22 | Nhận tệp CSV tải lên, chấm điểm toàn bộ, lưu kết quả và trả về tóm tắt lô | Bắt buộc |
| FR-23 | Từ chối dữ liệu vào không hợp lệ với mã lỗi rõ ràng và danh sách cột thiếu hoặc sai kiểu | Bắt buộc |
| FR-24 | Trả về giải thích SHAP cho một giao dịch: tối thiểu 5 yếu tố làm tăng và 3 yếu tố làm giảm điểm rủi ro | Bắt buộc |
| FR-25 | Cung cấp thư viện khoảng 200 giao dịch mẫu để thử nhanh, gồm cả mẫu gian lận và mẫu hợp lệ | Nên có |

### 3.3 Nhóm ngưỡng và chi phí

| Mã | Yêu cầu | Ưu tiên |
|---|---|---|
| FR-30 | Lưu ngưỡng hiện hành như một cấu hình thay đổi được lúc chạy, tách khỏi mô hình | Bắt buộc |
| FR-31 | Khi ngưỡng thay đổi, tính lại và trả về: số cảnh báo, TP, FP, FN, Precision, Recall, F1, chi phí kỳ vọng | Bắt buộc |
| FR-32 | Nhận tham số chi phí `cost_fn`, `cost_fp` do người dùng nhập và đề xuất ngưỡng tối thiểu hóa chi phí | Bắt buộc |
| FR-33 | Hỗ trợ chọn ngưỡng theo ràng buộc vận hành: "recall ≥ X" hoặc "tối đa K cảnh báo mỗi ngày" | Nên có |
| FR-34 | Hiển thị đồng thời các ngưỡng phương án: tối ưu chi phí, F1 cao nhất, recall ≥ 90%, mặc định 0,5 | Bắt buộc |

### 3.4 Nhóm thẩm định và trình bày

| Mã | Yêu cầu | Ưu tiên |
|---|---|---|
| FR-40 | Danh sách giao dịch vượt ngưỡng, sắp xếp giảm dần theo điểm rủi ro, có phân trang và bộ lọc | Bắt buộc |
| FR-41 | Ghi lại quyết định thẩm định của người dùng kèm ngưỡng tại thời điểm đó | Bắt buộc |
| FR-42 | Màn hình hiệu năng mô hình: đường PR, đường ROC, ma trận nhầm lẫn, đường cong chi phí, bảng 20 tổ hợp | Bắt buộc |
| FR-43 | Chế độ phát lại: phát dòng giao dịch theo thời gian mô phỏng, tốc độ điều chỉnh được | Nên có |
| FR-44 | Đối chiếu quyết định của người thẩm định với nhãn thật khi nhãn có sẵn | Tùy chọn |

## 4. Yêu cầu phi chức năng

| Mã | Yêu cầu | Ngưỡng chấp nhận | Cách đo |
|---|---|---|---|
| NFR-01 | Độ trễ chấm điểm một giao dịch | p95 < 50 ms | Đo 1.000 lần gọi liên tiếp |
| NFR-02 | Thông lượng chấm điểm hàng loạt | ≥ 10.000 giao dịch / 30 giây | Bấm giờ trên tệp mẫu |
| NFR-03 | Phản hồi khi kéo thanh trượt ngưỡng | < 200 ms | DevTools, tab Network |
| NFR-04 | Thời gian khởi động toàn hệ thống | < 15 giây từ `docker compose up` khi volume đã có dữ liệu; < 45 giây ở lần chạy đầu (PostgreSQL `initdb` + migration) | Bấm giờ tới khi `/health` trả 200 |
| NFR-05 | Bộ nhớ tiến trình API | < 1 GB ở trạng thái nghỉ; container PostgreSQL tính riêng, < 400 MB | `docker stats` |
| NFR-06 | Kích thước CSV tải lên tối đa | 100 MB | Kiểm thử biên |
| NFR-07 | Khả năng tái lập | Chạy lại notebook, PR-AUC lệch < 0,001 | So hai lần chạy |
| NFR-08 | Hoạt động ngoại tuyến | Sau khi có dữ liệu, không cần mạng | Ngắt mạng, chạy lại |
| NFR-09 | Trình duyệt hỗ trợ | Chrome, Firefox, Safari — hai phiên bản gần nhất | Thử thủ công |
| NFR-10 | Khả năng tiếp cận | Điều hướng được bằng bàn phím, tương phản WCAG AA | Kiểm tra bằng Tab và Lighthouse |
| NFR-11 | Khả năng truy vết | Mỗi kết quả chấm điểm gắn `model_version` | Rà soát phản hồi API |
| NFR-12 | Tài liệu API | Tự sinh, truy cập được tại `/docs` | Mở trình duyệt |

Ghi chú NFR-01: mô hình gradient boosting 500 cây chấm một mẫu mất khoảng 2–5 ms
trên CPU thường, nên 50 ms là biên rộng rãi. Điểm nghẽn thật nằm ở SHAP
(50–200 ms mỗi mẫu), vì vậy giải thích phải là endpoint riêng, chỉ gọi khi người
dùng mở chi tiết — không gọi cho mọi dòng trong danh sách.

## 5. Ràng buộc

| Mã | Ràng buộc | Nguồn |
|---|---|---|
| C-01 | Dữ liệu giữ nguyên, không thu thập thêm | Đề bài |
| C-02 | Chỉ dùng Python và các thư viện trong `requirements.txt` | Đề bài |
| C-03 | Toàn bộ dự án hoàn thành trong 21 ngày | Lịch môn học |
| C-04 | `creditcard.csv` (khoảng 144 MB) không được commit lên git | Kích thước repo |
| C-05 | Chạy được trên máy cá nhân, CPU, không yêu cầu GPU | Điều kiện thiết bị |
| C-06 | Giấy phép dữ liệu DbCL v1.0 — phải ghi nguồn trong báo cáo | Điều khoản Kaggle |

## 6. Giả định

| Mã | Giả định | Nếu sai thì sao |
|---|---|---|
| A-01 | Chi phí bỏ lọt xấp xỉ số tiền giao dịch trung bình của nhóm gian lận (khoảng 122 USD) | Ngưỡng tối ưu dịch chuyển; hệ thống cho phép người dùng nhập lại giá trị |
| A-02 | Chi phí thẩm định một cảnh báo khoảng 5 USD | Như trên |
| A-03 | Phân bố dữ liệu kiểm thử đại diện cho dữ liệu vận hành | Kết quả lạc quan hơn thực tế — phải nêu trong phần hạn chế |
| A-04 | Nhãn `Class` chính xác | Trần hiệu năng bị giới hạn bởi nhiễu nhãn |

## 7. Truy vết yêu cầu sang thiết kế

| Nhóm yêu cầu | Tài liệu thiết kế tương ứng |
|---|---|
| FR-01…FR-12 | [04 — Thiết kế mô hình ML](04-thiet-ke-mo-hinh-ml.md), [02 — Đặc tả dữ liệu](02-dac-ta-du-lieu.md) |
| FR-20…FR-25 | [05 — Thiết kế API](05-thiet-ke-api.md) |
| FR-30…FR-34 | [04 §6](04-thiet-ke-mo-hinh-ml.md), [05](05-thiet-ke-api.md), [07 UI-03](07-thiet-ke-giao-dien.md) |
| FR-40…FR-44 | [07 — Thiết kế giao diện](07-thiet-ke-giao-dien.md), [06 — Thiết kế lưu trữ](06-thiet-ke-luu-tru.md) |
| NFR-01…NFR-12 | [03 — Thiết kế kiến trúc](03-thiet-ke-kien-truc.md) |
