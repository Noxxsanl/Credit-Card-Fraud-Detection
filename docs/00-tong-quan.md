# 00 — Tổng quan dự án

## 1. Bối cảnh

Gian lận thẻ tín dụng là bài toán phân loại nhị phân với hai đặc điểm chi phối
mọi quyết định kỹ thuật về sau:

1. **Mất cân bằng cực đoan.** 492 giao dịch gian lận trên 284.807 giao dịch —
   0,172%. Một mô hình luôn trả lời "không gian lận" đạt 99,83% accuracy mà vô
   dụng hoàn toàn.
2. **Chi phí hai loại lỗi lệch nhau rất xa.** Bỏ lọt một giao dịch gian lận
   (false negative) mất trung bình toàn bộ số tiền giao dịch. Chặn nhầm một giao
   dịch hợp lệ (false positive) tốn chi phí thẩm định và làm phiền khách hàng.
   Tỷ lệ chi phí ước lượng khoảng 20–25 lần.

Hệ quả: bài toán không kết thúc ở chỗ "huấn luyện được mô hình có PR-AUC cao".
Nó kết thúc ở chỗ **chọn được ngưỡng quyết định phù hợp với ràng buộc chi phí và
năng lực thẩm định**, và giải thích được vì sao chọn ngưỡng đó.

## 2. Mục tiêu

### 2.1 Mục tiêu học thuật

| Mã | Mục tiêu |
|---|---|
| G-1 | So sánh có hệ thống các chiến lược xử lý mất cân bằng, có bằng chứng định lượng |
| G-2 | Chứng minh hiểu vì sao PR-AUC là metric chính, ROC-AUC gây hiểu nhầm, accuracy vô nghĩa |
| G-3 | Tránh triệt để rò rỉ dữ liệu — đặc biệt lỗi resample trước khi chia tập |
| G-4 | Chuyển từ xác suất sang quyết định nghiệp vụ qua phân tích chi phí |
| G-5 | Giải thích được dự đoán ở mức từng giao dịch bằng SHAP |

### 2.2 Mục tiêu sản phẩm

Xây dựng **Bảng điều khiển thẩm định gian lận** — ứng dụng web cho phép người
thẩm định xem hàng đợi giao dịch rủi ro, mở chi tiết kèm giải thích, chỉnh ngưỡng
và thấy ngay tác động lên số cảnh báo, chi phí kỳ vọng và độ phủ gian lận.

Ứng dụng không nhằm mô phỏng hệ thống sản xuất thật. Nó tồn tại để **làm luận
điểm chính của đồ án trở nên nhìn thấy được**: ngưỡng là một quyết định đánh đổi,
không phải hằng số 0,5.

## 3. Phạm vi

### 3.1 Trong phạm vi

- Phân tích khám phá và kiểm định thống kê trên `creditcard.csv`.
- Lưới thí nghiệm 5 chiến lược mất cân bằng × 4 thuật toán.
- Tinh chỉnh siêu tham số cho cấu hình thắng cuộc.
- Phân tích chi phí và tối ưu ngưỡng.
- Giải thích mô hình bằng SHAP ở mức toàn cục và mức từng giao dịch.
- Ứng dụng web bốn màn hình + API phục vụ.
- Đóng gói Docker Compose, chạy bằng một lệnh.
- Báo cáo và bộ tài liệu này.

### 3.2 Ngoài phạm vi

| Hạng mục | Lý do loại trừ |
|---|---|
| Huấn luyện trực tuyến / học tăng cường | Dữ liệu tĩnh, hai ngày, không có luồng thật |
| Kỹ thuật đặc trưng theo hành vi (số giao dịch/giờ của cùng thẻ) | Dữ liệu đã ẩn danh qua PCA, không có định danh thẻ |
| Xác thực người dùng, phân quyền | Demo cục bộ một người dùng |
| Triển khai lên đám mây, CI/CD, giám sát drift | Vượt quỹ thời gian 3 tuần |
| Chống gian lận thời gian thực ở quy mô nghìn TPS | Không phải mục tiêu học thuật |
| Hạng mục Chương 7 của chương trình học | Không vận dụng được với dữ liệu này |

### 3.3 Giới hạn của dữ liệu phải nêu trong báo cáo

- V1–V28 là thành phần chính sau PCA — **không diễn giải được về mặt nghiệp vụ**.
  Mọi kết luận SHAP chỉ nói được "đặc trưng V14 đẩy điểm rủi ro lên", không nói
  được "vì giao dịch diễn ra ở nước ngoài".
- Dữ liệu chỉ trải trong **hai ngày tháng 9/2013** tại châu Âu. Không đại diện
  cho mùa vụ, không đại diện cho thị trường khác.
- Nhãn `Class` được gán sau điều tra, có độ trễ; trong thực tế nhãn không có sẵn
  tại thời điểm chấm điểm.
- 492 mẫu dương là rất ít: khoảng tin cậy của mọi chỉ số sẽ rộng. Bắt buộc báo
  cáo khoảng tin cậy bootstrap, không báo cáo con số trần trụi.

## 4. Các bên liên quan

| Vai | Quan tâm chính |
|---|---|
| Sinh viên thực hiện | Hoàn thành đúng hạn, đạt tiêu chí nghiệm thu |
| Giảng viên chấm | Tính đúng đắn phương pháp, khả năng tái lập, chiều sâu phân tích |
| Người thẩm định (vai giả lập trong demo) | Hàng đợi ưu tiên đúng, giải thích dễ hiểu |
| Quản lý rủi ro (vai giả lập trong demo) | Đánh đổi precision–recall, chi phí kỳ vọng |

## 5. Sản phẩm bàn giao

| Mã | Sản phẩm | Tệp / vị trí |
|---|---|---|
| D-1 | Notebook phân tích, chạy được tuần tự | `notebooks/01..08` |
| D-2 | Mô hình đã lưu dạng pipeline đầy đủ | `models/model.joblib` |
| D-3 | Bảng so sánh 20 tổ hợp chiến lược × mô hình | `reports/bao-cao.md` |
| D-4 | Báo cáo trade-off Precision–Recall và phân tích chi phí | `reports/bao-cao.md` |
| D-5 | Ứng dụng web + API, chạy bằng `docker compose up` | `api/`, `web/` |
| D-6 | Bộ tài liệu đặc tả và thiết kế | `docs/` |
| D-7 | Bộ kiểm thử tự động | `tests/` |

## 6. Tiêu chí thành công tổng thể

Dự án được coi là thành công khi cả ba điều sau đồng thời đúng:

1. Mọi tiêu chí nghiệm thu bắt buộc trong [08 — Kế hoạch kiểm thử](08-ke-hoach-kiem-thu.md) đạt.
2. Người chấm chạy lại được toàn bộ từ máy sạch theo [10 — Vận hành và tái lập](10-van-hanh-tai-lap.md)
   và thu được PR-AUC lệch dưới 0,001 so với số trong báo cáo.
3. Báo cáo trả lời được câu hỏi "vì sao chọn ngưỡng này" bằng lập luận chi phí,
   không phải bằng "vì F1 cao nhất".

## 7. Thuật ngữ

| Thuật ngữ | Nghĩa dùng trong tài liệu này |
|---|---|
| Điểm rủi ro (risk score) | Xác suất dự đoán thuộc lớp gian lận, giá trị trong [0, 1] |
| Ngưỡng (threshold) | Mốc cắt biến điểm rủi ro thành quyết định chặn / cho qua |
| PR-AUC | Diện tích dưới đường Precision–Recall, tính bằng `average_precision_score` |
| FN / bỏ lọt | Giao dịch gian lận bị mô hình cho qua |
| FP / báo động giả | Giao dịch hợp lệ bị mô hình chặn |
| Chi phí kỳ vọng | `cost_fn × số_FN + cost_fp × số_FP` tại một ngưỡng cho trước |
| Hiện vật (artifact) | Tệp sinh ra từ giai đoạn huấn luyện, được API nạp lúc khởi động |
| Rò rỉ dữ liệu | Thông tin từ tập kiểm thử ảnh hưởng tới quá trình huấn luyện |
