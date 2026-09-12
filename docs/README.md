# Tài liệu dự án — Phát hiện gian lận thẻ tín dụng

Thư mục này chứa toàn bộ đặc tả và thiết kế của hệ thống. Mỗi tệp có một phạm vi
riêng, không lặp nội dung của nhau. Khi một quyết định thay đổi, sửa đúng tệp
sở hữu quyết định đó — các tệp khác chỉ tham chiếu tới.

## Đọc theo thứ tự

| # | Tệp | Trả lời câu hỏi | Đối tượng |
|---|---|---|---|
| 00 | [Tổng quan dự án](00-tong-quan.md) | Làm cái gì, vì sao, ranh giới đến đâu | Tất cả |
| 01 | [Đặc tả yêu cầu](01-dac-ta-yeu-cau.md) | Hệ thống phải làm được những gì | Tất cả |
| 02 | [Đặc tả dữ liệu](02-dac-ta-du-lieu.md) | Dữ liệu có gì, hợp đồng dữ liệu ra sao | Người làm mô hình |
| 03 | [Thiết kế kiến trúc](03-thiet-ke-kien-truc.md) | Hệ thống gồm khối nào, ghép ra sao | Người lập trình |
| 04 | [Thiết kế mô hình ML](04-thiet-ke-mo-hinh-ml.md) | Huấn luyện, đánh giá, chọn ngưỡng thế nào | Người làm mô hình |
| 05 | [Thiết kế API](05-thiet-ke-api.md) | Hợp đồng giữa backend và frontend | Người lập trình |
| 06 | [Thiết kế lưu trữ](06-thiet-ke-luu-tru.md) | Cơ sở dữ liệu và hiện vật mô hình | Người lập trình |
| 07 | [Thiết kế giao diện](07-thiet-ke-giao-dien.md) | Bốn màn hình trông và hoạt động thế nào | Người lập trình |
| 08 | [Kế hoạch kiểm thử](08-ke-hoach-kiem-thu.md) | Chứng minh hệ thống đúng bằng cách nào | Tất cả |
| 09 | [Kế hoạch triển khai](09-ke-hoach-trien-khai.md) | Ai làm gì, ngày nào, xong là thế nào | Tất cả |
| 10 | [Vận hành và tái lập](10-van-hanh-tai-lap.md) | Chạy lại toàn bộ từ đầu bằng lệnh gì | Người chấm, người mới |

## Quy ước mã định danh

Mọi hạng mục có mã riêng để trích dẫn chéo giữa các tài liệu và trong báo cáo:

| Tiền tố | Ý nghĩa | Nơi định nghĩa |
|---|---|---|
| `FR-xx` | Yêu cầu chức năng | 01 |
| `NFR-xx` | Yêu cầu phi chức năng | 01 |
| `DS-xx` | Quy tắc về dữ liệu | 02 |
| `ML-xx` | Ràng buộc mô hình | 04 |
| `API-xx` | Endpoint | 05 |
| `UI-xx` | Màn hình / thành phần giao diện | 07 |
| `AC-xx` | Tiêu chí nghiệm thu | 08 |
| `TC-xx` | Ca kiểm thử | 08 |
| `R-xx` | Rủi ro | 09 |

## Trạng thái tài liệu

| Phiên bản | Ngày | Nội dung |
|---|---|---|
| 1.0 | 2026-09-12 | Bản đầu tiên, tách từ hai tài liệu nháp ở thư mục gốc |
| 1.1 | 2026-09-12 | Chuyển cơ sở dữ liệu từ SQLite sang PostgreSQL 16: viết lại 06, cập nhật 01, 02, 03, 05, 08, 09, 10 |

## Quan hệ với các tệp ở thư mục gốc

Hai tệp dưới đây là bản nháp ban đầu, được giữ lại để tham chiếu. Khi có mâu
thuẫn, **tài liệu trong `docs/` là bản có hiệu lực**.

- `dac-ta-du-an-fraud-detection.md` — bản đặc tả gộp đầu tiên.
- `docs.md` — hướng dẫn triển khai chi tiết theo lộ trình 3 tuần, nặng về mã mẫu
  và giải thích lý thuyết. Vẫn hữu ích như tài liệu học, nhưng không phải đặc tả.
