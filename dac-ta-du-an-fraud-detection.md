# ĐẶC TẢ DỰ ÁN
# Hệ thống phát hiện gian lận thẻ tín dụng
### Phiên bản 1.0 — Đồ án cuối khóa AI/Data Science

---

## MỤC LỤC

1. [Tổng quan và phạm vi](#1-tổng-quan-và-phạm-vi)
2. [Người dùng và tình huống sử dụng](#2-người-dùng-và-tình-huống-sử-dụng)
3. [Kiến trúc hệ thống](#3-kiến-trúc-hệ-thống)
4. [Đặc tả phần mô hình học máy](#4-đặc-tả-phần-mô-hình-học-máy)
5. [Đặc tả ứng dụng demo](#5-đặc-tả-ứng-dụng-demo)
6. [Đặc tả API](#6-đặc-tả-api)
7. [Mô hình dữ liệu](#7-mô-hình-dữ-liệu)
8. [Yêu cầu phi chức năng](#8-yêu-cầu-phi-chức-năng)
9. [Công nghệ và lý do lựa chọn](#9-công-nghệ-và-lý-do-lựa-chọn)
10. [Cấu trúc mã nguồn](#10-cấu-trúc-mã-nguồn)
11. [Kế hoạch triển khai](#11-kế-hoạch-triển-khai)
12. [Tiêu chí nghiệm thu](#12-tiêu-chí-nghiệm-thu)
13. [Rủi ro](#13-rủi-ro)
14. [Ngoài phạm vi](#14-ngoài-phạm-vi)

---

## 1. TỔNG QUAN VÀ PHẠM VI

### 1.1 Mục đích

Dự án xây dựng một hệ thống hoàn chỉnh gồm hai phần:

- **Phần nghiên cứu:** huấn luyện và so sánh các mô hình phát hiện giao dịch gian lận trên dữ liệu mất cân bằng cực đoan (tỷ lệ lớp dương 0,172%).
- **Phần ứng dụng:** một công cụ web mô phỏng màn hình làm việc của nhân viên thẩm định gian lận, cho phép chấm điểm rủi ro giao dịch, điều chỉnh ngưỡng theo chi phí và xem giải thích cho từng quyết định.

Phần ứng dụng không phải là bản demo trang trí. Nó là nơi chứng minh rằng các lựa chọn kỹ thuật trong phần nghiên cứu — đặc biệt là việc chọn ngưỡng theo chi phí thay vì mặc định 0,5 — thực sự có ý nghĩa vận hành.

### 1.2 Nguồn dữ liệu

| Thuộc tính | Giá trị |
|---|---|
| Tên | Credit Card Fraud Detection |
| Chủ sở hữu | Machine Learning Group — Université Libre de Bruxelles (mlg-ulb) |
| Địa chỉ | https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud |
| Tệp | `creditcard.csv` — 284.807 dòng × 31 cột, ~144 MB |
| Giấy phép | Database Contents License (DbCL) v1.0 |
| Phạm vi | Giao dịch của chủ thẻ châu Âu, 2 ngày trong tháng 9/2013 |
| Nhãn | Có sẵn (cột `Class`), không cần gắn nhãn thủ công |

### 1.3 Ranh giới hệ thống

**Trong phạm vi:**
- Huấn luyện, đánh giá, so sánh mô hình phân loại nhị phân.
- Đóng gói mô hình thành dịch vụ dự đoán qua HTTP.
- Ứng dụng web cho phép chấm điểm hàng loạt, xem xét từng giao dịch, điều chỉnh ngưỡng, xem giải thích SHAP.
- Đóng gói toàn bộ bằng Docker Compose để chạy bằng một lệnh.

**Ngoài phạm vi:** xem [Phần 14](#14-ngoài-phạm-vi).

---

## 2. NGƯỜI DÙNG VÀ TÌNH HUỐNG SỬ DỤNG

### 2.1 Hai vai người dùng

**Nhân viên thẩm định (vai chính).** Mỗi ngày nhận một danh sách giao dịch bị hệ thống gắn cờ, cần quyết định chặn hay cho qua. Ngân sách thời gian có hạn — chỉ xử lý được khoảng 100 vụ mỗi ngày. Điều họ cần: danh sách xếp theo mức rủi ro giảm dần, và lý do vì sao mỗi giao dịch bị gắn cờ.

**Trưởng nhóm rủi ro (vai phụ).** Không xem từng giao dịch mà quan tâm tới cấu hình hệ thống: đặt ngưỡng ở đâu, mỗi ngày sinh ra bao nhiêu cảnh báo, tỷ lệ bắt đúng bao nhiêu, tổng chi phí ước tính là bao nhiêu.

Hai vai này dẫn tới hai nhóm màn hình khác nhau trong ứng dụng, và cũng là hai câu chuyện khác nhau khi trình bày bảo vệ.

### 2.2 User story

| Mã | Vai | Mô tả | Ưu tiên |
|---|---|---|---|
| US-01 | Thẩm định | Tải lên tệp CSV giao dịch và nhận điểm rủi ro cho từng dòng | Bắt buộc |
| US-02 | Thẩm định | Xem danh sách giao dịch bị gắn cờ, sắp xếp theo điểm rủi ro | Bắt buộc |
| US-03 | Thẩm định | Mở chi tiết một giao dịch và xem những yếu tố đẩy điểm rủi ro lên cao | Bắt buộc |
| US-04 | Thẩm định | Đánh dấu một giao dịch là “xác nhận gian lận” hoặc “báo động sai” | Nên có |
| US-05 | Trưởng nhóm | Kéo thanh trượt ngưỡng và thấy ngay số cảnh báo, Recall, Precision thay đổi | Bắt buộc |
| US-06 | Trưởng nhóm | Nhập chi phí bỏ lọt và chi phí thẩm định, hệ thống đề xuất ngưỡng tối ưu | Bắt buộc |
| US-07 | Trưởng nhóm | Xem tổng quan hiệu năng mô hình trên tập kiểm thử | Nên có |
| US-08 | Cả hai | Chạy chế độ phát lại dòng giao dịch của ngày thứ hai theo thời gian tăng tốc | Nên có |
| US-09 | Trưởng nhóm | So sánh kết quả của nhiều mô hình đã huấn luyện | Tùy chọn |

### 2.3 Vấn đề thiết kế quan trọng

Đặc trưng V1–V28 là thành phần chính sau PCA, không mang ý nghĩa nghiệp vụ. **Không thể yêu cầu người dùng nhập tay 28 con số vô nghĩa.** Ứng dụng vì vậy không có màn hình “nhập giao dịch thủ công”. Thay vào đó có ba cách đưa dữ liệu vào:

1. **Tải tệp CSV** đúng lược đồ dữ liệu gốc.
2. **Chọn từ thư viện mẫu** — khoảng 200 giao dịch trích từ tập kiểm thử, gồm cả gian lận lẫn bình thường, để trình bày nhanh khi bảo vệ.
3. **Phát lại dòng giao dịch** — chạy lại các giao dịch của ngày thứ hai theo trục thời gian được tăng tốc, mô phỏng hệ thống giám sát thời gian thực.

Cách thứ ba là điểm nhấn khi demo: hội đồng nhìn thấy giao dịch chảy qua màn hình, thỉnh thoảng một dòng chuyển đỏ và nhảy vào hàng đợi thẩm định.

---

## 3. KIẾN TRÚC HỆ THỐNG

### 3.1 Sơ đồ tổng thể

```
┌─────────────────────────────────────────────────────────┐
│  GIAI ĐOẠN OFFLINE — huấn luyện (chạy trong notebook)   │
│                                                          │
│  creditcard.csv                                          │
│       │                                                  │
│       ├─→ EDA + kiểm định thống kê                      │
│       │                                                  │
│       ├─→ Chia phân tầng 80/20  ──→ test_set.parquet    │
│       │                                                  │
│       └─→ Huấn luyện & tinh chỉnh                       │
│              │                                           │
│              ├─→ model.joblib      (pipeline đầy đủ)    │
│              ├─→ explainer.joblib  (SHAP TreeExplainer) │
│              ├─→ metrics.json      (kết quả đánh giá)   │
│              └─→ threshold.json    (ngưỡng đề xuất)     │
└─────────────────────────────────────────────────────────┘
                          │
                          ▼  (nạp lúc khởi động)
┌─────────────────────────────────────────────────────────┐
│  GIAI ĐOẠN ONLINE — phục vụ                             │
│                                                          │
│   ┌──────────────┐   HTTP/JSON   ┌──────────────────┐   │
│   │   Frontend   │ ◄───────────► │   API FastAPI    │   │
│   │              │               │                  │   │
│   │ • Hàng đợi   │               │ • /score         │   │
│   │ • Chi tiết   │               │ • /score/batch   │   │
│   │ • Ngưỡng     │               │ • /explain       │   │
│   │ • Bảng điều  │               │ • /threshold/*   │   │
│   │   khiển      │               │ • /metrics       │   │
│   └──────────────┘               │ • /review        │   │
│                                   └────────┬─────────┘   │
│                                            │             │
│                                   ┌────────▼─────────┐   │
│                                   │  SQLite          │   │
│                                   │  • transactions  │   │
│                                   │  • reviews       │   │
│                                   │  • settings      │   │
│                                   └──────────────────┘   │
└─────────────────────────────────────────────────────────┘
```

### 3.2 Nguyên tắc kiến trúc

**Tách hoàn toàn huấn luyện khỏi phục vụ.** API không bao giờ huấn luyện lại mô hình. Nó chỉ nạp các tệp hiện vật đã sinh ra từ notebook. Điều này giữ cho phần nghiên cứu và phần ứng dụng độc lập, và bảo đảm kết quả demo khớp chính xác với kết quả trong báo cáo.

**Mô hình lưu dạng pipeline hoàn chỉnh.** Tệp `model.joblib` chứa cả bước chuẩn hóa lẫn bộ phân loại. Nếu chỉ lưu bộ phân loại, phía phục vụ phải tự tái tạo bước tiền xử lý và sẽ lệch — đây là lỗi triển khai phổ biến nhất.

**Ngưỡng là cấu hình, không phải một phần của mô hình.** Mô hình luôn trả về xác suất. Việc chuyển xác suất thành quyết định chặn/cho qua diễn ra ở tầng ứng dụng, dựa trên ngưỡng có thể thay đổi lúc chạy. Nhờ vậy thanh trượt ngưỡng trong giao diện phản hồi tức thì mà không cần tính lại gì.

---

## 4. ĐẶC TẢ PHẦN MÔ HÌNH HỌC MÁY

### 4.1 Hợp đồng dữ liệu vào

Mọi dữ liệu đưa vào hệ thống phải tuân thủ lược đồ sau:

| Trường | Kiểu | Bắt buộc | Ràng buộc |
|---|---|---|---|
| `Time` | float | Có | ≥ 0, đơn vị giây |
| `V1` … `V28` | float | Có | Không giới hạn |
| `Amount` | float | Có | ≥ 0 |
| `Class` | int | Không | 0 hoặc 1; chỉ có khi đánh giá |

Thiếu bất kỳ cột nào trong 30 cột bắt buộc → trả lỗi 422 kèm danh sách cột thiếu.

### 4.2 Đặc trưng đưa vào mô hình

30 đặc trưng:

| Đặc trưng | Nguồn | Xử lý |
|---|---|---|
| `V1`–`V28` | Nguyên bản | Giữ nguyên (đã chuẩn hóa sẵn qua PCA) |
| `Amount` | Nguyên bản | RobustScaler |
| `hour` | Dẫn xuất: `(Time // 3600) % 24` | Chuẩn hóa vòng tròn: `sin(2πh/24)`, `cos(2πh/24)` |

Cột `Time` thô **không** được dùng làm đặc trưng. Nó là số giây tương đối trong một cửa sổ hai ngày cụ thể; đưa vào mô hình sẽ khiến mô hình học một đặc điểm không tồn tại khi triển khai. Chỉ giờ trong ngày mới mang ý nghĩa lặp lại được.

Mã hóa giờ theo sin/cos thay vì số nguyên 0–23 để mô hình hiểu 23 giờ và 0 giờ là kề nhau.

### 4.3 Quy trình huấn luyện

```
B1. Nạp dữ liệu, kiểm tra toàn vẹn, xử lý dòng trùng lặp
B2. Sinh đặc trưng hour_sin, hour_cos
B3. Chia phân tầng 80/20, random_state=42
      → lưu tập test ra test_set.parquet để ứng dụng dùng lại
B4. Định nghĩa pipeline imblearn: [scaler] → [resampler] → [classifier]
B5. Chạy lưới so sánh:  5 chiến lược mất cân bằng × 4 mô hình
      → StratifiedKFold(5), scoring='average_precision'
B6. Chọn cấu hình tốt nhất theo PR-AUC trung bình qua các fold
B7. Tinh chỉnh siêu tham số bằng RandomizedSearchCV (30 lần thử)
B8. Huấn luyện lại trên toàn bộ tập train, đánh giá trên tập test
B9. Tính đường cong chi phí, xác định ngưỡng tối ưu
B10. Sinh SHAP explainer
B11. Lưu hiện vật
```

### 4.4 Lưới thí nghiệm

**Chiến lược mất cân bằng:** không xử lý · `class_weight='balanced'` · giảm mẫu ngẫu nhiên · SMOTE · SMOTE + Tomek Links

**Mô hình:** Hồi quy logistic · Random Forest · XGBoost · LightGBM

Tổng cộng 20 tổ hợp. Ghi lại toàn bộ vào một bảng duy nhất — đây là bảng trung tâm của báo cáo.

### 4.5 Ràng buộc bắt buộc

| Ràng buộc | Lý do |
|---|---|
| Lấy mẫu lại chỉ nằm trong `imblearn.pipeline.Pipeline` | Nếu resample trước khi chia, mẫu nội suy từ tập test lọt vào tập train, mọi chỉ số tăng ảo lên gần 0,99 |
| `StratifiedKFold`, không dùng `KFold` | Với 0,172% lớp dương, một fold có thể không chứa mẫu gian lận nào |
| `scoring='average_precision'` khi tinh chỉnh | Nếu tối ưu theo accuracy, mô hình thắng cuộc sẽ là mô hình dự đoán toàn lớp 0 |
| Scaler `fit` chỉ trên tập train | Tránh rò rỉ thống kê từ tập test |
| Cố định `random_state=42` ở mọi bước ngẫu nhiên | Yêu cầu tái lập kết quả |

### 4.6 Hiện vật đầu ra

| Tệp | Nội dung | Bên sử dụng |
|---|---|---|
| `models/model.joblib` | Pipeline hoàn chỉnh (scaler + classifier) | API |
| `models/explainer.joblib` | SHAP TreeExplainer | API, endpoint `/explain` |
| `models/threshold.json` | Ngưỡng mặc định + tham số chi phí | API |
| `models/metrics.json` | Toàn bộ chỉ số trên tập test, đường PR, đường cong chi phí | API, màn hình bảng điều khiển |
| `data/test_set.parquet` | Tập kiểm thử kèm nhãn thật | Chế độ phát lại, thư viện mẫu |
| `data/sample_pool.json` | ~200 giao dịch chọn lọc để demo | Thư viện mẫu |

Cấu trúc `threshold.json`:

```json
{
  "default_threshold": 0.0473,
  "selection_method": "min_expected_cost",
  "cost_false_negative": 122.21,
  "cost_false_positive": 5.00,
  "alternatives": {
    "max_f1": 0.2140,
    "recall_at_least_90": 0.0089,
    "budget_200_alerts": 0.0612,
    "default_naive": 0.5000
  },
  "model_version": "xgb_scaleposweight_v3",
  "trained_at": "2026-09-15T10:22:41Z"
}
```

---

## 5. ĐẶC TẢ ỨNG DỤNG DEMO

### 5.1 Khái niệm sản phẩm

**Tên:** Bảng điều khiển thẩm định gian lận (Fraud Review Console)

Ứng dụng web một trang, bốn màn hình chính điều hướng bằng thanh bên. Không có đăng nhập — đây là bản demo cục bộ.

### 5.2 Màn hình 1 — Hàng đợi thẩm định

Màn hình mặc định khi mở ứng dụng. Danh sách giao dịch có điểm rủi ro vượt ngưỡng hiện tại, sắp xếp giảm dần.

```
┌────────────────────────────────────────────────────────────────┐
│  Hàng đợi thẩm định                    Ngưỡng: 0,047  [Đổi]    │
│  ────────────────────────────────────────────────────────────  │
│  187 cảnh báo · ước tính bắt được 84% gian lận                 │
│                                                                 │
│  [Tải CSV]  [Chọn mẫu có sẵn]  [▶ Phát lại ngày 2]            │
│  ────────────────────────────────────────────────────────────  │
│   Rủi ro │ Mã GD   │ Số tiền   │ Giờ   │ Trạng thái │        │
│  ─────────────────────────────────────────────────────────    │
│   ●99,7% │ TX-8841 │  1.809,68 │ 02:14 │ Chờ xử lý  │  ▸     │
│   ●98,2% │ TX-1207 │      0,76 │ 03:41 │ Chờ xử lý  │  ▸     │
│   ●94,1% │ TX-5563 │    529,00 │ 02:58 │ Đã chặn    │  ▸     │
│   ○71,3% │ TX-9930 │     14,20 │ 11:02 │ Báo động sai│ ▸     │
│   ○68,8% │ TX-4415 │    212,50 │ 22:37 │ Chờ xử lý  │  ▸     │
│  ─────────────────────────────────────────────────────────    │
│                                              ‹ 1 2 3 … 8 ›     │
└────────────────────────────────────────────────────────────────┘
```

**Yêu cầu chức năng:**
- Điểm rủi ro hiển thị dạng phần trăm, kèm chấm màu: đỏ ≥ 90%, cam 60–90%, vàng dưới 60%.
- Lọc theo trạng thái: chờ xử lý / đã chặn / báo động sai / tất cả.
- Sắp xếp theo điểm rủi ro hoặc số tiền.
- Phân trang 25 dòng mỗi trang.
- Đổi ngưỡng ở thanh trên cùng cập nhật lại danh sách ngay, không tải lại trang.
- Trạng thái rỗng khi chưa có dữ liệu: hiển thị hướng dẫn ba cách nạp dữ liệu.

### 5.3 Màn hình 2 — Chi tiết giao dịch

Mở ra khi bấm vào một dòng. Dạng ngăn kéo trượt từ phải hoặc trang riêng.

```
┌────────────────────────────────────────────────────────────────┐
│  TX-8841                                              [Đóng]   │
│  ════════════════════════════════════════════════════════════  │
│                                                                 │
│      Điểm rủi ro                                                │
│         99,7%          Vượt ngưỡng 0,047 → đề xuất CHẶN        │
│      ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓░               │
│                                                                 │
│      Số tiền   1.809,68        Giờ giao dịch   02:14           │
│  ────────────────────────────────────────────────────────────  │
│  Yếu tố đẩy điểm rủi ro lên                                    │
│                                                                 │
│    V14   -8,42  ████████████████████████  +0,31                │
│    V17   -6,17  ██████████████████        +0,24                │
│    V12   -5,88  ███████████████           +0,19                │
│    V10   -4,03  ██████████                +0,12                │
│  ────────────────────────────────────────────────────────────  │
│  Yếu tố kéo điểm rủi ro xuống                                  │
│                                                                 │
│    V4     1,12          ████                      -0,05        │
│  ────────────────────────────────────────────────────────────  │
│                                                                 │
│   [Xác nhận gian lận]        [Đánh dấu báo động sai]           │
└────────────────────────────────────────────────────────────────┘
```

**Yêu cầu chức năng:**
- Hiển thị 5 đặc trưng đóng góp dương và 3 đặc trưng đóng góp âm lớn nhất theo giá trị SHAP.
- Kèm chú thích cố định giải thích rằng V1–V28 là thành phần PCA nên không diễn giải được ý nghĩa nghiệp vụ — đây là hệ quả của việc ẩn danh dữ liệu. Không được bịa ra tên nghiệp vụ cho các biến này.
- Hai nút hành động ghi kết quả thẩm định vào cơ sở dữ liệu và cập nhật trạng thái ở hàng đợi.
- Nếu giao dịch đến từ tập kiểm thử và có nhãn thật, hiển thị nhãn đó **sau khi** người dùng đã quyết định, để có thể tự đối chiếu.

### 5.4 Màn hình 3 — Cấu hình ngưỡng

Màn hình dành cho vai trưởng nhóm rủi ro. Đây là màn hình có giá trị học thuật cao nhất.

```
┌────────────────────────────────────────────────────────────────┐
│  Cấu hình ngưỡng quyết định                                    │
│  ════════════════════════════════════════════════════════════  │
│                                                                 │
│  Ngưỡng   ├────────●──────────────────────────────┤   0,047    │
│         0,001                                    0,999          │
│                                                                 │
│  ┌──────────────┬──────────────┬──────────────┬─────────────┐ │
│  │ Cảnh báo/ngày│ Bắt được     │ Precision    │ Chi phí/ngày│ │
│  │      94      │  83 / 98     │    0,44      │  3.312 EUR  │ │
│  └──────────────┴──────────────┴──────────────┴─────────────┘ │
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
│    └──────────────────────────────────────────► ngưỡng         │
│                                                                 │
│  Ngưỡng gợi ý sẵn:                                              │
│   ( ) 0,500  Mặc định — không có cơ sở cho dữ liệu mất cân bằng│
│   (•) 0,047  Cực tiểu chi phí kỳ vọng                          │
│   ( ) 0,214  F1 lớn nhất                                        │
│   ( ) 0,009  Ràng buộc Recall ≥ 90%                            │
│   ( ) 0,061  Ngân sách 200 lượt thẩm định                      │
│                                                                 │
│                                     [Áp dụng ngưỡng này]        │
└────────────────────────────────────────────────────────────────┘
```

**Yêu cầu chức năng:**
- Kéo thanh trượt cập nhật cả bốn ô chỉ số trong vòng 200 ms. Tính toán thực hiện ở phía máy khách dựa trên mảng xác suất đã nạp sẵn của tập kiểm thử — không gọi API mỗi lần kéo.
- Thay đổi hai ô chi phí sẽ tính lại đường cong và vị trí cực tiểu.
- Chọn một ngưỡng gợi ý sẵn sẽ di chuyển thanh trượt tới đúng giá trị đó.
- Nút áp dụng ghi ngưỡng vào bảng `settings` và ảnh hưởng tới toàn bộ ứng dụng.

Màn hình này chính là câu trả lời trực quan cho câu hỏi *“vì sao không dùng ngưỡng 0,5”*. Khi trình bày, kéo thanh trượt từ 0,5 về 0,047 và cho hội đồng thấy số vụ bắt được nhảy từ khoảng 60 lên hơn 80 trong khi chi phí giảm — hiệu quả hơn nhiều so với giải thích bằng lời.

### 5.5 Màn hình 4 — Hiệu năng mô hình

Trình bày kết quả từ `metrics.json`. Màn hình tĩnh, không tương tác nhiều.

**Nội dung bắt buộc:**
- Bảng so sánh 20 tổ hợp mô hình × chiến lược, sắp xếp theo PR-AUC.
- Đường cong Precision–Recall của 5 chiến lược vẽ chồng.
- Ma trận nhầm lẫn tại ngưỡng hiện tại, cập nhật khi ngưỡng đổi.
- Bảng đối chiếu ba mô hình cho thấy accuracy gần như không phân biệt được chúng còn PR-AUC thì có.
- Xếp hạng đặc trưng theo SHAP, đặt cạnh xếp hạng theo thống kê KS để đối chiếu.

### 5.6 Chế độ phát lại

Truy cập từ nút trên màn hình hàng đợi.

**Cách hoạt động:** đọc các giao dịch của ngày thứ hai từ `test_set.parquet` theo thứ tự cột `Time` tăng dần, đẩy sang giao diện với tốc độ nén thời gian có thể chỉnh (mặc định 1 giờ dữ liệu = 2 giây thực).

**Giao diện:** dòng giao dịch chảy từ trên xuống. Giao dịch dưới ngưỡng trôi qua màu xám nhạt. Giao dịch vượt ngưỡng chuyển đỏ, dừng lại một nhịp và rơi vào hàng đợi thẩm định.

**Điều khiển:** phát / tạm dừng / đặt lại, thanh chỉnh tốc độ, đồng hồ hiển thị thời gian mô phỏng.

**Cài đặt kỹ thuật:** dùng Server-Sent Events từ FastAPI. Đơn giản hơn WebSocket và đủ dùng vì luồng chỉ đi một chiều. Nếu triển khai bằng Streamlit thì dùng vòng lặp với `st.empty()` và `time.sleep()`.

Chế độ này chiếm khoảng một ngày công nhưng là thứ hội đồng sẽ nhớ. Nếu thiếu thời gian, đây là hạng mục nên giữ lại và cắt các hạng mục khác.

---

## 6. ĐẶC TẢ API

Máy chủ FastAPI, tiền tố `/api/v1`, toàn bộ dữ liệu trao đổi dạng JSON.

### 6.1 Bảng endpoint

| Phương thức | Đường dẫn | Chức năng |
|---|---|---|
| GET | `/health` | Kiểm tra tình trạng, trả về phiên bản mô hình |
| POST | `/score` | Chấm điểm một giao dịch |
| POST | `/score/batch` | Chấm điểm nhiều giao dịch (tối đa 50.000) |
| POST | `/score/upload` | Nhận tệp CSV, chấm điểm, lưu vào cơ sở dữ liệu |
| POST | `/explain` | Trả về giá trị SHAP cho một giao dịch |
| GET | `/transactions` | Danh sách giao dịch, có lọc và phân trang |
| GET | `/transactions/{id}` | Chi tiết một giao dịch |
| POST | `/reviews` | Ghi kết quả thẩm định |
| GET | `/threshold` | Lấy ngưỡng hiện tại |
| PUT | `/threshold` | Đặt ngưỡng mới |
| POST | `/threshold/optimize` | Tính ngưỡng tối ưu theo tham số chi phí |
| GET | `/metrics` | Toàn bộ chỉ số đánh giá mô hình |
| GET | `/samples` | Thư viện giao dịch mẫu |
| GET | `/replay/stream` | Server-Sent Events cho chế độ phát lại |

### 6.2 Chi tiết các endpoint chính

**POST `/api/v1/score`**

Yêu cầu:
```json
{
  "transaction": {
    "Time": 7834.0,
    "V1": -1.3598, "V2": -0.0728, "V3": 2.5363,
    "V4": 1.3782,  "V5": -0.3383, "V6": 0.4624,
    "V7": 0.2396,  "V8": 0.0987,  "V9": 0.3638,
    "V10": 0.0908, "V11": -0.5516,"V12": -0.6178,
    "V13": -0.9914,"V14": -0.3112,"V15": 1.4682,
    "V16": -0.4704,"V17": 0.2080, "V18": 0.0258,
    "V19": 0.4040, "V20": 0.2514, "V21": -0.0183,
    "V22": 0.2778, "V23": -0.1105,"V24": 0.0669,
    "V25": 0.1285, "V26": -0.1891,"V27": 0.1336,
    "V28": -0.0211,
    "Amount": 149.62
  }
}
```

Phản hồi `200`:
```json
{
  "transaction_id": "TX-8841",
  "risk_score": 0.0037,
  "threshold": 0.0473,
  "decision": "allow",
  "risk_band": "low",
  "model_version": "xgb_scaleposweight_v3",
  "scored_at": "2026-09-20T14:03:11Z",
  "latency_ms": 4
}
```

Trường `decision` chỉ nhận `"allow"` hoặc `"review"`. Hệ thống **không** tự động chặn giao dịch — mọi trường hợp vượt ngưỡng đều chuyển sang thẩm định thủ công. Đây là lựa chọn thiết kế có chủ ý: với Precision khoảng 0,44 tại ngưỡng tối ưu, chặn tự động sẽ từ chối nhầm quá nhiều giao dịch hợp lệ.

**POST `/api/v1/explain`**

Phản hồi `200`:
```json
{
  "transaction_id": "TX-8841",
  "risk_score": 0.9971,
  "base_value": 0.0017,
  "contributions": [
    {"feature": "V14", "value": -8.4213, "shap": 0.3104},
    {"feature": "V17", "value": -6.1702, "shap": 0.2417},
    {"feature": "V12", "value": -5.8834, "shap": 0.1893},
    {"feature": "V10", "value": -4.0291, "shap": 0.1245},
    {"feature": "Amount", "value": 1809.68, "shap": 0.0388},
    {"feature": "V4", "value": 1.1204, "shap": -0.0512}
  ],
  "note": "V1–V28 là thành phần chính sau PCA, không có ý nghĩa nghiệp vụ trực tiếp."
}
```

**POST `/api/v1/threshold/optimize`**

Yêu cầu:
```json
{
  "cost_false_negative": 122.21,
  "cost_false_positive": 5.00,
  "constraint": null
}
```

Trường `constraint` tùy chọn, nhận `{"min_recall": 0.90}` hoặc `{"max_alerts": 200}`.

Phản hồi `200`:
```json
{
  "optimal_threshold": 0.0473,
  "expected_cost": 3312.44,
  "at_threshold": {
    "recall": 0.8367, "precision": 0.4415,
    "tp": 82, "fp": 105, "fn": 16, "tn": 56759,
    "alerts": 187
  },
  "comparison_default_05": {
    "threshold": 0.5, "recall": 0.6122, "precision": 0.8955,
    "expected_cost": 4832.19
  },
  "cost_curve": [
    {"threshold": 0.001, "cost": 9841.0},
    {"threshold": 0.011, "cost": 5120.3}
  ]
}
```

Trường `comparison_default_05` luôn được trả về để giao diện có thể đối chiếu trực tiếp với ngưỡng mặc định.

### 6.3 Xử lý lỗi

| Mã | Tình huống | Nội dung phản hồi |
|---|---|---|
| 400 | CSV sai định dạng | Mô tả lỗi phân tích cú pháp |
| 404 | Không tìm thấy giao dịch | Mã giao dịch đã yêu cầu |
| 413 | Tệp vượt 100 MB | Giới hạn kích thước |
| 422 | Thiếu cột bắt buộc | Danh sách cột thiếu |
| 422 | Ngưỡng ngoài khoảng (0, 1) | Khoảng hợp lệ |
| 503 | Chưa nạp được mô hình | Hướng dẫn chạy notebook huấn luyện trước |

Định dạng lỗi thống nhất:
```json
{
  "error": "missing_columns",
  "message": "Thiếu 3 cột bắt buộc trong tệp tải lên.",
  "details": {"missing": ["V7", "V19", "Amount"]}
}
```

---

## 7. MÔ HÌNH DỮ LIỆU

SQLite, một tệp duy nhất tại `data/app.db`. Đủ cho phạm vi demo; nếu cần đa người dùng thì thay bằng PostgreSQL mà không phải đổi mã nhờ dùng SQLAlchemy.

```sql
CREATE TABLE transactions (
    id              TEXT PRIMARY KEY,        -- 'TX-8841'
    time_offset     REAL NOT NULL,           -- cột Time gốc
    hour            INTEGER NOT NULL,        -- dẫn xuất
    amount          REAL NOT NULL,
    features        TEXT NOT NULL,           -- JSON mảng 28 giá trị V
    risk_score      REAL NOT NULL,
    model_version   TEXT NOT NULL,
    true_label      INTEGER,                 -- NULL nếu dữ liệu không có nhãn
    source          TEXT NOT NULL,           -- 'upload' | 'sample' | 'replay'
    batch_id        TEXT,
    created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_tx_risk   ON transactions(risk_score DESC);
CREATE INDEX idx_tx_batch  ON transactions(batch_id);

CREATE TABLE reviews (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    transaction_id  TEXT NOT NULL REFERENCES transactions(id),
    decision        TEXT NOT NULL,           -- 'confirmed_fraud' | 'false_alarm'
    threshold_used  REAL NOT NULL,
    reviewed_at     TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    note            TEXT
);

CREATE UNIQUE INDEX idx_review_tx ON reviews(transaction_id);

CREATE TABLE settings (
    key             TEXT PRIMARY KEY,
    value           TEXT NOT NULL,
    updated_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
-- các khóa: 'threshold', 'cost_fn', 'cost_fp', 'replay_speed'
```

Ghi chú thiết kế: bảng `reviews` cho phép so sánh quyết định của con người với nhãn thật ở cột `true_label`. Nếu có thời gian, đây là cơ sở cho một hạng mục mở rộng thú vị — đo xem người thẩm định có làm tốt hơn hay tệ hơn mô hình.

---

## 8. YÊU CẦU PHI CHỨC NĂNG

| Mã | Yêu cầu | Ngưỡng chấp nhận |
|---|---|---|
| NFR-01 | Độ trễ chấm điểm một giao dịch | p95 < 50 ms |
| NFR-02 | Thông lượng chấm điểm hàng loạt | ≥ 10.000 giao dịch trong 30 giây |
| NFR-03 | Thời gian phản hồi khi kéo thanh trượt ngưỡng | < 200 ms |
| NFR-04 | Thời gian khởi động toàn hệ thống | < 15 giây kể từ `docker compose up` |
| NFR-05 | Bộ nhớ tối đa của API | < 1 GB |
| NFR-06 | Kích thước tệp CSV tối đa | 100 MB |
| NFR-07 | Khả năng tái lập kết quả | Chạy lại notebook cho ra PR-AUC lệch < 0,001 |
| NFR-08 | Hoạt động ngoại tuyến | Sau khi tải dữ liệu, không cần kết nối mạng |
| NFR-09 | Trình duyệt hỗ trợ | Chrome, Firefox, Safari — hai phiên bản gần nhất |
| NFR-10 | Khả năng tiếp cận | Điều hướng được bằng bàn phím, tương phản đạt WCAG AA |

Ghi chú về NFR-01: mô hình cây với 500 cây gradient boosting chấm một mẫu mất khoảng 2–5 ms trên CPU thông thường, nên ngưỡng 50 ms là rộng rãi. Điểm nghẽn thực tế nằm ở việc tính SHAP — mất khoảng 50–200 ms mỗi mẫu — nên `/explain` là endpoint riêng, chỉ gọi khi người dùng mở chi tiết, không gọi cho mọi dòng trong danh sách.

---

## 9. CÔNG NGHỆ VÀ LÝ DO LỰA CHỌN

### 9.1 Phần huấn luyện

| Thành phần | Lựa chọn | Lý do |
|---|---|---|
| Ngôn ngữ | Python 3.11 | Hệ sinh thái ML |
| Xử lý dữ liệu | pandas, NumPy | Chuẩn ngành |
| Học máy | scikit-learn, XGBoost, LightGBM | Bao phủ cả mô hình tuyến tính lẫn ensemble |
| Mất cân bằng | imbalanced-learn | Pipeline xử lý resampling đúng cách trong CV |
| Thống kê | SciPy, statsmodels | Kiểm định phi tham số, hiệu chỉnh đa kiểm định |
| Giải thích | SHAP | TreeExplainer tính chính xác và nhanh cho mô hình cây |
| Lưu hiện vật | joblib | Chuẩn của scikit-learn |

### 9.2 Phần ứng dụng — hai phương án

Chọn một trong hai tùy quỹ thời gian còn lại.

**Phương án A — Streamlit đơn khối** *(khoảng 3 ngày)*

Toàn bộ giao diện và logic trong một ứng dụng Python. Không có API riêng.

*Ưu:* nhanh, ít mã, không cần biết frontend, tự động responsive.
*Nhược:* khó tùy biến giao diện, chế độ phát lại chạy giật do mô hình thực thi lại toàn trang mỗi lần tương tác, không tách được tầng phục vụ để nói về kiến trúc.

**Phương án B — FastAPI + giao diện riêng** *(khoảng 6 ngày)* — **khuyến nghị**

Backend FastAPI phục vụ JSON, frontend riêng gọi API.

*Ưu:* kiến trúc rõ ràng, có tài liệu API tự sinh tại `/docs`, chế độ phát lại chạy mượt qua Server-Sent Events, giao diện tùy biến hoàn toàn, thể hiện được năng lực kỹ thuật rộng hơn.
*Nhược:* nhiều việc hơn, cần biết cơ bản về frontend.

Cho phương án B, frontend nên chọn:
- **Next.js + React + Tailwind + Recharts** nếu muốn kết quả trau chuốt nhất.
- **HTML + Alpine.js + Chart.js** nếu muốn giữ đơn giản mà vẫn tách được backend — chỉ một tệp HTML, không cần công cụ đóng gói.

Phương án thứ hai này thường bị đánh giá thấp. Với bốn màn hình và không có trạng thái phức tạp, nó cho ra kết quả tương đương mà tốn ít hơn hai ngày công.

### 9.3 Đóng gói

`docker-compose.yml` với hai dịch vụ: `api` và `web`. Thư mục `models/` và `data/` gắn dưới dạng volume để không phải đóng gói tệp mô hình vào ảnh.

Điều kiện: chạy được toàn hệ thống bằng đúng một lệnh trên máy chưa cài gì ngoài Docker.

---

## 10. CẤU TRÚC MÃ NGUỒN

```
fraud-detection/
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_statistics.ipynb
│   ├── 03_baseline.ipynb
│   ├── 04_imbalance_strategies.ipynb
│   ├── 05_advanced_models.ipynb
│   ├── 06_threshold_and_cost.ipynb
│   ├── 07_explainability.ipynb
│   └── 08_export_artifacts.ipynb
│
├── src/
│   ├── features.py          # sinh hour_sin, hour_cos — DÙNG CHUNG train và serve
│   ├── data.py              # nạp dữ liệu, chia tập
│   ├── evaluate.py          # hàm đánh giá dùng chung
│   ├── threshold.py         # tối ưu ngưỡng theo chi phí
│   └── plots.py
│
├── api/
│   ├── main.py              # khởi tạo FastAPI, nạp mô hình lúc startup
│   ├── schemas.py           # mô hình Pydantic
│   ├── routes/
│   │   ├── scoring.py
│   │   ├── transactions.py
│   │   ├── threshold.py
│   │   ├── metrics.py
│   │   └── replay.py
│   ├── db.py
│   └── Dockerfile
│
├── web/                     # phương án B
│   ├── index.html
│   ├── app.js
│   ├── styles.css
│   └── Dockerfile
│
├── models/                  # hiện vật, không đưa vào git
├── data/                    # dữ liệu, không đưa vào git
├── tests/
│   ├── test_features.py     # kiểm tra tính nhất quán đặc trưng train/serve
│   ├── test_threshold.py
│   └── test_api.py
├── reports/
├── docker-compose.yml
├── requirements.txt
└── README.md
```

**Điểm quan trọng nhất trong cấu trúc này là `src/features.py`.** Cả notebook huấn luyện lẫn API đều import đúng hàm này để sinh đặc trưng. Nếu để hai bên tự viết logic riêng, sớm muộn chúng sẽ lệch nhau và mô hình sẽ nhận đầu vào khác với lúc huấn luyện — lỗi này rất khó phát hiện vì hệ thống vẫn chạy, chỉ là dự đoán sai. Tệp `tests/test_features.py` tồn tại để canh đúng điều đó.

---

## 11. KẾ HOẠCH TRIỂN KHAI

Tổng 21 ngày. Ứng dụng chiếm khoảng 6 ngày trong đó.

| Giai đoạn | Ngày | Nội dung | Sản phẩm |
|---|---|---|---|
| Nền tảng | 1–2 | Dựng repo, tải dữ liệu, khám phá dữ liệu | 8 biểu đồ |
| Thống kê | 3–4 | Kiểm định, hiệu chỉnh FDR, effect size | Bảng xếp hạng đặc trưng |
| Khung đánh giá | 5 | Chia tập, viết `evaluate()`, `features.py` | Hàm dùng chung |
| Mô hình cơ sở | 6–7 | Ba mô hình cơ sở | Bảng mốc so sánh |
| Mất cân bằng | 8–10 | 5 chiến lược × 4 mô hình | Bảng 20 dòng + biểu đồ PR |
| Tinh chỉnh | 11–12 | RandomizedSearchCV, chốt mô hình | `model.joblib` |
| Ngưỡng & SHAP | 13–14 | Đường cong chi phí, explainer | `threshold.json`, `metrics.json` |
| **API** | **15–16** | FastAPI, cơ sở dữ liệu, các endpoint | API chạy được, có `/docs` |
| **Giao diện** | **17–19** | Bốn màn hình + chế độ phát lại | Ứng dụng hoàn chỉnh |
| Đóng gói | 20 | Docker Compose, README, kiểm thử | Chạy bằng một lệnh |
| Báo cáo | 21 | Viết báo cáo, chuẩn bị bảo vệ | Tài liệu nộp |

**Thứ tự làm giao diện, nếu thiếu thời gian phải cắt bớt:** hàng đợi → cấu hình ngưỡng → chi tiết giao dịch → phát lại → hiệu năng mô hình. Màn hình cấu hình ngưỡng phải giữ bằng mọi giá vì đó là nơi thể hiện luận điểm chính của đồ án. Màn hình hiệu năng mô hình có thể thay tạm bằng ảnh chụp từ notebook.

---

## 12. TIÊU CHÍ NGHIỆM THU

### 12.1 Phần mô hình

| Mã | Tiêu chí | Cách kiểm chứng |
|---|---|---|
| AC-M1 | PR-AUC trên tập test ≥ 0,75 | Số liệu trong `metrics.json` |
| AC-M2 | Recall ≥ 0,75 tại ngưỡng tối ưu | Ma trận nhầm lẫn |
| AC-M3 | Đủ 20 tổ hợp mô hình × chiến lược | Bảng kết quả |
| AC-M4 | Không có rò rỉ dữ liệu | Rà soát mã: mọi resampler nằm trong `imblearn` pipeline |
| AC-M5 | Kết quả tái lập được | Chạy lại notebook, lệch PR-AUC < 0,001 |
| AC-M6 | Có khoảng tin cậy bootstrap cho chỉ số chính | Báo cáo |
| AC-M7 | Có đánh giá bổ sung theo cách chia theo thời gian | Bảng đối chiếu hai cách chia |

### 12.2 Phần ứng dụng

| Mã | Tiêu chí | Cách kiểm chứng |
|---|---|---|
| AC-A1 | Tải CSV 10.000 dòng, chấm điểm xong dưới 30 giây | Bấm giờ |
| AC-A2 | Hàng đợi sắp xếp đúng theo điểm rủi ro giảm dần | Kiểm tra thủ công |
| AC-A3 | Kéo thanh trượt ngưỡng cập nhật chỉ số dưới 200 ms | Đo bằng DevTools |
| AC-A4 | Màn hình chi tiết hiển thị đúng 5 yếu tố dương và 3 yếu tố âm | So với giá trị SHAP tính trong notebook |
| AC-A5 | Đổi tham số chi phí làm dịch chuyển ngưỡng tối ưu | Nhập cặp giá trị khác, quan sát |
| AC-A6 | Chế độ phát lại chạy liên tục 3 phút không lỗi | Chạy thử |
| AC-A7 | Quyết định thẩm định được lưu lại sau khi tải lại trang | Kiểm tra cơ sở dữ liệu |
| AC-A8 | `docker compose up` chạy được trên máy sạch | Thử trên máy khác |
| AC-A9 | Điều hướng được toàn bộ ứng dụng bằng bàn phím | Chỉ dùng Tab và Enter |
| AC-A10 | Endpoint sai đầu vào trả mã lỗi đúng, không sập | Bộ kiểm thử API |

---

## 13. RỦI RO

| Rủi ro | Khả năng | Tác động | Biện pháp |
|---|---|---|---|
| Rò rỉ dữ liệu do resample sai thứ tự | Cao | Nghiêm trọng — toàn bộ kết quả mất giá trị | Bắt buộc dùng `imblearn.pipeline`; rà soát mã trước khi chạy lưới thí nghiệm |
| Đặc trưng lệch giữa huấn luyện và phục vụ | Trung bình | Nghiêm trọng — dự đoán sai âm thầm | `src/features.py` dùng chung; có kiểm thử tự động đối chiếu |
| SHAP quá chậm làm nghẽn giao diện | Trung bình | Vừa | Chỉ tính khi mở chi tiết; nếu vẫn chậm thì tính trước cho tập mẫu |
| Không kịp làm giao diện | Trung bình | Vừa | Có sẵn phương án A (Streamlit) làm dự phòng; thứ tự cắt bớt đã định trước |
| Chỉ ~98 mẫu dương trong tập test làm chỉ số dao động mạnh | Cao | Vừa | Báo cáo kèm khoảng tin cậy; dùng CV lặp lại |
| Tệp CSV 144 MB làm tràn bộ nhớ khi tải lên | Thấp | Vừa | Đọc theo khối bằng `chunksize`; giới hạn 100 MB |
| Docker không chạy trên máy trình bày | Thấp | Cao | Chuẩn bị sẵn bản chạy cục bộ không cần Docker và video ghi màn hình dự phòng |

---

## 14. NGOÀI PHẠM VI

Những hạng mục sau **không** thuộc dự án. Nêu rõ trong báo cáo để tránh bị hỏi vặn khi bảo vệ:

- Xác thực, phân quyền, quản lý người dùng.
- Huấn luyện lại tự động hoặc học trực tuyến.
- Xử lý dòng dữ liệu thật (Kafka, hàng đợi tin nhắn). Chế độ phát lại là mô phỏng đọc từ tệp.
- Triển khai lên hạ tầng đám mây. Hệ thống chạy cục bộ.
- Giám sát độ trôi mô hình trong vận hành.
- Kiểm thử A/B giữa các mô hình.
- Đa ngôn ngữ giao diện.
- Kết nối với hệ thống thanh toán thật hoặc dữ liệu thật.

Về khả năng áp dụng thực tế: dữ liệu chỉ trải dài hai ngày của năm 2013 tại một thị trường châu Âu. Hình thái gian lận thay đổi liên tục, nên mô hình huấn luyện trên bộ này không dùng được cho hệ thống hiện tại. Giá trị của dự án nằm ở phương pháp — cách xử lý mất cân bằng, cách chọn thước đo, cách đặt ngưỡng theo chi phí — chứ không nằm ở bản thân mô hình đã huấn luyện.

---

## PHỤ LỤC A — LỆNH KHỞI ĐỘNG NHANH

```bash
# 1. Chuẩn bị
git clone <repo> && cd fraud-detection
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 2. Tải dữ liệu (cần Kaggle API token tại ~/.kaggle/kaggle.json)
kaggle datasets download -d mlg-ulb/creditcardfraud -p data/ --unzip

# 3. Huấn luyện — chạy lần lượt các notebook,
#    notebook 08 sinh toàn bộ hiện vật vào models/
jupyter lab

# 4. Chạy ứng dụng
docker compose up --build
# API      → http://localhost:8000/docs
# Giao diện → http://localhost:3000
```

## PHỤ LỤC B — LIÊN KẾT

| Nội dung | Địa chỉ |
|---|---|
| Dataset trên Kaggle | https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud |
| Tệp dữ liệu | https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud/data |
| Giấy phép và metadata | https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud/metadata |
| Tài liệu imbalanced-learn | https://imbalanced-learn.org/stable/ |
| Tài liệu SHAP | https://shap.readthedocs.io/ |
| Tài liệu FastAPI | https://fastapi.tiangolo.com/ |
