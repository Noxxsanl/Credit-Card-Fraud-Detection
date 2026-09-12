# 02 — Đặc tả dữ liệu

## 1. Nguồn

| Mục | Giá trị |
|---|---|
| Tên bộ dữ liệu | Credit Card Fraud Detection |
| Chủ sở hữu | Machine Learning Group — ULB (`mlg-ulb`) |
| Địa chỉ | https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud |
| Tệp | `creditcard.csv`, khoảng 144 MB |
| Giấy phép | Database Contents License (DbCL) v1.0 |
| Phạm vi | Giao dịch của chủ thẻ châu Âu, tháng 9/2013, cửa sổ hai ngày |
| Vị trí trong repo | `data/creditcard.csv` — **không commit**, xem `.gitignore` |

Cách tải: xem [10 — Vận hành và tái lập §2](10-van-hanh-tai-lap.md).

## 2. Lược đồ

284.807 dòng × 31 cột. Không có giá trị thiếu.

| Cột | Kiểu | Miền giá trị | Ghi chú |
|---|---|---|---|
| `Time` | float64 | 0 … 172.792 | Số giây kể từ giao dịch đầu tiên trong bộ dữ liệu |
| `V1` … `V28` | float64 | Không giới hạn, trung bình ≈ 0 | Thành phần chính sau PCA, đã ẩn danh |
| `Amount` | float64 | 0 … 25.691,16 | Số tiền giao dịch, lệch phải rất mạnh |
| `Class` | int64 | 0 hoặc 1 | 1 = gian lận |

### 2.1 Phân bố nhãn

| Lớp | Số lượng | Tỷ lệ |
|---|---|---|
| 0 — hợp lệ | 284.315 | 99,828% |
| 1 — gian lận | 492 | 0,172% |

Tỷ lệ mất cân bằng khoảng **1 : 578**.

Hệ quả trực tiếp lên thiết kế:
- Một mô hình rỗng luôn trả 0 đạt 99,83% accuracy → **accuracy bị loại khỏi mọi
  bảng kết quả chính** (xem [04 §5](04-thiet-ke-mo-hinh-ml.md)).
- Tập kiểm thử 20% chỉ chứa khoảng **98 mẫu gian lận**. Mỗi mẫu bắt hụt kéo recall
  xuống hơn 1 điểm phần trăm. Vì vậy mọi chỉ số phải kèm khoảng tin cậy bootstrap.
- Chia tập **bắt buộc phân tầng**; `KFold` thường có thể tạo ra fold không chứa
  mẫu dương nào.

## 3. Ba đặc điểm chi phối cách làm

**DS-01 — V1–V28 đã là thành phần chính.** Chúng trực giao với nhau, đã ở thang
đo tương đương, và **không diễn giải được về nghiệp vụ**. Không chạy PCA lần nữa.
Không cần chuẩn hóa lại. Chương 2 của môn học được vận dụng ở chỗ *đọc hiểu* ý
nghĩa của phép biến đổi này, không phải ở chỗ áp dụng thêm PCA.

**DS-02 — `Amount` lệch phải cực mạnh.** Trung vị khoảng 22 USD nhưng giá trị lớn
nhất hơn 25.000 USD. Dùng `RobustScaler` (dựa trên trung vị và IQR) thay vì
`StandardScaler`, vì giá trị ngoại lai sẽ kéo lệch trung bình và độ lệch chuẩn.
Khi vẽ biểu đồ, dùng thang log.

**DS-03 — `Time` không phải đặc trưng.** Đây là số giây tương đối trong một cửa
sổ hai ngày cụ thể. Đưa thẳng vào mô hình là dạy mô hình một đặc điểm không tồn
tại khi triển khai — chỉ số 86.400 không có ý nghĩa gì ngoài bộ dữ liệu này.
Thông tin dùng được duy nhất là **giờ trong ngày**, vì nó lặp lại.

## 4. Đặc trưng đưa vào mô hình

**31 đặc trưng**, cố định thứ tự (hợp đồng dữ liệu vào là 30 cột — xem §5; số 31
là số cột đầu ra sau khi bỏ `Time` và thêm hai cột mã hoá giờ):

| # | Đặc trưng | Nguồn | Phép biến đổi |
|---|---|---|---|
| 1–28 | `V1` … `V28` | Nguyên bản | Giữ nguyên |
| 29 | `Amount` | Nguyên bản | `RobustScaler` **trong pipeline**, fit chỉ trên tập huấn luyện |
| 30 | `hour_sin` | `(Time // 3600) % 24` | `sin(2π·h/24)` |
| 31 | `hour_cos` | `(Time // 3600) % 24` | `cos(2π·h/24)` |

Phân công trách nhiệm: `build_features()` sinh 31 cột với `Amount` còn **nguyên
giá trị**; việc chuẩn hoá thuộc về `make_preprocessor()` và phải nằm trong
pipeline để chỉ học thống kê của tập huấn luyện trong từng fold (ML-04). Sau
preprocessor, `ColumnTransformer` đưa cột đã biến đổi lên đầu, nên thứ tự cột lúc
đó là `Amount, V1…V28, hour_sin, hour_cos` — hằng số `MODEL_FEATURE_NAMES` trong
`src/features.py` giữ đúng thứ tự này để gán tên cho giá trị SHAP.

Mã hóa giờ theo sin/cos thay vì số nguyên 0–23 để mô hình hiểu 23 giờ và 0 giờ là
kề nhau. Nếu giữ số nguyên, mô hình tuyến tính coi hai mốc đó cách nhau 23 đơn vị.

**DS-04 — Logic sinh đặc trưng nằm ở một chỗ duy nhất: `src/features.py`.**
Cả notebook huấn luyện lẫn API đều import đúng hàm này. Đây là ràng buộc quan
trọng nhất trong toàn bộ thiết kế mã nguồn; lý do và cách canh giữ nằm ở
[03 §4](03-thiet-ke-kien-truc.md) và [08 TC-01](08-ke-hoach-kiem-thu.md).

## 5. Hợp đồng dữ liệu vào (dùng cho API)

Mọi giao dịch đưa vào hệ thống lúc chạy phải thỏa:

| Trường | Kiểu | Bắt buộc | Ràng buộc |
|---|---|---|---|
| `Time` | float | Có | ≥ 0 |
| `V1` … `V28` | float | Có | Hữu hạn, không NaN, không Inf |
| `Amount` | float | Có | ≥ 0 |
| `Class` | int | Không | 0 hoặc 1; chỉ xuất hiện khi dữ liệu có nhãn để đánh giá |

Quy tắc kiểm tra:

| Mã | Quy tắc | Hành vi khi vi phạm |
|---|---|---|
| DS-10 | Thiếu bất kỳ cột nào trong 30 cột bắt buộc | HTTP 422, liệt kê tên cột thiếu |
| DS-11 | Giá trị không ép được về float | HTTP 422, nêu tên cột và dòng đầu tiên sai |
| DS-12 | `Amount` âm | HTTP 422 |
| DS-13 | NaN hoặc Inf trong bất kỳ cột V nào | HTTP 422 |
| DS-14 | Có cột thừa ngoài lược đồ | Bỏ qua cột thừa, ghi cảnh báo vào log |
| DS-15 | Lô vượt 50.000 dòng | HTTP 413 |

## 6. Chất lượng dữ liệu

| Kiểm tra | Kỳ vọng | Hành động nếu lệch |
|---|---|---|
| Số dòng | 284.807 | Dừng lại, kiểm tra tệp tải về |
| Giá trị thiếu | 0 ô | Dừng lại |
| Dòng trùng lặp hoàn toàn | Khoảng 1.081 dòng | Ghi nhận trong EDA; loại bỏ trước khi chia tập, nêu rõ trong báo cáo |
| Số mẫu lớp 1 | 492 | Dừng lại |
| `Amount` bằng 0 | Có tồn tại (giao dịch kiểm tra thẻ) | Giữ nguyên — đây là tín hiệu có thật của gian lận |

**DS-20 — Xử lý trùng lặp.** Loại bỏ dòng trùng lặp **trước khi chia tập**. Nếu
để lại, một dòng có thể xuất hiện ở cả tập huấn luyện lẫn tập kiểm thử, tạo ra
rò rỉ nhẹ nhưng có thật. Báo cáo phải nêu số dòng đã loại và số mẫu gian lận còn
lại sau khi loại.

### 6.1 Số liệu thực đo (notebook 01, giai đoạn 1)

Đúng **1.081 dòng trùng lặp**, trong đó **19 dòng thuộc lớp gian lận**. Sau khi
loại còn **283.726 dòng / 473 gian lận**, tỷ lệ dương **0,167%** (1 trên 600).

Hệ quả cần nhớ: **mọi bảng kết quả của dự án tính trên 283.726 dòng, không phải
284.807**. Các con số mốc quen thuộc của bộ dữ liệu này thường được trích từ dữ
liệu thô nên lệch nhẹ so với số của dự án:

| Mốc | Dữ liệu thô | Sau khi loại trùng lặp — **số dùng trong báo cáo** |
|---|---|---|
| Tỷ lệ gian lận lúc 2h | 1,713% | **1,451%** (48/3.308) |
| Tỷ lệ gian lận lúc 10h | 0,048% | **0,048%** (8/16.548) |
| Trung vị `Amount` gian lận / hợp lệ | 9,25 / 22,00 | **9,82 / 22,00** |
| Trung bình `Amount` gian lận / hợp lệ | 122,21 / 88,29 | **123,87 / 88,41** |
| Số giao dịch gian lận có `Amount = 0` | 27 | **25** |
| Tương quan Pearson mạnh nhất (V17) | −0,326 | **−0,313** |
| Cohen's d của V17 | −8,32 | **−8,09** |

Hướng của mọi phát hiện giữ nguyên, chỉ độ lớn nhích nhẹ. Khi trích số thô để so
với tài liệu bên ngoài thì phải nói rõ đó là số thô. Bảng đối chiếu này được sinh
lại mỗi lần chạy `notebooks/01_eda.ipynb` §1.1.

## 7. Chiến lược chia tập

**Cách chính — chia ngẫu nhiên phân tầng 80/20.** Dùng cho toàn bộ bảng kết quả
trong báo cáo.

```
train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
```

**Cách phụ — chia theo thời gian.** Ngày 1 (`Time < 86400`) huấn luyện, ngày 2
kiểm thử. Dùng để kiểm tra xem mô hình có suy giảm khi phân bố dịch chuyển theo
thời gian hay không. Kết quả trình bày thành một bảng đối chiếu riêng, không trộn
vào bảng chính.

Mong đợi: hiệu năng ở cách chia theo thời gian **thấp hơn** cách chia ngẫu nhiên.
Chênh lệch này chính là phần thảo luận có giá trị, không phải lỗi cần giấu.

**Cross-validation:** `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)`
trên tập huấn luyện. Mỗi fold chứa khoảng 79 mẫu gian lận — đủ để chỉ số ổn định,
nhưng vẫn phải báo cáo độ lệch chuẩn giữa các fold.

## 8. Hiện vật dữ liệu sinh ra

| Tệp | Nội dung | Bên tiêu thụ |
|---|---|---|
| `data/test_set.parquet` | Tập kiểm thử kèm nhãn thật và điểm rủi ro đã tính | Chế độ phát lại, màn hình hiệu năng |
| `data/sample_pool.json` | Khoảng 200 giao dịch chọn lọc: 50 gian lận, 150 hợp lệ, trải đều các dải điểm | Thư viện mẫu trong giao diện |
| PostgreSQL (volume `pgdata`) | Trạng thái ứng dụng lúc chạy: giao dịch, thẩm định, cấu hình | API — xem [06](06-thiet-ke-luu-tru.md) |

Hai tệp đầu nằm ngoài git; cơ sở dữ liệu nằm trong volume Docker, ngoài cây thư mục repo.

## 9. Danh mục biểu đồ EDA bắt buộc

| # | Biểu đồ | Điều cần rút ra |
|---|---|---|
| 1 | Cột số lượng theo lớp, thang log | Mất cân bằng nhìn thấy được ngay |
| 2 | Phân bố `Amount` theo lớp, thang log | Gian lận tập trung ở khoảng tiền nào |
| 3 | Số giao dịch theo giờ, tách hai lớp, chuẩn hóa theo tỷ lệ | Gian lận có tập trung vào giờ vắng không |
| 4 | Ma trận tương quan giữa 30 đặc trưng | Xác nhận V1–V28 gần như trực giao |
| 5 | Tương quan từng đặc trưng với `Class`, xếp hạng | Ứng viên đặc trưng mạnh |
| 6 | Biểu đồ mật độ chồng hai lớp cho 6 đặc trưng phân tách mạnh nhất | Mức độ chồng lấn thực tế |
| 7 | Hộp (boxplot) `Amount` theo lớp | Ngoại lai và trung vị |
| 8 | Chiếu t-SNE hoặc UMAP trên mẫu con cân bằng | Gian lận có tách thành cụm không |

Biểu đồ lưu vào `reports/figures/` với tên có tiền tố số thứ tự để dễ chèn vào
báo cáo.
