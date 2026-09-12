# 08 — Kế hoạch kiểm thử và nghiệm thu

Ở dự án này, "kiểm thử" gồm hai phần khác hẳn nhau:

1. **Kiểm thử phần mềm** — mã chạy đúng như đặc tả (pytest).
2. **Kiểm chứng phương pháp** — kết quả mô hình đáng tin, không rò rỉ, tái lập được.

Phần thứ hai quan trọng hơn về mặt điểm số, và không có thư viện nào làm hộ.

## 1. Chiến lược

| Tầng | Công cụ | Phạm vi | Khi chạy |
|---|---|---|---|
| Đơn vị | pytest | `src/` — đặc trưng, ngưỡng, chỉ số | Mỗi lần sửa `src/` |
| Tích hợp | pytest + `TestClient` + PostgreSQL dùng một lần | `api/` — endpoint, xác thực dữ liệu, cơ sở dữ liệu | Trước khi commit |
| Đối chiếu | pytest | Máy khách và máy chủ cho cùng kết quả ngưỡng | Trước khi bảo vệ |
| Thủ công | Danh sách kiểm | Giao diện, khả năng tiếp cận | Ngày 20 |
| Phương pháp | Rà soát mã + notebook | Rò rỉ dữ liệu, tái lập | Ngày 12 và ngày 20 |

Không dựng CI. Với một người làm trong ba tuần, chạy `pytest` trước mỗi commit là
đủ; dựng GitHub Actions tốn thời gian mà không thêm giá trị cho bài nộp.

### 1.1 Cơ sở dữ liệu dùng cho kiểm thử

Kiểm thử **không bao giờ** chạy trên cơ sở dữ liệu phát triển. `tests/conftest.py`
dựng một cơ sở dữ liệu riêng và xóa sau khi xong:

```python
# Đọc từ TEST_DATABASE_URL, mặc định:
#   postgresql+psycopg://fraud:fraud@localhost:5432/fraud_test

@pytest.fixture(scope="session")
def engine():
    create_database(TEST_DATABASE_URL)        # CREATE DATABASE fraud_test
    alembic_upgrade_head()                    # cùng migration với môi trường thật
    yield create_engine(TEST_DATABASE_URL)
    drop_database(TEST_DATABASE_URL)

@pytest.fixture
def session(engine):
    # Mỗi ca kiểm thử chạy trong một giao dịch rồi rollback: không ca nào
    # nhìn thấy dữ liệu của ca khác, và không cần dọn dẹp thủ công.
    conn = engine.connect()
    tx = conn.begin()
    yield Session(bind=conn)
    tx.rollback()
    conn.close()
```

Hai điểm đáng lưu ý:

- **Dùng chính Alembic để dựng lược đồ kiểm thử**, không dùng `create_all()`. Nếu
  hai đường dựng lược đồ khác nhau, kiểm thử sẽ xanh trên một lược đồ mà môi
  trường thật không có.
- **Rollback thay vì `TRUNCATE`** giữa các ca: nhanh hơn nhiều và không phụ thuộc
  thứ tự chạy.

Nếu máy không có sẵn PostgreSQL, chạy riêng dịch vụ `db`:
`docker compose up -d db` rồi trỏ `TEST_DATABASE_URL` vào `localhost:5432`.

## 2. Ca kiểm thử tự động

### 2.1 `tests/test_features.py` — quan trọng nhất

| Mã | Ca kiểm thử | Kỳ vọng |
|---|---|---|
| TC-01 | **Kiểm thử vàng**: 20 dòng mẫu cố định qua `build_features()` | Khớp chính xác ma trận đặc trưng đã lưu kèm, sai số < 1e-9 |
| TC-02 | Thứ tự cột đầu ra | Khớp `FEATURE_ORDER`, đúng 30 cột |
| TC-03 | `Time = 0` và `Time = 86399` | `hour` bằng 0 và 23; `hour_sin/cos` nằm trong [-1, 1] |
| TC-04 | Tính tuần hoàn của giờ | Khoảng cách Euclid giữa (23h) và (0h) nhỏ hơn giữa (23h) và (12h) |
| TC-05 | Cột `Time` thô không có mặt trong đầu ra | Khẳng định `'Time' not in columns` (ML-07) |
| TC-06 | Đầu vào thiếu cột | Ném ngoại lệ nêu đúng tên cột thiếu |

TC-01 là lưới an toàn chống lệch train/serve (xem [03 §4](03-thiet-ke-kien-truc.md)).
Nếu ai đó thêm đặc trưng mà quên cập nhật một phía, ca này đỏ ngay lập tức.

### 2.2 `tests/test_threshold.py`

| Mã | Ca kiểm thử | Kỳ vọng |
|---|---|---|
| TC-10 | Chi phí tại ngưỡng 0 và ngưỡng 1 | Ngưỡng 0: mọi mẫu là cảnh báo, FN = 0; ngưỡng 1: không cảnh báo, FP = 0 |
| TC-11 | Ngưỡng tối ưu nằm trong dải quét | Nghiệm không rơi vào biên trừ khi chi phí đơn điệu |
| TC-12 | **Đối chiếu máy khách – máy chủ** | Hàm JavaScript ở UI-D1 và `src/threshold.py` cho cùng TP/FP/FN với sai lệch 0 trên 20 ngưỡng mẫu |
| TC-13 | Ràng buộc `max_alerts_per_day` | Số cảnh báo tại nghiệm không vượt ngân sách |
| TC-14 | Ràng buộc `min_recall` | Recall tại nghiệm ≥ mức yêu cầu |
| TC-15 | Chi phí tỷ lệ nghịch với ngưỡng khi `cost_fp = 0` | Ngưỡng tối ưu tiến về 0 |

TC-12 đáng để viết: nó bảo vệ quyết định UI-D1. Nếu hai bên lệch nhau, con số trên
màn hình demo sẽ không khớp con số trong báo cáo, và điều đó rất khó phát hiện
bằng mắt.

### 2.3 `tests/test_no_leakage.py`

| Mã | Ca kiểm thử | Kỳ vọng |
|---|---|---|
| TC-20 | Mọi pipeline có bước resample là `imblearn.pipeline.Pipeline` | Khẳng định kiểu (ML-01) |
| TC-21 | Scaler trong pipeline chưa được `fit` trước khi vào CV | `check_is_fitted` ném ngoại lệ |
| TC-22 | Tập train và test không giao nhau theo chỉ số | Giao rỗng |
| TC-23 | Sau khi loại trùng lặp, không có dòng trùng giữa hai tập | Giao rỗng theo giá trị băm của dòng |

### 2.4 `tests/test_api.py`

| Mã | Ca kiểm thử | Kỳ vọng |
|---|---|---|
| TC-30 | `GET /health` khi đã nạp mô hình | 200, có `model_version` |
| TC-31 | `POST /score` với giao dịch hợp lệ | 200, `risk_score` trong [0, 1] |
| TC-32 | `POST /score` thiếu `V13` | 422, `code = MISSING_FEATURES`, `details.missing = ["V13"]` |
| TC-33 | `POST /score` với `Amount` âm | 422, `code = INVALID_FEATURE_VALUE` |
| TC-34 | `POST /score` với `V5 = NaN` | 422 |
| TC-35 | `POST /score/batch` với 50.001 phần tử | 413 |
| TC-36 | `PUT /threshold` với giá trị 1,5 | 422, `code = INVALID_THRESHOLD` |
| TC-37 | `GET /transactions/{id}` không tồn tại | 404 |
| TC-38 | `POST /reviews` hai lần cho cùng giao dịch | Bản ghi thứ hai ghi đè, tổng số dòng không tăng |
| TC-39 | `POST /reviews` ghi đúng `threshold_used` | Khớp ngưỡng hiện hành tại thời điểm gửi |
| TC-40 | Đổi ngưỡng rồi gọi lại `GET /transactions` | Số dòng trả về thay đổi tương ứng |
| TC-41 | Mọi endpoint với thân yêu cầu rỗng | Trả 422, không ném 500 (AC-A10) |
| TC-42 | `GET /health` khi container `db` dừng | 503, `code = DATABASE_UNAVAILABLE`, tiến trình không sập |
| TC-43 | `POST /score/upload` có dòng lỗi ở giữa lô | Dòng hỏng bị loại trước khi ghi; toàn bộ dòng hợp lệ vào trong một giao dịch |
| TC-44 | Ghi trùng `transactions.id` | Không tạo bản ghi thứ hai — `ON CONFLICT DO NOTHING`, không ném 500 |
| TC-45 | Ràng buộc `CHECK` chặn dữ liệu rác | `INSERT` với `risk_score = 1.7` hoặc `hour = 25` bị cơ sở dữ liệu từ chối |
| TC-46 | Khóa ngoại `reviews.transaction_id` | Ghi thẩm định cho giao dịch không tồn tại bị từ chối |
| TC-47 | `upgrade head` → `downgrade base` → `upgrade head` | Chạy trọn vẹn, lược đồ cuối giống lược đồ đầu |

### 2.5 `tests/test_scoring.py`

| Mã | Ca kiểm thử | Kỳ vọng |
|---|---|---|
| TC-50 | Chấm điểm một mẫu và chấm điểm lô cho cùng kết quả | Sai số < 1e-12 trên cùng 100 giao dịch |
| TC-51 | `risk_band` suy ra đúng theo bảng ở [05 §2](05-thiet-ke-api.md) | Bốn dải với ngưỡng 0,05 |
| TC-52 | `model_version` xuất hiện trong mọi phản hồi chấm điểm | Khẳng định trường tồn tại |
| TC-53 | Chấm điểm 10.000 giao dịch | Dưới 30 giây (NFR-02) |
| TC-54 | Ghi 10.000 dòng bằng `COPY` | Dưới 3 giây; đo kèm số liệu của `INSERT` từng dòng để đưa vào báo cáo |
| TC-55 | `amount` đọc ra từ cơ sở dữ liệu là `Decimal` | Được ép về `float` trước khi vào `build_features()` (ST-07) |

## 3. Kiểm chứng phương pháp

Đây là phần không tự động hóa được hoàn toàn; làm theo danh sách kiểm.

### 3.1 Rà soát rò rỉ dữ liệu — thực hiện cuối ngày 12

- [ ] Mọi `fit_resample` chỉ xuất hiện bên trong định nghĩa pipeline, không đứng riêng.
- [ ] `train_test_split` xuất hiện **trước** mọi thao tác resample trong trình tự notebook.
- [ ] Không có `scaler.fit(X)` trên toàn bộ dữ liệu ở bất kỳ đâu.
- [ ] `StratifiedKFold` được dùng ở mọi chỗ có CV; tìm chuỗi `KFold(` không có tiền tố `Stratified`.
- [ ] Ngưỡng được chọn trên dữ liệu out-of-fold hoặc tập huấn luyện, không trên tập kiểm thử (ML-08).
- [ ] Dòng trùng lặp bị loại trước khi chia tập.
- [ ] `random_state=42` có mặt ở mọi đối tượng có tham số này.

### 3.2 Kiểm tra tái lập — thực hiện ngày 20

Chạy toàn bộ notebook từ đầu trong kernel sạch, ghi lại PR-AUC. So với lần chạy
trước. Chênh lệch phải < 0,001 (NFR-07, AC-M5).

Nếu lệch lớn hơn: tìm bước ngẫu nhiên chưa ghim seed, hoặc thứ tự dữ liệu bị đổi
bởi một thao tác không ổn định.

### 3.3 Kiểm tra tính hợp lý của kết quả

| Dấu hiệu | Kết luận |
|---|---|
| PR-AUC > 0,95 | Gần như chắc chắn có rò rỉ — kiểm tra lại §3.1 |
| Recall = 1,00 và Precision > 0,9 | Như trên |
| Accuracy được báo cáo là chỉ số chính | Sai phương pháp |
| Mô hình không xử lý mất cân bằng lại thắng rõ rệt | Kiểm tra lại cấu hình các chiến lược khác |
| ROC-AUC cao nhưng PR-AUC rất thấp | Bình thường ở bài này — giải thích trong báo cáo, không phải lỗi |

## 4. Tiêu chí nghiệm thu

### 4.1 Phần mô hình

| Mã | Tiêu chí | Cách kiểm chứng | Bắt buộc |
|---|---|---|---|
| AC-M1 | PR-AUC trên tập kiểm thử ≥ 0,75 | `metrics.json` | Có |
| AC-M2 | Recall ≥ 0,75 tại ngưỡng đề xuất | Ma trận nhầm lẫn | Có |
| AC-M3 | Đủ 20 tổ hợp mô hình × chiến lược | Bảng trong báo cáo | Có |
| AC-M4 | Không có rò rỉ dữ liệu | Danh sách kiểm §3.1 + TC-20…TC-23 | Có |
| AC-M5 | Tái lập: PR-AUC lệch < 0,001 giữa hai lần chạy | §3.2 | Có |
| AC-M6 | Khoảng tin cậy bootstrap cho chỉ số chính | Báo cáo | Có |
| AC-M7 | Bảng đối chiếu hai cách chia tập | Báo cáo | Nên |
| AC-M8 | Phân tích độ nhạy theo tỷ lệ chi phí | Báo cáo | Nên |

### 4.2 Phần ứng dụng

| Mã | Tiêu chí | Cách kiểm chứng | Bắt buộc |
|---|---|---|---|
| AC-A1 | Tải CSV 10.000 dòng, chấm điểm xong dưới 30 giây | Bấm giờ | Có |
| AC-A2 | Hàng đợi sắp xếp đúng theo điểm giảm dần | Kiểm tra thủ công | Có |
| AC-A3 | Kéo thanh trượt cập nhật chỉ số dưới 200 ms | DevTools | Có |
| AC-A4 | Chi tiết hiển thị đúng 5 yếu tố dương và 3 yếu tố âm | So với giá trị SHAP tính trong notebook | Có |
| AC-A5 | Đổi tham số chi phí làm dịch chuyển ngưỡng tối ưu | Nhập cặp giá trị khác, quan sát | Có |
| AC-A6 | Chế độ phát lại chạy liên tục 3 phút không lỗi | Chạy thử | Nên |
| AC-A7 | Quyết định thẩm định còn nguyên sau khi tải lại trang và sau `docker compose restart` | Truy vấn bảng `reviews` | Có |
| AC-A8 | `docker compose up` chạy được trên máy sạch | Thử trên máy khác | Có |
| AC-A9 | Điều hướng toàn bộ ứng dụng bằng bàn phím | Chỉ dùng `Tab` và `Enter` | Nên |
| AC-A10 | Đầu vào sai trả mã lỗi đúng, tiến trình không sập | TC-30…TC-41 | Có |

### 4.3 Phần tài liệu

| Mã | Tiêu chí | Bắt buộc |
|---|---|---|
| AC-D1 | Báo cáo trả lời "vì sao chọn ngưỡng này" bằng lập luận chi phí | Có |
| AC-D2 | Báo cáo nêu rõ hạn chế: PCA không diễn giải được, dữ liệu hai ngày, 492 mẫu dương | Có |
| AC-D3 | README hướng dẫn chạy lại từ đầu, có thể làm theo mà không hỏi thêm | Có |
| AC-D4 | Mọi biểu đồ có tiêu đề, nhãn trục, đơn vị | Có |
| AC-D5 | Ghi nguồn dữ liệu và giấy phép DbCL v1.0 | Có |

## 5. Danh sách kiểm trước buổi bảo vệ

- [ ] Chạy `pytest` — toàn bộ xanh.
- [ ] `docker compose down -v` rồi `docker compose up` trên máy sạch — lần đầu dưới 45 giây, lần thứ hai dưới 15 giây (NFR-04).
- [ ] Khởi động hệ thống một lần **trước** buổi bảo vệ để volume `pgdata` đã sẵn sàng.
- [ ] Chuẩn bị sẵn bản `pg_dump` dữ liệu demo để khôi phục nếu có sự cố ([06 §7.4](06-thiet-ke-luu-tru.md)).
- [ ] Mở cả bốn màn hình, không có lỗi trong console trình duyệt.
- [ ] Kịch bản trình bày ngưỡng (kéo 0,5 → 0,047) chạy mượt.
- [ ] Chuẩn bị sẵn 4 giao dịch mẫu: gian lận dễ, gian lận khó, hợp lệ dễ, hợp lệ khó.
- [ ] Có phương án dự phòng khi mạng hoặc máy chiếu hỏng: ảnh chụp màn hình và bản PDF báo cáo.
- [ ] Ôn ba câu hỏi chắc chắn bị hỏi:
      1. Vì sao PR-AUC chứ không phải ROC-AUC?
      2. SMOTE hoạt động thế nào và vì sao nó phải nằm trong pipeline?
      3. Ngưỡng này được chọn ra sao, và sẽ đổi thế nào nếu chi phí thay đổi?
