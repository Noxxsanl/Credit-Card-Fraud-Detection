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
| TC-02 | Thứ tự cột đầu ra | Khớp `FEATURE_ORDER`, đúng 31 cột; không phụ thuộc thứ tự cột đầu vào |
| TC-03 | `Time = 0` và `Time = 86399` | `hour` bằng 0 và 23; `hour_sin/cos` nằm trong [-1, 1] |
| TC-04 | Tính tuần hoàn của giờ | Khoảng cách Euclid giữa (23h) và (0h) nhỏ hơn giữa (23h) và (12h) |
| TC-05 | Cột `Time` thô không có mặt trong đầu ra | Khẳng định `Time` vắng mặt; dịch `Time` đi trọn ngày không làm đổi đặc trưng (ML-07) |
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

### 2.6 Trạng thái sau giai đoạn 7 (2026-09-28)

| Ca | Tệp | Trạng thái |
|---|---|---|
| TC-30…TC-44 | `tests/test_api.py` | xanh; TC-42 có ba biến thể: mất cơ sở dữ liệu, thiếu hiện vật, hiện vật hỏng |
| TC-45…TC-47 | `tests/test_db.py` | xanh; TC-47 chạy trên cơ sở dữ liệu riêng `fraud_test_migrations` |
| TC-50…TC-55 | `tests/test_scoring.py` (tầng dịch vụ) và `tests/test_api.py` (qua HTTP) | xanh |
| TC-12 | `tests/test_threshold_parity.py` | xanh từ giai đoạn 8 — xem §2.7 |

Toàn bộ: 311 xanh, 1 bỏ qua. Máy không có PostgreSQL: 88 ca cần cơ sở dữ liệu tự bỏ qua.

**Lệch khỏi §1.1:** giữa các ca, bảng được dọn bằng `TRUNCATE` rồi nạp lại ba khóa mặc định của
`settings`, thay vì rollback. Ca kiểm thử API đi qua máy chủ thật, còn đường ghi `COPY` và chế
độ phát lại tự mở giao dịch riêng, nên không bọc được trong một giao dịch ngoài. Dọn cả
`settings` là cần thiết: một ca đổi `cost_fn` từng làm đỏ ca xem trước ngưỡng chạy sau nó.

**Giới hạn của `TestClient`:** nó đọc hết thân phản hồi rồi mới trả, không stream thật. Ca phát lại
trong pytest vì vậy chỉ phát giờ cuối của ngày 2 ở tốc độ 3.600 (khoảng 1 giây). AC-A6 (3 phút liên
tục) kiểm trên uvicorn thật: 185 giây, 1.532 giao dịch, 0 lỗi.

### 2.7 Trạng thái sau giai đoạn 8 (2026-09-28)

| Ca | Tệp | Trạng thái |
|---|---|---|
| TC-12 | `tests/test_threshold_parity.py` | xanh cho **cả hai** bản giao diện (`web/threshold.js` và, từ khi có bản Next.js, `frontend/src/lib/threshold.mjs`): chạy chính tệp bằng Node, so với `src/threshold.py` trên 20 ngưỡng của tập kiểm thử thật (5 phương án của `threshold.json`, 5 điểm gian lận có thật, điểm cảnh báo nhỏ nhất và hai điểm lệch nó một ulp, hai đầu, ba lượng tử) và hai bộ dữ liệu tổng hợp có điểm trùng nhau đúng tại ngưỡng. TP/FP/FN/TN trùng tuyệt đối; precision, recall, F1, chi phí, cảnh báo/ngày trùng từng bit. Máy không có Node thì tự bỏ qua |
| — | `tests/test_threshold_parity.py` | canh thêm: `index.html` nạp `threshold.js` và `app.js` gọi đúng `FraudThreshold.prepare`/`metricsAt`; mã của `frontend/src` nạp `threshold.mjs` và gọi `prepare`/`metricsAt`; không nơi nào tự so điểm với ngưỡng — TC-12 chỉ có nghĩa khi giao diện thật sự dùng hàm vừa kiểm |
| — | `tests/test_artifacts.py` | 2 ca mới: đường PR sau khi rút mẫu giữ vùng precision cao (lỗi phát hiện qua UI-04, [lenh-chay §8.5](lenh-chay.md)) |

Toàn bộ: 323 xanh, 0 bỏ qua — TC-12 có 10 ca, 5 cho mỗi bản giao diện.

Phần giao diện (AC-A1…AC-A3, AC-A5, T-55, T-56) kiểm bằng Puppeteer điều khiển Chrome thật trên API và
PostgreSQL thật, không đưa vào pytest (cần trình duyệt và máy chủ đang chạy). Số đo ở
[lenh-chay §8.3](lenh-chay.md). AC-A9 mới thử sơ bộ; lượt đầy đủ thuộc T-61.

### 2.8 Trạng thái sau giai đoạn 9 (2026-10-02)

| Ca | Tệp | Trạng thái |
|---|---|---|
| — | `tests/test_packaging.py` | 12 ca, không cần Docker: `api/requirements*.txt` ghim bằng `==` và **trùng** `metrics.json → environment.packages`; xgboost chỉ nằm trong tệp `--no-deps`; mọi gói của container có trong `requirements.txt` gốc; `api/entrypoint.py` thử lại migration khi PostgreSQL từ chối kết nối, dừng khi hết lượt, không thử lại lỗi của chính migration; compose đủ 3 dịch vụ, `api` chờ `db` khỏe, healthcheck của `db` qua TCP, hiện vật gắn chỉ đọc; `.dockerignore` là danh sách trắng không chứa `data/`, `models/`, `.env` |

Toàn bộ: 335 xanh, 0 bỏ qua.

### 2.9 Sau rà soát (2026-10-09)

| Tệp | Ca mới hoặc đổi |
|---|---|
| `tests/test_threshold.py` | `missed_cost` và `fn_costs` (chi phí theo từng giao dịch) tính tay được; `fn_costs` hằng số trùng `cost_fn`; vụ lọt rẻ đẩy ngưỡng lên; tiêu chí `min_precision` đòi cả phần đuôi, kể cả khi precision chạm mức sớm rồi tụt |
| `tests/test_scoring.py` | TC-51 theo luật mới (`max(τ, τ_chặn)`); mức chặn không bao giờ dưới τ; `FastScorer` trùng **từng bit** `model.predict_proba` và margin trên cả 56.746 giao dịch; τ_chặn đạt precision ≥ 95% trên OOF, dải review của tập kiểm thử còn gian lận |
| `tests/test_api.py` | T-48 so SHAP của `/explain` (tính bằng `pred_contribs`) với `explainer.joblib` — trùng từng bit, cùng `base_value`. NFR-01 mang dấu `perf` |
| `tests/test_packaging.py` | `requirements.txt` gốc ghim `==` và trùng `metrics.json`; ảnh `api` không cài shap; ba cổng chỉ mở trên `127.0.0.1` theo mặc định; mật khẩu DB đọc từ biến môi trường |

Toàn bộ: 350 xanh, 1 bỏ chọn (`perf`), 2 phút 15 giây. `pytest -m perf`: NFR-01 p95 phía máy chủ
18,7 / 24,1 / 21,0 ms qua ba lần (trước rà soát: 60,6 / 51,3 / 49,9 ms).

Hai kịch bản Puppeteer của giai đoạn 8 nay nằm trong repo (`scripts/ui/flow.js` 21 bước, thêm
`scripts/ui/keyboard.js` 24 bước cho AC-A9), vẫn ngoài pytest vì cần Chrome và hệ thống đang chạy.
Kết quả trên hệ thống đóng gói: §4.4.

## 3. Kiểm chứng phương pháp

Đây là phần không tự động hóa được hoàn toàn; làm theo danh sách kiểm.

### 3.1 Rà soát rò rỉ dữ liệu — thực hiện cuối ngày 12

- [x] Mọi `fit_resample` chỉ xuất hiện bên trong định nghĩa pipeline, không đứng riêng. — `test_leak_1_…`
- [x] `train_test_split` xuất hiện **trước** mọi thao tác resample trong trình tự notebook. — `test_leak_2_and_6_…`
- [x] Không có `scaler.fit(X)` trên toàn bộ dữ liệu ở bất kỳ đâu. — `test_leak_3_…`, TC-21
- [x] `StratifiedKFold` được dùng ở mọi chỗ có CV; tìm chuỗi `KFold(` không có tiền tố `Stratified`. — `test_leak_4_…`
- [x] Ngưỡng được chọn trên dữ liệu out-of-fold hoặc tập huấn luyện, không trên tập kiểm thử (ML-08). — `test_leak_5_…`
- [x] Dòng trùng lặp bị loại trước khi chia tập. — `test_leak_2_and_6_…`, `test_leak_6_…`, TC-23
- [x] `random_state=42` có mặt ở mọi đối tượng có tham số này. — `test_leak_7_…` (3 ca)

**Rà soát ngày 12 (T-28): đạt 7/7.** Cả bảy ô được viết thành kiểm thử trong
`tests/test_no_leakage.py`, quét mã trong `src/`, `scripts/`, `app.py` và ô code của mọi notebook,
nên chúng còn đúng cả sau lần rà soát này. Lần quét tìm ra đúng một vi phạm — `app.py` lấy mẫu
demo với `random_state=0` — đã sửa về `RANDOM_STATE`. Chi tiết ở notebook 05 §6.

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

### 4.4 Kết quả nghiệm thu phần ứng dụng (T-61, 2026-10-02)

Kiểm trên **hệ thống đóng gói** — bản sao sạch của repo chạy bằng `docker compose up --build` — đi qua
đúng đường của người dùng: Chrome 154 → nginx (cổng 3000) → `api` → PostgreSQL. Lệnh và số đo chi tiết:
[lenh-chay §9.3–§9.4](lenh-chay.md).

| Mã | Cách kiểm | Kết quả | Đạt |
|---|---|---|---|
| AC-A1 | `flow.js`: tải `data/giao-dich-10000.csv` qua nút "Tải CSV" | 3,1 giây tới lúc hiện kết quả | Có |
| AC-A2 | `flow.js`: điểm của 25 dòng trang đầu | giảm dần | Có |
| AC-A3 | `flow.js`: 21 lần kéo 0,5 → τ\*, đo tới khi vẽ xong hai khung hình | lâu nhất 43–131 ms qua 4 lượt | Có |
| AC-A4 | `flow.js`: ngăn kéo UI-02 | 5 dương + 3 âm, chú thích PCA, nhãn thật ẩn tới khi quyết định. "Đúng 5 và 3" hiểu là "khi mô hình có đủ" — 77% giao dịch điểm thấp có ít hơn 5 yếu tố dương ([05 §7](05-thiet-ke-api.md)) | Có |
| AC-A5 | `flow.js`: chi phí bỏ lọt 122,21 → 600 EUR | ngưỡng tối ưu 0,02317 → 0,002262 | Có |
| AC-A6 | đọc `/replay/stream` qua nginx 185 giây | 1.605 giao dịch, 14 cảnh báo, 0 lỗi, khoảng lặng dài nhất 1,0 s | Có |
| AC-A7 | ghi 2 kết luận, `docker compose restart`, đọc lại qua API và `SELECT … FROM reviews` | còn nguyên, cùng `threshold_used`; cũng còn sau `pg_restore` | Có |
| AC-A8 | bản sao sạch (chỉ tệp git + 7 hiện vật), dự án Compose và volume mới | một lệnh dựng cả hệ thống; cold 13,7–14,9 s, warm 11,1–11,5 s. **Giới hạn:** cùng máy vật lý, ảnh nền có sẵn trong bộ đệm, cổng PostgreSQL đổi thành 5434 | Có, kèm giới hạn |
| AC-A9 | `keyboard.js` (24 bước) + Lighthouse | 24/24; Lighthouse Accessibility 100, Best Practices 100 trên 4 màn hình. Tab và Enter cho mọi thao tác; phím mũi tên trong ba nhóm có một điểm dừng Tab (bảng hàng đợi, thanh trượt, nhóm radio — mẫu WAI-ARIA) | Có |
| AC-A10 | 13 đầu vào sai gửi qua nginx | đúng mã HTTP và mã lỗi JSON: 422 ×8 (thân rỗng, không phải JSON, thiếu cột, sai kiểu, `Amount` âm, ngưỡng 1,5, hai tệp CSV hỏng), 400 ×2, 404, 405, và 413 do nginx trả cho tệp 115 MB; `RestartCount` của container vẫn 0 | Có |

## 5. Danh sách kiểm trước buổi bảo vệ

- [ ] Chạy `pytest` — toàn bộ xanh.
- [ ] `docker compose down -v` rồi `docker compose up` trên máy sạch — lần đầu dưới 45 giây, lần thứ hai dưới 15 giây (NFR-04).
- [ ] Khởi động hệ thống một lần **trước** buổi bảo vệ để volume `pgdata` đã sẵn sàng.
- [ ] Chuẩn bị sẵn bản `pg_dump` dữ liệu demo để khôi phục nếu có sự cố: `python scripts/demo_db.py seed` rồi `dump`; khôi phục bằng `restore`, khoảng 2,5 giây ([06 §7.4](06-thiet-ke-luu-tru.md)).
- [ ] Mở cả bốn màn hình, không có lỗi trong console trình duyệt.
- [ ] Kịch bản trình bày ngưỡng (kéo 0,5 → τ\* = 0,0232) chạy mượt — số thật ở [lenh-chay §8.3](lenh-chay.md).
- [ ] Chuẩn bị sẵn 4 giao dịch mẫu: gian lận dễ, gian lận khó, hợp lệ dễ, hợp lệ khó.
- [ ] Có phương án dự phòng khi mạng hoặc máy chiếu hỏng: ảnh chụp màn hình và bản PDF báo cáo.
- [ ] Ôn ba câu hỏi chắc chắn bị hỏi:
      1. Vì sao PR-AUC chứ không phải ROC-AUC?
      2. SMOTE hoạt động thế nào và vì sao nó phải nằm trong pipeline?
      3. Ngưỡng này được chọn ra sao, và sẽ đổi thế nào nếu chi phí thay đổi?
