# 09 — Kế hoạch triển khai

Tổng 21 ngày. Phần ứng dụng chiếm khoảng 6 ngày; phần còn lại dành cho mô hình,
phân tích và báo cáo.

Tài liệu này giải thích **vì sao** lộ trình có thứ tự như vậy. Bản thao tác hằng
ngày, tick được từng việc, nằm ở [TASKS.md](../TASKS.md).

## 1. Hiện trạng so với mục tiêu

Repo hiện có:

| Đã có | Còn thiếu |
|---|---|
| `src/data.py`, `src/evaluate.py`, `src/plots.py` | `src/features.py`, `src/threshold.py` |
| 7 notebook khung (01–07) | Nội dung notebook; notebook `08_export_artifacts.ipynb` |
| `app.py` (Streamlit tối giản) | Toàn bộ `api/`, `web/` |
| `requirements.txt`, `.gitignore` | `tests/`, `docker-compose.yml`, `api/migrations/` |
| `reports/bao-cao.md` (khung) | Nội dung báo cáo, `reports/figures/` |
| `docs/` (tài liệu này) | — |

Hai hạng mục nên làm sớm nhất vì mọi thứ khác phụ thuộc vào chúng:
`src/features.py` (ranh giới train/serve) và `src/threshold.py` (logic ngưỡng
dùng chung cho notebook, API và kiểm thử).

## 2. Lộ trình

| Giai đoạn | Ngày | Nội dung | Sản phẩm bàn giao |
|---|---|---|---|
| Nền tảng | 1–2 | Tải dữ liệu, kiểm tra toàn vẹn, EDA | 8 biểu đồ trong `reports/figures/` |
| Thống kê | 3–4 | Mann–Whitney, hiệu chỉnh BH, effect size | Bảng xếp hạng đặc trưng |
| Khung đánh giá | 5 | `features.py`, `threshold.py`, chia tập, `evaluate()` | Hàm dùng chung + TC-01…TC-06 xanh |
| Mô hình cơ sở | 6–7 | Logistic Regression, Decision Tree, mô hình rỗng | Bảng mốc so sánh |
| Mất cân bằng | 8–10 | Lưới 5 × 4 | Bảng 20 dòng + đồ thị PR chồng |
| Tinh chỉnh | 11–12 | RandomizedSearchCV, chốt mô hình, bootstrap CI | `model.joblib` |
| Ngưỡng & SHAP | 13–14 | Đường cong chi phí, explainer, notebook 08 | `threshold.json`, `metrics.json` |
| **Điểm quyết định** | **14** | **Chốt phương án A hay B cho ứng dụng** | Quyết định ghi vào README |
| API | 15–16 | FastAPI, PostgreSQL + Alembic, toàn bộ endpoint | API chạy, `/docs` mở được, TC-30…TC-47 xanh |
| Giao diện | 17–19 | UI-01, UI-03, UI-02, UI-05, UI-04 theo thứ tự này | Ứng dụng hoàn chỉnh |
| Đóng gói | 20 | Docker Compose, README, rà soát rò rỉ, kiểm tra tái lập | Chạy bằng một lệnh |
| Báo cáo | 21 | Viết báo cáo, chuẩn bị bảo vệ | Tài liệu nộp |

## 3. Phân rã công việc theo giai đoạn

### Ngày 1–2 — Nền tảng

- [x] Tải `creditcard.csv`, đặt vào `data/`, xác nhận 284.807 dòng.
- [x] Kiểm tra toàn vẹn theo [02 §6](02-dac-ta-du-lieu.md): giá trị thiếu, trùng lặp, phân bố nhãn.
- [x] Viết `01_eda.ipynb`, sinh đủ 8 biểu đồ bắt buộc.
- [x] Ghi nhận số dòng trùng lặp và quyết định xử lý (DS-20) — 1.081 dòng, 19 gian lận; xem [02 §6.1](02-dac-ta-du-lieu.md).

**Xong là khi:** 8 tệp PNG có trong `reports/figures/` và mỗi biểu đồ trả lời được
một câu hỏi cụ thể, không phải vẽ cho đủ.

### Ngày 3–4 — Thống kê

- [x] Mann–Whitney U cho 30 đặc trưng, hai lớp.
- [x] Hiệu chỉnh Benjamini–Hochberg, ghi rõ số kiểm định còn ý nghĩa sau hiệu chỉnh — 27/30 (Bonferroni chỉ giữ 24).
- [x] Tính Cliff's delta **và** Cohen's d, xếp hạng theo độ lớn hiệu ứng — 17 đặc trưng có |d| ≥ 0,8.
- [x] Đối chiếu xếp hạng thống kê với tương quan thô — thứ hạng trùng khít (Spearman 1,000), chỉ độ lớn bị tỷ lệ lớp dương nén xuống trần 0,33.

**Xong là khi:** có bảng xếp hạng đặc trưng dùng lại được ở notebook 07 để so với SHAP.

### Ngày 5 — Khung đánh giá (bước then chốt)

- [ ] `src/features.py`: `build_features()`, hằng số `FEATURE_ORDER`.
- [ ] `src/threshold.py`: quét ngưỡng, đường cong chi phí, bốn tiêu chí chọn ngưỡng.
- [ ] Mở rộng `src/evaluate.py`: bootstrap khoảng tin cậy.
- [ ] `tests/test_features.py`, `tests/test_threshold.py` — chạy xanh.

**Xong là khi:** TC-01…TC-06 và TC-10…TC-15 xanh. Đây là ngày trả lại nhiều thời
gian nhất về sau; đừng rút ngắn.

### Ngày 6–7 — Mô hình cơ sở

- [x] Chia tập phân tầng 80/20, lưu `test_set.parquet` — tập kiểm thử 56.746 dòng / 95 gian lận.
- [x] Mô hình rỗng (luôn dự đoán 0) — accuracy **99,8326%**, bỏ lọt toàn bộ 95 vụ.
- [x] Logistic Regression và Decision Tree, không xử lý mất cân bằng — PR-AUC 0,696 và 0,628.
- [x] Bảng mốc: accuracy, PR-AUC, ROC-AUC, Recall, Precision cạnh nhau — accuracy xếp hạng **ngược** với PR-AUC.

**Xong là khi:** bảng cho thấy rõ accuracy không phân biệt được các mô hình còn
PR-AUC thì có — đây là bằng chứng cho G-2.

### Ngày 8–10 — Chiến lược mất cân bằng

- [ ] Dựng 20 pipeline theo [04 §3](04-thiet-ke-mo-hinh-ml.md).
- [ ] Chạy `cross_val_score` với `StratifiedKFold(5)`, `scoring='average_precision'`.
- [ ] Ghi kết quả vào một DataFrame duy nhất, lưu ra `reports/grid_results.csv`.
- [ ] Vẽ đường PR của 5 chiến lược cho mô hình tốt nhất.
- [ ] `tests/test_no_leakage.py` — chạy xanh.

**Rủi ro thời gian:** SMOTE trên 227.000 dòng với Random Forest 300 cây chạy khá
lâu. Nếu quá 3 giờ cho cả lưới, giảm `n_estimators` xuống 100 ở giai đoạn so sánh
và chỉ tăng lại ở giai đoạn tinh chỉnh. Ghi rõ điều này trong báo cáo.

### Ngày 11–12 — Tinh chỉnh

- [ ] `RandomizedSearchCV` 30 lần thử cho cấu hình thắng cuộc.
- [ ] Huấn luyện lại trên toàn tập train, đánh giá trên tập test.
- [ ] Bootstrap 1.000 lần, tính khoảng tin cậy.
- [ ] Đánh giá bổ sung theo cách chia thời gian (FR-10).
- [ ] **Rà soát rò rỉ dữ liệu** theo danh sách kiểm [08 §3.1](08-ke-hoach-kiem-thu.md).

**Xong là khi:** AC-M1, AC-M2, AC-M4, AC-M6 đạt.

### Ngày 13–14 — Ngưỡng, SHAP, hiện vật

- [ ] Đường cong chi phí, bốn tiêu chí chọn ngưỡng, phân tích độ nhạy.
- [ ] SHAP toàn cục và cục bộ; phân tích các FN và FP điểm cao.
- [ ] `08_export_artifacts.ipynb` sinh toàn bộ hiện vật theo [06 §3](06-thiet-ke-luu-tru.md).
- [ ] Chọn 200 giao dịch mẫu cho `sample_pool.json`, đủ bốn nhóm.

**Điểm quyết định cuối ngày 14:** nếu mọi hiện vật đã sẵn sàng → làm phương án B.
Nếu còn nợ việc ở phần mô hình → chuyển sang phương án A (Streamlit) và dồn thời
gian cho báo cáo. Quyết định này ghi vào README, không để lửng.

### Ngày 15–16 — API

- [ ] `api/main.py`: nạp hiện vật lúc startup, cấu hình CORS, `/health`.
- [ ] `api/schemas.py`: toàn bộ mô hình Pydantic theo [05](05-thiet-ke-api.md).
- [ ] `api/db.py`: engine + pool kết nối PostgreSQL, bật `pool_pre_ping`.
- [ ] `api/models_orm.py` và migration `0001_initial_schema`: ba bảng, chỉ mục, ràng buộc `CHECK`.
- [ ] Migration `0002_seed_settings`: bốn khóa mặc định lấy từ `threshold.json`.
- [ ] `tests/conftest.py`: cơ sở dữ liệu `fraud_test` dùng một lần, fixture rollback.
- [ ] Các route theo thứ tự: scoring → threshold → transactions → metrics → replay.
- [ ] `tests/test_api.py` — chạy xanh.

**Xong là khi:** mở `/docs` thấy đủ 15 endpoint và thử được từng cái trên Swagger.

### Ngày 17–19 — Giao diện

Thứ tự làm đúng bằng thứ tự ưu tiên giữ lại ở [07 §1](07-thiet-ke-giao-dien.md):

- [ ] Ngày 17: khung ứng dụng + UI-01 hàng đợi.
- [ ] Ngày 18: UI-03 ngưỡng (gồm UI-D1 tính phía máy khách) + TC-12.
- [ ] Ngày 19 buổi sáng: UI-02 chi tiết + SHAP.
- [ ] Ngày 19 buổi chiều: UI-05 phát lại; nếu kịp thì UI-04 hiệu năng.

### Ngày 20 — Đóng gói và kiểm tra

- [ ] `docker-compose.yml` ba dịch vụ `db` + `api` + `web`, volume `pgdata`, healthcheck `pg_isready`, `depends_on: service_healthy`.
- [ ] Entrypoint của `api` chạy `alembic upgrade head` trước khi khởi động uvicorn.
- [ ] Bấm giờ khởi động cả lần đầu (cold) và lần sau (warm) — NFR-04.
- [ ] Tạo bản `pg_dump` dữ liệu demo để dự phòng cho buổi bảo vệ.
- [ ] Thử `docker compose up` trên máy sạch hoặc máy ảo (AC-A8).
- [ ] Kiểm tra tái lập: chạy lại notebook trong kernel sạch (AC-M5).
- [ ] Duyệt toàn bộ danh sách AC-A1…AC-A10.
- [ ] Viết README hướng dẫn chạy lại từ đầu.

### Ngày 21 — Báo cáo

- [ ] `reports/bao-cao.md` theo dàn ý ở §5.
- [ ] Chèn biểu đồ, kiểm tra mọi biểu đồ có tiêu đề và nhãn trục (AC-D4).
- [ ] Chuẩn bị kịch bản trình bày 10 phút.
- [ ] Danh sách kiểm trước bảo vệ ở [08 §5](08-ke-hoach-kiem-thu.md).

## 4. Rủi ro

| Mã | Rủi ro | Khả năng | Tác động | Ứng phó |
|---|---|---|---|---|
| R-01 | Lưới 20 tổ hợp chạy quá lâu | Trung bình | Trung bình | Giảm `n_estimators` ở giai đoạn so sánh; chạy qua đêm; ghi rõ trong báo cáo |
| R-02 | Phát hiện rò rỉ dữ liệu muộn, phải chạy lại toàn bộ | Thấp | Cao | Rà soát sớm ở ngày 12, không đợi ngày 20 |
| R-03 | Không kịp làm ứng dụng phương án B | Trung bình | Trung bình | Điểm quyết định ngày 14; `app.py` Streamlit là lưới an toàn |
| R-04 | PR-AUC không đạt 0,75 | Thấp | Cao | Kiểm tra lại đặc trưng và `scale_pos_weight`; nếu vẫn thấp thì báo cáo trung thực kèm phân tích nguyên nhân — trung thực vẫn tốt hơn số liệu đẹp đáng ngờ |
| R-05 | Đặc trưng lệch giữa notebook và API | Trung bình | Cao | TC-01 kiểm thử vàng; một hàm `build_features()` duy nhất |
| R-06 | SHAP quá chậm làm giao diện đứng | Thấp | Trung bình | Tách endpoint riêng (đã thiết kế); thêm khung xương khi đang tải |
| R-07 | Máy yếu, hết RAM khi chạy SMOTE | Trung bình | Trung bình | `sampling_strategy=0.1` thay vì 1:1; giảm mẫu lớp âm trước khi SMOTE |
| R-08 | Hết thời gian cho báo cáo vì sa đà vào ứng dụng | Trung bình | Cao | Ngày 21 là bất khả xâm phạm; ứng dụng đóng băng tính năng sau ngày 19 |
| R-09 | Mất thời gian vì PostgreSQL: cổng 5432 bị chiếm, migration lỗi, container chưa sẵn sàng | Trung bình | Thấp | Dựng dịch vụ `db` ngay đầu ngày 15, trước khi viết route; `pg_isready` + `depends_on` xử lý phần lớn vấn đề; đổi cổng máy chủ sang 5433 nếu trùng |
| R-10 | Chuyển sang phương án A nhưng vẫn cần cơ sở dữ liệu | Thấp | Thấp | Streamlit dùng chung `api/db.py`; chỉ cần `docker compose up -d db` |

## 5. Dàn ý báo cáo

| Phần | Nội dung | Trang ước tính |
|---|---|---|
| 1 | Đặt vấn đề, dữ liệu, hạn chế của dữ liệu | 1–2 |
| 2 | EDA — các phát hiện chính, không phải mọi biểu đồ đã vẽ | 2–3 |
| 3 | Kiểm định thống kê và xếp hạng đặc trưng | 1–2 |
| 4 | Khung đánh giá: vì sao PR-AUC, vì sao không accuracy | 1–2 |
| 5 | Mô hình cơ sở | 1 |
| 6 | **So sánh chiến lược mất cân bằng — bảng 20 dòng và thảo luận** | 3–4 |
| 7 | Mô hình cuối, tinh chỉnh, khoảng tin cậy | 2 |
| 8 | **Ngưỡng và trade-off Precision–Recall, phân tích chi phí** | 3–4 |
| 9 | Giải thích SHAP và phân tích lỗi | 2 |
| 10 | Ứng dụng demo — kiến trúc và ảnh chụp màn hình | 1–2 |
| 11 | Hạn chế và hướng mở rộng | 1 |

Phần 6 và phần 8 là trọng tâm; các phần khác phục vụ chúng.

## 6. Lỗi cần tránh

| Lỗi | Vì sao tệ |
|---|---|
| Dùng accuracy làm chỉ số chính | Mô hình rỗng đạt 99,83% |
| SMOTE trước khi chia tập | Rò rỉ; mọi chỉ số vô nghĩa |
| Cân bằng về đúng 1:1 | Méo xác suất tiên nghiệm, precision sụp |
| Báo cáo chỉ số ở ngưỡng 0,5 mà không bàn tới ngưỡng | Bỏ qua chính điểm thú vị của bài |
| Vẽ 30 biểu đồ mà không rút ra kết luận nào | Khối lượng không thay được chiều sâu |
| Gán ý nghĩa nghiệp vụ cho V1–V28 | Sai sự thật, dễ bị hỏi vặn |
| Báo cáo con số trần không kèm khoảng tin cậy với 98 mẫu dương | Không chứng minh được khác biệt là thật |
| Chỉ trình bày mô hình thắng cuộc, giấu các cấu hình kém | Mất đi phần so sánh vốn là trọng tâm đề tài |

## 7. Hướng mở rộng nếu còn thời gian

| Hạng mục | Giá trị thêm |
|---|---|
| Autoencoder (FR-11) | Đối chiếu có giám sát và không giám sát; vận dụng Chương 8 |
| Hiệu chỉnh xác suất (Platt, isotonic) | Điểm rủi ro trở thành xác suất thật, chi phí kỳ vọng chính xác hơn |
| Phân tích quyết định của người thẩm định (FR-44) | So người với mô hình — phần thảo luận rất hiếm gặp ở đồ án sinh viên |
| Học theo chi phí trực tiếp (`sample_weight` theo `Amount`) | Tối ưu thẳng thứ mà ta thật sự quan tâm |
| Mô phỏng drift: huấn luyện ngày 1, đánh giá theo từng giờ của ngày 2 | Cho thấy hiệu năng suy giảm theo thời gian thật ra sao |
