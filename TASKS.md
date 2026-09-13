# Danh sách công việc — Đề tài phát hiện gian lận thẻ tín dụng

Bảng theo dõi để tick từng việc. Kế hoạch theo ngày và lý do đằng sau thứ tự này
nằm ở [docs/09 — Kế hoạch triển khai](docs/09-ke-hoach-trien-khai.md); tệp này là
bản thao tác hằng ngày.

**Quy ước:** `[ ]` chưa làm · `[~]` đang làm · `[x]` xong.
Mỗi việc có điều kiện *xong là khi* — chưa đạt điều kiện đó thì chưa được tick.

## Tiến độ

| Giai đoạn | Việc | Xong | Trạng thái |
|---|---|---|---|
| 0. Nền tảng mã nguồn | 10 | 10 | ✅ |
| 1. EDA và thống kê | 5 | 5 | ✅ |
| 2. Mô hình cơ sở | 4 | 4 | ✅ |
| 3. Chiến lược mất cân bằng | 4 | 0 | |
| 4. Tinh chỉnh và kiểm chứng | 5 | 0 | |
| 5. Ngưỡng, chi phí, SHAP | 6 | 0 | |
| 6. Xuất hiện vật | 4 | 0 | |
| 🚩 Mốc quyết định A/B | 1 | 0 | |
| 7. API | 11 | 0 | |
| 8. Giao diện | 7 | 0 | |
| 9. Đóng gói | 5 | 0 | |
| 10. Báo cáo và bảo vệ | 5 | 0 | |
| **Tổng** | **67** | **19** | **28%** |

---

## Giai đoạn 0 — Nền tảng mã nguồn ✅

Phần này đã hoàn thành. Giữ lại để biết cái gì đã có mà không phải viết lại.

- [x] **T-01** Tạo `.venv` (Python 3.12.5) và cài `requirements.txt` — *xong là khi:* `pytest` chạy được.
- [x] **T-02** `scripts/download_data.py` — tải từ Kaggle API hoặc `--mirror`, tự kiểm tra toàn vẹn — *xong là khi:* `--check` báo đúng 284.807 dòng / 492 gian lận.
- [x] **T-03** `src/config.py` — hằng số dùng chung, `RANDOM_STATE=42`, chi phí mặc định.
- [x] **T-04** `src/features.py` — `build_features()`, `validate_raw()`, `make_preprocessor()` — *xong là khi:* TC-01…TC-06 xanh.
- [x] **T-05** `src/threshold.py` — `sweep()`, `cost_curve()`, `pick_threshold()` 4 tiêu chí — *xong là khi:* TC-10…TC-15 xanh.
- [x] **T-06** `src/data.py` — `check_integrity()`, `drop_duplicates()`, `split_data()`, `temporal_split()`.
- [x] **T-07** `src/evaluate.py` — `bootstrap_ci()` phân tầng, `headline_metrics()`, `curve_points()`.
- [x] **T-08** `src/modeling.py` — `build_pipeline()`, `run_grid()`, `oof_scores()` — *xong là khi:* TC-20…TC-23 xanh.
- [x] **T-09** `src/plots.py` — 8 biểu đồ EDA + 6 biểu đồ đánh giá — *xong là khi:* cả 14 hàm vẽ được.
- [x] **T-10** Tải và kiểm chứng `data/creditcard.csv` — *xong là khi:* SHA-256 khớp, 0 ô thiếu.

---

## Giai đoạn 1 — EDA và thống kê (ngày 1–4) ✅

- [x] **T-11** `notebooks/01_eda.ipynb`: nạp qua `load_prepared()`, in báo cáo toàn vẹn, ghi nhận 1.081 dòng trùng lặp — *xong là khi:* notebook chạy từ đầu đến cuối trong kernel sạch.
- [x] **T-12** Sinh đủ 8 biểu đồ bắt buộc vào `reports/figures/` với tiền tố `01_` — *xong là khi:* đủ 8 tệp PNG, mỗi biểu đồ có tiêu đề và nhãn trục.
- [x] **T-13** Viết nhận xét EDA: mỗi biểu đồ một đoạn trả lời một câu hỏi cụ thể — *xong là khi:* không có biểu đồ nào bị bỏ trống nhận xét. Ba phát hiện đã biết cần nêu: giờ 2h có tỷ lệ gian lận 1,713% so với 0,048% lúc 10h; trung vị `Amount` của gian lận (9,25) thấp hơn hợp lệ (22,00) nhưng trung bình lại cao hơn; 27 giao dịch gian lận có `Amount = 0`.
- [x] **T-14** `notebooks/02_statistics.ipynb`: Mann–Whitney U cho 30 đặc trưng + hiệu chỉnh Benjamini–Hochberg + Cohen's d — *xong là khi:* có bảng xếp hạng và nêu rõ số kiểm định còn ý nghĩa sau hiệu chỉnh.
- [x] **T-15** Lưu bảng xếp hạng ra `reports/feature_ranking.csv` để notebook 07 đối chiếu với SHAP — *xong là khi:* tệp tồn tại, có cột `cohens_d` và `p_adjusted`.

> Điểm cần viết trong T-14: tương quan Pearson cao nhất chỉ −0,326 (V17) trong khi Cohen's d là −8,32. Với 0,17% mẫu dương, tương quan điểm-nhị phân bị nén xuống — cùng một cái bẫy như accuracy 99,83%.

### Ghi chú khi làm xong giai đoạn 1

**Các con số mốc ở T-13 và T-14 là số trên dữ liệu THÔ.** Pipeline chạy trên dữ liệu đã loại
1.081 dòng trùng lặp (DS-20), nên số thực tế lệch nhẹ. Notebook 01 §1.1 in bảng đối chiếu hai
cột; quy ước đã chốt: **báo cáo lấy cột "sau khi loại trùng lặp"**.

| Mốc | Dữ liệu thô | Sau khi loại trùng lặp |
|---|---|---|
| Tỷ lệ gian lận lúc 2h | 1,713% | **1,451%** (48/3.308) |
| Tỷ lệ gian lận lúc 10h | 0,048% | **0,048%** (8/16.548) |
| Trung vị `Amount` gian lận | 9,25 | **9,82** (hợp lệ 22,00) |
| Trung bình `Amount` gian lận | 122,21 | **123,87** (hợp lệ 88,41) |
| Gian lận có `Amount = 0` | 27 | **25** |
| Tương quan Pearson mạnh nhất (V17) | −0,326 | **−0,313** |
| Cohen's d của V17 | −8,32 | **−8,09** |

Hướng của cả ba phát hiện giữ nguyên, chỉ độ lớn nhích nhẹ. Xem thêm [docs/02 §6](docs/02-dac-ta-du-lieu.md).

**Phát sinh thêm trong giai đoạn này** (không có trong kế hoạch ban đầu, nhưng cần để T-14 và
T-15 tái lập được):

- `src/stats.py` — `feature_ranking()`, `benjamini_hochberg()`, `cohens_d()`,
  `cliffs_delta_from_u()`, `pearson_from_cohens_d()`.
- `tests/test_stats.py` — 24 ca, chốt thủ tục BH và công thức effect size bằng ví dụ tính tay được.
- `src/plots.py` — bổ sung nhãn trục còn thiếu ở EDA 4, 5, 6, 7, 8 để đạt điều kiện của T-12.
- Ba hình phụ tiền tố `02_` trong `reports/figures/` (không tính vào 8 hình bắt buộc của T-12).

**Kết quả chính:** 27/30 kiểm định còn ý nghĩa sau hiệu chỉnh BH (Bonferroni chỉ giữ 24);
17 đặc trưng có |Cohen's d| ≥ 0,8. Cohen's d và Cliff's delta bất đồng về thứ hạng
(Spearman 0,921): V17 hạng 1 theo d nhưng hạng 11 theo delta — để T-32 đối chiếu với SHAP.

---

## Giai đoạn 2 — Mô hình cơ sở (ngày 5–7) ✅

- [x] **T-16** `notebooks/03_baseline.ipynb`: chia phân tầng 80/20, lưu `data/test_set.parquet` — *xong là khi:* tập test có đúng 95 mẫu gian lận (sau khi loại trùng lặp).
- [x] **T-17** Mô hình rỗng luôn dự đoán 0 — *xong là khi:* có con số accuracy 99,83% đưa vào báo cáo làm bằng chứng cho G-2.
- [x] **T-18** Logistic Regression và Decision Tree không xử lý mất cân bằng — *xong là khi:* có bảng mốc so sánh.
- [x] **T-19** Bảng đối chiếu accuracy / PR-AUC / Recall / Precision cho ba mô hình — *xong là khi:* bảng cho thấy accuracy gần như không phân biệt được chúng còn PR-AUC thì có.

### Ghi chú khi làm xong giai đoạn 2

**Bảng mốc** (tập kiểm thử 56.746 dòng / 95 gian lận, τ = 0,5 chưa tối ưu):

| mô hình | accuracy | PR-AUC | ROC-AUC | recall | precision | bắt được |
|---|---|---|---|---|---|---|
| mô hình rỗng | 99,8326% | 0,00167 | 0,5000 | 0,000 | — | 0/95 |
| logistic regression | 99,9154% | 0,6961 [0,598–0,791] | 0,9577 | 0,600 | 0,851 | 57/95 |
| decision tree | 99,9383% | 0,6275 [0,522–0,737] | 0,8746 | 0,716 | 0,895 | 68/95 |

**Bằng chứng cho G-2 mạnh hơn dự kiến.** Accuracy không chỉ *không phân biệt* được ba mô hình
(chênh 0,11 điểm phần trăm) mà còn **xếp hạng ngược**: cây có accuracy cao hơn hồi quy
(99,9383% so với 99,9154%) nhưng PR-AUC lại thấp hơn (0,6275 so với 0,6961). PR-AUC chênh
**416 lần** giữa mô hình tệ nhất và tốt nhất.

**Chưa được tuyên bố mô hình nào thắng.** Bootstrap hiệu PR-AUC theo cặp (04 §5.4) cho
`+0,069 [−0,031, +0,158]` — khoảng tin cậy **chứa 0**. Hồi quy thắng ở 92% số lần lặp, đủ để
nói "có xu hướng nhỉnh hơn", không đủ để kết luận. Quy tắc này phải áp lại cho bảng 20 tổ hợp
ở T-20.

**Phát hiện đáng mang sang giai đoạn 5.** Decision Tree `max_depth=6` chỉ phát ra **8 điểm rủi
ro khác nhau** (hồi quy: 56.537), nên đường PR của nó là cầu thang 9 điểm. Đó là lý do cơ học
khiến nó thua ở PR-AUC dù thắng ở τ = 0,5 — và là cảnh báo rằng **không thể tối ưu ngưỡng trên
một mô hình chỉ có 8 bậc**. Mô hình cuối cùng bắt buộc phải thuộc họ ensemble.

**Phát sinh thêm:** `src/plots.py` thêm `plot_roc_curves()` (bản sinh đôi của `plot_pr_curves`,
để đặt ROC cạnh PR trên cùng bộ mô hình) và nhãn trục tiếng Việt cho `plot_confusion()`.
Hiện vật mới: `data/test_set.parquet` (15,4 MB), `reports/baseline_results.csv`, 4 hình `03_*`.

**Còn cách mục tiêu:** AC-M1 cần PR-AUC ≥ 0,75 (đang 0,696), AC-M2 cần recall ≥ 0,75
(đang 0,60 và 0,72).

---

## Giai đoạn 3 — Chiến lược mất cân bằng (ngày 8–10)

- [ ] **T-20** `notebooks/04_imbalance_strategies.ipynb`: chạy `run_grid()` đủ 20 tổ hợp — *xong là khi:* `reports/grid_results.csv` có 20 dòng. **Cảnh báo thời gian:** 1–3 giờ. Nếu vượt 3 giờ thì hạ `n_estimators` xuống 100 ở giai đoạn so sánh và ghi rõ trong báo cáo.
- [ ] **T-21** Vẽ đường PR của 5 chiến lược chồng lên nhau cho mô hình tốt nhất — *xong là khi:* hình có đường cơ sở 0,00167.
- [ ] **T-22** Viết thảo luận so sánh `scale_pos_weight` với SMOTE — *xong là khi:* nêu được cái nào thắng, thắng bao nhiêu, và rẻ hơn bao nhiêu về thời gian huấn luyện.
- [ ] **T-23** Kiểm tra tính hợp lý: nếu PR-AUC > 0,95 thì dừng lại và rà rò rỉ — *xong là khi:* đã đối chiếu với dải kỳ vọng 0,80–0,87 ở [docs/04 §3.4](docs/04-thiet-ke-mo-hinh-ml.md).

---

## Giai đoạn 4 — Tinh chỉnh và kiểm chứng (ngày 11–12)

- [ ] **T-24** `notebooks/05_advanced_models.ipynb`: `RandomizedSearchCV` 30 lần thử, `scoring='average_precision'` — *xong là khi:* có bảng 5 cấu hình tốt nhất.
- [ ] **T-25** Huấn luyện lại trên toàn tập train, đánh giá trên tập test — *xong là khi:* AC-M1 (PR-AUC ≥ 0,75) và AC-M2 (Recall ≥ 0,75) đạt.
- [ ] **T-26** Bootstrap 1.000 lần cho PR-AUC, Recall, Precision — *xong là khi:* mọi chỉ số chính trong báo cáo đều có khoảng tin cậy (AC-M6).
- [ ] **T-27** Đánh giá theo cách chia thời gian (ngày 1 train, ngày 2 test) — *xong là khi:* có bảng đối chiếu hai cách chia và một đoạn giải thích vì sao cách theo thời gian thấp hơn (AC-M7).
- [ ] **T-28** **Rà soát rò rỉ dữ liệu** theo danh sách kiểm [docs/08 §3.1](docs/08-ke-hoach-kiem-thu.md) — *xong là khi:* cả 7 ô trong danh sách được tick và `pytest tests/test_no_leakage.py` xanh (AC-M4). **Làm ở ngày 12, không để đến ngày 20.**

---

## Giai đoạn 5 — Ngưỡng, chi phí, SHAP (ngày 13–14)

- [ ] **T-29** `notebooks/06_threshold_and_cost.ipynb`: đường cong chi phí + 4 phương án ngưỡng, chọn ngưỡng trên dữ liệu **out-of-fold** (`oof_scores`), không trên tập test — *xong là khi:* ML-08 được tôn trọng.
- [ ] **T-30** Bảng bắt buộc trong báo cáo: 4 ngưỡng × (cảnh báo/ngày, TP, FP, FN, Precision, Recall, chi phí) — *xong là khi:* có thêm một đoạn diễn giải bằng lời dạng "hạ ngưỡng từ 0,5 xuống τ\* bắt thêm N vụ, đổi lại M cảnh báo giả, tiết kiệm ròng X".
- [ ] **T-31** Phân tích độ nhạy theo tỷ lệ chi phí 5:1 → 100:1 bằng `sensitivity_analysis()` — *xong là khi:* có hình cho thấy ngưỡng tối ưu dịch chuyển thế nào (AC-M8).
- [ ] **T-32** `notebooks/07_explainability.ipynb`: SHAP toàn cục (`summary_plot` trên 2.000 mẫu) + xếp hạng `mean|SHAP|` — *xong là khi:* có bảng so SHAP với xếp hạng thống kê ở T-15.
- [ ] **T-33** Phân tích lỗi: giải thích SHAP cho các FN điểm cao nhất và FP điểm cao nhất — *xong là khi:* nêu được mẫu hình chung, kèm câu chốt rằng V1–V28 là PCA nên không diễn giải thành nguyên nhân nghiệp vụ.
- [ ] **T-34** *(Tùy chọn)* Autoencoder huấn luyện chỉ trên lớp bình thường — *xong là khi:* so được PR-AUC với mô hình có giám sát và giải thích **vì sao** nó thua (không dùng tới 492 nhãn có sẵn).

---

## Giai đoạn 6 — Xuất hiện vật (ngày 14)

- [ ] **T-35** `notebooks/08_export_artifacts.ipynb` sinh `models/model.joblib` (pipeline **hoàn chỉnh**, gồm cả scaler) — *xong là khi:* nạp lại tệp và `predict_proba` cho đúng kết quả như trong notebook (AR-02).
- [ ] **T-36** Sinh `models/explainer.joblib`, `models/metrics.json`, `models/threshold.json` — *xong là khi:* cấu trúc khớp [docs/06 §6](docs/06-thiet-ke-luu-tru.md).
- [ ] **T-37** Sinh `data/sample_pool.json` khoảng 200 giao dịch, đủ 4 nhóm `fraud_easy` / `fraud_hard` / `legit_easy` / `legit_hard` — *xong là khi:* mỗi nhóm "hard" có ít nhất 20 mẫu.
- [ ] **T-38** **Kiểm tra tái lập**: chạy lại toàn bộ notebook trong kernel sạch — *xong là khi:* PR-AUC lệch dưới 0,001 (AC-M5, NFR-07).

---

## 🚩 Mốc quyết định — cuối ngày 14

- [ ] **T-39** Chốt phương án ứng dụng và ghi vào README — *xong là khi:* quyết định được viết ra, không để lửng.
  - Hiện vật đã đủ, còn đủ 6 ngày → **phương án B** (FastAPI + web riêng), làm tiếp giai đoạn 7.
  - Còn nợ việc ở phần mô hình → **phương án A** (Streamlit trong `app.py`), bỏ giai đoạn 7, làm rút gọn giai đoạn 8, dồn thời gian cho báo cáo.

---

## Giai đoạn 7 — API (ngày 15–16, chỉ phương án B)

- [ ] **T-40** `docker-compose.yml` dịch vụ `db` (postgres:16-alpine) + volume `pgdata` + healthcheck `pg_isready` — *xong là khi:* `docker compose up -d db` và `pg_isready` trả `accepting connections`. **Làm đầu tiên trong giai đoạn này** (rủi ro R-09).
- [ ] **T-41** `api/db.py` engine + pool (`pool_pre_ping=True`) và `api/models_orm.py` — *xong là khi:* kết nối được bằng `DATABASE_URL` từ `.env`.
- [ ] **T-42** Alembic `0001_initial_schema` (3 bảng, chỉ mục, ràng buộc `CHECK`) và `0002_seed_settings` — *xong là khi:* `upgrade head` → `downgrade base` → `upgrade head` chạy trọn vẹn (TC-47).
- [ ] **T-43** `api/schemas.py` — toàn bộ Pydantic vào/ra theo [docs/05](docs/05-thiet-ke-api.md) — *xong là khi:* thiếu cột trả 422 kèm danh sách cột thiếu (TC-32).
- [ ] **T-44** `api/main.py` nạp hiện vật lúc startup + `/health` — *xong là khi:* `/health` trả 503 khi thiếu mô hình hoặc mất cơ sở dữ liệu (TC-42).
- [ ] **T-45** API-02, API-03: `/score`, `/score/batch` — *xong là khi:* chấm một mẫu và chấm lô cho cùng kết quả (TC-50).
- [ ] **T-46** API-04: `/score/upload` dùng `COPY` của PostgreSQL — *xong là khi:* 10.000 dòng chấm điểm và ghi xong dưới 30 giây (NFR-02, AC-A1, TC-54).
- [ ] **T-47** API-09…API-12: nhóm ngưỡng — *xong là khi:* `/threshold/preview` trả trong khoảng 10 ms và `/threshold/optimize` báo đúng `constraint_binding`.
- [ ] **T-48** API-05: `/explain` — *xong là khi:* trả đúng 5 yếu tố dương và 3 yếu tố âm, khớp giá trị SHAP tính trong notebook (AC-A4).
- [ ] **T-49** API-06…API-08: `/transactions`, `/reviews` với `ON CONFLICT DO UPDATE` — *xong là khi:* gửi thẩm định hai lần không tạo dòng thứ hai (TC-38) và `threshold_used` được ghi đúng (TC-39).
- [ ] **T-50** API-13…API-15: `/metrics`, `/samples`, `/replay/stream` (SSE) — *xong là khi:* phát lại chạy liên tục 3 phút không lỗi (AC-A6).

---

## Giai đoạn 8 — Giao diện (ngày 17–19)

Thứ tự dưới đây đồng thời là thứ tự **giữ lại** nếu thiếu thời gian.

- [ ] **T-51** Khung ứng dụng: thanh bên 4 mục, thanh trên luôn hiện ngưỡng và `model_version` — *xong là khi:* điều hướng được giữa các màn hình.
- [ ] **T-52** UI-01 hàng đợi thẩm định — *xong là khi:* sắp đúng theo điểm giảm dần (AC-A2), có trạng thái rỗng hướng dẫn 3 cách nạp dữ liệu.
- [ ] **T-53** UI-03 cấu hình ngưỡng, tính tại máy khách theo UI-D1 — *xong là khi:* kéo thanh trượt cập nhật dưới 200 ms (AC-A3) và đổi tham số chi phí làm dịch chuyển ngưỡng tối ưu (AC-A5).
- [ ] **T-54** Viết `tests/test_threshold_parity.py` cho TC-12 và bỏ `@pytest.mark.skip` — *xong là khi:* hàm JavaScript và `src/threshold.py` cho cùng TP/FP/FN trên 20 ngưỡng mẫu.
- [ ] **T-55** UI-02 chi tiết giao dịch + biểu đồ thác nước SHAP — *xong là khi:* chú thích PCA cố định hiện diện và nhãn thật chỉ hiện **sau** khi người dùng quyết định.
- [ ] **T-56** UI-05 chế độ phát lại qua SSE — *xong là khi:* điều khiển phát / tạm dừng / đặt lại và đồng hồ mô phỏng hoạt động.
- [ ] **T-57** UI-04 hiệu năng mô hình — *xong là khi:* có đủ 7 mục ở [docs/07 §6](docs/07-thiet-ke-giao-dien.md). **Được phép cắt đầu tiên**, thay tạm bằng ảnh PNG từ notebook.

---

## Giai đoạn 9 — Đóng gói và nghiệm thu (ngày 20)

- [ ] **T-58** `api/Dockerfile`, `web/Dockerfile`, compose 3 dịch vụ với `depends_on: service_healthy`, entrypoint chạy `alembic upgrade head` — *xong là khi:* một lệnh `docker compose up` dựng cả hệ thống.
- [ ] **T-59** Thử trên máy sạch hoặc máy ảo, bấm giờ cả lần đầu và lần sau — *xong là khi:* cold dưới 45 giây, warm dưới 15 giây (NFR-04, AC-A8).
- [ ] **T-60** Tạo bản `pg_dump` dữ liệu demo đẹp để dự phòng cho buổi bảo vệ — *xong là khi:* khôi phục thử được trong vài giây.
- [ ] **T-61** Duyệt toàn bộ AC-A1…AC-A10 và kiểm tra bàn phím (AC-A9) — *xong là khi:* mọi tiêu chí bắt buộc đạt hoặc được ghi rõ lý do không đạt.
- [ ] **T-62** Cập nhật README: cách chạy lại từ đầu trên máy sạch — *xong là khi:* người khác làm theo được mà không cần hỏi thêm (AC-D3).

---

## Giai đoạn 10 — Báo cáo và bảo vệ (ngày 21)

- [ ] **T-63** Viết `reports/bao-cao.md` theo dàn ý [docs/09 §5](docs/09-ke-hoach-trien-khai.md) — *xong là khi:* phần 6 (so sánh chiến lược) và phần 8 (ngưỡng và chi phí) là hai phần dài nhất.
- [ ] **T-64** Viết phần hạn chế — *xong là khi:* nêu đủ ba điều: V1–V28 là PCA nên không diễn giải được, dữ liệu chỉ hai ngày tháng 9/2013, và 492 mẫu dương làm khoảng tin cậy rất rộng (AC-D2).
- [ ] **T-65** Ghi nguồn dữ liệu và giấy phép DbCL v1.0 — *xong là khi:* có trích dẫn Kaggle mlg-ulb (AC-D5).
- [ ] **T-66** Chuẩn bị kịch bản demo 10 phút — *xong là khi:* đã diễn thử một lần trọn vẹn, gồm đoạn kéo thanh trượt ngưỡng từ 0,5 về τ\* để hội đồng thấy số vụ bắt được nhảy lên trong khi chi phí giảm.
- [ ] **T-67** Ôn ba câu hỏi chắc chắn bị hỏi — *xong là khi:* trả lời được trôi chảy không cần nhìn tài liệu:
  1. Vì sao PR-AUC chứ không phải ROC-AUC?
  2. SMOTE hoạt động thế nào và vì sao nó phải nằm trong pipeline?
  3. Ngưỡng này được chọn ra sao, sẽ đổi thế nào nếu chi phí thay đổi?

---

## Thứ tự cắt giảm khi thiếu thời gian

Cắt từ dưới lên. Không bao giờ cắt những việc in đậm.

| Ưu tiên | Hạng mục | Cắt được? |
|---|---|---|
| 1 | **T-28 rà soát rò rỉ** | Không — mất cái này thì toàn bộ kết quả vô giá trị |
| 2 | **T-20 bảng 20 tổ hợp** | Không — đây là trọng tâm đề tài |
| 3 | **T-29…T-31 ngưỡng và chi phí** | Không — đây là luận điểm chính |
| 4 | **T-38 kiểm tra tái lập** | Không — AC-M5 |
| 5 | **T-63…T-65 báo cáo** | Không — ngày 21 bất khả xâm phạm |
| 6 | T-53 UI-03 ngưỡng | Giữ bằng mọi giá nếu có làm ứng dụng |
| 7 | T-52 UI-01 hàng đợi | Giữ |
| 8 | T-55 UI-02 chi tiết | Cắt được, thay bằng ảnh notebook |
| 9 | T-56 UI-05 phát lại | Cắt được |
| 10 | T-57 UI-04 hiệu năng | Cắt trước nhất |
| 11 | T-34 autoencoder | Tùy chọn từ đầu |
| 12 | T-27 chia theo thời gian | Nên có, không bắt buộc |

## Việc lặp lại hằng ngày

- [ ] Chạy `pytest` trước mỗi lần commit.
- [ ] Cập nhật `[ ]` → `[x]` trong tệp này ngay khi đạt điều kiện *xong là khi*.
- [ ] Ghi lại mọi quyết định lệch khỏi đặc tả vào tài liệu tương ứng trong `docs/`, không để trong đầu.
