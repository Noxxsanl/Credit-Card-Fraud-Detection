# 11 — Review phần ML và kế hoạch sửa

**Thời điểm review:** 2026-10-10, trên commit `0c6e00e` ("validate, review, write report, update /web").
**Phạm vi:** toàn bộ `src/`, 9 notebook (cả output đã chạy), [`reports/bao-cao.md`](../reports/bao-cao.md),
[`SYSTEM_ARCHITECTURE.md`](../SYSTEM_ARCHITECTURE.md), `reports/*.csv`, `models/*` và `tests/`.
**Cách làm:** đọc mã và output, rồi kiểm chứng độc lập các nghi vấn bằng script riêng (không sửa repo
lúc review). Mọi số dòng trong liên kết dưới đây ứng với commit `0c6e00e`. "exec N" là
`execution_count` của ô code trong notebook.

Tiêu chí review: (1) dữ liệu và chia tập, (2) xử lý mất cân bằng, (3) metric, (4) chọn ngưỡng theo
chi phí, (5) mô hình và thí nghiệm, (6) tái lập và chất lượng code, (7) báo cáo. Mức độ vấn đề:
🔴 Critical — làm sai kết quả; 🟠 Major — ảnh hưởng độ tin cậy hoặc thiếu phần quan trọng; 🟡 Minor —
chất lượng code, trình bày.

Trạng thái sửa của từng vấn đề nằm ở [mục 6](#6-trạng-thái-triển-khai-kế-hoạch-sửa).

---

## 1. Tóm tắt

Đây là một đồ án có kỷ luật phương pháp tốt hơn hẳn mức thường gặp. Resampling nằm trong
`imblearn.Pipeline`, trùng lặp được loại trước khi chia tập, τ được chọn trên OOF, và mọi so sánh đều
theo quy tắc "KTC chứa 0 thì không tuyên bố". Điểm mạnh lớn nhất là sự trung thực về thống kê: quy tắc
chọn được đặt trước, kết quả âm tính được báo cáo đầy đủ, và dự án tự kiểm lại giả định chi phí tới mức
chỉ ra rằng lợi ích của τ\* đổi dấu. **Không tìm thấy lỗi Critical**: PR-AUC test tái tạo được đúng
0,8252, và không có bản ghi gian lận trùng gần đúng nào nằm ở cả train lẫn test. Vấn đề nghiêm trọng
nhất nằm ở khâu ra quyết định. τ\* = 0,0232 vẫn là ngưỡng mặc định, dù chính dự án đã cho thấy nó vượt
ngân sách thẩm định 44%, đắt hơn 0,5 khi tính chi phí theo `Amount`, và sinh khoảng 506 cảnh báo khi
gặp một ngày mới. Nếu tôn trọng ngân sách 200 cảnh báo/ngày thì recall test chỉ còn 0,726, tức AC-M2
không đạt.

## 2. Bảng chấm điểm

| # | Nhóm | Điểm | Nhận xét |
|---|---|---|---|
| 1 | Dữ liệu & chia tập | 8/10 | Kiểm toàn vẹn, loại trùng trước split, stratify và scaler trong pipeline đều đúng; bị trừ điểm vì đánh giá chính vẫn dùng random split dù chính dự án đã đo được drift giữa hai ngày. |
| 2 | Xử lý mất cân bằng | 7/10 | Lưới 5×4 có baseline `none` và resampling đặt đúng chỗ, nhưng các chiến lược được so ở các mức cân bằng khác nhau (1:1 so với 1:10), và việc chọn `class_weight` thay vì `none` không có số liệu ủng hộ. |
| 3 | Metric | 9/10 | PR-AUC là metric chính, có lý do và có bằng chứng để loại accuracy/ROC-AUC, kèm CV mean ± std, bootstrap CI và paired bootstrap; chỉ thiếu phần phương sai do huấn luyện (repeated CV). |
| 4 | Chọn ngưỡng theo chi phí | 7/10 | Phân tích rất tốt (chọn trên OOF, độ nhạy 5:1→100:1, regret, chi phí theo Amount, calibration), nhưng τ mặc định lại mâu thuẫn với chính phân tích đó, và UI cho phép chọn ngưỡng trên tập test. |
| 5 | Mô hình & thí nghiệm | 8/10 | Có baseline dummy/LR/DT, RandomizedSearchCV với scoring AP và quy tắc chọn đặt trước; tập test được mở ở nhiều notebook (dù không dùng để chọn), và tài liệu lại mô tả sai điều này. |
| 6 | Tái lập & code | 8/10 | Seed cố định, phiên bản thư viện được ghim, chạy lại cho kết quả bit-exact, 220 test ML pass; bị trừ điểm vì cache theo sự tồn tại của tệp, lưới/search không được chạy lại, leakage test chủ yếu là quét regex, và không có experiment tracking. |
| 7 | Báo cáo & trình bày | 8/10 | Hạn chế và kết quả âm tính được nêu đầy đủ; bị trừ điểm vì vài khẳng định sai so với chính số liệu ("nhanh nhất… mịn nhất", "tập kiểm thử chưa từng dùng"). |

## 3. Danh sách vấn đề

### 🔴 Critical: không phát hiện

Những gì đã kiểm:

- **Resampling:** chỉ chạy bên trong `ImbPipeline` ([`src/modeling.py:157-163`](../src/modeling.py#L157-L163));
  không có `fit_resample` nào bên ngoài pipeline.
- **Trùng lặp:** 1.081 dòng trùng hoàn toàn được loại trước khi chia. Kiểm thêm trùng **gần đúng** (cùng
  V1–V28 + Amount, chỉ khác `Time`): 2.218 dòng test có bản sao trong train, trong đó 806 dòng trùng hẳn
  cả vector 31 đặc trưng. **Không dòng nào là gian lận.** Bỏ các dòng này đi, PR-AUC chỉ đổi từ 0,8252
  thành 0,8254.
- **Tỷ lệ gian lận từng tập:** train 378/226.980 (0,1665%), test 95/56.746 (0,1674%), ngày 1 là 0,189%,
  ngày 2 là 0,144%.
- **Ngưỡng:** τ được chọn trên OOF. PR-AUC test tính lại từ `models/model.joblib` ra đúng 0,8252.
- **Quy tắc chọn tuned/default:** đã có ngay trong commit đầu tiên của notebook 05 (`ea57507`). Thứ tự
  thời gian trước commit đó: **không đủ thông tin để đánh giá**, nhưng không thấy dấu hiệu quyết định bị
  đảo lại.

### 🟠 Major

#### M1. Ngưỡng mặc định mâu thuẫn với chính phân tích chi phí; AC-M2 chỉ đạt khi vượt ngân sách thẩm định

- **Mô tả:** Báo cáo kết luận nên "chọn trong đáy chi phí theo năng lực thẩm định"
  ([`bao-cao.md:866-868`](../reports/bao-cao.md#L866-L868)). Thế nhưng hệ thống vẫn xuất τ\* tự do làm
  mặc định ([`bao-cao.md:638`](../reports/bao-cao.md#L638)).
- **Bằng chứng:**
  - τ\* sinh 287 cảnh báo/ngày, trong khi ngân sách là 200
    ([`bao-cao.md:556-558`](../reports/bao-cao.md#L556-L558)).
  - Tính lại trên OOF ngưỡng cực tiểu chi phí với ràng buộc ≤ 200 cảnh báo/ngày: τ = 0,9657. Tại ngưỡng
    này recall test chỉ **0,726** (69/95), dưới mức 0,75 của AC-M2.
  - Khi chi phí bỏ lọt là `Amount`, τ\* đắt hơn 0,5 trên test: +160 [+103; +217] EUR
    ([`bao-cao.md:620`](../reports/bao-cao.md#L620)).
  - [`src/threshold.py:236`](../src/threshold.py#L236) không tính phí thẩm định cho TP. Phần này tương
    đương dùng c_FN − c_FP, tỷ lệ 23,4:1 vẫn nằm trong dải ổn định, nên không làm đổi τ\*. Hàm này cũng
    không tách FP "cần thẩm định" khỏi FP "đề xuất chặn": khách hợp lệ bị chặn tốn nhiều hơn 5 EUR.
- **Tác động:** Cả "τ\* tối ưu chi phí" lẫn "AC-M2 đạt" đều chỉ đúng có điều kiện, dựa trên những giả
  định mà chính dự án đã chỉ ra là yếu. Người đọc sẽ đánh giá sai mức sẵn sàng triển khai.
- **Cách sửa:**
  1. Trong notebook 08, chọn τ mặc định bằng tối ưu có ràng buộc. Logic này đã có sẵn ở
     [`api/services/thresholding.py:43-52`](../api/services/thresholding.py#L43-L52):
     ```python
     t = cost_curve(y_train, oof, thresholds=np.unique(oof), sample_fraction=PHAN_TRAIN)
     ok = t[t["alerts_per_day"] <= NGAN_SACH_CANH_BAO]
     TAU_MAC_DINH = float(ok.loc[ok["expected_cost"].idxmin(), "threshold"])   # 0,9657
     ```
  2. Ghi AC-M2 kèm điều kiện: "đạt khi chấp nhận 287 cảnh báo/ngày; ở mức 200/ngày recall là 0,726".
  3. Nếu nghiệp vụ muốn tính chi phí theo số tiền, cần xác suất đã hiệu chỉnh và quyết định theo tổn thất
     kỳ vọng của từng giao dịch:
     ```python
     iso = IsotonicRegression(out_of_bounds="clip").fit(oof, y_train)   # chỉ fit trên OOF
     p = iso.predict(model.predict_proba(X)[:, 1])
     flag = p * (amount + C_FIXED) >= (1 - p) * C_REVIEW              # C_FIXED: phụ phí cho vụ thử thẻ (Amount ≈ 0)
     ```
     Thử nhanh với C_FIXED = 0 (chấm trên test nên chỉ để minh hoạ, không được dùng để chọn): chi phí theo
     Amount là 3.024 EUR so với 3.813 EUR ở τ = 0,5 (giảm 21%, KTC [−2.566; +417] chứa 0), nhưng recall
     chỉ còn 0,695. Đây là đánh đổi recall lấy tiền, và nghiệp vụ phải là bên quyết định.

#### M2. τ\* và mọi con số vận hành dựa trên random split; gặp một ngày mới thì tải cảnh báo gần gấp đôi

- **Bằng chứng:**
  - Dòng "ngày 1 → ngày 2" trong [`reports/split_comparison.csv`](../reports/split_comparison.csv):
    τ\* = 0,0233, chọn trên OOF của ngày 1. Kết quả là TP 168 + FP 338 = **506 cảnh báo trong đúng một
    ngày** (139.490 giao dịch). Con số ước tính từ random split là 287/ngày, còn ngân sách là 200.
  - Precision giảm từ 0,670 xuống 0,332, và hai KTC tách rời nhau.
  - PR-AUC OOF trong ngày 1 là 0,877, nhưng trên test ngày 2 chỉ còn 0,782.
  - Chênh lệch prevalence (0,167% so với 0,144%) chỉ giải thích −0,004 trong tổng −0,044 PR-AUC, nên phần
    còn lại đúng là do thời gian.
  - Báo cáo có nêu điều này ở mục hạn chế ([`bao-cao.md:838-839`](../reports/bao-cao.md#L838-L839))
    nhưng không đổi quyết định nào.
- **Tác động:** Khi lên production, đội thẩm định sẽ nhận khoảng 1,8 lần số cảnh báo dự kiến. Các con
  số 0,825 và 0,811 là cận trên lạc quan.
- **Cách sửa:** Chọn τ bằng validation forward-chaining theo `Time`. Báo cáo số cảnh báo/ngày theo cả hai
  cách chia, và theo dõi alerts/day cùng precision trên bảng `reviews` để chỉnh lại τ định kỳ.
  ```python
  order = np.argsort(X_train_tho["Time"].to_numpy(), kind="stable")
  oof_t = np.full(len(y_train), np.nan)
  for tr, va in TimeSeriesSplit(n_splits=4).split(order):     # fold validation luôn ở "tương lai"
      tr, va = order[tr], order[va]
      pipe = build_pipeline("xgboost", "class_weight", y=y_train[tr]).fit(X_train.iloc[tr], y_train[tr])
      oof_t[va] = pipe.predict_proba(X_train.iloc[va])[:, 1]
  ok = ~np.isnan(oof_t)   # chọn τ trên (y_train[ok], oof_t[ok]); quy đổi cảnh báo/ngày theo đúng cửa sổ thời gian
  ```

#### M3. Công cụ chọn ngưỡng trên UI đo trên tập test

- **Bằng chứng:**
  - `preview` đo trên `test_scores`
    ([`api/services/thresholding.py:21-23`](../api/services/thresholding.py#L21-L23)).
  - Thanh trượt UI-D1 trong trình duyệt cũng tính trên `test_scores`
    ([`frontend/src/lib/threshold.mjs:28`](../frontend/src/lib/threshold.mjs#L28)).
  - Dự án đã tự ghi nhận đây là rủi ro R5 ([`SYSTEM_ARCHITECTURE.md:903`](../SYSTEM_ARCHITECTURE.md#L903))
    nhưng chưa sửa.
- **Tác động:** Mọi τ mà người quản lý rủi ro chọn "bằng mắt" đều đã được khớp theo 95 mẫu dương của
  test. Số liệu test cho ngưỡng đó vì thế không còn khách quan, đúng loại lỗi mà quy tắc ML-08 của chính
  dự án cấm.
- **Cách sửa:** Cho `preview` và UI-D1 chạy trên `oof_scores.npz` (API đã nạp sẵn tệp này). Số liệu test
  chỉ hiện ở một panel báo cáo chỉ đọc, cho τ đã lưu. Khi lên production, thay tập test bằng cửa sổ nhãn
  review gần nhất.

#### M4. Chọn `class_weight` (scale_pos_weight = 599,5) thay vì `none` không có số liệu ủng hộ, và lý do ghi trong báo cáo là sai

- **Bằng chứng:**
  - Theo [`reports/grid_results.csv`](../reports/grid_results.csv), XGBoost + `none` đạt 0,8514 ± 0,030,
    còn `class_weight` đạt 0,8549 ± 0,030. Paired bootstrap cho KTC chứa 0 (nb04 §5.5).

    | XGBoost | `none` | `class_weight` |
    |---|---|---|
    | Thời gian 5 fold | 33,4 s | 36,1 s |
    | Precision tại τ\* | 0,866 | 0,702 |
    | Cảnh báo/ngày tại τ\* | 229 | 287 |
    | Chi phí OOF | 7.578 EUR | 7.529 EUR |
  - Trong RandomizedSearchCV, `scale_pos_weight = 1` có PR-AUC trung bình không kém: 0,846, so với 0,841
    ở mức 599,5 (nb05 exec 5).
  - Câu "vì nhanh nhất trong nhóm và cho điểm rủi ro mịn nhất"
    ([`bao-cao.md:17-18`](../reports/bao-cao.md#L17-L18)) sai ở cả hai vế. `none` chạy nhanh hơn, còn
    XGBoost + SMOTE có 222.236 mức điểm, nhiều hơn 221.674 mức của `class_weight`.
  - [`SYSTEM_ARCHITECTURE.md:74`](../SYSTEM_ARCHITECTURE.md#L74) ghi rằng lựa chọn đi "theo quy tắc đặt
    trước". Thực ra quy tắc đặt trước chỉ áp cho việc chọn giữa tuned và default.
  - Tính thêm trên OOF: `none` có calibration tốt hơn — Brier 0,000378 so với 0,000408, log-loss 0,00286
    so với 0,00301. Ở dải điểm ≥ 0,9, `none` có điểm trung bình 0,988 với tỷ lệ gian lận thật 0,979, còn
    `class_weight` là 0,997 với 0,936.
- **Tác động:** Kết luận trung tâm của đề tài bị trình bày lệch. Với XGBoost, xử lý mất cân bằng không
  cho lợi ích đo được, vậy mà hệ thống lại triển khai phiên bản có calibration kém hơn và sinh nhiều
  cảnh báo hơn.
- **Cách sửa:** Ghi trước một quy tắc tie-break, ví dụ: trong nhóm có KTC chứa 0, chọn cấu hình đơn giản
  nhất, rồi đến Brier thấp nhất trên OOF, rồi đến ít cảnh báo/ngày nhất. Thêm cột Brier vào bảng lưới:
  ```python
  grid["brier_oof"] = [brier_score_loss(y_train, oof[f"{m}__{s}"]) for m, s in zip(grid.model, grid.strategy)]
  ```

#### M5. So sánh chiến lược bị lẫn biến: mức cân bằng khác nhau và chỉ có một điểm cấu hình

- **Bằng chứng:**
  - Undersample và SMOTE đưa dữ liệu về 1:10 (`SAMPLING_STRATEGY = 0.1`,
    [`src/modeling.py:43`](../src/modeling.py#L43)).
  - `class_weight` lại cân bằng hoàn toàn: `"balanced"` cho LR/DT/RF
    ([`src/modeling.py:55-71`](../src/modeling.py#L55-L71)) và `scale_pos_weight` ≈ 599,5 cho XGBoost
    ([`src/modeling.py:87-88`](../src/modeling.py#L87-L88)).
  - Không có lần chạy nào ở tỷ lệ khác, và hyperparameter giữ nguyên cho mọi chiến lược.
- **Tác động:** Với thiết kế hiện tại, không tách được *loại* kỹ thuật khỏi *mức độ* cân bằng. Ví dụ
  kết luận "class_weight làm Decision Tree sụp từ 0,695 xuống 0,359"
  ([`bao-cao.md:341-343`](../reports/bao-cao.md#L341-L343)) có thể do trọng số gấp 600 lần chứ không do
  bản thân phương pháp trọng số lớp.
- **Cách sửa:** Chạy mọi chiến lược trên cùng một lưới tỷ lệ mục tiêu r, trong CV:
  ```python
  ir = imbalance_ratio(y_train)                                   # N_âm / N_dương ≈ 599,5
  for r in (0.02, 0.1, 0.3, 1.0):                                 # tỷ lệ dương:âm sau cân bằng
      cw = {"scale_pos_weight": r * ir} if model == "xgboost" else {"class_weight": {0: 1, 1: r * ir}}
      p_cw = build_pipeline(model, "none", **cw)
      # SMOTE/undersample: thêm tham số ratio=r cho _make_resampler thay cho hằng SAMPLING_STRATEGY
  ```

### 🟡 Minor

#### m1. Tập test được dùng nhiều lần, nhưng tài liệu khẳng định ngược lại

- **Bằng chứng:**
  - Notebook 03 chấm LR/DT trên test ([`bao-cao.md:228`](../reports/bao-cao.md#L228)). Notebook 05 chấm
    cả bản tuned lẫn default. Notebook 06, 06b và 07 cũng dùng test.
  - Trong khi đó báo cáo viết "tập kiểm thử chưa từng dùng" ([`bao-cao.md:18`](../reports/bao-cao.md#L18)),
    và tài liệu kiến trúc viết "tập test chỉ được mở ở notebook 05"
    ([`SYSTEM_ARCHITECTURE.md:414`](../SYSTEM_ARCHITECTURE.md#L414)).
  - Bảng ở §8.6 ([`bao-cao.md:648-653`](../reports/bao-cao.md#L648-L653)) chỉ dùng các dòng test để biện
    minh cho việc đổi luật chặn. Bằng chứng OOF thì đã có sẵn trong
    [`reports/block_threshold.csv`](../reports/block_threshold.csv): luật 3τ\* trên OOF có precision 0,814
    và chặn nhầm 73 khách hợp lệ.
- **Tác động:** Số liệu không sai, vì không lựa chọn nào dựa trên test. Nhưng đây là khẳng định sai
  trong tài liệu đem đi chấm.
- **Cách sửa:** Đổi thành "không dùng để **chọn**" và liệt kê các lần mở test. Ở §8.6, đưa các dòng OOF
  lên làm căn cứ, còn test chỉ để xác nhận.

#### m2. Độ bất định khi so mô hình bị ước lượng thấp

- **Bằng chứng:**
  - Paired bootstrap trên OOF gộp (nb04 exec 14) chỉ lấy mẫu lại các dòng. Nó bỏ qua phương sai do huấn
    luyện, và CV chỉ chạy một lần, dù độ lệch chuẩn giữa các fold khoảng 0,03.
  - Hai khẳng định "khác biệt thật" dựa vào phép này: RF + SMOTE hơn RF + `class_weight` +0,0146, và
    XGBoost + `class_weight` hơn RF + `class_weight` +0,0152.
  - nb04 còn lấy `h.mean()` làm ước lượng điểm thay vì hiệu quan sát.
- **Tác động:** Hai khẳng định "khác biệt thật" có thể không còn đứng khi tính thêm phương sai huấn luyện.
- **Cách sửa:** Dùng `RepeatedStratifiedKFold(5, n_repeats=5)` cùng corrected resampled t-test
  (Nadeau & Bengio) cho các cặp cần kết luận.

#### m3. Vệ sinh giả định chi phí và EDA

- **Bằng chứng:**
  - `DEFAULT_COST_FN = 122.21` ([`src/config.py:35`](../src/config.py#L35)) là trung bình trên dữ liệu
    thô, có cả phần test. Dự án đã kiểm rằng dùng 115,94 (chỉ train) cho cùng τ\*.
  - Quyết định thêm `hour_sin`/`hour_cos` dựa trên tỷ lệ gian lận theo giờ tính trên toàn bộ dữ liệu
    (nb01 exec 9).
  - Trùng gần đúng theo đặc trưng chưa từng được dự án kiểm (đã kiểm khi review: không có ảnh hưởng, xem
    phần Critical).
- **Tác động:** Hiện không đổi kết quả nào, nhưng là các điểm rò thông tin từ test mà giám khảo có thể
  bắt lỗi.
- **Cách sửa:** Dùng con số chỉ tính trên train cho `DEFAULT_COST_FN`; làm phần EDA phục vụ quyết định
  đặc trưng trên train; thêm vào nb05 §6 một phép kiểm trùng theo `[*V_COLUMNS, "Amount"]` giữa hai tập.

#### m4. Leakage test chủ yếu là quét regex nên dễ lọt

- **Bằng chứng:**
  - [`tests/test_no_leakage.py:285`](../tests/test_no_leakage.py#L285) chỉ bắt được khi tham số đầu tiên
    có chữ "test". Một lệnh như `pick_threshold(y_n2, ...)`, dùng nhãn test ngày 2 ở nb05, sẽ lọt qua.
  - Test scaler ([`tests/test_no_leakage.py:108-120`](../tests/test_no_leakage.py#L108-L120)) chỉ chứng
    minh rằng fit trên 400 dòng khác fit trên 600 dòng.
- **Tác động:** Bộ test có thể vẫn xanh khi có rò rỉ thật, ví dụ chọn τ trên nhãn test mang tên khác.
- **Cách sửa:** Thêm canary test theo hành vi:
  ```python
  def test_canary_shuffled_labels(dataset):
      X, y = dataset
      y_perm = y.sample(frac=1, random_state=42).reset_index(drop=True)
      s = oof_scores(build_pipeline("xgboost", "smote", y=y_perm), X, y_perm)
      assert average_precision_score(y_perm, s) < 3 * y_perm.mean()   # SMOTE rò rỉ sẽ làm số này vọt lên
  ```

#### m5. Tái lập: cache theo sự tồn tại của tệp, thiếu tracking

- **Bằng chứng:**
  - nb04 đọc lại lưới từ điểm lưu thay vì chạy lại ("đã có 20/20 tổ hợp … 0.0 phút", exec 4).
  - nb05 bỏ qua bước search nếu CSV đã tồn tại (`CHAY_LAI_TIM_KIEM = not ….exists()`).
  - [`scripts/check_reproducibility.py:13-15`](../scripts/check_reproducibility.py#L13-L15) không chạy
    lại hai bước này, và nb08 chỉ đối chiếu 1/20 tổ hợp.
  - `metrics.json` không ghi git SHA.
  - Các hằng `RECALL_TOI_THIEU` và `NGAN_SACH_CANH_BAO` bị khai báo lặp ở nb06 và nb08.
- **Tác động:** Sửa `src/modeling.py` mà không xoá điểm lưu thì vẫn có thể ra kết quả cũ mà không ai biết.
- **Cách sửa:** Ghi git SHA và hash của `src/modeling.py` vào điểm lưu; khi hash đổi thì bỏ cache. Chuyển
  các hằng số trên vào `src/config.py`. Log params, metrics và SHA của mỗi lần chạy.

## 4. Điểm làm tốt

- **Chống rò rỉ bằng cấu trúc:** `build_pipeline` luôn trả về `ImbPipeline`
  ([`src/modeling.py:122-163`](../src/modeling.py#L122-L163)), scaler nằm trong pipeline, và việc loại
  trùng được ghi số liệu rõ ràng (1.081 dòng, 19 vụ gian lận).
- **Quy tắc thống kê áp dụng nhất quán:** "KTC của hiệu chứa 0 thì không tuyên bố" được dùng từ LR so với
  DT (nb03) tới tuned so với default (nb05).
- **Kết quả âm tính kèm cơ chế:** tuning không hơn cấu hình mặc định; SMOTE-Tomek xoá 0 cặp, và dự án giải
  thích được vì sao SMOTE đã xoá các cặp Tomek trước đó.
- **Tự phản biện giả định chi phí:** dự án chỉ ra kết luận đổi dấu khi tính theo Amount
  ([`bao-cao.md:603-638`](../reports/bao-cao.md#L603-L638)), và phát hiện lưới 200 điểm làm hỏng tiêu chí
  ngân sách (172,5 so với 199,4 cảnh báo/ngày).
- **Temporal split có nhóm đối chứng** (train thu nhỏ bằng cỡ ngày 1), nên tách được ảnh hưởng của lượng
  dữ liệu khỏi ảnh hưởng của thời gian.
- **Khẳng định ở EDA kiểm chứng được và đúng:** cả 21 vụ gian lận "không có láng giềng gian lận" đều là FN
  tại τ\* (đã kiểm: 21/21).
- **Tái lập và train/serve:** 9 notebook chạy lại cho kết quả bit-exact
  ([`reports/reproducibility.csv`](../reports/reproducibility.csv)); `build_features` dùng chung cho cả
  train lẫn serve; SHAP được kiểm tính cộng với sai số 2,2·10⁻⁵.

## 5. Kế hoạch sửa (theo thứ tự ưu tiên)

1. **Quyết định lại τ mặc định (M1):** dùng tối ưu chi phí có ràng buộc ngân sách trên OOF khi xuất
   `threshold.json`; ghi AC-M2 kèm điều kiện ngân sách; nhờ nghiệp vụ xác nhận dạng hàm chi phí (cố định,
   theo Amount, hay Amount cộng phí cố định).
2. **Validation theo thời gian cho ngưỡng và số vận hành (M2):** chọn τ bằng forward-chaining theo `Time`;
   báo cáo cảnh báo/ngày theo cả hai cách chia; theo dõi drift trên bảng `reviews`.
3. **Chuyển preview và UI-D1 sang điểm OOF (M3):** sửa ít, nhưng gỡ được việc chọn ngưỡng trên test.
4. **Chạy lại so sánh chiến lược và chọn lại mô hình (M5, M4, m2):** chạy mọi chiến lược ở cùng các mức
   cân bằng r; chọn lại giữa `none` và `class_weight` bằng quy tắc tie-break có Brier; dùng repeated CV
   cho các cặp cần kết luận.
5. **Sửa tài liệu và vệ sinh (m1, m3, m4, m5):** bỏ các câu "chưa từng dùng", "chỉ mở ở notebook 05",
   "nhanh nhất… mịn nhất"; đưa các dòng OOF vào §8.6; đổi c_FN sang giá trị chỉ tính trên train; thêm
   canary test; thêm git SHA và bỏ cache khi code đổi.

### Phụ lục — các phép kiểm chứng độc lập lúc review

| Kiểm chứng | Kết quả |
|---|---|
| PR-AUC test chấm lại từ `models/model.joblib` + `data/test_set.parquet` | 0,8252 — khớp báo cáo |
| Trùng gần đúng (V1–V28 + Amount) giữa train và test | 2.218 dòng test, 806 trùng cả 31 đặc trưng, 0 gian lận; bỏ đi thì PR-AUC 0,8254 |
| Phần chênh PR-AUC random → temporal do prevalence (trọng số lại lớp âm về 0,144%) | −0,004 trong −0,044 |
| 21 gian lận "cô lập" của EDA có phải FN tại τ\* không (OOF cho train, test cho test) | 21/21 là FN |
| Calibration OOF của XGBoost `none` so với `class_weight` | Brier 0,000378 so với 0,000408; log-loss 0,00286 so với 0,00301 |
| Ngưỡng cực tiểu chi phí có ràng buộc ≤ 200 cảnh báo/ngày trên OOF | τ = 0,9657 → test: TP 69, FP 1, recall 0,726, 175 cảnh báo/ngày |
| Bộ test ML (9 tệp) | 220 passed |

## 6. Trạng thái triển khai kế hoạch sửa

Triển khai ngày 2026-10-10 → 2026-10-11 trên cây làm việc, **chưa commit**. Số liệu dưới đây là của lần chạy
lại toàn bộ:

- lưới 20 tổ hợp, giờ có dấu vân tay cấu hình — cho đúng từng bit các điểm out-of-fold cũ;
- lưới theo mức cân bằng (48 tổ hợp, 86 phút);
- CV lặp (81 phút);
- tìm kiếm siêu tham số (13 phút);
- notebook 01 và 04–08, mỗi notebook một kernel mới.

Kiểm tra tái lập: chạy lại 9 notebook 01 → 08 trong kernel sạch (46,5 phút, 2026-10-11): PR-AUC 0,8278893033 → 0,8278893033, lệch 0; 13 con số chính và ngưỡng mặc định lệch 0; dấu vân tay điểm test và điểm chọn ngưỡng trùng; 20/20 bảng `reports/*.csv` trùng hoàn toàn — AC-M5 đạt.

### 6.1 Từng vấn đề

| Mục | Đã làm | Kết quả | Tệp chính |
|---|---|---|---|
| **M1** — τ mặc định | Ngưỡng mặc định = chi phí nhỏ nhất trong các ngưỡng có ≤ 200 cảnh báo/ngày (`selection_method: min_cost_within_budget`). AC-M2 không còn là `assert`: kết quả ghi vào `metrics.json → acceptance`, kèm số tại τ\* tự do và năng lực cần cho recall 0,75 | τ = **0,2675**. Trên test: recall 0,758 [0,674; 0,842] = 72/95, precision 0,960, 187 cảnh báo/ngày. AC-M2 đạt theo ước lượng điểm, sát nút. Cần ~185 cảnh báo/ngày cho recall 0,75 | `src/threshold.py` (`constrained_min_cost`, `policy_threshold`, `capacity_for_recall`), `src/config.py`, notebook 06 §7, notebook 08 |
| **M2** — chọn ngưỡng theo thời gian | Điểm chọn ngưỡng sinh bằng forward-chaining theo `Time` (`TimeSeriesSplit(4)` trên tập huấn luyện đã sắp; khối đầu không có điểm); cảnh báo/ngày quy theo đúng cửa sổ thời gian. Có đối chứng giao thức ngày 1 → ngày 2. Thêm `GET /monitoring` (API-16) và thẻ "Theo dõi vận hành" ở UI-04: so tỷ lệ cảnh báo, precision thẩm định, recall theo nhãn với kỳ vọng, kèm KTC Wilson | 181.584 điểm (249 gian lận), 1,45 ngày, PR-AUC 0,783. Với τ\*, ước tính tải lệch ×1,46 so với ngày 2 thật (CV ngẫu nhiên: ×1,65). Ngưỡng trong ngân sách giữ được tải, nhưng recall ngày 2 chỉ ~0,70 | `src/modeling.py` (`temporal_oof_scores`, `selection_scores`), `api/services/monitoring.py`, `api/routes/monitoring.py`, `frontend/src/components/MonitoringCard.tsx`, `web/` |
| **M3** — UI chọn ngưỡng trên test | `preview` (API-11), thanh trượt UI-D1 và `optimize` (API-12) cùng tính trên điểm chọn ngưỡng. Thêm `GET /threshold/oof-scores` (API-17) gửi điểm đã sắp dạng nhị phân, float32. Số test chỉ còn ở một dòng chỉ đọc cho ngưỡng đang áp dụng | R5 đóng. Payload 0,7 MB. TC-12 đối chiếu cả đường nhị phân với Python, trùng từng bit | `api/services/thresholding.py`, `api/routes/threshold.py`, `src/artifacts.py` (`sorted_scores_payload`), `web/threshold.js`, `frontend/src/lib/threshold.mjs` |
| **M4, M5, m2** — so chiến lược, chọn mô hình | Lưới theo mức r ∈ {0,02; 0,1; 0,3; 1} cho `class_weight`/SMOTE/undersample (48 tổ hợp). Thêm cột Brier ở mọi lưới. CV lặp 5 × 5 (RF 3 × 5) với kiểm định t hiệu chỉnh Nadeau–Bengio. Quy tắc tie-break đặt trước (`select_configuration`) ghi kết quả vào `src/config.py` (`FINAL_*`); notebook 05–08 dựng mô hình qua `final_pipeline()` | Cả 9 ứng viên XGBoost hoà với cấu hình tốt nhất → **`xgboost + none`** (`xgb_none_v2`). "Class weight làm Decision Tree sụp" là do mức cân bằng: ở r = 0,1, hai kỹ thuật cho 0,642 và 0,640. Hai khẳng định "khác biệt thật" cũ có KTC chứa 0 | `scripts/run_ratio_grid.py`, `scripts/run_repeated_cv.py`, `src/modeling.py`, `src/evaluate.py`, notebook 04 §5.6–5.8 |
| **m1** — tập test | Bỏ câu "chưa từng dùng" / "chỉ mở ở notebook 05"; liệt kê mọi lần mở test ("chỉ để đo, không để chọn"). Ở §8.6, đưa các dòng trên điểm chọn ngưỡng lên làm căn cứ | — | `reports/bao-cao.md` §7.2, §8.6; `SYSTEM_ARCHITECTURE.md` §7.4 |
| **m3** — rò thông tin từ test | `DEFAULT_COST_FN = 115.94` (chỉ train) kèm migration `0003`. Phát hiện "gian lận theo giờ" kiểm lại chỉ trên train. Notebook 05 §6.1 thêm kiểm trùng gần đúng | Cả τ\* lẫn ngưỡng mặc định không đổi theo nguồn c_FN. 2h: 1,365% so với 10h: 0,045%, vẫn chênh 30 lần. 2.218 dòng test có bản sao, 0 gian lận | `src/config.py`, `api/migrations/versions/0003_cost_fn_train_only.py`, notebook 01 §4.1, `src/data.py` (`near_duplicate_mask`) |
| **m4** — leakage test | Thêm kiểm thử canary: xáo nhãn thì PR-AUC ở mức may rủi, với cả 4 chiến lược. Có đối chứng dương: SMOTE trước CV phải bị bắt. Regex `test_leak_5` nhận thêm `policy_threshold`, `capacity_for_recall` | Xanh | `tests/test_no_leakage.py` |
| **m5** — tái lập | Điểm lưu chỉ dùng lại khi `config_hash` khớp (tham số pipeline, CV, dữ liệu, phiên bản thư viện), áp cho lưới, lưới theo mức, CV lặp, tìm kiếm. `metrics.json → environment.git` ghi commit và cờ có thay đổi chưa commit. Hằng số ngưỡng chuyển vào `src/config.py` | Lưới chạy lại khớp từng bit với điểm lưu cũ | `src/modeling.py`, `src/artifacts.py`, `scripts/*.py` |

Kiểm thử: **406 xanh** (93 ca cần PostgreSQL, chạy với `docker compose up -d db`); `pytest -m perf` cho p95 phía
máy chủ 16,8 ms. Bộ test có thêm `tests/test_experiments.py` và nhiều ca mới ở các tệp khác ([08 §2.10](08-ke-hoach-kiem-thu.md)).
Frontend: `tsc --noEmit` và ESLint sạch.

### 6.2 Con số nào thay đổi

| | Lúc review (`xgb_scaleposweight_v1`) | Sau sửa (`xgb_none_v2`) |
|---|---|---|
| Cấu hình | XGBoost + `scale_pos_weight` 599,5 | XGBoost, không xử lý mất cân bằng |
| Ngưỡng mặc định | τ\* = 0,0232 (cực tiểu chi phí, OOF ngẫu nhiên) | 0,2675 (chi phí nhỏ nhất trong 200 cảnh báo/ngày, forward-chaining) |
| PR-AUC test | 0,825 [0,747; 0,896] | 0,828 [0,747; 0,900] |
| Recall test tại ngưỡng mặc định | 0,811 (77/95) | 0,758 (72/95) |
| Precision test tại ngưỡng mặc định | 0,670 | 0,960 |
| Cảnh báo/ngày (test) tại ngưỡng mặc định | 287 — vượt ngân sách 44% | 187 — trong ngân sách |
| AC-M2 | "đạt", nhưng chỉ khi vượt ngân sách | đạt theo ước lượng điểm, sát nút, trong ngân sách; ngày mới ~0,70 |
| c_FN | 122,21 (dữ liệu thô) | 115,94 (chỉ tập huấn luyện) |
| τ_chặn (precision ≥ 95%) | 0,9735 | 0,9995 — chặn tự động 7 gian lận, 0 hợp lệ trên test |
| Hai khẳng định "khác biệt thật" về Random Forest | +0,0146 và +0,0152, KTC không chứa 0 | +0,0106 và +0,0143 theo CV lặp, KTC chứa 0 |
| Chi phí theo số tiền, τ\* − 0,5 trên test | +160 [+103; +217] EUR — τ\* đắt hơn | −169 [−673; +105] EUR — chứa 0 |
| PR-AUC theo thời gian (ngày 1 → ngày 2) | 0,782 | 0,773 |

### 6.3 Chưa làm, hoặc cần quyết định ngoài phạm vi kỹ thuật

1. **Nghiệp vụ xác nhận hàm chi phí và ngân sách** (M1). Phải chọn chi phí bỏ lọt là hằng số, bằng số tiền,
   hay số tiền cộng phí cố định cho vụ thử thẻ; và năng lực thẩm định thật mỗi ngày. Ngưỡng mặc định và việc
   AC-M2 đạt đều phụ thuộc hai điều này. Báo cáo ghi rõ chúng là giả định.
2. **Quyết định theo tổn thất kỳ vọng từng giao dịch** (M1.3) cần xác suất đã hiệu chỉnh — chưa làm. Điểm của
   mô hình mới gần xác suất hơn (Brier 0,000439 trên điểm chọn ngưỡng), nhưng vẫn lệch ở hai đầu.
3. **Experiment tracking kiểu MLflow** (m5) — không làm. Thay bằng `config_hash` trong mọi bảng kết quả, git SHA
   trong `metrics.json` và các bảng `reports/*.csv`.
4. ~~Ảnh giao diện trong báo cáo~~ — đã chụp lại 2026-10-11 bằng `scripts/ui/flow.js` (21/21 bước đạt) trên hệ
   thống đóng gói với `xgb_none_v2`; kịch bản bỏ các số ghi cứng của mô hình cũ.
5. ~~AC-A3 chưa đo lại~~ — đo lại: lâu nhất 44–68 ms trên điểm chọn ngưỡng (yêu cầu < 200 ms).
6. **Quy tắc tie-break được đề xuất sau khi đã thấy bảng 20 tổ hợp**, dù trước khi chạy CV lặp và lưới theo mức.
   Đây là hạn chế đã ghi trong báo cáo §11.2.
7. **Đánh giá out-of-time làm mặc định** cần dữ liệu dài hơn hai ngày — ngoài phạm vi bộ dữ liệu.
