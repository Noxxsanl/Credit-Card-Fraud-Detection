# Báo cáo: Phát hiện gian lận thẻ tín dụng

> Bản nháp — điền số liệu sau khi chạy xong các notebook.

## 1. Giới thiệu bài toán

- Bối cảnh, thiệt hại do gian lận thẻ.
- Mục tiêu: tối đa hoá số gian lận bắt được với số báo động nhầm chấp nhận được.

## 2. Dữ liệu

- Nguồn: Kaggle — Credit Card Fraud Detection (ULB).
- 284.807 giao dịch, 492 gian lận (0,172%).
- Đặc trưng: `Time`, `Amount`, và `V1`–`V28` đã qua PCA (ẩn danh hoá).

## 3. EDA (notebook 01)

- Phân phối nhãn: _(hình `reports/figures/class_balance.png`)_
- Phân phối `Amount` theo nhãn:
- Mẫu theo `Time`:

## 4. Phân tích thống kê (notebook 02)

- Kiểm định khác biệt giữa hai nhóm, các biến `V*` phân tách mạnh nhất:

## 5. Mô hình baseline (notebook 03)

| Model | PR-AUC | ROC-AUC | Precision | Recall |
|---|---|---|---|---|
| Logistic Regression | | | | |
| Decision Tree | | | | |

## 6. Xử lý mất cân bằng (notebook 04)

| Chiến lược | PR-AUC | Precision | Recall |
|---|---|---|---|
| Không xử lý | | | |
| class_weight='balanced' | | | |
| Random undersampling | | | |
| SMOTE | | | |

Nhận xét: _(SMOTE thường tăng recall nhưng giảm precision — cần cân nhắc theo chi phí)_

## 7. Mô hình nâng cao (notebook 05)

| Model | PR-AUC | ROC-AUC | Precision | Recall |
|---|---|---|---|---|
| Random Forest | | | | |
| XGBoost | | | | |
| LightGBM | | | | |

Model được chọn: _____ → lưu tại `models/best_model.pkl`.

## 8. Ngưỡng quyết định và chi phí (notebook 06)

- Giả định chi phí: FN = ___ , FP = ___
- Ngưỡng tối ưu: ___ (so với mặc định 0,5)
- Tiết kiệm ước tính:

## 9. Giải thích mô hình (notebook 07)

- Đặc trưng quan trọng nhất theo SHAP:
- Phân tích các ca dự đoán sai:

## 10. Kết luận và hướng phát triển

- Kết quả chính:
- Hạn chế: dữ liệu đã PCA nên khó diễn giải nghiệp vụ; không có thông tin thời gian thực.
- Hướng tiếp: cập nhật mô hình theo thời gian (concept drift), đặc trưng theo hành vi người dùng.
