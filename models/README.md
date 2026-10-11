# models/ — hiện vật mô hình

Mọi tệp ở đây do `notebooks/08_export_artifacts.ipynb` sinh ra (T-35, T-36). Không notebook nào
khác ghi vào thư mục này, và API chỉ đọc (AR-01). Cấu trúc chi tiết:
[docs/06 §6](../docs/06-thiet-ke-luu-tru.md).

| Tệp | Nội dung | Trong git? |
|---|---|---|
| `model.joblib` | pipeline hoàn chỉnh: `RobustScaler` cho `Amount` → `XGBClassifier`, cấu hình theo `FINAL_MODEL`, `FINAL_STRATEGY`, `FINAL_RATIO` của `src/config.py`. Đầu vào là `build_features(df)` | Không — sinh lại từ notebook 08 |
| `explainer.joblib` | `shap.TreeExplainer` của bước `clf`. Đầu vào là `model[:-1].transform(build_features(df))`. Notebook và kiểm thử dùng; API **không** nạp (SHAP bằng `pred_contribs` của XGBoost, trùng từng bit) | Không |
| `threshold.json` | ngưỡng mặc định — chi phí nhỏ nhất trong ngân sách 200 cảnh báo/ngày (`selection_method: min_cost_within_budget`) — và các phương án khác, tất cả chọn trên điểm chọn ngưỡng (`selected_on: temporal_out_of_fold`); tham số chi phí, ràng buộc | Nên có (dưới 1 KB) |
| `metrics.json` | chỉ số kèm khoảng tin cậy, đường PR/ROC/chi phí, điểm tập kiểm thử, bảng 20 tổ hợp, SHAP toàn cục, kết quả nghiệm thu (`acceptance`), cách sinh điểm chọn ngưỡng (`selection`), commit git | Có cân nhắc (1,6 MB) |
| `oof_scores.npz` | **điểm chọn ngưỡng**: điểm out-of-fold forward-chaining theo `Time` của tập huấn luyện, kèm nhãn, `method`, `train_fraction`, `days`. API chọn ngưỡng (API-12), xem trước (API-11) và gửi cho thanh trượt UI-D1 (API-17) trên bộ điểm này — không trên tập kiểm thử (ML-08, [docs/11](../docs/11-review-ml-va-ke-hoach-sua.md) M2, M3) | Không |

Sinh lại:

```powershell
.\.venv\Scripts\python.exe -m jupyter nbconvert --to notebook --execute --inplace `
    --ExecutePreprocessor.timeout=3600 notebooks\08_export_artifacts.ipynb
.\.venv\Scripts\python.exe -m pytest tests\test_artifacts.py
```

Hai tệp `.joblib` là pickle, chỉ nạp được với đúng phiên bản thư viện ghi trong
`metrics.json → environment.packages`. Đổi phiên bản thì chạy lại notebook 08, và sửa
`api/requirements.txt` cho khớp (`tests/test_packaging.py` báo đỏ nếu quên).

Trong Docker Compose, thư mục này được gắn **chỉ đọc** vào container `api`. Xuất lại xong thì
`docker compose restart api` để nạp, không phải build lại ảnh. Chép hiện vật sang máy khác:
gói `hien-vat.tgz` ở [README](../README.md#chạy-lại-từ-đầu-trên-máy-sạch), bước 3.
