# models/ — hiện vật mô hình

Mọi tệp ở đây do `notebooks/08_export_artifacts.ipynb` sinh ra (T-35, T-36). Không notebook nào
khác ghi vào thư mục này, và API chỉ đọc (AR-01). Cấu trúc chi tiết:
[docs/06 §6](../docs/06-thiet-ke-luu-tru.md).

| Tệp | Nội dung | Trong git? |
|---|---|---|
| `model.joblib` | pipeline hoàn chỉnh: `RobustScaler` cho `Amount` → `XGBClassifier`. Đầu vào là `build_features(df)` | Không — sinh lại từ notebook 08 |
| `explainer.joblib` | `shap.TreeExplainer` của bước `clf`. Đầu vào là `model[:-1].transform(build_features(df))` | Không |
| `threshold.json` | ngưỡng mặc định τ\* và 4 phương án, chọn trên out-of-fold; tham số chi phí | Nên có (dưới 1 KB) |
| `metrics.json` | chỉ số kèm khoảng tin cậy, đường PR/ROC/chi phí, điểm tập kiểm thử, bảng 20 tổ hợp, SHAP toàn cục | Có cân nhắc (1,6 MB) |
| `oof_scores.npz` | 226.980 điểm out-of-fold và nhãn của tập huấn luyện — API chọn ngưỡng trên đây (ML-08) | Không |

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
