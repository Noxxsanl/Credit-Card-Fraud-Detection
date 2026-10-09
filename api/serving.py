"""Đường chấm điểm lúc phục vụ — cùng phép tính với ``model.predict_proba``, bỏ chi phí pandas.

Pipeline xuất ra là ``ColumnTransformer(RobustScaler cho Amount, phần còn lại giữ nguyên) →
XGBClassifier``. Gọi nó cho **một** giao dịch tốn khoảng 20 ms, gần hết là chi phí dựng
DataFrame trong ``ColumnTransformer`` và trong lớp sklearn của XGBoost; phép tính thật chỉ
khoảng 1 ms. ``FastScorer`` làm đúng phép tính đó trên mảng numpy:

- ``Amount`` chuẩn hoá bằng ``(x - center_) / scale_`` — đúng hai phép mà ``RobustScaler.transform``
  thực hiện, cùng thứ tự, trên float64;
- cột xếp theo ``get_feature_names_out()`` của ``ColumnTransformer``;
- ``Booster.inplace_predict`` thay cho ``XGBClassifier.predict_proba``.

Đặc trưng vẫn chỉ sinh bằng ``src.features.build_features`` (ranh giới chống lệch train/serve
không đổi). Đường tắt này chỉ thay bước tiền xử lý + dự đoán, và **phải trùng từng bit** với
pipeline: ``api/loader.py`` đối chiếu lúc khởi động và từ chối nạp nếu lệch.

SHAP lấy bằng ``pred_contribs=True`` của chính XGBoost — cùng thuật toán TreeSHAP (đường dẫn
cây) mà ``shap.TreeExplainer(clf)`` dùng khi không có dữ liệu nền, trùng từng bit với
``explainer.joblib`` (``tests/test_api.py``). Nhờ vậy ảnh ``api`` không cần thư viện shap.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.preprocessing import FunctionTransformer, RobustScaler

from src.features import AMOUNT_COLUMN, FEATURE_ORDER


class UnsupportedPipeline(ValueError):
    """Pipeline không đúng dạng mà đường tắt hiểu được — dùng lại pipeline đầy đủ."""


def _is_passthrough(transformer) -> bool:
    """``remainder="passthrough"`` sau khi fit: chuỗi "passthrough", hoặc (scikit-learn ≥ 1.5)
    một ``FunctionTransformer`` đồng nhất — không hàm, không hàm ngược."""
    if transformer == "passthrough":
        return True
    return (isinstance(transformer, FunctionTransformer) and transformer.func is None
            and transformer.inverse_func is None)


class FastScorer:
    def __init__(self, model):
        names = [name for name, _ in model.steps]
        if names != ["preprocess", "clf"]:
            raise UnsupportedPipeline(f"các bước {names}, cần ['preprocess', 'clf']")
        preprocess = model.named_steps["preprocess"]
        fitted = [(name, transformer, list(columns)) for name, transformer, columns in preprocess.transformers_
                  if not (name == "remainder" and transformer == "drop")]
        amount = [t for t in fitted if t[0] == "amount"]
        others = [t for t in fitted if t[0] != "amount"]
        if (len(amount) != 1 or not isinstance(amount[0][1], RobustScaler) or amount[0][2] != [AMOUNT_COLUMN]
                or len(others) != 1 or not _is_passthrough(others[0][1])):
            raise UnsupportedPipeline(f"ColumnTransformer có dạng lạ: {[(n, type(t).__name__) for n, t, _ in fitted]}")
        scaler = amount[0][1]
        self.center = float(scaler.center_[0]) if scaler.with_centering else 0.0
        self.scale = float(scaler.scale_[0]) if scaler.with_scaling else 1.0

        #: Thứ tự cột mà bộ phân loại nhìn thấy (Amount đã chuẩn hoá đứng đầu)
        self.feature_names = [str(n) for n in preprocess.get_feature_names_out()]
        self._source = [FEATURE_ORDER.index(name) for name in self.feature_names]
        self._amount = self.feature_names.index(AMOUNT_COLUMN)
        self.booster = model.named_steps["clf"].get_booster()

    def matrix(self, features: pd.DataFrame) -> np.ndarray:
        """Đầu vào của bộ phân loại, float64, theo ``feature_names``. ``features`` là kết quả của
        ``build_features``."""
        values = features[FEATURE_ORDER].to_numpy(dtype="float64")[:, self._source]
        values[:, self._amount] = (values[:, self._amount] - self.center) / self.scale
        return values

    def predict(self, features: pd.DataFrame) -> np.ndarray:
        """Điểm rủi ro — ``model.predict_proba(features)[:, 1]``, float64."""
        if len(features) == 0:
            return np.array([], dtype="float64")
        return self.booster.inplace_predict(self.matrix(features)).astype("float64")

    def margin(self, features: pd.DataFrame) -> np.ndarray:
        """Logit của điểm rủi ro (thang log-odds của SHAP)."""
        return self.booster.inplace_predict(self.matrix(features), predict_type="margin").astype("float64")

    def contributions(self, features: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        """``(shap, bias)``: SHAP của từng đặc trưng theo ``feature_names`` và giá trị cơ sở."""
        data = xgb.DMatrix(self.matrix(features), feature_names=self.feature_names)
        out = self.booster.predict(data, pred_contribs=True).astype("float64")
        return out[:, :-1], out[:, -1]
