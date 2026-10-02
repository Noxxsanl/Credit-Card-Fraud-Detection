"""Giải thích SHAP cho một giao dịch — API-05, FR-24 (docs/03 §5.3).

``explainer.joblib`` bọc riêng bước ``clf``, nên đầu vào là ma trận **sau** tiền xử lý:
``model[:-1].transform(build_features(x))``. Đưa ``build_features(x)`` thẳng vào thì
SHAP của ``Amount`` sai mà không báo lỗi (bẫy đã ghi ở giai đoạn 5).

Giá trị SHAP ở thang log-odds: ``base_value + Σ SHAP`` bằng margin của booster, tức
logit của điểm rủi ro — tính chất mà biểu đồ thác nước ở UI-02 dựa vào.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.features import build_features

from ..loader import Artifacts

N_POSITIVE, N_NEGATIVE = 5, 3


def explain(loaded: Artifacts, frame: pd.DataFrame) -> dict:
    """``frame`` là **một** dòng 30 cột thô."""
    features = build_features(frame)
    transformed = loaded.model[:-1].transform(features)
    values = np.asarray(loaded.explainer.shap_values(transformed), dtype="float64")[0]
    names = list(transformed.columns)
    # Giá trị hiển thị là giá trị gốc (Amount theo đơn vị tiền), không phải giá trị đã chuẩn hoá
    shown = features.iloc[0]

    contributions = [
        {"feature": name, "value": float(shown[name]), "shap": float(value)}
        for name, value in zip(names, values)
    ]
    by_magnitude = sorted(contributions, key=lambda c: -abs(c["shap"]))
    positive = sorted((c for c in contributions if c["shap"] > 0), key=lambda c: -c["shap"])[:N_POSITIVE]
    negative = sorted((c for c in contributions if c["shap"] < 0), key=lambda c: c["shap"])[:N_NEGATIVE]
    shown_names = {c["feature"] for c in positive + negative}

    base = loaded.base_value
    margin = float(loaded.model[-1].predict(transformed, output_margin=True)[0])
    return {
        "risk_score": float(loaded.model.predict_proba(features)[:, 1][0]),
        "base_value": base,
        "margin": margin,
        "top_positive": positive,
        "top_negative": negative,
        "remaining_shap": float(sum(c["shap"] for c in contributions if c["feature"] not in shown_names)),
        "contributions": by_magnitude,
        "model_version": loaded.model_version,
    }
