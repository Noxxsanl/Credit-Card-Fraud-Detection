"""Giải thích SHAP cho một giao dịch — API-05, FR-24 (docs/03 §5.3).

SHAP tính bằng ``pred_contribs=True`` của chính XGBoost (``FastScorer.contributions``) trên ma
trận **sau** tiền xử lý — cùng thuật toán TreeSHAP mà ``shap.TreeExplainer(clf)`` của notebook
07/08 dùng, trùng từng bit với ``explainer.joblib`` (``tests/test_api.py``), nhưng ảnh ``api``
không phải cài thư viện shap. Đưa ``build_features(x)`` chưa chuẩn hoá vào thì SHAP của
``Amount`` sai mà không báo lỗi (bẫy đã ghi ở giai đoạn 5) — ``FastScorer.matrix`` lo việc đó.

Giá trị SHAP ở thang log-odds: ``base_value + Σ SHAP`` bằng margin của booster, tức
logit của điểm rủi ro — tính chất mà biểu đồ thác nước ở UI-02 dựa vào.
"""

from __future__ import annotations

import pandas as pd

from src.features import build_features

from ..loader import Artifacts

N_POSITIVE, N_NEGATIVE = 5, 3


def explain(loaded: Artifacts, frame: pd.DataFrame) -> dict:
    """``frame`` là **một** dòng 30 cột thô."""
    features = build_features(frame)
    values, bias = loaded.scorer.contributions(features)
    names = loaded.scorer.feature_names
    # Giá trị hiển thị là giá trị gốc (Amount theo đơn vị tiền), không phải giá trị đã chuẩn hoá
    shown = features.iloc[0]

    contributions = [
        {"feature": name, "value": float(shown[name]), "shap": float(value)}
        for name, value in zip(names, values[0])
    ]
    by_magnitude = sorted(contributions, key=lambda c: -abs(c["shap"]))
    positive = sorted((c for c in contributions if c["shap"] > 0), key=lambda c: -c["shap"])[:N_POSITIVE]
    negative = sorted((c for c in contributions if c["shap"] < 0), key=lambda c: c["shap"])[:N_NEGATIVE]
    shown_names = {c["feature"] for c in positive + negative}

    return {
        "risk_score": float(loaded.scorer.predict(features)[0]),
        "base_value": float(bias[0]),
        "margin": float(loaded.scorer.margin(features)[0]),
        "top_positive": positive,
        "top_negative": negative,
        "remaining_shap": float(sum(c["shap"] for c in contributions if c["feature"] not in shown_names)),
        "contributions": by_magnitude,
        "model_version": loaded.model_version,
    }
