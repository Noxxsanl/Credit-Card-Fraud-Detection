"""Dựng pipeline và chạy lưới thí nghiệm 5 chiến lược × 4 mô hình (04 §3).

Mọi pipeline sinh ra ở đây đều là ``imblearn.pipeline.Pipeline``. Đó không phải
lựa chọn thẩm mỹ: khi đưa vào ``cross_validate``, imblearn TỰ ĐỘNG bỏ qua bước
lấy mẫu lại lúc transform tập validation, còn ``sklearn.pipeline.Pipeline`` thì
không. Resample ngoài pipeline là lỗi rò rỉ phổ biến nhất ở bài toán này: mẫu
SMOTE nội suy từ mẫu validation lọt vào tập huấn luyện và mọi chỉ số vọt lên
khoảng 0,99 (ML-01, 04 §2.1).
"""

from __future__ import annotations

import time
import warnings

import numpy as np
import pandas as pd
from imblearn.combine import SMOTETomek
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.under_sampling import RandomUnderSampler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict, cross_validate
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from .config import CV_FOLDS, RANDOM_STATE
from .features import make_preprocessor

#: Tỷ lệ lớp thiểu số sau khi lấy mẫu lại. KHÔNG đưa về 1:1 — cân bằng hoàn toàn
#: làm méo xác suất tiên nghiệm rất mạnh, mô hình trả xác suất lệch cao và
#: precision sụp (04 §3.1).
SAMPLING_STRATEGY = 0.1

MODEL_NAMES = ("logistic_regression", "decision_tree", "random_forest", "xgboost", "lightgbm")
STRATEGY_NAMES = ("none", "class_weight", "undersample", "smote", "smote_tomek")

#: Lưới chuẩn trong báo cáo: 5 chiến lược × 4 mô hình = 20 tổ hợp (FR-05)
DEFAULT_MODELS = ("logistic_regression", "decision_tree", "random_forest", "xgboost")
DEFAULT_STRATEGIES = STRATEGY_NAMES

#: Mô hình cây nhận mất cân bằng qua scale_pos_weight thay vì class_weight
_BOOSTED = {"xgboost", "lightgbm"}


def _make_estimator(name: str, *, balanced: bool, pos_weight: float, random_state: int, **kwargs):
    """Khởi tạo một bộ phân loại theo cấu hình khởi điểm ở 04 §3.2."""
    if name == "logistic_regression":
        params = dict(max_iter=1000, solver="liblinear", random_state=random_state)
        if balanced:
            params["class_weight"] = "balanced"
        return LogisticRegression(**{**params, **kwargs})

    if name == "decision_tree":
        params = dict(max_depth=6, random_state=random_state)
        if balanced:
            params["class_weight"] = "balanced"
        return DecisionTreeClassifier(**{**params, **kwargs})

    if name == "random_forest":
        params = dict(n_estimators=300, n_jobs=-1, random_state=random_state)
        if balanced:
            params["class_weight"] = "balanced"
        return RandomForestClassifier(**{**params, **kwargs})

    if name == "xgboost":
        from xgboost import XGBClassifier

        params = dict(
            n_estimators=500,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            tree_method="hist",
            eval_metric="aucpr",
            n_jobs=-1,
            random_state=random_state,
        )
        if balanced:
            params["scale_pos_weight"] = pos_weight
        return XGBClassifier(**{**params, **kwargs})

    if name == "lightgbm":
        from lightgbm import LGBMClassifier

        params = dict(
            n_estimators=500,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            n_jobs=-1,
            verbose=-1,
            random_state=random_state,
        )
        if balanced:
            params["scale_pos_weight"] = pos_weight
        return LGBMClassifier(**{**params, **kwargs})

    raise ValueError(f"Mô hình không hỗ trợ: {name!r}. Chọn trong {MODEL_NAMES}.")


def _make_resampler(strategy: str, random_state: int):
    """Bước lấy mẫu lại tương ứng với chiến lược, hoặc ``None`` nếu không có."""
    if strategy in ("none", "class_weight"):
        return None
    if strategy == "undersample":
        return RandomUnderSampler(
            sampling_strategy=SAMPLING_STRATEGY, random_state=random_state
        )
    if strategy == "smote":
        return SMOTE(
            sampling_strategy=SAMPLING_STRATEGY, k_neighbors=5, random_state=random_state
        )
    if strategy == "smote_tomek":
        return SMOTETomek(
            sampling_strategy=SAMPLING_STRATEGY, random_state=random_state
        )
    raise ValueError(f"Chiến lược không hỗ trợ: {strategy!r}. Chọn trong {STRATEGY_NAMES}.")


def imbalance_ratio(y) -> float:
    """Tỷ lệ âm/dương — giá trị dùng cho ``scale_pos_weight``."""
    y = np.asarray(y)
    n_pos = int((y == 1).sum())
    if n_pos == 0:
        raise ValueError("Không có mẫu dương nào trong y.")
    return float((y == 0).sum() / n_pos)


def build_pipeline(
    model: str = "xgboost",
    strategy: str = "none",
    *,
    y=None,
    pos_weight: float | None = None,
    random_state: int = RANDOM_STATE,
    **model_kwargs,
) -> ImbPipeline:
    """Pipeline hoàn chỉnh: tiền xử lý → lấy mẫu lại → phân loại.

    Luôn trả về pipeline của imblearn, kể cả khi không có bước lấy mẫu lại, để
    cả lưới thí nghiệm đồng nhất về kiểu và ``tests/test_no_leakage.py`` kiểm tra
    được bằng một điều kiện duy nhất.

    ``pos_weight`` suy từ ``y`` nếu không truyền vào; chỉ dùng cho chiến lược
    ``class_weight`` với mô hình boosting.
    """
    if model not in MODEL_NAMES:
        raise ValueError(f"Mô hình không hỗ trợ: {model!r}. Chọn trong {MODEL_NAMES}.")
    if strategy not in STRATEGY_NAMES:
        raise ValueError(f"Chiến lược không hỗ trợ: {strategy!r}. Chọn trong {STRATEGY_NAMES}.")

    if pos_weight is None:
        pos_weight = imbalance_ratio(y) if y is not None else 1.0

    balanced = strategy == "class_weight"
    estimator = _make_estimator(
        model,
        balanced=balanced,
        pos_weight=pos_weight,
        random_state=random_state,
        **model_kwargs,
    )

    steps = [("preprocess", make_preprocessor())]
    resampler = _make_resampler(strategy, random_state)
    if resampler is not None:
        steps.append(("resample", resampler))
    steps.append(("clf", estimator))

    return ImbPipeline(steps)


def make_cv(n_splits: int = CV_FOLDS, random_state: int = RANDOM_STATE) -> StratifiedKFold:
    """Phân tầng là bắt buộc: với 0,172% lớp dương, ``KFold`` thường có thể tạo ra
    fold không chứa mẫu gian lận nào và ``average_precision`` trên fold đó không
    xác định (ML-02)."""
    return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)


def oof_scores(pipeline, X, y, *, cv=None, random_state: int = RANDOM_STATE) -> np.ndarray:
    """Điểm rủi ro out-of-fold — dùng để chọn ngưỡng mà không đụng tập kiểm thử.

    Chọn ngưỡng trên chính tập kiểm thử rồi báo cáo chỉ số cũng trên tập đó là
    một dạng rò rỉ nhẹ; ML-08 yêu cầu dùng dữ liệu out-of-fold cho việc này.
    """
    cv = cv or make_cv(random_state=random_state)
    probabilities = cross_val_predict(pipeline, X, y, cv=cv, method="predict_proba")
    return probabilities[:, 1]


def run_grid(
    X,
    y,
    *,
    models=DEFAULT_MODELS,
    strategies=DEFAULT_STRATEGIES,
    cv=None,
    random_state: int = RANDOM_STATE,
    verbose: bool = True,
) -> pd.DataFrame:
    """Chạy toàn bộ lưới và trả về bảng kết quả trung tâm của báo cáo (FR-05).

    Ghi lại **mọi** tổ hợp, kể cả những cấu hình cho kết quả kém — phần so sánh
    mới là trọng tâm đề tài, không phải riêng mô hình thắng cuộc (US-10).
    """
    cv = cv or make_cv(random_state=random_state)
    pos_weight = imbalance_ratio(y)
    scoring = {"pr_auc": "average_precision", "roc_auc": "roc_auc"}

    rows = []
    for model in models:
        for strategy in strategies:
            pipeline = build_pipeline(
                model, strategy, pos_weight=pos_weight, random_state=random_state
            )
            started = time.perf_counter()
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                results = cross_validate(
                    pipeline, X, y, cv=cv, scoring=scoring, error_score="raise"
                )
            elapsed = time.perf_counter() - started

            row = {
                "model": model,
                "strategy": strategy,
                "pr_auc_mean": float(np.mean(results["test_pr_auc"])),
                "pr_auc_std": float(np.std(results["test_pr_auc"])),
                "roc_auc_mean": float(np.mean(results["test_roc_auc"])),
                "roc_auc_std": float(np.std(results["test_roc_auc"])),
                "fit_seconds": float(np.mean(results["fit_time"])),
                "total_seconds": elapsed,
            }
            rows.append(row)

            if verbose:
                print(
                    f"{model:20s} {strategy:14s} "
                    f"PR-AUC {row['pr_auc_mean']:.4f} ± {row['pr_auc_std']:.4f}  "
                    f"({elapsed:5.1f}s)"
                )

    return (
        pd.DataFrame(rows)
        .sort_values("pr_auc_mean", ascending=False)
        .reset_index(drop=True)
    )
