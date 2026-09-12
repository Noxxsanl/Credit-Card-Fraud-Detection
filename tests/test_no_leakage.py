"""TC-20…TC-21 — chặn rò rỉ dữ liệu ngay ở mức cấu trúc pipeline.

Rò rỉ ở bài toán này không làm chương trình đổ; nó chỉ làm mọi chỉ số đẹp lên
một cách vô nghĩa. Vì vậy phải bắt bằng kiểm thử chứ không thể trông vào việc
đọc lại mã (AC-M4).
"""

import numpy as np
import pandas as pd
import pytest
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.exceptions import NotFittedError
from sklearn.pipeline import Pipeline as SklearnPipeline
from sklearn.utils.validation import check_is_fitted

from src.modeling import (
    DEFAULT_MODELS,
    DEFAULT_STRATEGIES,
    SAMPLING_STRATEGY,
    build_pipeline,
    imbalance_ratio,
    make_cv,
    oof_scores,
)
from src.features import V_COLUMNS, build_features

RESAMPLING_STRATEGIES = ("undersample", "smote", "smote_tomek")


@pytest.fixture
def dataset():
    """600 giao dịch, 30 gian lận — đủ để SMOTE và StratifiedKFold(5) chạy được."""
    rng = np.random.default_rng(11)
    n_rows, n_fraud = 600, 30

    raw = {"Time": rng.uniform(0, 172_000, n_rows)}
    for column in V_COLUMNS:
        raw[column] = rng.normal(size=n_rows)
    raw["Amount"] = rng.lognormal(3.0, 1.2, n_rows)

    y = np.zeros(n_rows, dtype=int)
    fraud_idx = rng.choice(n_rows, size=n_fraud, replace=False)
    y[fraud_idx] = 1

    frame = pd.DataFrame(raw)
    # Cho lớp dương một tín hiệu thật, nếu không mô hình không học được gì
    frame.loc[fraud_idx, "V14"] -= 3.0
    frame.loc[fraud_idx, "V17"] -= 2.0

    return build_features(frame), pd.Series(y)


# --------------------------------------------------------------------------
# TC-20 — resampler bắt buộc nằm trong pipeline của imblearn
# --------------------------------------------------------------------------

@pytest.mark.parametrize("strategy", DEFAULT_STRATEGIES)
def test_tc20_every_pipeline_is_an_imblearn_pipeline(strategy, dataset):
    X, y = dataset
    pipeline = build_pipeline("logistic_regression", strategy, y=y)

    assert isinstance(pipeline, ImbPipeline)
    # sklearn.pipeline.Pipeline KHÔNG bỏ qua bước resample khi transform tập
    # validation, nên dùng nhầm lớp này là rò rỉ ngay
    assert not type(pipeline) is SklearnPipeline


@pytest.mark.parametrize("strategy", RESAMPLING_STRATEGIES)
def test_tc20_resampler_sits_inside_the_pipeline(strategy, dataset):
    X, y = dataset
    pipeline = build_pipeline("logistic_regression", strategy, y=y)

    assert "resample" in pipeline.named_steps
    assert list(pipeline.named_steps) == ["preprocess", "resample", "clf"]


@pytest.mark.parametrize("strategy", ("none", "class_weight"))
def test_tc20_strategies_without_resampling_have_no_resampler(strategy, dataset):
    X, y = dataset
    pipeline = build_pipeline("logistic_regression", strategy, y=y)

    assert "resample" not in pipeline.named_steps


def test_tc20_resamplers_do_not_balance_to_one_to_one(dataset):
    """Cân bằng hoàn toàn làm méo xác suất tiên nghiệm — dừng ở 1:10 (04 §3.1)."""
    X, y = dataset
    for strategy in RESAMPLING_STRATEGIES:
        resampler = build_pipeline("logistic_regression", strategy, y=y).named_steps["resample"]
        assert resampler.sampling_strategy == SAMPLING_STRATEGY


# --------------------------------------------------------------------------
# TC-21 — không bước nào được fit sẵn trước khi vào cross-validation
# --------------------------------------------------------------------------

@pytest.mark.parametrize("strategy", DEFAULT_STRATEGIES)
def test_tc21_pipeline_steps_are_unfitted_before_cv(strategy, dataset):
    X, y = dataset
    pipeline = build_pipeline("logistic_regression", strategy, y=y)

    with pytest.raises(NotFittedError):
        check_is_fitted(pipeline.named_steps["preprocess"])
    with pytest.raises(NotFittedError):
        check_is_fitted(pipeline.named_steps["clf"])


def test_tc21_scaler_statistics_come_only_from_training_data(dataset):
    """Scaler phải học trung vị/IQR của tập huấn luyện, không phải toàn bộ dữ liệu (ML-04)."""
    X, y = dataset
    train_only = build_pipeline("logistic_regression", "none", y=y)
    train_only.fit(X.iloc[:400], y.iloc[:400])

    everything = build_pipeline("logistic_regression", "none", y=y)
    everything.fit(X, y)

    center_train = train_only.named_steps["preprocess"].named_transformers_["amount"].center_
    center_all = everything.named_steps["preprocess"].named_transformers_["amount"].center_

    assert center_train != pytest.approx(center_all)


def test_resampling_does_not_change_prediction_row_count(dataset):
    """Lấy mẫu lại chỉ tác động lúc fit; lúc dự đoán số dòng phải giữ nguyên."""
    X, y = dataset
    pipeline = build_pipeline("logistic_regression", "smote", y=y).fit(X, y)

    assert pipeline.predict_proba(X).shape[0] == len(X)


def test_fitting_does_not_mutate_input_frame(dataset):
    X, y = dataset
    snapshot = X.copy(deep=True)

    build_pipeline("logistic_regression", "smote", y=y).fit(X, y)

    pd.testing.assert_frame_equal(X, snapshot)


# --------------------------------------------------------------------------
# Hành vi của cross-validation
# --------------------------------------------------------------------------

def test_cv_is_stratified_and_every_fold_has_positives(dataset):
    X, y = dataset
    cv = make_cv()

    for _, test_idx in cv.split(X, y):
        assert y.iloc[test_idx].sum() > 0


def test_oof_scores_cover_every_row_exactly_once(dataset):
    X, y = dataset

    scores = oof_scores(build_pipeline("logistic_regression", "none", y=y), X, y)

    assert scores.shape == (len(X),)
    assert np.all((scores >= 0) & (scores <= 1))


def test_imbalance_ratio_matches_definition():
    y = np.array([0] * 578 + [1])
    assert imbalance_ratio(y) == pytest.approx(578.0)


def test_unknown_model_or_strategy_is_rejected(dataset):
    X, y = dataset

    with pytest.raises(ValueError, match="Mô hình không hỗ trợ"):
        build_pipeline("mang_no_ron", "none", y=y)
    with pytest.raises(ValueError, match="Chiến lược không hỗ trợ"):
        build_pipeline("logistic_regression", "xoa_bot_lop_am", y=y)


@pytest.mark.parametrize("model", DEFAULT_MODELS)
def test_every_model_in_the_grid_can_fit_and_score(model, dataset):
    X, y = dataset
    pipeline = build_pipeline(model, "class_weight", y=y).fit(X, y)

    scores = pipeline.predict_proba(X)[:, 1]

    assert scores.shape == (len(X),)
    assert np.isfinite(scores).all()
