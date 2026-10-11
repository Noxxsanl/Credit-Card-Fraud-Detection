"""Kiểm thử các phần thêm sau review (docs/11): mức cân bằng, điểm lưu theo dấu vân tay,
CV lặp và kiểm định t hiệu chỉnh, OOF forward-chaining, quy tắc tie-break.

Mọi ca chạy trên dữ liệu tổng hợp nhỏ — các thí nghiệm thật chạy hàng giờ nên lỗi phải bị
bắt ở đây trước.
"""

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import brier_score_loss

from src.evaluate import compare_repeated_cv, corrected_resampled_ttest
from src.features import V_COLUMNS, build_features
from src.modeling import (
    GRID_COLUMNS,
    RATIO_GRID_COLUMNS,
    SAMPLING_STRATEGY,
    build_pipeline,
    config_fingerprint,
    data_fingerprint,
    effective_ratio,
    imbalance_ratio,
    make_cv,
    repeated_cv_scores,
    run_grid,
    run_ratio_grid,
    score_key,
    select_configuration,
    temporal_oof_scores,
)


@pytest.fixture
def dataset():
    """600 giao dịch, 30 gian lận, có cột Time để thử chia theo thời gian."""
    rng = np.random.default_rng(11)
    n_rows, n_fraud = 600, 30
    raw = {"Time": np.sort(rng.uniform(0, 172_000, n_rows))}
    for column in V_COLUMNS:
        raw[column] = rng.normal(size=n_rows)
    raw["Amount"] = rng.lognormal(3.0, 1.2, n_rows)
    y = np.zeros(n_rows, dtype=int)
    fraud_idx = rng.choice(n_rows, size=n_fraud, replace=False)
    y[fraud_idx] = 1
    frame = pd.DataFrame(raw)
    frame.loc[fraud_idx, "V14"] -= 3.0
    frame.loc[fraud_idx, "V17"] -= 2.0
    return build_features(frame), pd.Series(y), frame["Time"]


# --------------------------------------------------------------------------
# Mức cân bằng (docs/11 M5)
# --------------------------------------------------------------------------

def test_ratio_none_keeps_the_standard_grid_configuration(dataset):
    """ratio=None phải cho đúng pipeline của lưới chuẩn — điểm lưu cũ vẫn khớp dấu vân tay."""
    X, y, _ = dataset
    for model in ("logistic_regression", "xgboost"):
        cw = build_pipeline(model, "class_weight", y=y).named_steps["clf"]
        if model == "xgboost":
            assert cw.scale_pos_weight == pytest.approx(imbalance_ratio(y))
        else:
            assert cw.class_weight == "balanced"
    for strategy in ("undersample", "smote", "smote_tomek"):
        resampler = build_pipeline("logistic_regression", strategy, y=y).named_steps["resample"]
        assert resampler.sampling_strategy == SAMPLING_STRATEGY


@pytest.mark.parametrize("ratio", (0.02, 0.1, 0.3, 1.0))
def test_class_weight_at_ratio_r_weights_positives_by_r_times_the_imbalance(dataset, ratio):
    X, y, _ = dataset
    ir = imbalance_ratio(y)
    xgb = build_pipeline("xgboost", "class_weight", y=y, ratio=ratio).named_steps["clf"]
    assert xgb.scale_pos_weight == pytest.approx(ratio * ir)
    for model in ("logistic_regression", "decision_tree", "random_forest"):
        clf = build_pipeline(model, "class_weight", y=y, ratio=ratio).named_steps["clf"]
        assert clf.class_weight == {0: 1.0, 1: pytest.approx(ratio * ir)}


@pytest.mark.parametrize("strategy", ("undersample", "smote", "smote_tomek"))
def test_resamplers_at_ratio_r_use_sampling_strategy_r(dataset, strategy):
    X, y, _ = dataset
    resampler = build_pipeline("logistic_regression", strategy, y=y, ratio=0.3).named_steps["resample"]
    assert resampler.sampling_strategy == 0.3


def test_ratio_is_rejected_for_none_and_out_of_range(dataset):
    X, y, _ = dataset
    with pytest.raises(ValueError, match="none"):
        build_pipeline("xgboost", "none", y=y, ratio=0.1)
    with pytest.raises(ValueError, match=r"\(0, 1\]"):
        build_pipeline("xgboost", "smote", y=y, ratio=1.5)


def test_effective_ratio():
    assert effective_ratio("none", None, 1 / 600) == pytest.approx(1 / 600)
    assert effective_ratio("class_weight", None, 1 / 600) == 1.0
    assert effective_ratio("smote", None, 1 / 600) == SAMPLING_STRATEGY
    assert effective_ratio("undersample", 0.3, 1 / 600) == 0.3


def test_ratio_grid_has_one_row_per_model_strategy_ratio(dataset, tmp_path):
    X, y, _ = dataset
    table, scores = run_ratio_grid(
        X, y, models=("logistic_regression",), strategies=("class_weight", "undersample"),
        ratios=(0.1, 1.0), checkpoint_path=tmp_path / "ratio_grid.csv", return_scores=True, verbose=False,
    )
    assert list(table.columns) == RATIO_GRID_COLUMNS
    assert len(table) == 4
    assert set(scores) == {score_key("logistic_regression", s, r)
                           for s in ("class_weight", "undersample") for r in (0.1, 1.0)}
    assert (tmp_path / "ratio_grid.npz").exists()


# --------------------------------------------------------------------------
# Brier và điểm lưu theo dấu vân tay (docs/11 M4, m5)
# --------------------------------------------------------------------------

def test_grid_brier_is_the_brier_score_of_the_oof_scores(dataset):
    X, y, _ = dataset
    table, scores = run_grid(X, y, models=("logistic_regression",), strategies=("none", "class_weight"),
                             return_scores=True, verbose=False)
    assert "brier" in GRID_COLUMNS and "config_hash" in GRID_COLUMNS
    for _, row in table.iterrows():
        oof = scores[score_key(row["model"], row["strategy"])]
        assert row["brier"] == pytest.approx(brier_score_loss(y, oof))


def test_config_fingerprint_is_deterministic_and_sensitive(dataset):
    X, y, _ = dataset
    data = data_fingerprint(X, y)
    cv = make_cv()
    a = config_fingerprint(build_pipeline("xgboost", "none", y=y), cv=cv, data=data)
    b = config_fingerprint(build_pipeline("xgboost", "none", y=y), cv=cv, data=data)
    assert a == b and len(a) == 16
    assert config_fingerprint(build_pipeline("xgboost", "none", y=y, max_depth=3), cv=cv, data=data) != a
    assert config_fingerprint(build_pipeline("xgboost", "none", y=y), cv=make_cv(n_splits=3), data=data) != a
    assert config_fingerprint(build_pipeline("xgboost", "none", y=y), cv=cv,
                              data=data_fingerprint(X.iloc[::-1], y.iloc[::-1])) != a


def test_checkpoint_is_reused_only_when_the_fingerprint_matches(dataset, tmp_path):
    """Trước review điểm lưu được dùng lại chỉ vì tệp tồn tại (docs/11 m5)."""
    X, y, _ = dataset
    path = tmp_path / "grid.csv"
    first = run_grid(X, y, models=("logistic_regression",), strategies=("none",),
                     checkpoint_path=path, verbose=False)

    # Cùng cấu hình: dùng lại điểm, giữ nguyên thời gian huấn luyện đã ghi
    again = run_grid(X, y, models=("logistic_regression",), strategies=("none",),
                     checkpoint_path=path, verbose=False)
    # CSV ghi số thực 16 chữ số có nghĩa: thời gian đọc lại chỉ khớp tới chữ số cuối
    assert again.loc[0, "total_seconds"] == pytest.approx(first.loc[0, "total_seconds"], rel=1e-12)
    assert again.loc[0, "config_hash"] == first.loc[0, "config_hash"]
    assert again.loc[0, "pr_auc_mean"] == first.loc[0, "pr_auc_mean"]     # điểm lấy từ .npz: trùng từng bit

    # Đổi cấu hình mô hình: dấu vân tay đổi, phải huấn luyện lại
    changed = run_grid(X, y, models=("logistic_regression",), strategies=("none",),
                       model_params={"logistic_regression": {"C": 0.01}}, checkpoint_path=path, verbose=False)
    assert changed.loc[0, "config_hash"] != first.loc[0, "config_hash"]
    assert changed.loc[0, "total_seconds"] != first.loc[0, "total_seconds"]


def test_legacy_checkpoint_without_fingerprint_is_retrained(dataset, tmp_path):
    X, y, _ = dataset
    path = tmp_path / "grid.csv"
    run_grid(X, y, models=("logistic_regression",), strategies=("none",), checkpoint_path=path, verbose=False)
    legacy = pd.read_csv(path).drop(columns=["config_hash"])
    legacy["total_seconds"] = 123456.0               # dấu hiệu: nếu dùng lại thì con số này còn nguyên
    legacy.to_csv(path, index=False)

    table = run_grid(X, y, models=("logistic_regression",), strategies=("none",), checkpoint_path=path,
                     verbose=False)
    assert table.loc[0, "total_seconds"] != 123456.0


def test_changing_costs_recomputes_threshold_columns_without_retraining(dataset, tmp_path):
    X, y, _ = dataset
    path = tmp_path / "grid.csv"
    cheap = run_grid(X, y, models=("logistic_regression",), strategies=("none",), cost_fn=10.0, cost_fp=10.0,
                     checkpoint_path=path, verbose=False)
    dear = run_grid(X, y, models=("logistic_regression",), strategies=("none",), cost_fn=1000.0, cost_fp=1.0,
                    checkpoint_path=path, verbose=False)
    assert dear.loc[0, "total_seconds"] == pytest.approx(cheap.loc[0, "total_seconds"], rel=1e-12)  # không huấn luyện lại
    assert dear.loc[0, "pr_auc_mean"] == cheap.loc[0, "pr_auc_mean"]
    assert dear.loc[0, "threshold"] <= cheap.loc[0, "threshold"]               # FN đắt hơn → ngưỡng hạ


# --------------------------------------------------------------------------
# CV lặp và kiểm định t hiệu chỉnh (docs/11 m2)
# --------------------------------------------------------------------------

def test_corrected_ttest_matches_the_formula():
    diffs = np.array([0.01, 0.02, -0.005, 0.015, 0.0, 0.01])
    out = corrected_resampled_ttest(diffs, n_train=400, n_test=100)
    k = diffs.size
    se = np.sqrt((1 / k + 100 / 400) * diffs.var(ddof=1))
    assert out["mean_diff"] == pytest.approx(diffs.mean())
    assert out["se"] == pytest.approx(se)
    assert out["t"] == pytest.approx(diffs.mean() / se)
    assert out["df"] == k - 1
    assert out["ci_low"] < out["mean_diff"] < out["ci_high"]


def test_corrected_ttest_is_wider_than_the_naive_ttest():
    rng = np.random.default_rng(3)
    diffs = rng.normal(0.01, 0.01, 25)
    corrected = corrected_resampled_ttest(diffs, n_train=4, n_test=1)
    naive_se = diffs.std(ddof=1) / np.sqrt(diffs.size)
    assert corrected["se"] > naive_se


def test_repeated_cv_uses_the_same_folds_for_every_configuration(dataset):
    X, y, _ = dataset
    a = repeated_cv_scores(build_pipeline("logistic_regression", "none", y=y), X, y, n_repeats=2)
    b = repeated_cv_scores(build_pipeline("decision_tree", "none", y=y), X, y, n_repeats=3)
    assert len(a) == 10 and len(b) == 15
    assert set(zip(a["repeat"], a["fold"])) <= set(zip(b["repeat"], b["fold"]))
    assert (a["n_test"].to_numpy() == b["n_test"].to_numpy()[:10]).all()

    same = compare_repeated_cv(a, a)
    assert same["mean_diff"] == 0.0 and same["contains_zero"]
    diff = compare_repeated_cv(a, b)
    assert diff["n_folds"] == 10                                    # chỉ so trên fold chung


# --------------------------------------------------------------------------
# OOF forward-chaining (docs/11 M2)
# --------------------------------------------------------------------------

def test_temporal_oof_never_trains_on_the_future(dataset):
    X, y, time_values = dataset
    oof, folds = temporal_oof_scores(build_pipeline("logistic_regression", "none", y=y), X, y, time_values,
                                     n_splits=4)
    assert len(folds) == 4
    assert (folds["train_to"] <= folds["valid_from"]).all()
    assert np.isnan(oof).sum() == len(y) - folds["n_valid"].sum()        # khối đầu không có điểm
    first_valid = folds["valid_from"].min()
    assert np.isnan(oof[time_values.to_numpy() < first_valid]).all()
    assert np.isfinite(oof[time_values.to_numpy() >= first_valid]).all()


def test_temporal_oof_works_on_unsorted_rows(dataset):
    X, y, time_values = dataset
    shuffled = np.random.default_rng(0).permutation(len(y))
    pipe = build_pipeline("logistic_regression", "none", y=y)
    sorted_oof, _ = temporal_oof_scores(pipe, X, y, time_values)
    shuffled_oof, _ = temporal_oof_scores(pipe, X.iloc[shuffled], y.iloc[shuffled], time_values.iloc[shuffled])
    np.testing.assert_allclose(shuffled_oof, sorted_oof[shuffled], equal_nan=True)


# --------------------------------------------------------------------------
# Quy tắc tie-break (docs/11 M4)
# --------------------------------------------------------------------------

def _candidates(**overrides):
    base = pd.DataFrame({
        "strategy": ["class_weight", "none", "smote", "undersample"],
        "tie_with_best": [True, True, True, False],
        "brier": [0.00041, 0.00038, 0.00039, 0.00030],
        "alerts_per_day": [287.0, 229.0, 316.0, 150.0],
    })
    for column, values in overrides.items():
        base[column] = values
    return base


def test_selection_prefers_the_simplest_strategy_among_ties():
    assert select_configuration(_candidates())["strategy"] == "none"


def test_selection_ignores_configurations_that_are_not_tied_with_the_best():
    pick = select_configuration(_candidates(tie_with_best=[True, False, True, True]))
    assert pick["strategy"] == "class_weight"


def test_selection_breaks_equal_complexity_by_brier_then_alerts():
    table = pd.DataFrame({
        "strategy": ["class_weight", "class_weight", "class_weight"],
        "tie_with_best": [True, True, True],
        "brier": [0.0004, 0.0003, 0.0003],
        "alerts_per_day": [100.0, 250.0, 200.0],
        "ratio": [1.0, 0.1, 0.3],
    })
    assert select_configuration(table)["ratio"] == 0.3


def test_selection_needs_a_tied_configuration():
    with pytest.raises(ValueError):
        select_configuration(_candidates(tie_with_best=[False] * 4))
