"""Kiểm thử tinh chỉnh siêu tham số (T-24, 04 §4) và bootstrap theo cặp (04 §5.4).

Tìm kiếm thật chạy 150 lần huấn luyện trên 227.000 dòng; lỗi ở phần dựng bảng
hay đọc tham số thắng mà chỉ lộ ra sau khi chạy xong thì mất cả lượt. Kiểm ở đây
trên dữ liệu tổng hợp nhỏ trước.
"""

import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import ParameterSampler

from src.config import RANDOM_STATE
from src.evaluate import bootstrap_diff
from src.features import V_COLUMNS, build_features
from src.modeling import (
    SEARCH_PARAMS,
    best_params_from,
    build_pipeline,
    imbalance_ratio,
    run_search,
    search_space,
    search_table,
)


@pytest.fixture
def dataset():
    """600 giao dịch, 30 gian lận — đủ để StratifiedKFold(5) luôn có mẫu dương."""
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
    frame.loc[fraud_idx, "V14"] -= 3.0
    frame.loc[fraud_idx, "V17"] -= 2.0
    return build_features(frame), pd.Series(y)


@pytest.fixture(scope="module")
def search_result():
    rng = np.random.default_rng(3)
    n_rows = 500
    raw = {"Time": rng.uniform(0, 172_000, n_rows)}
    for column in V_COLUMNS:
        raw[column] = rng.normal(size=n_rows)
    raw["Amount"] = rng.lognormal(3.0, 1.2, n_rows)
    y = np.zeros(n_rows, dtype=int)
    y[rng.choice(n_rows, size=25, replace=False)] = 1
    frame = pd.DataFrame(raw)
    frame.loc[y == 1, "V14"] -= 3.0
    X = build_features(frame)
    return X, pd.Series(y), run_search(X, pd.Series(y), n_iter=3, n_jobs=1, verbose=0)


# --------------------------------------------------------------------------
# Không gian tìm kiếm — đúng bảng ở 04 §4
# --------------------------------------------------------------------------

def test_search_space_covers_the_seven_parameters_of_the_spec():
    space = search_space(pos_weight=599.0)
    assert set(space) == {f"clf__{name}" for name in SEARCH_PARAMS}


def test_search_space_samples_stay_inside_the_ranges_of_the_spec():
    samples = list(ParameterSampler(search_space(pos_weight=599.0), n_iter=300,
                                    random_state=RANDOM_STATE))
    col = lambda name: np.array([s[f"clf__{name}"] for s in samples])  # noqa: E731

    assert col("n_estimators").min() >= 200 and col("n_estimators").max() <= 800
    assert col("max_depth").min() >= 3 and col("max_depth").max() <= 8
    assert col("learning_rate").min() >= 0.01 and col("learning_rate").max() <= 0.3
    assert col("subsample").min() >= 0.6 and col("subsample").max() <= 1.0
    assert col("colsample_bytree").min() >= 0.6 and col("colsample_bytree").max() <= 1.0
    assert col("min_child_weight").min() >= 1 and col("min_child_weight").max() <= 10
    assert set(col("scale_pos_weight")) == {1.0, 10.0, 100.0, 599.0}


def test_search_space_uses_the_real_training_ratio(dataset):
    _, y = dataset
    weights = search_space(y)["clf__scale_pos_weight"]
    assert weights[-1] == pytest.approx(round(imbalance_ratio(y), 1))


def test_search_space_needs_y_or_pos_weight():
    with pytest.raises(ValueError, match="scale_pos_weight"):
        search_space()


# --------------------------------------------------------------------------
# run_search và bảng kết quả
# --------------------------------------------------------------------------

def test_run_search_tries_exactly_n_iter_configurations(search_result):
    _, _, search = search_result
    assert len(search.cv_results_["params"]) == 3
    assert search.scoring == "average_precision"


def test_run_search_is_reproducible(search_result):
    X, y, search = search_result
    again = run_search(X, y, n_iter=3, n_jobs=1, verbose=0)
    np.testing.assert_allclose(search.cv_results_["mean_test_score"],
                               again.cv_results_["mean_test_score"])


def test_search_table_is_sorted_and_typed(search_result):
    _, _, search = search_result
    table = search_table(search.cv_results_)

    assert len(table) == 3
    assert list(table["rank"]) == sorted(table["rank"])
    assert table["pr_auc_mean"].is_monotonic_decreasing
    assert table["n_estimators"].dtype.kind == "i"
    assert (table["overfit_gap"] == table["pr_auc_train"] - table["pr_auc_mean"]).all()


def test_search_table_roundtrips_through_csv(search_result, tmp_path):
    """Notebook đọc lại bảng từ CSV do script ghi — hai đường phải cho cùng tham số."""
    _, _, search = search_result
    table = search_table(search.cv_results_)
    path = tmp_path / "search.csv"
    table.to_csv(path, index=False)

    assert best_params_from(pd.read_csv(path)) == pytest.approx(best_params_from(table))


def test_best_params_rebuild_the_winning_pipeline(search_result):
    X, y, search = search_result
    params = best_params_from(search_table(search.cv_results_))

    # n_jobs=1 như trong run_search: XGBoost "hist" cộng dồn histogram theo thứ tự
    # khác nhau khi số luồng khác nhau, nên n_jobs=-1 cho điểm lệch nhẹ
    pipeline = build_pipeline("xgboost", "class_weight", y=y, n_jobs=1, **params)
    clf = pipeline.named_steps["clf"]
    assert clf.get_params()["n_estimators"] == search.best_params_["clf__n_estimators"]
    assert clf.get_params()["scale_pos_weight"] == search.best_params_["clf__scale_pos_weight"]

    # refit=True của run_search và tự huấn luyện lại phải cho cùng điểm
    np.testing.assert_allclose(pipeline.fit(X, y).predict_proba(X)[:, 1],
                               search.best_estimator_.predict_proba(X)[:, 1], rtol=1e-6)


# --------------------------------------------------------------------------
# bootstrap_diff — so hai mô hình theo cặp
# --------------------------------------------------------------------------

def test_bootstrap_diff_of_a_model_with_itself_is_exactly_zero():
    rng = np.random.default_rng(0)
    y = np.r_[np.ones(20), np.zeros(500)].astype(int)
    s = rng.random(y.size)

    out = bootstrap_diff(y, s, s, n_boot=200)

    assert out["value"] == 0.0
    assert out["ci_low"] == 0.0 and out["ci_high"] == 0.0
    assert out["contains_zero"]


def test_bootstrap_diff_detects_a_clearly_better_model():
    rng = np.random.default_rng(1)
    y = np.r_[np.ones(40), np.zeros(800)].astype(int)
    good = y + rng.normal(0, 0.3, y.size)
    noise = rng.random(y.size)

    out = bootstrap_diff(y, good, noise, n_boot=300)

    assert out["ci_low"] > 0
    assert not out["contains_zero"]
    assert out["a_wins"] == 1.0


def test_bootstrap_diff_is_antisymmetric():
    rng = np.random.default_rng(2)
    y = np.r_[np.ones(30), np.zeros(600)].astype(int)
    a, b = y + rng.normal(0, 0.8, y.size), y + rng.normal(0, 1.0, y.size)

    ab = bootstrap_diff(y, a, b, n_boot=200)
    ba = bootstrap_diff(y, b, a, n_boot=200)

    assert ab["value"] == pytest.approx(-ba["value"])
    assert ab["ci_low"] == pytest.approx(-ba["ci_high"])
