"""Kiểm tra toàn vẹn, loại trùng lặp và chia tập (TC-22, TC-23 phần không cần dữ liệu thật)."""

import numpy as np
import pandas as pd
import pytest

from src.config import TARGET
from src.data import (
    check_integrity,
    class_balance,
    drop_duplicates,
    observed_days,
    split_data,
    temporal_split,
)
from src.features import V_COLUMNS


def make_dataset(n_rows: int = 400, n_fraud: int = 20, *, seed: int = 7) -> pd.DataFrame:
    """Bộ dữ liệu tổng hợp trải đúng hai ngày, mất cân bằng như bài toán thật."""
    rng = np.random.default_rng(seed)

    data = {"Time": np.linspace(0, 172_000, n_rows)}
    for column in V_COLUMNS:
        data[column] = rng.normal(size=n_rows)
    data["Amount"] = rng.uniform(0, 1_000, size=n_rows)

    labels = np.zeros(n_rows, dtype=int)
    labels[rng.choice(n_rows, size=n_fraud, replace=False)] = 1
    data[TARGET] = labels

    return pd.DataFrame(data)


def test_integrity_report_flags_row_count_mismatch():
    report = check_integrity(make_dataset())

    assert not report.ok
    assert any("số dòng" in problem for problem in report.problems)
    assert report.n_missing_cells == 0


def test_integrity_report_flags_missing_values():
    df = make_dataset()
    df.loc[0, "V3"] = np.nan

    report = check_integrity(df)

    assert report.n_missing_cells == 1
    assert any("ô thiếu giá trị" in problem for problem in report.problems)


def test_strict_mode_raises():
    with pytest.raises(ValueError, match="toàn vẹn"):
        check_integrity(make_dataset(), strict=True)


# --------------------------------------------------------------------------
# DS-20 / ML-06 — loại trùng lặp trước khi chia tập
# --------------------------------------------------------------------------

def test_duplicates_are_removed(capsys):
    df = make_dataset(n_rows=100, n_fraud=10)
    with_duplicates = pd.concat([df, df.iloc[:5]], ignore_index=True)

    cleaned = drop_duplicates(with_duplicates)

    assert len(cleaned) == len(df)
    assert not cleaned.duplicated().any()
    assert "Loại 5" in capsys.readouterr().out


def test_tc23_no_duplicate_rows_shared_between_splits():
    df = drop_duplicates(make_dataset(), verbose=False)
    X_train, X_test, _, _ = split_data(df)

    train_hashes = set(pd.util.hash_pandas_object(X_train, index=False))
    test_hashes = set(pd.util.hash_pandas_object(X_test, index=False))

    assert train_hashes & test_hashes == set()


def test_tc22_split_indices_do_not_overlap():
    df = make_dataset()
    X_train, X_test, y_train, y_test = split_data(df)

    assert set(X_train.index) & set(X_test.index) == set()
    assert len(X_train) + len(X_test) == len(df)
    assert len(y_train) + len(y_test) == len(df)


def test_split_is_stratified():
    df = make_dataset(n_rows=1_000, n_fraud=50)
    _, _, y_train, y_test = split_data(df)

    assert y_train.mean() == pytest.approx(y_test.mean(), abs=0.005)


def test_split_is_reproducible():
    df = make_dataset()
    first = split_data(df)[0]
    second = split_data(df)[0]

    pd.testing.assert_frame_equal(first, second)


def test_split_returns_validation_set_when_requested():
    df = make_dataset(n_rows=1_000, n_fraud=50)
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(df, val_size=0.1)

    assert len(X_train) + len(X_val) + len(X_test) == len(df)
    assert len(X_val) == pytest.approx(100, abs=2)


# --------------------------------------------------------------------------
# FR-10 — chia theo thời gian
# --------------------------------------------------------------------------

def test_temporal_split_cuts_at_end_of_day_one():
    df = make_dataset()
    X_train, X_test, y_train, y_test = temporal_split(df, as_features=False)

    assert (X_train["Time"] < 86_400).all()
    assert (X_test["Time"] >= 86_400).all()
    assert len(X_train) + len(X_test) == len(df)


def test_temporal_split_can_return_feature_matrices():
    X_train, X_test, _, _ = temporal_split(make_dataset())

    assert "Time" not in X_train.columns
    assert "hour_sin" in X_test.columns


def test_class_balance_reports_imbalance_ratio():
    df = make_dataset(n_rows=1_000, n_fraud=10)

    balance = class_balance(df[TARGET])

    assert balance.loc[1, "count"] == 10
    assert balance.loc[1, "ratio"] == pytest.approx(0.01)
    assert balance.loc[1, "one_in"] == pytest.approx(100.0)


def test_observed_days_matches_two_day_window():
    assert observed_days(make_dataset()) == pytest.approx(2.0, abs=0.01)
