"""Autoencoder tùy chọn (T-34, 04 §8) — kiểm trên dữ liệu tổng hợp nhỏ."""

import numpy as np
import pandas as pd
import pytest

from src.anomaly import ReconstructionScorer
from src.config import RANDOM_STATE


@pytest.fixture
def normal_and_outliers():
    """Lớp bình thường nằm trên một mặt phẳng 3 chiều trong không gian 8 chiều;
    điểm bất thường lệch khỏi mặt phẳng đó."""
    rng = np.random.default_rng(7)
    basis = rng.normal(size=(3, 8))
    normal = rng.normal(size=(1500, 3)) @ basis
    outliers = rng.normal(size=(40, 3)) @ basis + rng.normal(0, 3.0, size=(40, 8))
    columns = [f"V{i}" for i in range(1, 8)] + ["Amount"]
    normal[:, -1] = np.abs(normal[:, -1]) * 20
    outliers[:, -1] = np.abs(outliers[:, -1]) * 20
    return pd.DataFrame(normal, columns=columns), pd.DataFrame(outliers, columns=columns)


def test_outliers_reconstruct_worse_than_normal_rows(normal_and_outliers):
    normal, outliers = normal_and_outliers
    model = ReconstructionScorer(max_iter=60).fit(normal.iloc[:1200])

    held_out = model.score_samples(normal.iloc[1200:])
    anomalous = model.score_samples(outliers)

    assert np.median(anomalous) > 3 * np.median(held_out)


def test_standardization_is_learned_from_training_rows_only(normal_and_outliers):
    normal, outliers = normal_and_outliers
    model = ReconstructionScorer(max_iter=5).fit(normal)

    expected = np.log1p(normal["Amount"]).mean()
    assert model.mean_[-1] == pytest.approx(expected)
    # Chấm thêm dữ liệu mới không được làm thay đổi thống kê đã học
    model.score_samples(outliers)
    assert model.mean_[-1] == pytest.approx(expected)


def test_scorer_uses_shared_seed_and_is_deterministic(normal_and_outliers):
    normal, _ = normal_and_outliers
    a = ReconstructionScorer(max_iter=5).fit(normal).score_samples(normal)
    b = ReconstructionScorer(max_iter=5).fit(normal).score_samples(normal)

    assert ReconstructionScorer().random_state == RANDOM_STATE
    np.testing.assert_array_equal(a, b)
    assert a.shape == (len(normal),)
