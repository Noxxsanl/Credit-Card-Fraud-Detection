"""TC-10…TC-15 — logic chọn ngưỡng.

Dữ liệu dùng trong các ca này là điểm rủi ro tổng hợp có cấu trúc giống thực tế:
lớp âm dồn gần 0, lớp dương trải rộng. Không cần mô hình thật.
"""

import numpy as np
import pandas as pd
import pytest

from src.threshold import (
    NAIVE_THRESHOLD,
    alerts_per_day,
    compare_thresholds,
    confusion_counts,
    cost_curve,
    metrics_at_threshold,
    pick_threshold,
    sweep,
    threshold_alternatives,
    threshold_grid,
)


@pytest.fixture
def scores():
    """5.000 giao dịch, 50 gian lận — mô hình xếp hạng tốt nhưng KHÔNG tách hẳn.

    Phần chồng lấn giữa hai lớp là bắt buộc phải có: nếu điểm của hai lớp tách
    rời nhau, mọi ngưỡng trong khoảng trống đều cho chi phí bằng 0 và các ca kiểm
    thử về đánh đổi chi phí trở nên vô nghĩa. Vì vậy 3% lớp âm là "âm khó" (điểm
    cao) và 30% lớp dương là "dương khó" (điểm thấp), giống phân bố thật.
    """
    rng = np.random.default_rng(42)
    n_neg, n_pos = 4_950, 50
    n_hard_neg, n_hard_pos = int(n_neg * 0.03), int(n_pos * 0.3)

    negatives = np.concatenate(
        [
            rng.beta(1.0, 80.0, size=n_neg - n_hard_neg),  # dồn sát 0
            rng.beta(3.0, 6.0, size=n_hard_neg),  # âm khó
        ]
    )
    positives = np.concatenate(
        [
            rng.beta(6.0, 3.0, size=n_pos - n_hard_pos),  # dương rõ
            rng.beta(1.5, 20.0, size=n_hard_pos),  # dương khó, lẫn vào lớp âm
        ]
    )

    y_true = np.concatenate([np.zeros(n_neg, dtype=int), np.ones(n_pos, dtype=int)])
    return y_true, np.concatenate([negatives, positives])


def test_fixture_has_genuine_overlap(scores):
    """Canh chính fixture: nếu hai lớp tách rời, phần lớn ca dưới đây mất ý nghĩa."""
    y_true, y_scores = scores

    assert y_scores[y_true == 0].max() > y_scores[y_true == 1].min()
    assert metrics_at_threshold(y_true, y_scores, 0.05).expected_cost > 0


# --------------------------------------------------------------------------
# TC-10 — hai biên
# --------------------------------------------------------------------------

def test_tc10_threshold_zero_flags_everything(scores):
    y_true, y_scores = scores
    metrics = metrics_at_threshold(y_true, y_scores, 0.0)

    assert metrics.alerts == y_true.size
    assert metrics.fn == 0
    assert metrics.recall == 1.0


def test_tc10_threshold_above_max_flags_nothing(scores):
    y_true, y_scores = scores
    metrics = metrics_at_threshold(y_true, y_scores, 1.01)

    assert metrics.alerts == 0
    assert metrics.fp == 0
    assert metrics.precision == 0.0
    assert metrics.recall == 0.0


def test_confusion_counts_sum_to_total(scores):
    y_true, y_scores = scores
    tp, fp, fn, tn = confusion_counts(y_true, y_scores, 0.1)

    assert tp + fp + fn + tn == y_true.size
    assert tp + fn == int(y_true.sum())


def test_sweep_matches_point_by_point_counting(scores):
    """Cài đặt vector hoá bằng searchsorted phải khớp phép đếm trực tiếp."""
    y_true, y_scores = scores
    thresholds = [0.001, 0.01, 0.05, 0.2, 0.5, 0.9]

    table = sweep(y_true, y_scores, thresholds=thresholds)

    for row in table.itertuples():
        tp, fp, fn, tn = confusion_counts(y_true, y_scores, row.threshold)
        assert (row.tp, row.fp, row.fn, row.tn) == (tp, fp, fn, tn)


# --------------------------------------------------------------------------
# TC-11 — nghiệm tối ưu nằm trong dải quét
# --------------------------------------------------------------------------

def test_tc11_cost_optimal_threshold_is_interior(scores):
    y_true, y_scores = scores
    grid = threshold_grid(y_scores, n_steps=200)

    threshold = pick_threshold(y_true, y_scores, "min_expected_cost")

    assert grid.min() <= threshold <= grid.max()
    assert threshold not in (grid.min(), grid.max())


def test_tc11_cost_at_optimum_beats_naive_threshold(scores):
    """Luận điểm trung tâm của đồ án: ngưỡng 0,5 tốn kém hơn ngưỡng đã chọn."""
    y_true, y_scores = scores
    optimal = pick_threshold(y_true, y_scores, "min_expected_cost")

    cost_optimal = metrics_at_threshold(y_true, y_scores, optimal).expected_cost
    cost_naive = metrics_at_threshold(y_true, y_scores, NAIVE_THRESHOLD).expected_cost

    assert cost_optimal <= cost_naive


def test_threshold_grid_always_contains_naive_threshold(scores):
    _, y_scores = scores
    assert NAIVE_THRESHOLD in threshold_grid(y_scores)


# --------------------------------------------------------------------------
# TC-13 — ràng buộc ngân sách cảnh báo
# --------------------------------------------------------------------------

def test_tc13_alert_budget_is_respected(scores):
    y_true, y_scores = scores
    budget = 200.0

    threshold = pick_threshold(
        y_true, y_scores, "max_alerts_per_day", value=budget, sample_fraction=1.0, days=2.0
    )
    metrics = metrics_at_threshold(
        y_true, y_scores, threshold, sample_fraction=1.0, days=2.0
    )

    assert metrics.alerts_per_day <= budget


def test_alerts_per_day_extrapolates_from_sample(scores):
    """20% dữ liệu của hai ngày: 40 cảnh báo tương ứng 100 cảnh báo mỗi ngày."""
    assert alerts_per_day(40, sample_fraction=0.2, days=2.0) == pytest.approx(100.0)


@pytest.fixture
def shifted_scores(scores):
    """Điểm bị đẩy sát 1, như mô hình dùng ``class_weight`` / ``scale_pos_weight``.

    Trên dữ liệu thật, Logistic Regression + ``class_weight='balanced'`` cho
    precision 0,057 ngay tại ngưỡng 0,5 — toàn bộ vùng quyết định nằm ở đuôi
    trên. Lưới ngưỡng phải phủ được vùng đó.
    """
    y_true, y_scores = scores
    return y_true, 1 - (1 - y_scores) ** 8


def test_grid_resolves_the_upper_tail(shifted_scores):
    """Hồi quy: lưới log cũ chỉ đặt vài điểm ở đuôi trên và bỏ sót vùng quyết định."""
    _, y_scores = shifted_scores
    grid = threshold_grid(y_scores)

    cutoff = np.quantile(y_scores, 0.98)
    assert (grid >= cutoff).sum() >= 50


def test_alert_budget_respected_for_shifted_scores(shifted_scores):
    """Hồi quy: ràng buộc ngân sách từng trả về nghiệm vi phạm chính nó."""
    y_true, y_scores = shifted_scores
    budget = 200.0

    threshold = pick_threshold(
        y_true, y_scores, "max_alerts_per_day", value=budget, sample_fraction=1.0, days=2.0
    )
    metrics = metrics_at_threshold(y_true, y_scores, threshold, sample_fraction=1.0, days=2.0)

    assert metrics.alerts_per_day <= budget


def test_min_recall_respected_for_shifted_scores(shifted_scores):
    y_true, y_scores = shifted_scores

    threshold = pick_threshold(y_true, y_scores, "min_recall", value=0.90)

    assert metrics_at_threshold(y_true, y_scores, threshold).recall >= 0.90


# --------------------------------------------------------------------------
# TC-14 — ràng buộc recall tối thiểu
# --------------------------------------------------------------------------

def test_tc14_min_recall_constraint_is_met(scores):
    y_true, y_scores = scores

    threshold = pick_threshold(y_true, y_scores, "min_recall", value=0.90)

    assert metrics_at_threshold(y_true, y_scores, threshold).recall >= 0.90


def test_tc14_min_recall_picks_highest_feasible_threshold(scores):
    """Đạt recall yêu cầu mà sinh ít cảnh báo giả nhất."""
    y_true, y_scores = scores
    threshold = pick_threshold(y_true, y_scores, "min_recall", value=0.90)

    table = cost_curve(y_true, y_scores)
    feasible = table[table["recall"] >= 0.90]

    assert threshold == pytest.approx(feasible["threshold"].max())
    assert metrics_at_threshold(y_true, y_scores, threshold).fp <= int(
        feasible["fp"].min()
    )


# --------------------------------------------------------------------------
# TC-15 — chi phí báo động giả bằng 0 thì ngưỡng tiến về 0
# --------------------------------------------------------------------------

def test_tc15_zero_false_positive_cost_drives_threshold_down(scores):
    y_true, y_scores = scores

    threshold = pick_threshold(y_true, y_scores, "min_expected_cost", cost_fp=0.0)

    assert threshold == pytest.approx(threshold_grid(y_scores).min())
    assert metrics_at_threshold(y_true, y_scores, threshold).recall == 1.0


def test_expensive_false_positives_push_threshold_up(scores):
    y_true, y_scores = scores

    cheap = pick_threshold(y_true, y_scores, "min_expected_cost", cost_fn=122.21, cost_fp=1.0)
    expensive = pick_threshold(y_true, y_scores, "min_expected_cost", cost_fn=122.21, cost_fp=50.0)

    assert expensive > cheap


# --------------------------------------------------------------------------
# Bảng phương án và bảng so sánh
# --------------------------------------------------------------------------

def test_alternatives_contains_all_four_options_plus_naive(scores):
    y_true, y_scores = scores

    alternatives = threshold_alternatives(y_true, y_scores)

    assert set(alternatives) == {
        "min_expected_cost",
        "max_f1",
        "recall_at_least_90",
        "budget_200_alerts",
        "default_naive",
    }
    assert all(0 < value < 1 for value in alternatives.values())


def test_compare_thresholds_returns_one_row_per_option(scores):
    y_true, y_scores = scores
    alternatives = threshold_alternatives(y_true, y_scores)

    table = compare_thresholds(y_true, y_scores, alternatives)

    assert isinstance(table, pd.DataFrame)
    assert len(table) == len(alternatives)
    assert {"precision", "recall", "expected_cost", "alerts_per_day"} <= set(table.columns)


def test_invalid_criterion_is_rejected(scores):
    y_true, y_scores = scores

    with pytest.raises(ValueError, match="Tiêu chí không hợp lệ"):
        pick_threshold(y_true, y_scores, "cai_gi_do")


def test_constrained_criterion_requires_value(scores):
    y_true, y_scores = scores

    with pytest.raises(ValueError, match="cần tham số"):
        pick_threshold(y_true, y_scores, "min_recall")


@pytest.mark.skip(reason="TC-12: cần hàm JavaScript ở web/ — viết khi làm UI-03 (ngày 18)")
def test_tc12_client_server_parity():
    """Máy khách và máy chủ phải cho cùng TP/FP/FN trên 20 ngưỡng mẫu (UI-D1)."""
