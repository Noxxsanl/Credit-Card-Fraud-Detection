"""Chọn ngưỡng quyết định theo chi phí nghiệp vụ.

Mô hình trả về xác suất; việc biến xác suất thành quyết định chặn/cho qua là một
lựa chọn riêng, thay đổi được lúc chạy (AR-03). Module này là nơi duy nhất chứa
logic đó — notebook, API và bộ kiểm thử đều dùng chung (03 §3.1).

Ngưỡng mặc định 0,5 không có cơ sở gì ở bài toán mất cân bằng 1:578: nó ngầm giả
định hai loại lỗi tốn kém như nhau và hai lớp cân bằng nhau, cả hai đều sai
(04 §6.1).
"""

from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np
import pandas as pd

from .config import (
    DATASET_DAYS,
    DEFAULT_COST_FN,
    DEFAULT_COST_FP,
    TEST_SIZE,
)

#: Các tiêu chí chọn ngưỡng được hỗ trợ (04 §6.3)
CRITERIA = (
    "min_expected_cost",
    "max_f1",
    "min_recall",
    "max_alerts_per_day",
    "naive",
)

NAIVE_THRESHOLD = 0.5


@dataclass(frozen=True) 
class ThresholdMetrics:
    """Toàn bộ con số cần thiết để đánh giá một ngưỡng."""

    threshold: float
    tp: int
    fp: int
    fn: int
    tn: int
    alerts: int
    precision: float
    recall: float
    f1: float
    expected_cost: float
    alerts_per_day: float

    def as_dict(self) -> dict:
        return asdict(self)


def _as_arrays(y_true, y_scores) -> tuple[np.ndarray, np.ndarray]:
    y_true = np.asarray(y_true, dtype="int8").ravel()
    y_scores = np.asarray(y_scores, dtype="float64").ravel()
    if y_true.shape != y_scores.shape:
        raise ValueError(
            f"y_true và y_scores khác độ dài: {y_true.shape} vs {y_scores.shape}"
        )
    return y_true, y_scores


def confusion_counts(y_true, y_scores, threshold: float) -> tuple[int, int, int, int]:
    """Trả về (tp, fp, fn, tn) tại một ngưỡng. Quy ước: dự đoán dương khi score >= τ."""
    y_true, y_scores = _as_arrays(y_true, y_scores)
    predicted = y_scores >= threshold
    actual = y_true == 1

    tp = int(np.count_nonzero(predicted & actual))
    fp = int(np.count_nonzero(predicted & ~actual))
    fn = int(np.count_nonzero(~predicted & actual))
    tn = int(np.count_nonzero(~predicted & ~actual))
    return tp, fp, fn, tn


def sweep(
    y_true,
    y_scores,
    thresholds=None,
    *,
    n_steps: int = 200,
) -> pd.DataFrame:
    """Tính tp/fp/fn/tn cho một dãy ngưỡng cùng lúc.

    Cài đặt bằng ``searchsorted`` trên mảng đã sắp xếp: mỗi ngưỡng tốn O(log n)
    thay vì O(n). Với 57.000 giao dịch và 200 ngưỡng, cả bảng tính xong trong vài
    mili giây — đây là cái làm cho thanh trượt ngưỡng phản hồi tức thì (NFR-03).
    """
    y_true, y_scores = _as_arrays(y_true, y_scores)
    if thresholds is None:
        thresholds = threshold_grid(y_scores, n_steps=n_steps)
    thresholds = np.asarray(thresholds, dtype="float64").ravel()

    n_total = y_scores.size
    positives = np.sort(y_scores[y_true == 1])
    n_pos = positives.size
    all_scores = np.sort(y_scores)

    # Số phần tử >= τ  =  n - vị trí chèn bên trái của τ
    alerts = n_total - np.searchsorted(all_scores, thresholds, side="left")
    tp = n_pos - np.searchsorted(positives, thresholds, side="left")
    fp = alerts - tp
    fn = n_pos - tp
    tn = n_total - alerts - fn

    return pd.DataFrame(
        {
            "threshold": thresholds,
            "tp": tp.astype("int64"),
            "fp": fp.astype("int64"),
            "fn": fn.astype("int64"),
            "tn": tn.astype("int64"),
            "alerts": alerts.astype("int64"),
        }
    )


#: Phần đuôi trên của phân bố điểm được lấy mẫu dày hơn — mọi quyết định chặn
#: đều nằm trong vùng này.
TAIL_QUANTILE = 0.98


def threshold_grid(y_scores, n_steps: int = 200) -> np.ndarray:
    """Dãy ngưỡng ứng viên, bám theo phân bố điểm thực tế.

    Lưới cố định (đều hoặc log) không dùng được ở đây vì hình dạng phân bố điểm
    thay đổi hẳn theo chiến lược huấn luyện: mô hình không xử lý mất cân bằng dồn
    điểm sát 0, còn mô hình ``class_weight`` hoặc ``scale_pos_weight`` đẩy điểm
    lên sát 1. Lưới log phủ tốt trường hợp đầu và bỏ sót vùng quyết định của
    trường hợp sau — đủ để một ràng buộc ngân sách cảnh báo trả về nghiệm vi phạm
    chính ràng buộc đó.

    Vì vậy ngưỡng ứng viên lấy theo **lượng tử của chính điểm rủi ro**: một nửa
    trải đều trên toàn bộ phân bố, một nửa dồn vào 2% điểm cao nhất là nơi mọi
    cảnh báo thực sự phát sinh. Cách này tự thích nghi với mọi mô hình.
    """
    y_scores = np.asarray(y_scores, dtype="float64").ravel()
    if y_scores.size == 0:
        return np.array([NAIVE_THRESHOLD])

    half = max(n_steps // 2, 2)
    candidates = [np.quantile(y_scores, np.linspace(0.0, 1.0, half))]

    tail = y_scores[y_scores >= np.quantile(y_scores, TAIL_QUANTILE)]
    if tail.size > 1:
        candidates.append(np.quantile(tail, np.linspace(0.0, 1.0, half)))

    # Luôn có mặt ngưỡng mặc định để bảng so sánh nào cũng đối chiếu được với nó
    candidates.append(np.array([NAIVE_THRESHOLD]))

    grid = np.unique(np.concatenate(candidates))
    return grid[(grid > 0) & (grid < 1)]


def alerts_per_day(
    n_alerts: int,
    *,
    sample_fraction: float = TEST_SIZE,
    days: float = DATASET_DAYS,
) -> float:
    """Quy đổi số cảnh báo trên tập mẫu thành số cảnh báo mỗi ngày trên toàn luồng.

    Tập kiểm thử là 20% dữ liệu của một cửa sổ hai ngày, nên khối lượng công việc
    thực tế mỗi ngày gấp ``1 / 0.2 / 2`` lần số cảnh báo đếm được. Con số này mới
    là thứ người quản lý rủi ro quan tâm (US-07).
    """
    if sample_fraction <= 0 or days <= 0:
        raise ValueError("sample_fraction và days phải dương")
    return float(n_alerts) / sample_fraction / days


def metrics_at_threshold(
    y_true,
    y_scores,
    threshold: float,
    *,
    cost_fn: float = DEFAULT_COST_FN,
    cost_fp: float = DEFAULT_COST_FP,
    sample_fraction: float = TEST_SIZE,
    days: float = DATASET_DAYS,
) -> ThresholdMetrics:
    """Toàn bộ chỉ số tại một ngưỡng, gồm cả chi phí kỳ vọng."""
    tp, fp, fn, tn = confusion_counts(y_true, y_scores, threshold)
    alerts = tp + fp

    precision = tp / alerts if alerts else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

    return ThresholdMetrics(
        threshold=float(threshold),
        tp=tp,
        fp=fp,
        fn=fn,
        tn=tn,
        alerts=alerts,
        precision=precision,
        recall=recall,
        f1=f1,
        expected_cost=fn * cost_fn + fp * cost_fp,
        alerts_per_day=alerts_per_day(
            alerts, sample_fraction=sample_fraction, days=days
        ),
    )


def cost_curve(
    y_true,
    y_scores,
    *,
    cost_fn: float = DEFAULT_COST_FN,
    cost_fp: float = DEFAULT_COST_FP,
    n_steps: int = 200,
    thresholds=None,
    sample_fraction: float = TEST_SIZE,
    days: float = DATASET_DAYS,
) -> pd.DataFrame:
    """Bảng ngưỡng × (chi phí, precision, recall, f1, số cảnh báo mỗi ngày)."""
    table = sweep(y_true, y_scores, thresholds=thresholds, n_steps=n_steps)

    tp = table["tp"].to_numpy()
    fp = table["fp"].to_numpy()
    fn = table["fn"].to_numpy()
    alerts = table["alerts"].to_numpy()

    with np.errstate(divide="ignore", invalid="ignore"):
        precision = np.where(alerts > 0, tp / np.maximum(alerts, 1), 0.0)
        recall = np.where((tp + fn) > 0, tp / np.maximum(tp + fn, 1), 0.0)
        denominator = precision + recall
        f1 = np.where(denominator > 0, 2 * precision * recall / np.maximum(denominator, 1e-12), 0.0)

    table["precision"] = precision
    table["recall"] = recall
    table["f1"] = f1
    table["expected_cost"] = fn * cost_fn + fp * cost_fp
    table["alerts_per_day"] = alerts / sample_fraction / days
    return table


def pick_threshold(
    y_true,
    y_scores,
    criterion: str = "min_expected_cost",
    *,
    value: float | None = None,
    cost_fn: float = DEFAULT_COST_FN,
    cost_fp: float = DEFAULT_COST_FP,
    n_steps: int = 200,
    sample_fraction: float = TEST_SIZE,
    days: float = DATASET_DAYS,
) -> float:
    """Chọn ngưỡng theo một tiêu chí.

    Tiêu chí
    --------
    ``min_expected_cost``
        Cực tiểu ``cost_fn × FN + cost_fp × FP``. Mặc định của hệ thống.
    ``max_f1``
        Cực đại F1. Dùng khi không ước lượng được chi phí.
    ``min_recall``
        Ngưỡng LỚN NHẤT còn đạt ``recall >= value`` — tức bắt đủ gian lận yêu cầu
        mà sinh ít cảnh báo giả nhất.
    ``max_alerts_per_day``
        Ngưỡng NHỎ NHẤT mà số cảnh báo mỗi ngày vẫn nằm trong ngân sách ``value``
        — tức tận dụng hết năng lực thẩm định.
    ``naive``
        Trả về 0,5. Có mặt để bảng so sánh luôn có mốc tham chiếu.

    Lưu ý ML-08: hàm này phải chạy trên dữ liệu huấn luyện hoặc out-of-fold. Dò
    ngưỡng trên chính tập kiểm thử rồi báo cáo chỉ số cũng trên tập đó là một
    dạng rò rỉ nhẹ, làm mọi con số lạc quan hơn thực tế.
    """
    if criterion not in CRITERIA:
        raise ValueError(f"Tiêu chí không hợp lệ: {criterion!r}. Chọn trong {CRITERIA}.")

    if criterion == "naive":
        return NAIVE_THRESHOLD

    table = cost_curve(
        y_true,
        y_scores,
        cost_fn=cost_fn,
        cost_fp=cost_fp,
        n_steps=n_steps,
        sample_fraction=sample_fraction,
        days=days,
    )

    if criterion == "min_expected_cost":
        return float(table.loc[table["expected_cost"].idxmin(), "threshold"])

    if criterion == "max_f1":
        return float(table.loc[table["f1"].idxmax(), "threshold"])

    if value is None:
        raise ValueError(f"Tiêu chí {criterion!r} cần tham số `value`.")

    if criterion == "min_recall":
        feasible = table[table["recall"] >= value]
        if feasible.empty:
            # Không ngưỡng nào đạt: trả ngưỡng thấp nhất, tức recall cao nhất có thể
            return float(table["threshold"].min())
        return float(feasible["threshold"].max())

    # max_alerts_per_day
    feasible = table[table["alerts_per_day"] <= value]
    if feasible.empty:
        return float(table["threshold"].max())
    return float(feasible["threshold"].min())


def threshold_alternatives(
    y_true,
    y_scores,
    *,
    cost_fn: float = DEFAULT_COST_FN,
    cost_fp: float = DEFAULT_COST_FP,
    min_recall: float = 0.90,
    alert_budget: float = 200.0,
    n_steps: int = 200,
    sample_fraction: float = TEST_SIZE,
    days: float = DATASET_DAYS,
) -> dict[str, float]:
    """Bốn phương án ngưỡng trình bày cạnh nhau ở màn hình UI-03 (FR-34)."""
    kwargs = dict(
        cost_fn=cost_fn,
        cost_fp=cost_fp,
        n_steps=n_steps,
        sample_fraction=sample_fraction,
        days=days,
    )
    return {
        "min_expected_cost": pick_threshold(y_true, y_scores, "min_expected_cost", **kwargs),
        "max_f1": pick_threshold(y_true, y_scores, "max_f1", **kwargs),
        f"recall_at_least_{int(min_recall * 100)}": pick_threshold(
            y_true, y_scores, "min_recall", value=min_recall, **kwargs
        ),
        f"budget_{int(alert_budget)}_alerts": pick_threshold(
            y_true, y_scores, "max_alerts_per_day", value=alert_budget, **kwargs
        ),
        "default_naive": NAIVE_THRESHOLD,
    }


def compare_thresholds(
    y_true,
    y_scores,
    thresholds: dict[str, float],
    *,
    cost_fn: float = DEFAULT_COST_FN,
    cost_fp: float = DEFAULT_COST_FP,
    sample_fraction: float = TEST_SIZE,
    days: float = DATASET_DAYS,
) -> pd.DataFrame:
    """Bảng bắt buộc có trong báo cáo (04 §6.4)."""
    rows = []
    for name, threshold in thresholds.items():
        metrics = metrics_at_threshold(
            y_true,
            y_scores,
            threshold,
            cost_fn=cost_fn,
            cost_fp=cost_fp,
            sample_fraction=sample_fraction,
            days=days,
        )
        rows.append({"name": name, **metrics.as_dict()})
    return pd.DataFrame(rows)


def sensitivity_analysis(
    y_true,
    y_scores,
    ratios=(5, 10, 25, 50, 100),
    *,
    cost_fp: float = DEFAULT_COST_FP,
    n_steps: int = 200,
) -> pd.DataFrame:
    """Ngưỡng tối ưu dịch chuyển thế nào khi tỷ lệ chi phí thay đổi (AC-M8).

    Vì ``cost_fn`` và ``cost_fp`` đều là giả định, kết luận chỉ vững nếu ngưỡng
    tối ưu ổn định trong một dải tỷ lệ rộng.
    """
    rows = []
    for ratio in ratios:
        cost_fn = cost_fp * ratio
        threshold = pick_threshold(
            y_true,
            y_scores,
            "min_expected_cost",
            cost_fn=cost_fn,
            cost_fp=cost_fp,
            n_steps=n_steps,
        )
        metrics = metrics_at_threshold(
            y_true, y_scores, threshold, cost_fn=cost_fn, cost_fp=cost_fp
        )
        rows.append(
            {
                "cost_ratio": ratio,
                "cost_fn": cost_fn,
                "cost_fp": cost_fp,
                **metrics.as_dict(),
            }
        )
    return pd.DataFrame(rows)
