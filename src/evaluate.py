"""Khung đánh giá dùng chung.

Với tỷ lệ lớp dương 0,172%, accuracy vô nghĩa: mô hình rỗng luôn trả 0 đã đạt
99,83%. Metric chính là **PR-AUC**; ROC-AUC chỉ báo cáo kèm để đối chiếu, vì
mẫu số ``FP + TN`` gần như không đổi khiến đường ROC che mất thay đổi lớn về
khối lượng thẩm định (04 §5.2).

Tập kiểm thử chỉ có khoảng 98 mẫu gian lận, nên mọi con số phải kèm khoảng tin
cậy bootstrap (AC-M6). Hai mô hình có khoảng tin cậy chồng lấn nhiều thì không
được kết luận cái nào tốt hơn.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from .config import DEFAULT_COST_FN, DEFAULT_COST_FP, RANDOM_STATE
from .threshold import (  # noqa: F401  (tái xuất cho notebook dùng lại)
    NAIVE_THRESHOLD,
    compare_thresholds,
    cost_curve,
    metrics_at_threshold,
    pick_threshold,
    threshold_alternatives,
)


def predict_scores(model, X) -> np.ndarray:
    """Điểm rủi ro liên tục: ``predict_proba`` nếu có, ngược lại ``decision_function``."""
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    return model.decision_function(X)


def baseline_pr_auc(y_true) -> float:
    """Đường cơ sở của PR-AUC bằng đúng tỷ lệ lớp dương.

    Con số này (khoảng 0,0017) phải xuất hiện trong báo cáo: nó cho thấy PR-AUC
    0,84 là gấp gần 500 lần mức ngẫu nhiên, điều mà bản thân con số 0,84 không
    nói lên được.
    """
    y_true = np.asarray(y_true)
    return float(np.mean(y_true == 1))


def compute_metrics(y_true, y_scores, threshold: float = NAIVE_THRESHOLD, name: str = "model") -> pd.Series:
    """Một dòng metric ứng với ngưỡng cho trước."""
    y_pred = (np.asarray(y_scores) >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    return pd.Series(
        {
            "model": name,
            "threshold": float(threshold),
            "pr_auc": average_precision_score(y_true, y_scores),
            "roc_auc": roc_auc_score(y_true, y_scores),
            "precision": precision_score(y_true, y_pred, zero_division=0),
            "recall": recall_score(y_true, y_pred, zero_division=0),
            "f1": f1_score(y_true, y_pred, zero_division=0),
            "tp": int(tp),
            "fp": int(fp),
            "fn": int(fn),
            "tn": int(tn),
        }
    )


def evaluate_model(
    model,
    X_test,
    y_test,
    threshold: float = NAIVE_THRESHOLD,
    name: str = "model",
    verbose: bool = True,
) -> pd.Series:
    """Đánh giá đầy đủ một model đã fit."""
    y_scores = predict_scores(model, X_test)
    metrics = compute_metrics(y_test, y_scores, threshold=threshold, name=name)

    if verbose:
        y_pred = (y_scores >= threshold).astype(int)
        print(f"=== {name} (ngưỡng = {threshold:.4f}) ===")
        print(classification_report(y_test, y_pred, digits=4, zero_division=0))
        print("Ma trận nhầm lẫn [[TN, FP], [FN, TP]]:")
        print(confusion_matrix(y_test, y_pred, labels=[0, 1]))
        print(f"PR-AUC : {metrics['pr_auc']:.4f}  (đường cơ sở {baseline_pr_auc(y_test):.5f})")
        print(f"ROC-AUC: {metrics['roc_auc']:.4f}")

    return metrics


def bootstrap_ci(
    y_true,
    y_scores,
    metric: str = "pr_auc",
    *,
    threshold: float | None = None,
    n_boot: int = 1000,
    alpha: float = 0.05,
    random_state: int = RANDOM_STATE,
) -> dict[str, float]:
    """Khoảng tin cậy bootstrap phân tầng cho một chỉ số (AC-M6).

    Lấy mẫu lại **riêng trong từng lớp** để mỗi lần lặp luôn giữ đúng số mẫu
    dương. Bootstrap thường trên dữ liệu 0,17% dương có thể sinh ra mẫu không có
    giao dịch gian lận nào, khiến chỉ số không xác định.

    ``metric`` nhận: ``pr_auc``, ``roc_auc``, ``precision``, ``recall``, ``f1``.
    Ba chỉ số sau cần ``threshold``.
    """
    y_true = np.asarray(y_true).ravel()
    y_scores = np.asarray(y_scores, dtype="float64").ravel()

    threshold_metrics = {"precision", "recall", "f1"}
    if metric in threshold_metrics and threshold is None:
        raise ValueError(f"Chỉ số {metric!r} cần tham số `threshold`.")

    pos_idx = np.flatnonzero(y_true == 1)
    neg_idx = np.flatnonzero(y_true == 0)
    rng = np.random.default_rng(random_state)

    def score(idx: np.ndarray) -> float:
        yt, ys = y_true[idx], y_scores[idx]
        if metric == "pr_auc":
            return float(average_precision_score(yt, ys))
        if metric == "roc_auc":
            return float(roc_auc_score(yt, ys))
        yp = (ys >= threshold).astype(int)
        if metric == "precision":
            return float(precision_score(yt, yp, zero_division=0))
        if metric == "recall":
            return float(recall_score(yt, yp, zero_division=0))
        if metric == "f1":
            return float(f1_score(yt, yp, zero_division=0))
        raise ValueError(f"Chỉ số không hỗ trợ: {metric!r}")

    samples = np.empty(n_boot, dtype="float64")
    for i in range(n_boot):
        idx = np.concatenate(
            [
                rng.choice(pos_idx, size=pos_idx.size, replace=True),
                rng.choice(neg_idx, size=neg_idx.size, replace=True),
            ]
        )
        samples[i] = score(idx)

    lo, hi = np.quantile(samples, [alpha / 2, 1 - alpha / 2])
    return {
        "metric": metric,
        "value": score(np.arange(y_true.size)),
        "ci_low": float(lo),
        "ci_high": float(hi),
        "n_boot": n_boot,
    }


def bootstrap_diff(
    y_true,
    scores_a,
    scores_b,
    *,
    n_boot: int = 1000,
    alpha: float = 0.05,
    random_state: int = RANDOM_STATE,
) -> dict[str, float]:
    """Bootstrap hiệu PR-AUC ``A − B`` **theo cặp** trên cùng một tập (04 §5.4).

    Mỗi lần lặp dùng CÙNG một mẫu cho cả hai mô hình. So hai khoảng tin cậy
    riêng lẻ thì bỏ mất tương quan giữa hai bộ điểm và luôn bảo thủ quá mức;
    phép theo cặp mới trả lời đúng câu hỏi "A có hơn B không".

    Khoảng tin cậy chứa 0 thì **không** được kết luận mô hình nào tốt hơn.
    """
    y_true = np.asarray(y_true).ravel()
    scores_a = np.asarray(scores_a, dtype="float64").ravel()
    scores_b = np.asarray(scores_b, dtype="float64").ravel()

    pos_idx = np.flatnonzero(y_true == 1)
    neg_idx = np.flatnonzero(y_true == 0)
    rng = np.random.default_rng(random_state)

    diffs = np.empty(n_boot, dtype="float64")
    for i in range(n_boot):
        idx = np.concatenate(
            [
                rng.choice(pos_idx, size=pos_idx.size, replace=True),
                rng.choice(neg_idx, size=neg_idx.size, replace=True),
            ]
        )
        diffs[i] = average_precision_score(y_true[idx], scores_a[idx]) - average_precision_score(
            y_true[idx], scores_b[idx]
        )

    lo, hi = np.quantile(diffs, [alpha / 2, 1 - alpha / 2])
    return {
        "value": float(
            average_precision_score(y_true, scores_a) - average_precision_score(y_true, scores_b)
        ),
        "ci_low": float(lo),
        "ci_high": float(hi),
        "a_wins": float(np.mean(diffs > 0)),
        "contains_zero": bool(lo <= 0 <= hi),
        "n_boot": n_boot,
    }


def headline_metrics(
    y_true,
    y_scores,
    threshold: float,
    *,
    n_boot: int = 1000,
    random_state: int = RANDOM_STATE,
) -> dict[str, dict]:
    """Bộ chỉ số chính kèm khoảng tin cậy, dùng cho ``metrics.json`` (06 §6.2)."""
    out: dict[str, dict] = {}
    for metric in ("pr_auc", "roc_auc", "recall", "precision"):
        out[metric] = bootstrap_ci(
            y_true,
            y_scores,
            metric,
            threshold=threshold,
            n_boot=n_boot,
            random_state=random_state,
        )
    out["f1"] = {"value": float(f1_score(y_true, (np.asarray(y_scores) >= threshold).astype(int), zero_division=0))}
    out["baseline_pr_auc"] = {"value": baseline_pr_auc(y_true)}
    return out


def curve_points(y_true, y_scores, *, max_points: int = 500) -> dict[str, dict]:
    """Điểm trên đường PR và ROC, đã lấy mẫu thưa để nhét vừa ``metrics.json``."""
    precision, recall, pr_thresholds = precision_recall_curve(y_true, y_scores)
    fpr, tpr, _ = roc_curve(y_true, y_scores)

    def thin(array: np.ndarray) -> list[float]:
        if array.size <= max_points:
            return [float(v) for v in array]
        idx = np.linspace(0, array.size - 1, max_points).astype(int)
        return [float(v) for v in array[idx]]

    return {
        "pr_curve": {
            "recall": thin(recall),
            "precision": thin(precision),
            "thresholds": thin(np.append(pr_thresholds, 1.0)),
        },
        "roc_curve": {"fpr": thin(fpr), "tpr": thin(tpr)},
    }


def expected_cost(
    y_true,
    y_scores,
    threshold: float,
    cost_fn: float = DEFAULT_COST_FN,
    cost_fp: float = DEFAULT_COST_FP,
) -> float:
    """Chi phí nghiệp vụ tại một ngưỡng: bỏ lọt đắt hơn báo động giả nhiều lần."""
    return metrics_at_threshold(
        y_true, y_scores, threshold, cost_fn=cost_fn, cost_fp=cost_fp
    ).expected_cost


def best_threshold_by_f1(y_true, y_scores) -> tuple[float, float]:
    """Ngưỡng tối đa hoá F1, kèm giá trị F1 đạt được."""
    threshold = pick_threshold(y_true, y_scores, "max_f1")
    return threshold, metrics_at_threshold(y_true, y_scores, threshold).f1


def best_threshold_by_cost(
    y_true,
    y_scores,
    cost_fn: float = DEFAULT_COST_FN,
    cost_fp: float = DEFAULT_COST_FP,
    n_steps: int = 200,
) -> tuple[float, float]:
    """Ngưỡng cực tiểu hoá chi phí kỳ vọng, kèm chi phí tại đó."""
    threshold = pick_threshold(
        y_true, y_scores, "min_expected_cost", cost_fn=cost_fn, cost_fp=cost_fp, n_steps=n_steps
    )
    cost = metrics_at_threshold(
        y_true, y_scores, threshold, cost_fn=cost_fn, cost_fp=cost_fp
    ).expected_cost
    return threshold, cost


def compare_models(rows) -> pd.DataFrame:
    """Gộp nhiều Series metric thành bảng so sánh, sắp theo PR-AUC giảm dần."""
    return pd.DataFrame(rows).sort_values("pr_auc", ascending=False).reset_index(drop=True)
