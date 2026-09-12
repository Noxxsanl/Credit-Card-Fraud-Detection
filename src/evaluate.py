"""Hàm đánh giá dùng chung cho mọi notebook.

Với dữ liệu mất cân bằng nặng, accuracy vô nghĩa: ưu tiên PR-AUC,
recall và precision tại ngưỡng đã chọn.
"""

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
)


def predict_scores(model, X):
    """Lấy điểm liên tục từ model (predict_proba nếu có, ngược lại decision_function)."""
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    return model.decision_function(X)


def compute_metrics(y_true, y_scores, threshold=0.5, name="model"):
    """Một dòng metric ứng với ngưỡng cho trước."""
    y_pred = (y_scores >= threshold).astype(int)
    return pd.Series(
        {
            "model": name,
            "threshold": threshold,
            "pr_auc": average_precision_score(y_true, y_scores),
            "roc_auc": roc_auc_score(y_true, y_scores),
            "precision": precision_score(y_true, y_pred, zero_division=0),
            "recall": recall_score(y_true, y_pred, zero_division=0),
            "f1": f1_score(y_true, y_pred, zero_division=0),
        }
    )


def evaluate_model(model, X_test, y_test, threshold=0.5, name="model", verbose=True):
    """Đánh giá đầy đủ một model đã fit."""
    y_scores = predict_scores(model, X_test)
    metrics = compute_metrics(y_test, y_scores, threshold=threshold, name=name)

    if verbose:
        y_pred = (y_scores >= threshold).astype(int)
        print(f"=== {name} (threshold={threshold:.3f}) ===")
        print(classification_report(y_test, y_pred, digits=4, zero_division=0))
        print("Confusion matrix:")
        print(confusion_matrix(y_test, y_pred))
        print(f"PR-AUC : {metrics['pr_auc']:.4f}")
        print(f"ROC-AUC: {metrics['roc_auc']:.4f}")

    return metrics


def best_threshold_by_f1(y_true, y_scores):
    """Ngưỡng tối đa hoá F1 trên đường precision-recall."""
    precision, recall, thresholds = precision_recall_curve(y_true, y_scores)
    f1 = 2 * precision * recall / np.clip(precision + recall, 1e-12, None)
    # precision/recall dài hơn thresholds đúng 1 phần tử
    idx = int(np.nanargmax(f1[:-1]))
    return float(thresholds[idx]), float(f1[idx])


def expected_cost(y_true, y_scores, threshold, cost_fn=100.0, cost_fp=1.0):
    """Chi phí nghiệp vụ: bỏ sót gian lận (FN) đắt hơn nhiều báo động nhầm (FP)."""
    y_pred = (y_scores >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return fn * cost_fn + fp * cost_fp


def best_threshold_by_cost(y_true, y_scores, cost_fn=100.0, cost_fp=1.0, n_steps=200):
    """Quét ngưỡng để cực tiểu hoá chi phí nghiệp vụ."""
    candidates = np.linspace(y_scores.min(), y_scores.max(), n_steps)
    costs = [expected_cost(y_true, y_scores, t, cost_fn, cost_fp) for t in candidates]
    idx = int(np.argmin(costs))
    return float(candidates[idx]), float(costs[idx])


def compare_models(rows):
    """Gộp nhiều Series metric thành bảng so sánh, sắp theo PR-AUC."""
    return pd.DataFrame(rows).sort_values("pr_auc", ascending=False).reset_index(drop=True)
