"""Biểu đồ dùng chung; mọi hàm trả về Figure để notebook tự lưu."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    confusion_matrix,
    precision_recall_curve,
    roc_curve,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
FIGURES_DIR = PROJECT_ROOT / "reports" / "figures"


def save_fig(fig, filename, dpi=150):
    """Lưu figure vào reports/figures/."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    path = FIGURES_DIR / filename
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    return path


def plot_class_balance(y, ax=None):
    if ax is None:
        _, ax = plt.subplots(figsize=(4, 3))
    counts = y.value_counts().sort_index()
    sns.barplot(x=counts.index.astype(str), y=counts.values, ax=ax)
    ax.set_yscale("log")
    ax.set_xlabel("Class (0 = hợp lệ, 1 = gian lận)")
    ax.set_ylabel("Số giao dịch (log)")
    ax.set_title("Phân phối nhãn")
    return ax.figure


def plot_pr_curve(y_true, y_scores, label="model", ax=None):
    if ax is None:
        _, ax = plt.subplots(figsize=(5, 4))
    precision, recall, _ = precision_recall_curve(y_true, y_scores)
    ax.plot(recall, precision, label=label)
    baseline = float(np.mean(y_true))
    ax.axhline(baseline, ls="--", c="grey", lw=1, label=f"baseline = {baseline:.4f}")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall curve")
    ax.legend()
    return ax.figure


def plot_roc_curve(y_true, y_scores, label="model", ax=None):
    if ax is None:
        _, ax = plt.subplots(figsize=(5, 4))
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    ax.plot(fpr, tpr, label=label)
    ax.plot([0, 1], [0, 1], ls="--", c="grey", lw=1)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("ROC curve")
    ax.legend()
    return ax.figure


def plot_confusion(y_true, y_pred, ax=None):
    if ax is None:
        _, ax = plt.subplots(figsize=(4, 4))
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    ConfusionMatrixDisplay(cm, display_labels=["hợp lệ", "gian lận"]).plot(
        ax=ax, colorbar=False
    )
    ax.set_title("Confusion matrix")
    return ax.figure


def plot_threshold_sweep(y_true, y_scores, ax=None):
    """Precision / recall / F1 thay đổi theo ngưỡng."""
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 4))
    precision, recall, thresholds = precision_recall_curve(y_true, y_scores)
    f1 = 2 * precision * recall / np.clip(precision + recall, 1e-12, None)
    ax.plot(thresholds, precision[:-1], label="precision")
    ax.plot(thresholds, recall[:-1], label="recall")
    ax.plot(thresholds, f1[:-1], label="f1")
    ax.set_xlabel("Threshold")
    ax.set_ylabel("Score")
    ax.set_title("Ảnh hưởng của ngưỡng quyết định")
    ax.legend()
    return ax.figure
