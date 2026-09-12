"""Biểu đồ dùng chung; mọi hàm trả về Figure để notebook tự lưu.

Danh mục 8 biểu đồ EDA bắt buộc nằm ở 02 §9. Nguyên tắc chung: mỗi biểu đồ phải
trả lời được một câu hỏi cụ thể, không vẽ cho đủ số lượng (09 §6).
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    confusion_matrix,
    precision_recall_curve,
    roc_curve,
)

from .config import FIGURES_DIR, TARGET
from .data import add_hour
from .features import V_COLUMNS
from .threshold import cost_curve

CLASS_LABELS = {0: "hợp lệ", 1: "gian lận"}
CLASS_PALETTE = {0: "#4C78A8", 1: "#E45756"}


def save_fig(fig, filename, dpi=150):
    """Lưu figure vào ``reports/figures/``."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    path = FIGURES_DIR / filename
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    return path


# --------------------------------------------------------------------------
# EDA 1–8 (02 §9)
# --------------------------------------------------------------------------

def plot_class_balance(y, ax=None):
    """EDA 1 — mất cân bằng nhìn thấy được ngay, thang log."""
    if ax is None:
        _, ax = plt.subplots(figsize=(4, 3))

    counts = pd.Series(y).value_counts().sort_index()
    ax.bar(
        [CLASS_LABELS[i] for i in counts.index],
        counts.to_numpy(),
        color=[CLASS_PALETTE[i] for i in counts.index],
    )
    ax.set_yscale("log")
    ax.set_xlabel("Nhãn")
    ax.set_ylabel("Số giao dịch (thang log)")
    ax.set_title("Phân bố nhãn")

    total = counts.sum()
    for i, value in enumerate(counts.to_numpy()):
        ax.text(i, value, f"{value:,}\n({value / total:.3%})", ha="center", va="bottom")
    return ax.figure


def plot_amount_distribution(df, ax=None):
    """EDA 2 — gian lận tập trung ở khoảng tiền nào. Thang log vì lệch phải rất mạnh."""
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 4))

    for label, group in df.groupby(TARGET):
        # +0.01 để giữ lại các giao dịch Amount = 0 (thử thẻ) trên thang log
        ax.hist(
            np.log10(group["Amount"] + 0.01),
            bins=60,
            density=True,
            alpha=0.55,
            label=CLASS_LABELS[label],
            color=CLASS_PALETTE[label],
        )
    ax.set_xlabel("log10(Amount + 0,01)")
    ax.set_ylabel("Mật độ")
    ax.set_title("Phân bố số tiền theo nhãn")
    ax.legend()
    return ax.figure


def plot_hourly_rate(df, ax=None):
    """EDA 3 — gian lận có dồn vào giờ vắng không.

    Vẽ TỶ LỆ gian lận theo giờ, không vẽ số lượng tuyệt đối: đường số lượng của
    lớp dương luôn bị lớp âm lấn át và không nói lên điều gì.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 4))

    hourly = add_hour(df).groupby("hour")[TARGET].agg(["size", "sum"])
    rate = hourly["sum"] / hourly["size"]

    ax.bar(rate.index, rate.to_numpy() * 100, color=CLASS_PALETTE[1], alpha=0.8)
    ax.axhline(
        df[TARGET].mean() * 100,
        ls="--",
        c="grey",
        lw=1,
        label=f"tỷ lệ chung {df[TARGET].mean():.3%}",
    )
    ax.set_xticks(range(0, 24))
    ax.set_xlabel("Giờ trong ngày")
    ax.set_ylabel("Tỷ lệ gian lận (%)")
    ax.set_title("Tỷ lệ gian lận theo giờ")
    ax.legend()

    secondary = ax.twinx()
    secondary.plot(hourly.index, hourly["size"], color="#4C78A8", lw=1, alpha=0.6)
    secondary.set_ylabel("Tổng số giao dịch", color="#4C78A8")
    return ax.figure


def plot_correlation_matrix(df, ax=None):
    """EDA 4 — xác nhận V1–V28 gần như trực giao (hệ quả của PCA)."""
    if ax is None:
        _, ax = plt.subplots(figsize=(9, 7))

    columns = [*V_COLUMNS, "Amount"]
    corr = df[columns].corr()
    sns.heatmap(corr, cmap="RdBu_r", center=0, vmin=-1, vmax=1, ax=ax, square=True, cbar_kws={"shrink": 0.7})
    ax.set_title("Ma trận tương quan giữa các đặc trưng")
    return ax.figure


def plot_target_correlation(df, top_n=15, ax=None):
    """EDA 5 — xếp hạng đặc trưng theo tương quan với nhãn."""
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 5))

    corr = df[[*V_COLUMNS, "Amount", TARGET]].corr()[TARGET].drop(TARGET)
    ranked = corr.reindex(corr.abs().sort_values(ascending=False).index).head(top_n)

    colors = [CLASS_PALETTE[1] if v < 0 else CLASS_PALETTE[0] for v in ranked]
    ax.barh(ranked.index[::-1], ranked.to_numpy()[::-1], color=colors[::-1])
    ax.axvline(0, c="black", lw=0.8)
    ax.set_xlabel(f"Tương quan Pearson với {TARGET}")
    ax.set_title(f"{top_n} đặc trưng tương quan mạnh nhất với nhãn")
    return ax.figure


def plot_feature_densities(df, features=None, top_n=6, axes=None):
    """EDA 6 — mức độ chồng lấn thực tế của các đặc trưng phân tách mạnh nhất."""
    if features is None:
        corr = df[[*V_COLUMNS, TARGET]].corr()[TARGET].drop(TARGET).abs()
        features = list(corr.sort_values(ascending=False).head(top_n).index)

    n_cols = 3
    n_rows = int(np.ceil(len(features) / n_cols))
    if axes is None:
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3 * n_rows))
    else:
        fig = np.ravel(axes)[0].figure

    for ax, feature in zip(np.ravel(axes), features):
        for label, group in df.groupby(TARGET):
            sns.kdeplot(
                group[feature],
                ax=ax,
                fill=True,
                alpha=0.4,
                label=CLASS_LABELS[label],
                color=CLASS_PALETTE[label],
                warn_singular=False,
            )
        ax.set_title(feature)
        ax.set_xlabel("")
        ax.legend(fontsize="small")

    for ax in np.ravel(axes)[len(features):]:
        ax.set_visible(False)

    fig.suptitle("Phân bố hai lớp ở các đặc trưng phân tách mạnh nhất")
    fig.tight_layout()
    return fig


def plot_amount_boxplot(df, ax=None):
    """EDA 7 — trung vị và ngoại lai của số tiền theo nhãn."""
    if ax is None:
        _, ax = plt.subplots(figsize=(5, 4))

    data = [df.loc[df[TARGET] == label, "Amount"] for label in (0, 1)]
    ax.boxplot(data, tick_labels=[CLASS_LABELS[0], CLASS_LABELS[1]], showfliers=True)
    ax.set_yscale("symlog")
    ax.set_ylabel("Amount (thang symlog)")
    ax.set_title("Số tiền giao dịch theo nhãn")
    return ax.figure


def plot_projection(df, n_negative=2000, random_state=42, ax=None):
    """EDA 8 — chiếu t-SNE trên mẫu con cân bằng: gian lận có tách thành cụm không.

    Lấy mẫu con vì t-SNE trên 284.000 điểm là bất khả thi và cũng vô ích — lớp
    dương sẽ chìm hoàn toàn.
    """
    from sklearn.manifold import TSNE

    if ax is None:
        _, ax = plt.subplots(figsize=(6, 5))

    fraud = df[df[TARGET] == 1]
    legit = df[df[TARGET] == 0].sample(
        n=min(n_negative, int((df[TARGET] == 0).sum())), random_state=random_state
    )
    sample = pd.concat([fraud, legit])

    embedded = TSNE(
        n_components=2, random_state=random_state, init="pca", perplexity=30
    ).fit_transform(sample[V_COLUMNS].to_numpy())

    for label in (0, 1):
        mask = (sample[TARGET] == label).to_numpy()
        ax.scatter(
            embedded[mask, 0],
            embedded[mask, 1],
            s=8 if label == 0 else 18,
            alpha=0.4 if label == 0 else 0.9,
            label=CLASS_LABELS[label],
            color=CLASS_PALETTE[label],
        )
    ax.set_title(f"Chiếu t-SNE ({len(fraud)} gian lận + {len(legit)} hợp lệ)")
    ax.legend()
    return ax.figure


# --------------------------------------------------------------------------
# Biểu đồ đánh giá mô hình
# --------------------------------------------------------------------------

def plot_pr_curve(y_true, y_scores, label="model", ax=None):
    """Đường Precision–Recall kèm đường cơ sở bằng tỷ lệ lớp dương (04 §5.2)."""
    if ax is None:
        _, ax = plt.subplots(figsize=(5, 4))

    precision, recall, _ = precision_recall_curve(y_true, y_scores)
    ax.plot(recall, precision, label=label)

    baseline = float(np.mean(np.asarray(y_true) == 1))
    ax.axhline(baseline, ls="--", c="grey", lw=1, label=f"đường cơ sở = {baseline:.5f}")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Đường Precision–Recall")
    ax.legend()
    return ax.figure


def plot_pr_curves(curves, y_true=None, ax=None):
    """Vẽ chồng nhiều đường PR: ``curves`` là dict {tên: điểm rủi ro}."""
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 5))

    for name, scores in curves.items():
        precision, recall, _ = precision_recall_curve(y_true, scores)
        ax.plot(recall, precision, label=name, lw=1.6)

    baseline = float(np.mean(np.asarray(y_true) == 1))
    ax.axhline(baseline, ls="--", c="grey", lw=1, label=f"đường cơ sở = {baseline:.5f}")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("So sánh các chiến lược xử lý mất cân bằng")
    ax.legend(fontsize="small")
    return ax.figure


def plot_roc_curve(y_true, y_scores, label="model", ax=None):
    """Đường ROC — báo cáo kèm để đối chiếu, không dùng làm metric chính."""
    if ax is None:
        _, ax = plt.subplots(figsize=(5, 4))

    fpr, tpr, _ = roc_curve(y_true, y_scores)
    ax.plot(fpr, tpr, label=label)
    ax.plot([0, 1], [0, 1], ls="--", c="grey", lw=1, label="ngẫu nhiên")
    ax.set_xlabel("Tỷ lệ dương tính giả")
    ax.set_ylabel("Tỷ lệ dương tính thật")
    ax.set_title("Đường ROC")
    ax.legend()
    return ax.figure


def plot_confusion(y_true, y_pred, ax=None):
    """Bốn con số tuyệt đối — không giấu được gì."""
    if ax is None:
        _, ax = plt.subplots(figsize=(4, 4))

    matrix = confusion_matrix(y_true, y_pred, labels=[0, 1])
    ConfusionMatrixDisplay(matrix, display_labels=list(CLASS_LABELS.values())).plot(
        ax=ax, colorbar=False, values_format=","
    )
    ax.set_title("Ma trận nhầm lẫn")
    return ax.figure


def plot_threshold_sweep(y_true, y_scores, ax=None):
    """Precision / recall / F1 thay đổi thế nào theo ngưỡng."""
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 4))

    table = cost_curve(y_true, y_scores)
    ax.plot(table["threshold"], table["precision"], label="precision")
    ax.plot(table["threshold"], table["recall"], label="recall")
    ax.plot(table["threshold"], table["f1"], label="F1")
    ax.set_xscale("log")
    ax.set_xlabel("Ngưỡng (thang log)")
    ax.set_ylabel("Giá trị")
    ax.set_title("Ảnh hưởng của ngưỡng quyết định")
    ax.legend()
    return ax.figure


def plot_cost_curve(y_true, y_scores, *, cost_fn=None, cost_fp=None, ax=None):
    """Đường cong chi phí kèm vị trí cực tiểu — hình trung tâm của phần ngưỡng.

    Đánh dấu cả ngưỡng mặc định 0,5 để người đọc thấy ngay khoảng cách giữa
    "không chọn gì" và "chọn theo chi phí".
    """
    from .config import DEFAULT_COST_FN, DEFAULT_COST_FP

    cost_fn = DEFAULT_COST_FN if cost_fn is None else cost_fn
    cost_fp = DEFAULT_COST_FP if cost_fp is None else cost_fp

    if ax is None:
        _, ax = plt.subplots(figsize=(7, 4))

    table = cost_curve(y_true, y_scores, cost_fn=cost_fn, cost_fp=cost_fp)
    optimal = table.loc[table["expected_cost"].idxmin()]

    ax.plot(table["threshold"], table["expected_cost"], lw=1.8)
    ax.axvline(
        optimal["threshold"],
        c=CLASS_PALETTE[1],
        ls="--",
        label=f"cực tiểu τ = {optimal['threshold']:.4f}",
    )
    ax.axvline(0.5, c="grey", ls=":", label="mặc định τ = 0,5")
    ax.set_xscale("log")
    ax.set_xlabel("Ngưỡng (thang log)")
    ax.set_ylabel(f"Chi phí kỳ vọng (FN={cost_fn:g}, FP={cost_fp:g})")
    ax.set_title("Chi phí kỳ vọng theo ngưỡng")
    ax.legend()
    return ax.figure
