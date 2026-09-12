"""Kiểm định thống kê so sánh hai lớp cho từng đặc trưng (FR-03, 04 §5.4).

Ba quyết định phương pháp, đều vì cùng một lý do — hai nhóm chênh lệch kích thước
600 lần:

* **Mann–Whitney U** thay vì t-test. Không giả định phân phối chuẩn, và với
  n₁ = 473 so với n₀ = 283.253 thì giả định phương sai bằng nhau của t-test
  không có gì bảo đảm.
* **Hiệu chỉnh Benjamini–Hochberg.** 30 kiểm định chạy cùng lúc ở mức α = 0,05
  thì kỳ vọng khoảng 1,5 kết quả dương giả. BH kiểm soát tỷ lệ phát hiện sai
  (FDR) thay vì xác suất sai họ (FWER) như Bonferroni — phù hợp hơn cho bước
  sàng lọc đặc trưng, nơi bỏ sót tốn kém hơn một chút dương giả.
* **Effect size bắt buộc đi kèm p-value.** Với n gần 284.000, gần như mọi khác
  biệt đều "có ý nghĩa thống kê"; p-value chỉ nói khác biệt *có thật*, còn
  Cohen's d và Cliff's delta mới nói khác biệt *lớn đến đâu*.

Cohen's d và Cliff's delta có thể xếp hạng khác nhau, và đó là thông tin chứ
không phải mâu thuẫn: d so sánh trung bình nên nhạy với đuôi phân bố, delta so
sánh thứ hạng nên nhạy với phần thân. Đặc trưng nào lệch dấu giữa hai chỉ số là
đặc trưng có trung bình và trung vị đi ngược nhau.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu

from .config import TARGET
from .features import FEATURE_ORDER, RAW_REQUIRED_COLUMNS

#: Ngưỡng α mặc định, áp lên p-value ĐÃ hiệu chỉnh
DEFAULT_ALPHA = 0.05

#: Mốc quy ước của Cohen cho |d|
COHEN_BOUNDS = ((0.2, "không đáng kể"), (0.5, "nhỏ"), (0.8, "trung bình"))

#: Mốc quy ước của Romano và cộng sự cho |delta|
CLIFF_BOUNDS = ((0.147, "không đáng kể"), (0.33, "nhỏ"), (0.474, "trung bình"))


def benjamini_hochberg(p_values) -> np.ndarray:
    """p-value hiệu chỉnh theo thủ tục step-up Benjamini–Hochberg.

    Trả về mảng cùng thứ tự với đầu vào. Giá trị trả về là *p-value hiệu chỉnh*
    (còn gọi là q-value): so trực tiếp với α để kết luận, không cần so với
    ``α·i/m`` nữa.

    Thủ tục: sắp tăng dần, nhân mỗi ``p₍ᵢ₎`` với ``m/i``, rồi lấy cực tiểu tích
    luỹ từ phải sang trái để giữ tính đơn điệu (một đặc trưng có p thô nhỏ hơn
    không bao giờ được hiệu chỉnh thành lớn hơn), cuối cùng cắt về [0, 1].
    """
    p = np.asarray(p_values, dtype="float64")
    if p.ndim != 1:
        raise ValueError("p_values phải là mảng một chiều")
    if p.size == 0:
        return p.copy()
    if np.isnan(p).any():
        raise ValueError("p_values chứa NaN")

    m = p.size
    order = np.argsort(p, kind="stable")
    scaled = p[order] * m / np.arange(1, m + 1)
    monotone = np.minimum.accumulate(scaled[::-1])[::-1]

    adjusted = np.empty(m, dtype="float64")
    adjusted[order] = np.clip(monotone, 0.0, 1.0)
    return adjusted


def cohens_d(x, y) -> float:
    """Chênh lệch trung bình chuẩn hoá theo độ lệch chuẩn gộp.

    ``d = (mean(x) - mean(y)) / s_pooled``. Dấu dương nghĩa là nhóm ``x``
    (gian lận) có giá trị cao hơn.
    """
    x = np.asarray(x, dtype="float64")
    y = np.asarray(y, dtype="float64")
    n_x, n_y = x.size, y.size
    if n_x < 2 or n_y < 2:
        raise ValueError("mỗi nhóm cần ít nhất 2 quan sát")

    pooled_var = ((n_x - 1) * x.var(ddof=1) + (n_y - 1) * y.var(ddof=1)) / (n_x + n_y - 2)
    if pooled_var == 0:
        return 0.0
    return float((x.mean() - y.mean()) / np.sqrt(pooled_var))


def cliffs_delta_from_u(u_statistic: float, n_x: int, n_y: int) -> float:
    """Cliff's delta suy ra từ thống kê U của Mann–Whitney.

    ``delta = 2U/(n_x·n_y) − 1``, nằm trong [−1, 1]. Đây là xác suất một mẫu
    ngẫu nhiên của nhóm ``x`` lớn hơn một mẫu của nhóm ``y``, trừ đi xác suất
    ngược lại — không phụ thuộc thang đo và không bị ngoại lai kéo lệch.

    Dùng lại ``U`` đã tính thay vì so sánh từng cặp, vì so cặp trực tiếp trên
    473 × 283.253 là hơn 134 triệu phép so sánh.
    """
    if n_x <= 0 or n_y <= 0:
        raise ValueError("kích thước nhóm phải dương")
    return float(2.0 * u_statistic / (n_x * n_y) - 1.0)


def _label(value: float, bounds) -> str:
    magnitude = abs(value)
    for limit, name in bounds:
        if magnitude < limit:
            return name
    return "lớn"


def effect_size_label(d: float) -> str:
    """Nhãn quy ước cho |Cohen's d|."""
    return _label(d, COHEN_BOUNDS)


def cliff_label(delta: float) -> str:
    """Nhãn quy ước cho |Cliff's delta|."""
    return _label(delta, CLIFF_BOUNDS)


def pearson_from_cohens_d(d, positive_rate: float) -> np.ndarray:
    """Tương quan điểm-nhị phân tương ứng với một Cohen's d cho trước.

    ``r = d / sqrt(d² + 1/(p·q))`` với ``p`` là tỷ lệ lớp dương. Đây là lý do
    tương quan Pearson thô gần như vô dụng trên bộ dữ liệu này: khi ``p·q`` rất
    bé, số hạng ``1/(p·q)`` áp đảo ``d²``, nên ``|r| ≈ |d|·sqrt(p·q)``. Với
    p ≈ 0,0017 thì ``sqrt(p·q) ≈ 0,041`` — một đặc trưng tách hai lớp bằng 8 độ
    lệch chuẩn vẫn chỉ cho tương quan khoảng 0,33.

    Hàm này dùng để *chứng minh* điều đó trong notebook 02, không dùng để thay
    thế phép tính tương quan thật.
    """
    d = np.asarray(d, dtype="float64")
    pq = positive_rate * (1.0 - positive_rate)
    if pq <= 0:
        raise ValueError("positive_rate phải nằm trong khoảng (0, 1)")
    return d / np.sqrt(d**2 + 1.0 / pq)


def compare_groups(
    df: pd.DataFrame,
    feature: str,
    *,
    target: str = TARGET,
) -> dict:
    """Toàn bộ số liệu so sánh hai lớp cho MỘT đặc trưng."""
    values = df[feature].to_numpy(dtype="float64")
    labels = df[target].to_numpy()

    fraud = values[labels == 1]
    legit = values[labels == 0]

    u_statistic, p_value = mannwhitneyu(fraud, legit, alternative="two-sided")

    return {
        "feature": feature,
        "n_fraud": int(fraud.size),
        "n_legit": int(legit.size),
        "mean_fraud": float(fraud.mean()),
        "mean_legit": float(legit.mean()),
        "median_fraud": float(np.median(fraud)),
        "median_legit": float(np.median(legit)),
        "u_statistic": float(u_statistic),
        "p_value": float(p_value),
        "cohens_d": cohens_d(fraud, legit),
        "cliffs_delta": cliffs_delta_from_u(u_statistic, fraud.size, legit.size),
        "pearson_r": float(np.corrcoef(values, labels.astype("float64"))[0, 1]),
    }


def feature_ranking(
    df: pd.DataFrame,
    features=None,
    *,
    target: str = TARGET,
    alpha: float = DEFAULT_ALPHA,
) -> pd.DataFrame:
    """Bảng xếp hạng đặc trưng: Mann–Whitney + BH + effect size.

    Tham số
    -------
    df
        DataFrame có cột nhãn ``target`` và các cột đặc trưng.
    features
        Danh sách cột cần kiểm định. Mặc định là 30 cột của hợp đồng dữ liệu vào
        (``RAW_REQUIRED_COLUMNS``) — đúng con số 30 kiểm định mà 04 §5.4 nêu.
    alpha
        Mức ý nghĩa áp lên p-value **đã hiệu chỉnh**.

    Trả về
    ------
    DataFrame sắp theo ``|cohens_d|`` giảm dần, một dòng mỗi đặc trưng. Cột
    ``p_adjusted`` và ``cohens_d`` là hai cột mà notebook 07 đối chiếu với SHAP.
    """
    if target not in df.columns:
        raise KeyError(f"thiếu cột nhãn {target!r}")

    features = list(RAW_REQUIRED_COLUMNS if features is None else features)
    missing = [c for c in features if c not in df.columns]
    if missing:
        raise KeyError("thiếu cột đặc trưng: " + ", ".join(missing))

    table = pd.DataFrame([compare_groups(df, f, target=target) for f in features])

    table["p_adjusted"] = benjamini_hochberg(table["p_value"].to_numpy())
    table["significant"] = table["p_adjusted"] < alpha
    table["abs_cohens_d"] = table["cohens_d"].abs()
    table["abs_cliffs_delta"] = table["cliffs_delta"].abs()
    table["effect_size"] = table["cohens_d"].map(effect_size_label)
    table["cliff_size"] = table["cliffs_delta"].map(cliff_label)
    table["in_model"] = table["feature"].isin(FEATURE_ORDER)

    table = table.sort_values("abs_cohens_d", ascending=False).reset_index(drop=True)
    table.insert(0, "rank_cohens_d", np.arange(1, len(table) + 1))
    table["rank_cliffs_delta"] = (
        table["abs_cliffs_delta"].rank(ascending=False, method="first").astype(int)
    )
    table["rank_pearson"] = (
        table["pearson_r"].abs().rank(ascending=False, method="first").astype(int)
    )

    column_order = [
        "rank_cohens_d",
        "feature",
        "in_model",
        "n_fraud",
        "n_legit",
        "mean_fraud",
        "mean_legit",
        "median_fraud",
        "median_legit",
        "u_statistic",
        "p_value",
        "p_adjusted",
        "significant",
        "cohens_d",
        "abs_cohens_d",
        "effect_size",
        "cliffs_delta",
        "abs_cliffs_delta",
        "cliff_size",
        "pearson_r",
        "rank_cliffs_delta",
        "rank_pearson",
    ]
    return table[column_order]


def significance_summary(ranking: pd.DataFrame, *, alpha: float = DEFAULT_ALPHA) -> str:
    """Câu tóm tắt để dán thẳng vào báo cáo."""
    n_total = len(ranking)
    n_raw = int((ranking["p_value"] < alpha).sum())
    n_adjusted = int((ranking["p_adjusted"] < alpha).sum())
    n_large = int((ranking["abs_cohens_d"] >= 0.8).sum())

    return (
        f"{n_total} kiểm định Mann–Whitney U, α = {alpha}. "
        f"Trước hiệu chỉnh: {n_raw} có ý nghĩa. "
        f"Sau hiệu chỉnh Benjamini–Hochberg: {n_adjusted} có ý nghĩa "
        f"({n_raw - n_adjusted} bị loại). "
        f"Trong đó {n_large} đặc trưng có |Cohen's d| ≥ 0,8 (hiệu ứng lớn)."
    )
