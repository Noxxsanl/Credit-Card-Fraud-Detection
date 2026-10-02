"""Dựng, ghi và kiểm tra hiện vật mô hình — ranh giới giữa huấn luyện và phục vụ.

Notebook 08 là nơi DUY NHẤT ghi vào ``models/`` và ``data/sample_pool.json``; API
chỉ đọc (AR-01). Module này giữ cho hai phía nói cùng một ngôn ngữ:

* các hàm ``*_payload`` dựng nội dung JSON theo đúng cấu trúc ở 06 §6;
* các hàm ``validate_*`` kiểm tra ngược lại cấu trúc đó. Notebook gọi trước khi
  ghi, kiểm thử gọi sau khi ghi, và API sẽ gọi lúc khởi động (T-44).

JSON ghi ra là JSON **chặt**: không có ``NaN`` hay ``Infinity``. ``JSON.parse`` của
trình duyệt từ chối hai giá trị đó, mà UI-D1 đọc ``metrics.json`` ngay trong trình
duyệt.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import re
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path

import numpy as np
import pandas as pd

from .config import DATASET_DAYS, DEFAULT_COST_FN, DEFAULT_COST_FP, TEST_SIZE
from .features import RAW_REQUIRED_COLUMNS
from .threshold import NAIVE_THRESHOLD, cost_curve

#: Bốn nhóm của thư viện mẫu (06 §6.3), theo thứ tự hiển thị
SAMPLE_CATEGORIES = ("fraud_easy", "fraud_hard", "legit_easy", "legit_hard")

#: Mỗi nhóm "hard" phải có ít nhất chừng này mẫu (T-37)
MIN_HARD_SAMPLES = 20

#: Chỉ số chính phải kèm khoảng tin cậy bootstrap (AC-M6)
HEADLINE_CI_METRICS = ("pr_auc", "roc_auc", "recall", "precision")

THRESHOLD_KEYS = (
    "default_threshold",
    "selection_method",
    "cost_false_negative",
    "cost_false_positive",
    "alternatives",
    "model_version",
    "trained_at",
)

METRICS_KEYS = (
    "model_version",
    "trained_at",
    "dataset",
    "headline",
    "confusion_at_default",
    "pr_curve",
    "roc_curve",
    "cost_curve",
    "test_scores",
    "grid_results",
    "shap_global",
    "split_comparison",
)

_TIMESTAMP = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$")

#: Thư viện mà việc nạp lại ``.joblib`` phụ thuộc vào — ghi kèm hiện vật để khi
#: API báo lỗi unpickle thì biết ngay lệch phiên bản nào
_PACKAGES = ("numpy", "pandas", "scikit-learn", "imbalanced-learn", "xgboost", "shap", "joblib")


# --------------------------------------------------------------------------
# Tiện ích
# --------------------------------------------------------------------------

def utc_timestamp(moment: datetime | None = None) -> str:
    """Thời điểm dạng ISO 8601, múi UTC, hậu tố ``Z`` (quy ước 05 §5)."""
    moment = moment or datetime.now(timezone.utc)
    return moment.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def environment_info() -> dict:
    """Phiên bản Python và thư viện lúc xuất hiện vật, kèm số lõi CPU.

    Số lõi có mặt vì XGBoost ``hist`` cho điểm lệch nhẹ theo số luồng (10 §6).
    """
    packages = {}
    for name in _PACKAGES:
        try:
            packages[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            packages[name] = None
    return {"python": platform.python_version(), "cpu_count": os.cpu_count(), "packages": packages}


def to_jsonable(obj):
    """Đổi cấu trúc lồng nhau chứa kiểu NumPy/pandas sang kiểu JSON thuần.

    Số thực không hữu hạn thành ``None`` để JSON ghi ra luôn là JSON chặt.
    """
    if isinstance(obj, dict):
        return {str(key): to_jsonable(value) for key, value in obj.items()}
    if isinstance(obj, pd.DataFrame):
        return [to_jsonable(row) for row in obj.to_dict("records")]
    if isinstance(obj, (list, tuple, np.ndarray, pd.Series)):
        return [to_jsonable(value) for value in obj]
    if isinstance(obj, (bool, np.bool_)):
        return bool(obj)
    if isinstance(obj, (int, np.integer)):
        return int(obj)
    if isinstance(obj, (float, np.floating)):
        value = float(obj)
        return value if np.isfinite(value) else None
    if obj is None or isinstance(obj, str):
        return obj
    if isinstance(obj, Path):
        return obj.as_posix()
    raise TypeError(f"Không đổi được sang JSON: {type(obj).__name__}")


def write_json(path, payload, *, indent: int | None = None) -> Path:
    """Ghi JSON chặt, nguyên tử: ghi ra tệp tạm rồi đổi tên.

    API nạp hiện vật lúc khởi động; ghi nguyên tử bảo đảm nó không bao giờ đọc
    phải một tệp đang ghi dở.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(to_jsonable(payload), ensure_ascii=False, allow_nan=False, indent=indent)
    tmp = path.with_name(path.name + ".tmp")
    # Ghi byte chứ không ghi văn bản: Windows sẽ tự đổi \n thành \r\n, và tệp sinh ra trên hai
    # hệ điều hành sẽ khác nhau dù nội dung như nhau
    tmp.write_bytes(text.encode("utf-8"))
    os.replace(tmp, path)
    return path


def read_json(path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def score_fingerprint(scores) -> str:
    """Dấu vân tay 16 ký tự của một mảng điểm rủi ro.

    Hai lần chạy cho cùng dấu vân tay nghĩa là mô hình cho điểm giống nhau tới
    từng bit trên cùng một tập — mạnh hơn nhiều so với so PR-AUC (T-38).
    """
    array = np.ascontiguousarray(np.asarray(scores, dtype="float64"))
    return hashlib.sha256(array.tobytes()).hexdigest()[:16]


# --------------------------------------------------------------------------
# threshold.json — 06 §6.1
# --------------------------------------------------------------------------

def threshold_payload(
    *,
    default_threshold: float,
    alternatives: dict[str, float],
    model_version: str,
    trained_at: str,
    cost_fn: float = DEFAULT_COST_FN,
    cost_fp: float = DEFAULT_COST_FP,
    selection_method: str = "min_expected_cost",
    selected_on: str = "out_of_fold",
    constraints: dict | None = None,
) -> dict:
    """Nội dung ``threshold.json``: ngưỡng mặc định là cấu hình, không phải mô hình (AR-03)."""
    payload = {
        "default_threshold": float(default_threshold),
        "selection_method": selection_method,
        "selected_on": selected_on,
        "cost_false_negative": float(cost_fn),
        "cost_false_positive": float(cost_fp),
        "alternatives": {name: float(value) for name, value in alternatives.items()},
        "model_version": model_version,
        "trained_at": trained_at,
    }
    if constraints:
        payload["constraints"] = dict(constraints)
    return payload


def validate_threshold(payload: dict) -> list[str]:
    """Danh sách vấn đề của ``threshold.json`` (rỗng nếu đúng cấu trúc)."""
    problems = [f"thiếu khóa: {key}" for key in THRESHOLD_KEYS if key not in payload]
    if problems:
        return problems

    tau = payload["default_threshold"]
    if not (isinstance(tau, (int, float)) and 0 < tau < 1):
        problems.append(f"default_threshold phải nằm trong (0, 1): {tau!r}")
    for key in ("cost_false_negative", "cost_false_positive"):
        if not (isinstance(payload[key], (int, float)) and payload[key] > 0):
            problems.append(f"{key} phải dương: {payload[key]!r}")

    alternatives = payload["alternatives"]
    if not isinstance(alternatives, dict) or not alternatives:
        problems.append("alternatives phải là dict khác rỗng")
    else:
        for name in ("min_expected_cost", "default_naive"):
            if name not in alternatives:
                problems.append(f"alternatives thiếu phương án {name}")
        for name, value in alternatives.items():
            if not (isinstance(value, (int, float)) and 0 < value < 1):
                problems.append(f"alternatives.{name} phải nằm trong (0, 1): {value!r}")
        chosen = alternatives.get(payload["selection_method"])
        if chosen is not None and chosen != tau:
            problems.append(
                f"default_threshold {tau} khác phương án {payload['selection_method']} = {chosen}"
            )

    if not _TIMESTAMP.match(str(payload["trained_at"])):
        problems.append(f"trained_at không theo ISO 8601 UTC: {payload['trained_at']!r}")
    return problems


# --------------------------------------------------------------------------
# metrics.json — 06 §6.2
# --------------------------------------------------------------------------

def headline_block(raw: dict) -> dict:
    """Rút gọn kết quả của ``evaluate.headline_metrics`` về dạng ở 06 §6.2."""
    out = {}
    for metric in HEADLINE_CI_METRICS:
        out[metric] = {key: raw[metric][key] for key in ("value", "ci_low", "ci_high")}
    out["f1"] = {"value": raw["f1"]["value"]}
    baseline = raw["baseline_pr_auc"]
    out["baseline_pr_auc"] = baseline["value"] if isinstance(baseline, dict) else baseline
    if "n_boot" in raw["pr_auc"]:
        out["n_boot"] = raw["pr_auc"]["n_boot"]
    return out


def cost_curve_points(
    y_true,
    y_scores,
    *,
    cost_fn: float = DEFAULT_COST_FN,
    cost_fp: float = DEFAULT_COST_FP,
    n_steps: int = 200,
    sample_fraction: float = TEST_SIZE,
    days: float = DATASET_DAYS,
) -> list[dict]:
    """Khoảng 200 điểm của đường chi phí, cùng lưới ngưỡng mà API-12 dùng."""
    table = cost_curve(
        y_true,
        y_scores,
        cost_fn=cost_fn,
        cost_fp=cost_fp,
        n_steps=n_steps,
        sample_fraction=sample_fraction,
        days=days,
    )
    table = table.rename(columns={"expected_cost": "cost"})
    columns = ["threshold", "cost", "alerts", "alerts_per_day", "tp", "fp", "fn", "precision", "recall"]
    return to_jsonable(table[columns])


def metrics_payload(
    *,
    model_version: str,
    trained_at: str,
    dataset: dict,
    headline: dict,
    confusion_at_default: dict,
    curves: dict,
    cost_curve: list[dict],
    y_true,
    y_scores,
    grid_results: list[dict],
    shap_global: list[dict],
    split_comparison: dict,
    **extra,
) -> dict:
    """Nội dung ``metrics.json``, khóa theo đúng thứ tự ở 06 §6.2.

    ``test_scores`` giữ nguyên độ chính xác của float64: làm tròn sẽ làm lệch
    TP/FP/FN giữa trình duyệt và máy chủ ở đúng những điểm nằm sát ngưỡng (TC-12).
    Các mục bổ sung cho UI-04 đi qua ``extra`` và nằm sau các khóa bắt buộc.
    """
    y_true = np.asarray(y_true).astype(int)
    y_scores = np.asarray(y_scores, dtype="float64")
    payload = {
        "model_version": model_version,
        "trained_at": trained_at,
        "dataset": dataset,
        "headline": headline,
        "confusion_at_default": confusion_at_default,
        "pr_curve": curves["pr_curve"],
        "roc_curve": curves["roc_curve"],
        "cost_curve": cost_curve,
        "test_scores": {"y_true": y_true.tolist(), "y_score": y_scores.tolist()},
        "grid_results": grid_results,
        "shap_global": shap_global,
        "split_comparison": split_comparison,
    }
    payload.update(extra)
    return payload


def validate_metrics(payload: dict) -> list[str]:
    """Danh sách vấn đề của ``metrics.json`` (rỗng nếu đúng cấu trúc)."""
    problems = [f"thiếu khóa: {key}" for key in METRICS_KEYS if key not in payload]
    if problems:
        return problems

    headline = payload["headline"]
    for metric in HEADLINE_CI_METRICS:
        block = headline.get(metric)
        if not isinstance(block, dict) or not {"value", "ci_low", "ci_high"} <= block.keys():
            problems.append(f"headline.{metric} phải có value, ci_low, ci_high")
        elif not 0 <= block["ci_low"] <= block["ci_high"] <= 1 or not 0 <= block["value"] <= 1:
            # Không đòi value nằm trong khoảng: khoảng bách phân vị của bootstrap
            # không bảo đảm chứa ước lượng điểm
            problems.append(f"headline.{metric}: giá trị hoặc khoảng tin cậy nằm ngoài [0, 1]")
    if "value" not in headline.get("f1", {}):
        problems.append("headline.f1 thiếu value")
    if not isinstance(headline.get("baseline_pr_auc"), (int, float)):
        problems.append("headline.baseline_pr_auc phải là số")

    scores = payload["test_scores"]
    y_true, y_score = scores.get("y_true", []), scores.get("y_score", [])
    if not y_true or len(y_true) != len(y_score):
        problems.append("test_scores.y_true và y_score phải khác rỗng và cùng độ dài")
    else:
        if set(y_true) - {0, 1}:
            problems.append("test_scores.y_true chỉ được chứa 0 và 1")
        if min(y_score) < 0 or max(y_score) > 1:
            problems.append("test_scores.y_score phải nằm trong [0, 1]")
        dataset = payload["dataset"]
        if dataset.get("n_test") != len(y_true):
            problems.append("dataset.n_test khác độ dài test_scores")
        if dataset.get("n_fraud_test") != sum(y_true):
            problems.append("dataset.n_fraud_test khác số nhãn 1 trong test_scores")
        confusion = payload["confusion_at_default"]
        if sum(confusion.get(k, 0) for k in ("tp", "fp", "fn", "tn")) != len(y_true):
            problems.append("confusion_at_default không cộng về đúng n_test")

    pr, roc = payload["pr_curve"], payload["roc_curve"]
    if not pr.get("recall") or not len(pr["recall"]) == len(pr.get("precision", [])) == len(pr.get("thresholds", [])):
        problems.append("pr_curve phải có recall, precision, thresholds cùng độ dài")
    if not roc.get("fpr") or len(roc["fpr"]) != len(roc.get("tpr", [])):
        problems.append("roc_curve phải có fpr, tpr cùng độ dài")

    if not payload["cost_curve"] or not {"threshold", "cost", "alerts"} <= payload["cost_curve"][0].keys():
        problems.append("cost_curve phải là danh sách {threshold, cost, alerts}")
    if not payload["grid_results"] or not {"model", "strategy", "pr_auc_mean", "pr_auc_std"} <= payload["grid_results"][0].keys():
        problems.append("grid_results phải là danh sách {model, strategy, pr_auc_mean, pr_auc_std, …}")
    if not payload["shap_global"] or not {"feature", "mean_abs_shap"} <= payload["shap_global"][0].keys():
        problems.append("shap_global phải là danh sách {feature, mean_abs_shap}")
    if not {"random_stratified", "temporal"} <= payload["split_comparison"].keys():
        problems.append("split_comparison phải có random_stratified và temporal")
    if not _TIMESTAMP.match(str(payload["trained_at"])):
        problems.append(f"trained_at không theo ISO 8601 UTC: {payload['trained_at']!r}")
    return problems


# --------------------------------------------------------------------------
# models/oof_scores.npz — 06 §6.4
# --------------------------------------------------------------------------

def write_oof_scores(path, y_true, y_scores, *, train_fraction: float, days: float = DATASET_DAYS) -> Path:
    """Điểm out-of-fold của tập huấn luyện — nơi API-12 chọn ngưỡng (ML-08).

    Giữ nguyên kiểu float32 của ``predict_proba``: ngưỡng ứng viên là chính các giá
    trị này, đổi kiểu là lệch nghiệm ở chữ số cuối so với ``threshold.json``.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.stem + ".tmp.npz")
    np.savez_compressed(
        tmp,
        y_true=np.asarray(y_true).astype("int8"),
        y_score=np.asarray(y_scores),
        train_fraction=np.float64(train_fraction),
        days=np.float64(days),
    )
    os.replace(tmp, path)
    return path


def read_oof_scores(path) -> dict:
    with np.load(path) as npz:
        return {
            "y_true": npz["y_true"].astype(int),
            "y_score": npz["y_score"],
            "train_fraction": float(npz["train_fraction"]),
            "days": float(npz["days"]),
        }


def validate_oof_scores(oof: dict, *, fingerprint: str | None = None) -> list[str]:
    """Vấn đề của ``oof_scores.npz``; ``fingerprint`` lấy từ ``metrics.json`` nếu có."""
    y_true, y_score = oof["y_true"], oof["y_score"]
    problems = []
    if y_true.shape != y_score.shape or y_true.size == 0:
        problems.append("y_true và y_score phải khác rỗng và cùng độ dài")
    elif set(np.unique(y_true)) - {0, 1}:
        problems.append("y_true chỉ được chứa 0 và 1")
    elif y_score.min() < 0 or y_score.max() > 1:
        problems.append("y_score phải nằm trong [0, 1]")
    if not 0 < oof["train_fraction"] <= 1:
        problems.append(f"train_fraction phải nằm trong (0, 1]: {oof['train_fraction']}")
    if fingerprint is not None and not problems and score_fingerprint(y_score) != fingerprint:
        problems.append("dấu vân tay khác metrics.json — hai tệp không cùng một lần xuất")
    return problems


# --------------------------------------------------------------------------
# data/sample_pool.json — 06 §6.3
# --------------------------------------------------------------------------

def _evenly_spaced(indices: np.ndarray, scores: np.ndarray, k: int) -> np.ndarray:
    """``k`` phần tử trải đều theo thứ hạng điểm — tất định, phủ cả dải điểm."""
    ordered = indices[np.argsort(scores[indices], kind="stable")]
    if k >= ordered.size:
        return ordered
    if k <= 0:
        return ordered[:0]
    return ordered[np.unique(np.linspace(0, ordered.size - 1, k).round().astype(int))]


def _so(value: float, fmt: str = "g") -> str:
    """Số viết theo tiếng Việt, dấu phẩy thập phân — giao diện hiển thị nguyên văn các chuỗi này."""
    return format(value, fmt).replace(".", ",")


def _describe(category: str, score: float, reference: float, fraud_cutoff: float) -> str:
    if category == "fraud_easy":
        if score >= 0.9:
            return "Gian lận điển hình — điểm rủi ro rất cao"
        return f"Gian lận — điểm rủi ro cao, vượt cả ngưỡng {_so(fraud_cutoff)}"
    if category == "fraud_hard":
        if score < reference:
            return "Gian lận bị bỏ lọt — điểm dưới cả ngưỡng đề xuất τ*"
        return f"Gian lận ở vùng biên — chỉ bắt được khi hạ ngưỡng từ {_so(fraud_cutoff)} về τ*"
    if category == "legit_easy":
        if score < reference / 2:
            return "Hợp lệ điển hình — điểm rủi ro rất thấp"
        return "Hợp lệ sát ngưỡng — vẫn được cho qua ở τ*"
    if score >= fraud_cutoff:
        return f"Hợp lệ nhưng điểm rất cao — cảnh báo giả ngay cả ở ngưỡng {_so(fraud_cutoff)}"
    return "Hợp lệ nhưng điểm cao — cảnh báo giả ở ngưỡng đề xuất τ*"


def select_sample_pool(
    raw: pd.DataFrame,
    y_true,
    y_scores,
    *,
    reference_threshold: float,
    fraud_cutoff: float = NAIVE_THRESHOLD,
    n_fraud: int = 50,
    n_legit: int = 150,
    max_legit_hard: int = 40,
) -> list[dict]:
    """Chọn thư viện mẫu bốn nhóm cho buổi trình diễn (FR-25, T-37).

    Luật chia nhóm, với τ* = ``reference_threshold``:

    ``fraud_hard``
        Gian lận có điểm dưới ``fraud_cutoff`` (0,5): ngưỡng mặc định bỏ lọt.
        Gồm cả vụ bị bỏ lọt ở τ* lẫn vụ chỉ bắt được khi hạ ngưỡng về τ* — nhóm
        sau là thứ làm cảnh kéo thanh trượt ở buổi bảo vệ có ý nghĩa.
    ``fraud_easy``
        Gian lận còn lại, lấy trải đều theo thứ hạng điểm cho đủ ``n_fraud``.
    ``legit_hard``
        Hợp lệ có điểm từ τ* trở lên: cảnh báo giả ở ngưỡng đề xuất. Tối đa
        ``max_legit_hard`` mẫu, trải đều theo điểm.
    ``legit_easy``
        Hợp lệ còn lại, trải đều theo thứ hạng điểm cho đủ ``n_legit``.

    Hai nhóm "hard" lấy ngưỡng khác nhau vì chúng khó theo hai nghĩa khác nhau:
    gian lận khó là vụ mô hình không tự tin (điểm dưới 0,5), còn hợp lệ khó là
    giao dịch thật sự bị chặn nhầm ở chính sách đang chạy.

    Chọn tất định — không lấy mẫu ngẫu nhiên — nên chạy lại cho đúng cùng một
    thư viện. ``raw`` là dữ liệu thô 30 cột, cùng thứ tự với ``y_true`` và
    ``y_scores``; ``test_row`` của mỗi mẫu là vị trí dòng trong ``raw``.
    """
    y_true = np.asarray(y_true).astype(int)
    y_scores = np.asarray(y_scores, dtype="float64")
    if not (len(raw) == y_true.size == y_scores.size):
        raise ValueError("raw, y_true và y_scores phải cùng số dòng")
    missing = [c for c in RAW_REQUIRED_COLUMNS if c not in raw.columns]
    if missing:
        raise ValueError("raw thiếu cột: " + ", ".join(missing))

    rows = np.arange(y_true.size)
    fraud, legit = rows[y_true == 1], rows[y_true == 0]
    fraud_hard = fraud[y_scores[fraud] < fraud_cutoff]
    fraud_hard = _evenly_spaced(fraud_hard, y_scores, n_fraud)
    fraud_easy = _evenly_spaced(fraud[y_scores[fraud] >= fraud_cutoff], y_scores, n_fraud - fraud_hard.size)
    legit_hard = _evenly_spaced(legit[y_scores[legit] >= reference_threshold], y_scores, max_legit_hard)
    legit_easy = _evenly_spaced(
        legit[y_scores[legit] < reference_threshold], y_scores, n_legit - legit_hard.size
    )

    values = raw[RAW_REQUIRED_COLUMNS].to_numpy(dtype="float64")
    items = []
    groups = dict(zip(SAMPLE_CATEGORIES, (fraud_easy, fraud_hard, legit_easy, legit_hard)))
    for category, members in groups.items():
        for row in members[np.argsort(-y_scores[members], kind="stable")]:
            score = float(y_scores[row])
            items.append(
                {
                    "id": f"S-{len(items) + 1:03d}",
                    "category": category,
                    "label": int(y_true[row]),
                    "risk_score": score,
                    "description": _describe(category, score, reference_threshold, fraud_cutoff),
                    "test_row": int(row),
                    "features": dict(zip(RAW_REQUIRED_COLUMNS, map(float, values[row]))),
                }
            )
    return items


def sample_pool_payload(
    items: list[dict],
    *,
    generated_at: str,
    model_version: str,
    reference_threshold: float,
    fraud_cutoff: float = NAIVE_THRESHOLD,
) -> dict:
    """Nội dung ``data/sample_pool.json``, kèm luật chia nhóm để người đọc tệp hiểu được."""
    counts = {c: sum(item["category"] == c for item in items) for c in SAMPLE_CATEGORIES}
    rules = {
        "fraud_easy": f"gian lận, điểm ≥ {_so(fraud_cutoff)}",
        "fraud_hard": f"gian lận, điểm < {_so(fraud_cutoff)} (ngưỡng mặc định bỏ lọt)",
        "legit_easy": f"hợp lệ, điểm < τ* = {_so(reference_threshold, '.5f')}",
        "legit_hard": f"hợp lệ, điểm ≥ τ* = {_so(reference_threshold, '.5f')} (cảnh báo giả)",
    }
    return {
        "generated_at": generated_at,
        "model_version": model_version,
        "reference_threshold": float(reference_threshold),
        "fraud_hard_cutoff": float(fraud_cutoff),
        "source": "data/test_set.parquet",
        "categories": {c: {"count": counts[c], "rule": rules[c]} for c in SAMPLE_CATEGORIES},
        "items": items,
    }


def validate_sample_pool(payload: dict, *, min_hard: int = MIN_HARD_SAMPLES) -> list[str]:
    """Danh sách vấn đề của ``sample_pool.json`` (rỗng nếu đạt T-37)."""
    items = payload.get("items")
    if not items:
        return ["items phải là danh sách khác rỗng"]

    problems = []
    ids = [item.get("id") for item in items]
    if len(set(ids)) != len(ids):
        problems.append("id bị trùng")
    rows = [item.get("test_row") for item in items]
    if len(set(rows)) != len(rows):
        problems.append("một giao dịch xuất hiện hai lần (test_row trùng)")

    for item in items:
        category = item.get("category")
        where = item.get("id", "?")
        if category not in SAMPLE_CATEGORIES:
            problems.append(f"{where}: category không hợp lệ {category!r}")
            continue
        if item.get("label") != (1 if category.startswith("fraud") else 0):
            problems.append(f"{where}: label không khớp category {category}")
        if not 0 <= item.get("risk_score", -1) <= 1:
            problems.append(f"{where}: risk_score ngoài [0, 1]")
        features = item.get("features", {})
        if list(features) != RAW_REQUIRED_COLUMNS:
            problems.append(f"{where}: features phải có đúng 30 cột thô theo thứ tự Time, V1…V28, Amount")

    counts = {c: sum(item.get("category") == c for item in items) for c in SAMPLE_CATEGORIES}
    for category, count in counts.items():
        if count == 0:
            problems.append(f"thiếu nhóm {category}")
    for category in ("fraud_hard", "legit_hard"):
        if counts[category] < min_hard:
            problems.append(f"nhóm {category} chỉ có {counts[category]} mẫu, cần ít nhất {min_hard}")
    return problems
