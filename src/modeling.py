"""Dựng pipeline và chạy lưới thí nghiệm 5 chiến lược × 4 mô hình (04 §3).

Mọi pipeline sinh ra ở đây đều là ``imblearn.pipeline.Pipeline``. Đó không phải
lựa chọn thẩm mỹ: khi đưa vào ``cross_validate``, imblearn TỰ ĐỘNG bỏ qua bước
lấy mẫu lại lúc transform tập validation, còn ``sklearn.pipeline.Pipeline`` thì
không. Resample ngoài pipeline là lỗi rò rỉ phổ biến nhất ở bài toán này: mẫu
SMOTE nội suy từ mẫu validation lọt vào tập huấn luyện và mọi chỉ số vọt lên
khoảng 0,99 (ML-01, 04 §2.1).
"""

from __future__ import annotations

import hashlib
import json
import time
import warnings
from importlib import metadata
from pathlib import Path

import numpy as np
import pandas as pd
from imblearn.combine import SMOTETomek
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.under_sampling import RandomUnderSampler
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import (
    RepeatedStratifiedKFold,
    StratifiedKFold,
    TimeSeriesSplit,
    cross_val_predict,
)
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from .config import (
    CV_FOLDS,
    DATASET_DAYS,
    DEFAULT_COST_FN,
    DEFAULT_COST_FP,
    RANDOM_STATE,
    TEST_SIZE,
)
from .features import make_preprocessor
from .threshold import metrics_at_threshold, pick_threshold

#: Tỷ lệ lớp thiểu số sau khi lấy mẫu lại. KHÔNG đưa về 1:1 — cân bằng hoàn toàn
#: làm méo xác suất tiên nghiệm rất mạnh, mô hình trả xác suất lệch cao và
#: precision sụp (04 §3.1).
SAMPLING_STRATEGY = 0.1

MODEL_NAMES = ("logistic_regression", "decision_tree", "random_forest", "xgboost")
STRATEGY_NAMES = ("none", "class_weight", "undersample", "smote", "smote_tomek")

#: Lưới chuẩn trong báo cáo: 5 chiến lược × 4 mô hình = 20 tổ hợp (FR-05)
DEFAULT_MODELS = ("logistic_regression", "decision_tree", "random_forest", "xgboost")
DEFAULT_STRATEGIES = STRATEGY_NAMES

#: Lưới theo mức cân bằng (docs/11 M5): mọi chiến lược chạy ở CÙNG các tỷ lệ dương:âm mục tiêu
#: r sau khi cân bằng, để tách *loại* kỹ thuật khỏi *mức độ* cân bằng. r = 1 là cân bằng hoàn toàn
#: (``class_weight="balanced"`` của lưới chuẩn), r = 0,1 là mức của SMOTE/undersample trong lưới chuẩn.
RATIO_GRID_RATIOS = (0.02, 0.1, 0.3, 1.0)
RATIO_GRID_STRATEGIES = ("class_weight", "undersample", "smote")

#: Độ phức tạp của chiến lược, dùng cho quy tắc tie-break đặt trước (docs/11 M4): không xử lý
#: đơn giản nhất; trọng số lớp không đổi dữ liệu; undersample bỏ dữ liệu; SMOTE sinh dữ liệu.
STRATEGY_COMPLEXITY = {"none": 0, "class_weight": 1, "undersample": 2, "smote": 3, "smote_tomek": 4}


def _make_estimator(name: str, *, weight, pos_weight: float, random_state: int, **kwargs):
    """Khởi tạo một bộ phân loại theo cấu hình khởi điểm ở 04 §3.2.

    ``weight`` là trọng số lớp dương: ``None`` (không trọng số), ``"balanced"`` (cân bằng hoàn toàn —
    ``class_weight="balanced"``, riêng XGBoost là ``scale_pos_weight = pos_weight``), hoặc một số
    ``w`` (lớp dương nặng gấp ``w`` lớp âm).
    """
    def class_weight():
        return "balanced" if weight == "balanced" else {0: 1.0, 1: float(weight)}

    if name == "logistic_regression":
        params = dict(max_iter=1000, solver="liblinear", random_state=random_state)
        if weight is not None:
            params["class_weight"] = class_weight()
        return LogisticRegression(**{**params, **kwargs})

    if name == "decision_tree":
        params = dict(max_depth=6, random_state=random_state)
        if weight is not None:
            params["class_weight"] = class_weight()
        return DecisionTreeClassifier(**{**params, **kwargs})

    if name == "random_forest":
        params = dict(n_estimators=300, n_jobs=-1, random_state=random_state)
        if weight is not None:
            params["class_weight"] = class_weight()
        return RandomForestClassifier(**{**params, **kwargs})

    if name == "xgboost":
        from xgboost import XGBClassifier

        params = dict(
            n_estimators=500,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            tree_method="hist",
            eval_metric="aucpr",
            n_jobs=-1,
            random_state=random_state,
        )
        if weight is not None:
            params["scale_pos_weight"] = pos_weight if weight == "balanced" else float(weight)
        return XGBClassifier(**{**params, **kwargs})

    raise ValueError(f"Mô hình không hỗ trợ: {name!r}. Chọn trong {MODEL_NAMES}.")


def _make_resampler(strategy: str, random_state: int, ratio: float = SAMPLING_STRATEGY):
    """Bước lấy mẫu lại tương ứng với chiến lược, hoặc ``None`` nếu không có.

    ``ratio`` là tỷ lệ dương:âm sau khi lấy mẫu lại (``sampling_strategy`` của imblearn).
    """
    if strategy in ("none", "class_weight"):
        return None
    if strategy == "undersample":
        return RandomUnderSampler(sampling_strategy=ratio, random_state=random_state)
    if strategy == "smote":
        return SMOTE(sampling_strategy=ratio, k_neighbors=5, random_state=random_state)
    if strategy == "smote_tomek":
        return SMOTETomek(sampling_strategy=ratio, random_state=random_state)
    raise ValueError(f"Chiến lược không hỗ trợ: {strategy!r}. Chọn trong {STRATEGY_NAMES}.")


def effective_ratio(strategy: str, ratio: float | None, base_ratio: float) -> float:
    """Tỷ lệ dương:âm mà bộ phân loại thực sự "thấy" sau khi xử lý mất cân bằng.

    ``base_ratio`` là tỷ lệ gốc của dữ liệu (≈ 1/600). Trọng số ``w`` cho lớp dương tương đương
    nhân số mẫu dương lên ``w`` lần, nên ``class_weight="balanced"`` ứng với 1:1.
    """
    if strategy == "none":
        return float(base_ratio)
    if ratio is not None:
        return float(ratio)
    return 1.0 if strategy == "class_weight" else float(SAMPLING_STRATEGY)


def imbalance_ratio(y) -> float:
    """Tỷ lệ âm/dương — giá trị dùng cho ``scale_pos_weight``."""
    y = np.asarray(y)
    n_pos = int((y == 1).sum())
    if n_pos == 0:
        raise ValueError("Không có mẫu dương nào trong y.")
    return float((y == 0).sum() / n_pos)


def build_pipeline(
    model: str = "xgboost",
    strategy: str = "none",
    *,
    y=None,
    pos_weight: float | None = None,
    ratio: float | None = None,
    random_state: int = RANDOM_STATE,
    **model_kwargs,
) -> ImbPipeline:
    """Pipeline hoàn chỉnh: tiền xử lý → lấy mẫu lại → phân loại.

    Luôn trả về pipeline của imblearn, kể cả khi không có bước lấy mẫu lại, để
    cả lưới thí nghiệm đồng nhất về kiểu và ``tests/test_no_leakage.py`` kiểm tra
    được bằng một điều kiện duy nhất.

    ``pos_weight`` (tỷ lệ âm/dương) suy từ ``y`` nếu không truyền vào.

    ``ratio`` đặt **mức** cân bằng: tỷ lệ dương:âm mục tiêu sau khi xử lý (docs/11 M5).
    ``None`` giữ cấu hình của lưới chuẩn — ``class_weight`` cân bằng hoàn toàn, lấy mẫu lại về
    1:10. Với ``class_weight``, mức ``r`` nghĩa là trọng số lớp dương ``r × pos_weight``; với
    các chiến lược lấy mẫu lại, ``sampling_strategy = r``.
    """
    if model not in MODEL_NAMES:
        raise ValueError(f"Mô hình không hỗ trợ: {model!r}. Chọn trong {MODEL_NAMES}.")
    if strategy not in STRATEGY_NAMES:
        raise ValueError(f"Chiến lược không hỗ trợ: {strategy!r}. Chọn trong {STRATEGY_NAMES}.")
    if ratio is not None:
        if strategy == "none":
            raise ValueError("ratio chỉ dùng cho chiến lược có xử lý mất cân bằng, không dùng với 'none'.")
        if not 0 < ratio <= 1:
            raise ValueError(f"ratio phải nằm trong (0, 1]: {ratio!r}")

    if pos_weight is None:
        pos_weight = imbalance_ratio(y) if y is not None else 1.0

    if strategy != "class_weight":
        weight = None
    elif ratio is None:
        weight = "balanced"
    else:
        weight = ratio * pos_weight
    estimator = _make_estimator(
        model,
        weight=weight,
        pos_weight=pos_weight,
        random_state=random_state,
        **model_kwargs,
    )

    steps = [("preprocess", make_preprocessor())]
    resampler = _make_resampler(strategy, random_state, SAMPLING_STRATEGY if ratio is None else ratio)
    if resampler is not None:
        steps.append(("resample", resampler))
    steps.append(("clf", estimator))

    return ImbPipeline(steps)


def make_cv(n_splits: int = CV_FOLDS, random_state: int = RANDOM_STATE) -> StratifiedKFold:
    """Phân tầng là bắt buộc: với 0,172% lớp dương, ``KFold`` thường có thể tạo ra
    fold không chứa mẫu gian lận nào và ``average_precision`` trên fold đó không
    xác định (ML-02)."""
    return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)


def oof_scores(pipeline, X, y, *, cv=None, random_state: int = RANDOM_STATE) -> np.ndarray:
    """Điểm rủi ro out-of-fold — dùng để chọn ngưỡng mà không đụng tập kiểm thử.

    Chọn ngưỡng trên chính tập kiểm thử rồi báo cáo chỉ số cũng trên tập đó là
    một dạng rò rỉ nhẹ; ML-08 yêu cầu dùng dữ liệu out-of-fold cho việc này.
    """
    cv = cv or make_cv(random_state=random_state)
    probabilities = cross_val_predict(pipeline, X, y, cv=cv, method="predict_proba")
    return probabilities[:, 1]


#: Thứ tự cột của bảng kết quả, khớp định dạng bắt buộc ở 04 §3.3. Hai cột thêm sau review
#: (docs/11): ``brier`` — điểm Brier của điểm out-of-fold, để so độ hiệu chỉnh giữa các chiến lược
#: (M4); ``config_hash`` — dấu vân tay của cấu hình huấn luyện, dữ liệu và phiên bản thư viện, để
#: điểm lưu chỉ được dùng lại khi đúng là cùng một phép tính (m5).
GRID_COLUMNS = [
    "model",
    "strategy",
    "pr_auc_mean",
    "pr_auc_std",
    "roc_auc_mean",
    "roc_auc_std",
    "threshold",
    "recall",
    "precision",
    "f1",
    "tp",
    "fp",
    "fn",
    "alerts_per_day",
    "expected_cost",
    "brier",
    "n_distinct_scores",
    "seconds_per_fold",
    "total_seconds",
    "config_hash",
]

#: Bảng của lưới theo mức cân bằng: thêm ``ratio`` (tỷ lệ dương:âm mục tiêu) ngay sau ``strategy``
RATIO_GRID_COLUMNS = [*GRID_COLUMNS[:2], "ratio", *GRID_COLUMNS[2:]]

#: Gói mà kết quả huấn luyện phụ thuộc vào — đổi phiên bản thì điểm lưu cũ không còn dùng được
_FINGERPRINT_PACKAGES = ("numpy", "scikit-learn", "imbalanced-learn", "xgboost")


def _fold_metrics(y, scores, cv, X) -> tuple[list[float], list[float]]:
    """PR-AUC và ROC-AUC của từng fold, tính lại từ điểm out-of-fold.

    ``cross_val_predict`` không trả chỉ số theo fold, nhưng ``cv`` là
    ``StratifiedKFold`` có ``random_state`` cố định nên gọi ``split`` lần nữa cho
    đúng các fold mà nó đã dùng. Nhờ vậy chỉ cần MỘT lượt huấn luyện mà vẫn có độ
    lệch chuẩn giữa các fold — con số mà 04 §3.3 bắt buộc phải báo cáo.
    """
    y = np.asarray(y)
    pr_auc, roc_auc = [], []
    for _, test_idx in cv.split(X, y):
        pr_auc.append(float(average_precision_score(y[test_idx], scores[test_idx])))
        roc_auc.append(float(roc_auc_score(y[test_idx], scores[test_idx])))
    return pr_auc, roc_auc


def _describe(obj):
    """Mô tả tất định của một estimator và tham số của nó — đầu vào cho dấu vân tay."""
    if hasattr(obj, "get_params") and not isinstance(obj, type):
        params = obj.get_params(deep=False)
        return {
            "class": f"{type(obj).__module__}.{type(obj).__qualname__}",
            "params": {key: _describe(value) for key, value in sorted(params.items())},
        }
    if isinstance(obj, dict):
        return {str(key): _describe(value) for key, value in sorted(obj.items(), key=lambda kv: str(kv[0]))}
    if isinstance(obj, (list, tuple)):
        return [_describe(value) for value in obj]
    if isinstance(obj, (bool, np.bool_)):
        return bool(obj)
    if isinstance(obj, (int, np.integer)):
        return int(obj)
    if isinstance(obj, (float, np.floating)):
        return repr(float(obj))
    if obj is None or isinstance(obj, str):
        return obj
    return repr(obj)


def data_fingerprint(X, y) -> str:
    """Dấu vân tay 16 ký tự của ``(X, y)``: đổi một giá trị hay thứ tự một dòng là đổi dấu vân tay."""
    digest = hashlib.sha256()
    if hasattr(X, "columns"):
        digest.update(",".join(map(str, X.columns)).encode("utf-8"))
    digest.update(np.ascontiguousarray(np.asarray(X, dtype="float64")).tobytes())
    digest.update(np.ascontiguousarray(np.asarray(y, dtype="int64")).tobytes())
    return digest.hexdigest()[:16]


def config_fingerprint(estimator, *, cv=None, data: str = "", extra=None) -> str:
    """Dấu vân tay của một phép huấn luyện: cấu hình estimator, cách chia fold, dữ liệu và phiên
    bản thư viện.

    Hai lần chạy cùng dấu vân tay trên cùng một máy cho điểm trùng nhau từng bit — đó là điều kiện
    để dùng lại điểm lưu thay cho việc huấn luyện lại. Trước review, điểm lưu được dùng lại chỉ vì
    tệp tồn tại, nên sửa mã mà quên xoá tệp thì vẫn ra số cũ mà không ai biết (docs/11 m5).
    """
    versions = {}
    for name in _FINGERPRINT_PACKAGES:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = None
    payload = {
        "estimator": _describe(estimator),
        "cv": repr(cv),
        "data": data,
        "versions": versions,
        "extra": _describe(extra),
    }
    text = json.dumps(payload, sort_keys=True, ensure_ascii=True)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def _load_checkpoint(path):
    """Đọc bảng đã chạy dở và ma trận điểm kèm theo, nếu có."""
    done, scores = pd.DataFrame(), {}
    if path is None:
        return done, scores

    path = Path(path)
    if path.exists():
        done = pd.read_csv(path)
    npz = path.with_suffix(".npz")
    if npz.exists():
        with np.load(npz) as data:
            scores = {key: data[key] for key in data.files}
    return done, scores


def _save_checkpoint(path, rows, scores):
    """Ghi lại ngay sau mỗi tổ hợp — lưới này chạy 1–3 giờ, mất giữa chừng là mất hết."""
    if path is None:
        return
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False, encoding="utf-8")
    if scores:
        np.savez_compressed(path.with_suffix(".npz"), **scores)


def score_key(model: str, strategy: str, ratio: float | None = None) -> str:
    """Khoá của mảng điểm trong tệp ``.npz``: ``model__strategy``, thêm ``__r0.1`` nếu có mức cân bằng."""
    return f"{model}__{strategy}" if ratio is None else f"{model}__{strategy}__r{ratio:g}"


def _summarize_oof(y, oof, cv, X, *, criterion, cost_fn, cost_fp, sample_fraction, days) -> dict:
    """Mọi cột chỉ số của một dòng bảng, suy ra từ điểm out-of-fold — không huấn luyện gì.

    Tách khỏi phần huấn luyện để điểm lưu chỉ cần giữ ĐIỂM: đổi tham số chi phí thì τ\\* và các cột
    theo ngưỡng được tính lại ngay, không phải chạy lại lưới.
    """
    pr_auc, roc_auc = _fold_metrics(y, oof, cv, X)
    nguong = pick_threshold(y, oof, criterion, cost_fn=cost_fn, cost_fp=cost_fp,
                            sample_fraction=sample_fraction, days=days)
    met = metrics_at_threshold(y, oof, nguong, cost_fn=cost_fn, cost_fp=cost_fp,
                               sample_fraction=sample_fraction, days=days)
    return {
        "pr_auc_mean": float(np.mean(pr_auc)),
        "pr_auc_std": float(np.std(pr_auc)),
        "roc_auc_mean": float(np.mean(roc_auc)),
        "roc_auc_std": float(np.std(roc_auc)),
        "threshold": met.threshold,
        "recall": met.recall,
        "precision": met.precision,
        "f1": met.f1,
        "tp": met.tp,
        "fp": met.fp,
        "fn": met.fn,
        "alerts_per_day": met.alerts_per_day,
        "expected_cost": met.expected_cost,
        "brier": float(brier_score_loss(y, oof)),
        "n_distinct_scores": int(np.unique(oof).size),
    }


def _run_combinations(X, y, combos, *, model_params, cv, criterion, cost_fn, cost_fp, sample_fraction,
                      days, random_state, checkpoint_path, resume, verbose):
    """Phần chung của ``run_grid`` và ``run_ratio_grid``; ``combos`` là các bộ (model, strategy, ratio).

    Điểm lưu giữ ĐIỂM out-of-fold kèm dấu vân tay cấu hình. Một tổ hợp chỉ được bỏ qua khi điểm lưu
    có đúng tổ hợp đó, đúng dấu vân tay, và có mảng điểm trong ``.npz``. Mọi cột chỉ số luôn được
    tính lại từ điểm.
    """
    cv = cv or make_cv(random_state=random_state)
    pos_weight = imbalance_ratio(y)
    y_array = np.asarray(y)
    data_fp = data_fingerprint(X, y_array)

    done, cached_scores = _load_checkpoint(checkpoint_path) if resume else (pd.DataFrame(), {})
    cached = {}
    if not done.empty and "config_hash" in done.columns:
        for row in done.to_dict("records"):
            ratio = None if "ratio" not in row or pd.isna(row["ratio"]) else float(row["ratio"])
            cached[score_key(row["model"], row["strategy"], ratio)] = row

    rows, scores = [], {}
    tong = len(combos)
    bat_dau = time.perf_counter()
    da_huan_luyen = 0
    for thu_tu, (model, strategy, ratio) in enumerate(combos, start=1):
        pipeline = build_pipeline(model, strategy, pos_weight=pos_weight, ratio=ratio,
                                  random_state=random_state, **(model_params or {}).get(model, {}))
        key = score_key(model, strategy, ratio)
        fingerprint = config_fingerprint(pipeline, cv=cv, data=data_fp)
        old = cached.get(key)
        reused = old is not None and old.get("config_hash") == fingerprint and key in cached_scores
        if reused:
            oof = np.asarray(cached_scores[key])
            elapsed = float(old["total_seconds"])
        else:
            started = time.perf_counter()
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                oof = cross_val_predict(pipeline, X, y, cv=cv, method="predict_proba")[:, 1]
            elapsed = time.perf_counter() - started
            da_huan_luyen += 1

        row = {
            "model": model,
            "strategy": strategy,
            "ratio": ratio,
            **_summarize_oof(y_array, oof, cv, X, criterion=criterion, cost_fn=cost_fn, cost_fp=cost_fp,
                             sample_fraction=sample_fraction, days=days),
            "seconds_per_fold": elapsed / cv.get_n_splits(),
            "total_seconds": elapsed,
            "config_hash": fingerprint,
        }
        rows.append(row)
        scores[key] = oof
        if not reused:
            # Giữ cả các dòng cũ chưa tới lượt: chạy dở rồi chết vẫn không mất gì
            other_rows = [r for k, r in cached.items() if k not in scores]
            other_scores = {k: v for k, v in cached_scores.items() if k not in scores}
            _save_checkpoint(checkpoint_path, rows + other_rows, {**other_scores, **scores})

        if verbose:
            troi = time.perf_counter() - bat_dau
            con_lai = (tong - thu_tu) * troi / max(da_huan_luyen, 1) if da_huan_luyen else 0.0
            muc = "" if ratio is None else f" r={ratio:g}"
            nguon = "dùng lại điểm lưu" if reused else f"{elapsed / 60:5.1f} phút"
            print(
                f"[{thu_tu:2d}/{tong}] {model:20s} {strategy + muc:18s} "
                f"PR-AUC {row['pr_auc_mean']:.4f} ± {row['pr_auc_std']:.4f}  "
                f"R@τ* {row['recall']:.3f}  P@τ* {row['precision']:.3f}  Brier {row['brier']:.6f}  "
                f"({nguon}, còn khoảng {con_lai / 60:.0f} phút)"
            )

    other_rows = [r for k, r in cached.items() if k not in scores]
    other_scores = {k: v for k, v in cached_scores.items() if k not in scores}
    _save_checkpoint(checkpoint_path, rows + other_rows, {**other_scores, **scores})
    return rows, scores


def run_grid(
    X,
    y,
    *,
    models=DEFAULT_MODELS,
    strategies=DEFAULT_STRATEGIES,
    model_params: dict | None = None,
    cv=None,
    criterion: str = "min_expected_cost",
    cost_fn: float = DEFAULT_COST_FN,
    cost_fp: float = DEFAULT_COST_FP,
    sample_fraction: float = 1.0 - TEST_SIZE,
    days: float = DATASET_DAYS,
    random_state: int = RANDOM_STATE,
    checkpoint_path=None,
    resume: bool = True,
    return_scores: bool = False,
    verbose: bool = True,
):
    """Chạy toàn bộ lưới và trả về bảng kết quả trung tâm của báo cáo (FR-05).

    Ghi lại **mọi** tổ hợp, kể cả những cấu hình cho kết quả kém — phần so sánh
    mới là trọng tâm đề tài, không phải riêng mô hình thắng cuộc (US-10).

    Mỗi tổ hợp chỉ huấn luyện **một lượt**. ``cross_val_predict`` cho điểm
    out-of-fold, và từ đó suy ra tất cả phần còn lại: chỉ số theo fold (để có độ
    lệch chuẩn), ngưỡng τ\\* theo tiêu chí chi phí, recall/precision/F1 tại τ\\* đó,
    và điểm Brier. Cách cũ — ``cross_validate`` để lấy chỉ số rồi ``cross_val_predict``
    để lấy điểm — phải huấn luyện hai lượt, tức gấp đôi 1–3 giờ.

    τ\\* được chọn trên chính điểm out-of-fold chứ không đụng tập kiểm thử (ML-08).

    Tham số đáng chú ý
    ------------------
    model_params
        Tham số ghi đè theo từng mô hình, dạng ``{"xgboost": {"n_estimators": 100}}``.
        Có mặt để dùng phương án dự phòng đã ghi sẵn ở T-20: nếu lưới vượt 3 giờ
        thì hạ ``n_estimators`` xuống 100 cho riêng hai mô hình ensemble và nêu
        rõ trong báo cáo. Mô hình không có tên trong dict thì giữ cấu hình gốc.
    sample_fraction
        Phần dữ liệu mà ``X`` chiếm trong toàn luồng, dùng để quy đổi số cảnh báo
        mỗi ngày. Mặc định 0,8 vì lưới chạy trên tập huấn luyện.
    checkpoint_path
        Đường dẫn CSV. Sau **mỗi** tổ hợp, bảng hiện có được ghi lại và ma trận
        điểm out-of-fold ghi vào tệp ``.npz`` cùng tên. Với ``resume=True``, một tổ
        hợp được bỏ qua khi điểm lưu có đúng tổ hợp đó với cùng ``config_hash`` — tức
        cùng cấu hình, cùng dữ liệu, cùng phiên bản thư viện. Các cột chỉ số luôn được
        tính lại từ điểm, nên đổi tham số chi phí không cần chạy lại lưới.
    return_scores
        Trả thêm ``dict`` ánh xạ ``"{model}__{strategy}"`` sang mảng điểm
        out-of-fold — đầu vào cho hình PR chồng của T-21.
    """
    combos = [(model, strategy, None) for model in models for strategy in strategies]
    rows, scores = _run_combinations(
        X, y, combos, model_params=model_params, cv=cv, criterion=criterion, cost_fn=cost_fn,
        cost_fp=cost_fp, sample_fraction=sample_fraction, days=days, random_state=random_state,
        checkpoint_path=checkpoint_path, resume=resume, verbose=verbose,
    )
    table = (
        pd.DataFrame(rows)[GRID_COLUMNS]
        .sort_values("pr_auc_mean", ascending=False)
        .reset_index(drop=True)
    )
    return (table, scores) if return_scores else table


def run_ratio_grid(
    X,
    y,
    *,
    models=DEFAULT_MODELS,
    strategies=RATIO_GRID_STRATEGIES,
    ratios=RATIO_GRID_RATIOS,
    model_params: dict | None = None,
    cv=None,
    criterion: str = "min_expected_cost",
    cost_fn: float = DEFAULT_COST_FN,
    cost_fp: float = DEFAULT_COST_FP,
    sample_fraction: float = 1.0 - TEST_SIZE,
    days: float = DATASET_DAYS,
    random_state: int = RANDOM_STATE,
    checkpoint_path=None,
    resume: bool = True,
    return_scores: bool = False,
    verbose: bool = True,
):
    """Lưới theo mức cân bằng (docs/11 M5): mọi chiến lược ở CÙNG các tỷ lệ dương:âm mục tiêu.

    Lưới chuẩn so ``class_weight`` cân bằng hoàn toàn (1:1) với SMOTE/undersample đưa về 1:10, nên
    không tách được *loại* kỹ thuật khỏi *mức độ* cân bằng. Ở đây mỗi (mô hình, chiến lược) chạy ở
    từng ``ratio``: ``class_weight`` dùng trọng số lớp dương ``ratio × N_âm/N_dương``,
    undersample/SMOTE dùng ``sampling_strategy = ratio``. Mốc "không xử lý" lấy từ lưới chuẩn.

    Cùng cách chia fold, cùng cách chọn τ\\*, cùng cơ chế điểm lưu theo ``config_hash`` như
    ``run_grid``; khoá điểm trong ``.npz`` là ``"{model}__{strategy}__r{ratio}"``.
    """
    combos = [(model, strategy, float(ratio)) for model in models for strategy in strategies for ratio in ratios]
    rows, scores = _run_combinations(
        X, y, combos, model_params=model_params, cv=cv, criterion=criterion, cost_fn=cost_fn,
        cost_fp=cost_fp, sample_fraction=sample_fraction, days=days, random_state=random_state,
        checkpoint_path=checkpoint_path, resume=resume, verbose=verbose,
    )
    table = (
        pd.DataFrame(rows)[RATIO_GRID_COLUMNS]
        .sort_values(["model", "strategy", "ratio"])
        .reset_index(drop=True)
    )
    return (table, scores) if return_scores else table


# --------------------------------------------------------------------------
# Độ bất định khi so cấu hình, validation theo thời gian, quy tắc chọn (docs/11)
# --------------------------------------------------------------------------

def repeated_cv_scores(
    pipeline,
    X,
    y,
    *,
    n_splits: int = CV_FOLDS,
    n_repeats: int = 5,
    random_state: int = RANDOM_STATE,
    verbose: bool = False,
) -> pd.DataFrame:
    """PR-AUC và ROC-AUC trên từng fold của ``RepeatedStratifiedKFold`` (docs/11 m2).

    Một lần CV 5 fold chỉ cho 5 con số, và bootstrap trên điểm OOF gộp bỏ qua phương sai do huấn
    luyện. Lặp lại CV với các cách chia khác nhau cho phương sai đó. Cùng ``random_state`` thì mọi
    cấu hình gặp đúng cùng các fold — kể cả khi ``n_repeats`` khác nhau, vì ``n_repeats`` lần đầu
    luôn trùng — nên so được theo cặp trên các khoá ``(repeat, fold)`` chung.
    """
    y_array = np.asarray(y)
    splitter = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=random_state)
    rows = []
    for i, (train_idx, test_idx) in enumerate(splitter.split(X, y_array)):
        started = time.perf_counter()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fitted = clone(pipeline).fit(X.iloc[train_idx], y_array[train_idx])
            scores = fitted.predict_proba(X.iloc[test_idx])[:, 1]
        rows.append({
            "repeat": i // n_splits,
            "fold": i % n_splits,
            "pr_auc": float(average_precision_score(y_array[test_idx], scores)),
            "roc_auc": float(roc_auc_score(y_array[test_idx], scores)),
            "n_train": int(train_idx.size),
            "n_test": int(test_idx.size),
            "seconds": time.perf_counter() - started,
        })
        if verbose:
            print(f"    lặp {i // n_splits + 1}, fold {i % n_splits + 1}: PR-AUC {rows[-1]['pr_auc']:.4f}")
    return pd.DataFrame(rows)


def temporal_oof_scores(pipeline, X, y, time_values, *, n_splits: int = 4):
    """Điểm "out-of-fold" theo kiểu forward-chaining (docs/11 M2).

    Sắp giao dịch theo ``time_values`` rồi chia ``n_splits + 1`` khối liên tiếp (``TimeSeriesSplit``):
    fold k huấn luyện trên mọi giao dịch **trước** khối k và chấm khối k. Khác cross-validation ngẫu
    nhiên, mô hình không bao giờ thấy giao dịch của "tương lai" — đúng hoàn cảnh triển khai, nơi ngưỡng
    chọn hôm nay sẽ áp cho giao dịch ngày mai.

    Trả về ``(oof, folds)``: ``oof`` cùng độ dài ``y``, khối đầu tiên là ``NaN`` vì không có quá khứ để
    huấn luyện; ``folds`` mô tả từng fold (mốc thời gian, cỡ, số gian lận).
    """
    times = np.asarray(time_values, dtype="float64")
    y_array = np.asarray(y)
    order = np.argsort(times, kind="stable")
    oof = np.full(y_array.size, np.nan)
    folds = []
    for k, (train_pos, valid_pos) in enumerate(TimeSeriesSplit(n_splits=n_splits).split(order)):
        train_idx, valid_idx = order[train_pos], order[valid_pos]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fitted = clone(pipeline).fit(X.iloc[train_idx], y_array[train_idx])
            oof[valid_idx] = fitted.predict_proba(X.iloc[valid_idx])[:, 1]
        folds.append({
            "fold": k,
            "train_from": float(times[train_idx].min()),
            "train_to": float(times[train_idx].max()),
            "valid_from": float(times[valid_idx].min()),
            "valid_to": float(times[valid_idx].max()),
            "n_train": int(train_idx.size),
            "n_fraud_train": int(y_array[train_idx].sum()),
            "n_valid": int(valid_idx.size),
            "n_fraud_valid": int(y_array[valid_idx].sum()),
        })
    return oof, pd.DataFrame(folds)


def final_pipeline(y=None, *, pos_weight: float | None = None, random_state: int = RANDOM_STATE, **model_kwargs):
    """Pipeline của cấu hình cuối ghi trong ``src/config.py`` (``FINAL_MODEL``, ``FINAL_STRATEGY``,
    ``FINAL_RATIO``) — notebook 05–08 dựng mô hình qua hàm này, không tự ghi tên mô hình."""
    from .config import FINAL_MODEL, FINAL_RATIO, FINAL_STRATEGY

    return build_pipeline(FINAL_MODEL, FINAL_STRATEGY, y=y, pos_weight=pos_weight, ratio=FINAL_RATIO,
                          random_state=random_state, **model_kwargs)


def model_version_name(model: str, strategy: str, ratio: float | None = None, *, revision: int = 2) -> str:
    """Tên phiên bản mô hình đi vào mọi phản hồi của API (NFR-11), ví dụ ``xgb_none_v2``."""
    short = {"xgboost": "xgb", "random_forest": "rf", "logistic_regression": "lr", "decision_tree": "dt"}[model]
    tail = {"none": "none", "class_weight": "scaleposweight" if model == "xgboost" else "classweight",
            "undersample": "undersample", "smote": "smote", "smote_tomek": "smotetomek"}[strategy]
    level = "" if ratio is None else f"_r{ratio:g}".replace(".", "")
    return f"{short}_{tail}{level}_v{revision}"


def selection_scores(pipeline, X, y, time_values, *, sample_fraction: float, n_splits: int | None = None) -> dict:
    """Điểm chọn ngưỡng của hệ thống: forward-chaining theo thời gian trên tập huấn luyện (docs/11 M2).

    Trả về những gì ``threshold.json`` và ``oof_scores.npz`` cần: nhãn và điểm của các giao dịch có
    điểm (khối thời gian đầu không có), phần luồng mà chúng đại diện, và số ngày mà các cửa sổ kiểm
    định trải qua — để quy đổi cảnh báo ra mỗi ngày.

    ``sample_fraction`` là phần của toàn luồng mà ``X`` chiếm (0,8 với tập huấn luyện chia ngẫu nhiên):
    trong mỗi cửa sổ thời gian, tập huấn luyện vẫn là một mẫu ngẫu nhiên 80% của luồng.
    """
    from .config import TEMPORAL_SPLITS

    oof, folds = temporal_oof_scores(pipeline, X, y, time_values, n_splits=n_splits or TEMPORAL_SPLITS)
    scored = np.isfinite(oof)
    y_array = np.asarray(y)
    span = float(folds["valid_to"].max() - folds["valid_from"].min())
    scores = oof[scored]
    # Điểm của XGBoost là float32: trả về đúng kiểu đó để ngưỡng ứng viên trùng từng bit với API
    if np.array_equal(scores.astype("float32").astype("float64"), scores):
        scores = scores.astype("float32")
    return {
        "y_true": y_array[scored],
        "y_score": scores,
        "index": np.flatnonzero(scored),
        "sample_fraction": float(sample_fraction),
        "days": span / 86_400,
        "folds": folds,
        "method": "temporal_forward_chaining",
    }


def select_configuration(candidates: pd.DataFrame) -> pd.Series:
    """Quy tắc tie-break đặt trước (docs/11 M4), áp lên các cấu hình ứng viên.

    ``candidates`` cần các cột ``strategy``, ``tie_with_best`` (bool — hiệu PR-AUC so với cấu hình
    tốt nhất có KTC chứa 0 theo repeated CV), ``brier`` và ``alerts_per_day``. Thứ tự ưu tiên:

    1. chỉ xét cấu hình hoà với cấu hình tốt nhất — chất lượng xếp hạng là điều kiện cần;
    2. chiến lược đơn giản nhất (``STRATEGY_COMPLEXITY``): không chứng minh được lợi ích thì không
       trả giá phức tạp;
    3. Brier thấp nhất trên OOF — điểm gần xác suất hơn;
    4. ít cảnh báo/ngày nhất tại τ\\* của chính nó.
    """
    pool = candidates[candidates["tie_with_best"].astype(bool)]
    if pool.empty:
        raise ValueError("Không cấu hình nào được đánh dấu tie_with_best — thiếu cấu hình tốt nhất?")
    ranked = pool.assign(_complexity=pool["strategy"].map(STRATEGY_COMPLEXITY)).sort_values(
        ["_complexity", "brier", "alerts_per_day"], kind="stable"
    )
    return ranked.drop(columns="_complexity").iloc[0]


# --------------------------------------------------------------------------
# Tinh chỉnh siêu tham số (04 §4, T-24)
# --------------------------------------------------------------------------

#: Số lần thử của RandomizedSearchCV (04 §4)
SEARCH_N_ITER = 30

#: Cột tham số trong bảng kết quả tìm kiếm, đã bỏ tiền tố ``clf__``
SEARCH_PARAMS = (
    "n_estimators",
    "max_depth",
    "learning_rate",
    "subsample",
    "colsample_bytree",
    "scale_pos_weight",
    "min_child_weight",
)


def search_space(y=None, *, pos_weight: float | None = None) -> dict:
    """Không gian tìm kiếm cho XGBoost, đúng bảng ở 04 §4.

    Giá trị lớn nhất của ``scale_pos_weight`` là tỷ lệ âm/dương **của chính tập
    huấn luyện**. Con số 578 trong đặc tả là tỷ lệ trên dữ liệu thô; sau khi loại
    trùng lặp và chia 80/20 thì tỷ lệ thật khoảng 599. Dùng số thật để lựa chọn
    "cân bằng đầy đủ" đúng nghĩa của nó.
    """
    from scipy.stats import loguniform, randint, uniform

    if pos_weight is None:
        if y is None:
            raise ValueError("Cần `y` hoặc `pos_weight` để đặt miền của scale_pos_weight.")
        pos_weight = imbalance_ratio(y)

    return {
        "clf__n_estimators": randint(200, 801),
        "clf__max_depth": randint(3, 9),
        "clf__learning_rate": loguniform(0.01, 0.3),
        "clf__subsample": uniform(0.6, 0.4),
        "clf__colsample_bytree": uniform(0.6, 0.4),
        "clf__scale_pos_weight": [1.0, 10.0, 100.0, round(float(pos_weight), 1)],
        "clf__min_child_weight": randint(1, 11),
    }


def search_fingerprint(X, y, *, n_iter: int = SEARCH_N_ITER, cv=None, random_state: int = RANDOM_STATE) -> str:
    """Dấu vân tay của một lần ``run_search`` — cấu hình, miền tìm kiếm, cách chia fold, dữ liệu.

    ``scripts/run_search.py`` ghi nó vào ``best_params.json``; notebook 05 chỉ dùng lại kết quả đã
    lưu khi dấu vân tay khớp, thay vì chỉ vì tệp tồn tại (docs/11 m5).
    """
    cv = cv or make_cv(random_state=random_state)
    pipeline = build_pipeline("xgboost", "class_weight", y=y, random_state=random_state, n_jobs=1)
    space = {}
    for name, dist in search_space(y).items():
        if hasattr(dist, "dist"):        # phân phối scipy đã cố định tham số
            space[name] = {"dist": dist.dist.name, "args": [repr(float(a)) for a in dist.args],
                           "kwds": {k: repr(v) for k, v in sorted(dist.kwds.items())}}
        else:
            space[name] = _describe(list(dist))
    return config_fingerprint(
        pipeline, cv=cv, data=data_fingerprint(X, y),
        extra={"space": space, "n_iter": n_iter, "random_state": random_state, "scoring": "average_precision"},
    )


def run_search(
    X,
    y,
    *,
    n_iter: int = SEARCH_N_ITER,
    cv=None,
    n_jobs: int = -1,
    refit: bool = True,
    random_state: int = RANDOM_STATE,
    verbose: int = 1,
):
    """``RandomizedSearchCV`` trên pipeline ``xgboost`` + ``class_weight`` (T-24).

    Pipeline là đúng cái đã thắng ở giai đoạn 3, nên bước tiền xử lý nằm trong
    từng fold như mọi chỗ khác (ML-04). Chiến lược ``class_weight`` không có bước
    lấy mẫu lại; ``scale_pos_weight`` là một trong các tham số được tìm, nên
    giá trị 1 trong miền tương ứng với "không xử lý mất cân bằng".

    Song song ở tầng ngoài: ``n_jobs`` của tìm kiếm là -1, còn XGBoost bên trong
    chạy **một luồng**. 150 lần huấn luyện độc lập chia đều cho các lõi nhanh
    hơn nhiều so với để mỗi lần huấn luyện tự chia luồng, và tránh 12 × 12 luồng
    tranh nhau CPU.
    """
    from sklearn.model_selection import RandomizedSearchCV

    cv = cv or make_cv(random_state=random_state)
    pipeline = build_pipeline(
        "xgboost", "class_weight", y=y, random_state=random_state, n_jobs=1
    )
    search = RandomizedSearchCV(
        pipeline,
        search_space(y),
        n_iter=n_iter,
        scoring="average_precision",
        cv=cv,
        n_jobs=n_jobs,
        refit=refit,
        random_state=random_state,
        return_train_score=True,
        verbose=verbose,
        error_score="raise",
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        search.fit(X, y)
    return search


def search_table(cv_results) -> pd.DataFrame:
    """Bảng kết quả tìm kiếm, sắp theo PR-AUC giảm dần.

    Nhận ``search.cv_results_`` hoặc một DataFrame đã đọc lại từ CSV. Cột
    ``overfit_gap`` là chênh PR-AUC train − validation: cấu hình đứng đầu mà
    khoảng cách này quá lớn thì thứ hạng của nó kém tin cậy hơn con số cho thấy.
    """
    raw = pd.DataFrame(cv_results)
    table = pd.DataFrame({
        "rank": raw["rank_test_score"].astype(int),
        "pr_auc_mean": raw["mean_test_score"],
        "pr_auc_std": raw["std_test_score"],
        "pr_auc_train": raw["mean_train_score"],
    })
    table["overfit_gap"] = table["pr_auc_train"] - table["pr_auc_mean"]
    for name in SEARCH_PARAMS:
        table[name] = raw[f"param_clf__{name}"].astype(float)
    for name in ("n_estimators", "max_depth", "min_child_weight"):
        table[name] = table[name].astype(int)
    table["fit_seconds"] = raw["mean_fit_time"]
    # Sắp ổn định: khi hoà hạng, giữ thứ tự gốc — đúng quy tắc sklearn dùng để chọn
    # ``best_index_``, nhờ vậy dòng đầu bảng luôn là cấu hình mà ``refit`` huấn luyện
    return table.sort_values("rank", kind="stable").reset_index(drop=True)


def best_params_from(table: pd.DataFrame, rank: int = 0) -> dict:
    """Tham số của dòng thứ ``rank`` trong bảng ``search_table``, dạng truyền được
    thẳng vào ``build_pipeline(..., **params)``."""
    row = table.iloc[rank]
    params = {name: row[name] for name in SEARCH_PARAMS}
    for name in ("n_estimators", "max_depth", "min_child_weight"):
        params[name] = int(params[name])
    for name in ("learning_rate", "subsample", "colsample_bytree", "scale_pos_weight"):
        params[name] = float(params[name])
    return params
