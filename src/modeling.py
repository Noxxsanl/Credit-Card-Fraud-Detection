"""Dựng pipeline và chạy lưới thí nghiệm 5 chiến lược × 4 mô hình (04 §3).

Mọi pipeline sinh ra ở đây đều là ``imblearn.pipeline.Pipeline``. Đó không phải
lựa chọn thẩm mỹ: khi đưa vào ``cross_validate``, imblearn TỰ ĐỘNG bỏ qua bước
lấy mẫu lại lúc transform tập validation, còn ``sklearn.pipeline.Pipeline`` thì
không. Resample ngoài pipeline là lỗi rò rỉ phổ biến nhất ở bài toán này: mẫu
SMOTE nội suy từ mẫu validation lọt vào tập huấn luyện và mọi chỉ số vọt lên
khoảng 0,99 (ML-01, 04 §2.1).
"""

from __future__ import annotations

import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from imblearn.combine import SMOTETomek
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.under_sampling import RandomUnderSampler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
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

MODEL_NAMES = ("logistic_regression", "decision_tree", "random_forest", "xgboost", "lightgbm")
STRATEGY_NAMES = ("none", "class_weight", "undersample", "smote", "smote_tomek")

#: Lưới chuẩn trong báo cáo: 5 chiến lược × 4 mô hình = 20 tổ hợp (FR-05)
DEFAULT_MODELS = ("logistic_regression", "decision_tree", "random_forest", "xgboost")
DEFAULT_STRATEGIES = STRATEGY_NAMES

#: Mô hình cây nhận mất cân bằng qua scale_pos_weight thay vì class_weight
_BOOSTED = {"xgboost", "lightgbm"}


def _make_estimator(name: str, *, balanced: bool, pos_weight: float, random_state: int, **kwargs):
    """Khởi tạo một bộ phân loại theo cấu hình khởi điểm ở 04 §3.2."""
    if name == "logistic_regression":
        params = dict(max_iter=1000, solver="liblinear", random_state=random_state)
        if balanced:
            params["class_weight"] = "balanced"
        return LogisticRegression(**{**params, **kwargs})

    if name == "decision_tree":
        params = dict(max_depth=6, random_state=random_state)
        if balanced:
            params["class_weight"] = "balanced"
        return DecisionTreeClassifier(**{**params, **kwargs})

    if name == "random_forest":
        params = dict(n_estimators=300, n_jobs=-1, random_state=random_state)
        if balanced:
            params["class_weight"] = "balanced"
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
        if balanced:
            params["scale_pos_weight"] = pos_weight
        return XGBClassifier(**{**params, **kwargs})

    if name == "lightgbm":
        from lightgbm import LGBMClassifier

        params = dict(
            n_estimators=500,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            n_jobs=-1,
            verbose=-1,
            random_state=random_state,
        )
        if balanced:
            params["scale_pos_weight"] = pos_weight
        return LGBMClassifier(**{**params, **kwargs})

    raise ValueError(f"Mô hình không hỗ trợ: {name!r}. Chọn trong {MODEL_NAMES}.")


def _make_resampler(strategy: str, random_state: int):
    """Bước lấy mẫu lại tương ứng với chiến lược, hoặc ``None`` nếu không có."""
    if strategy in ("none", "class_weight"):
        return None
    if strategy == "undersample":
        return RandomUnderSampler(
            sampling_strategy=SAMPLING_STRATEGY, random_state=random_state
        )
    if strategy == "smote":
        return SMOTE(
            sampling_strategy=SAMPLING_STRATEGY, k_neighbors=5, random_state=random_state
        )
    if strategy == "smote_tomek":
        return SMOTETomek(
            sampling_strategy=SAMPLING_STRATEGY, random_state=random_state
        )
    raise ValueError(f"Chiến lược không hỗ trợ: {strategy!r}. Chọn trong {STRATEGY_NAMES}.")


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
    random_state: int = RANDOM_STATE,
    **model_kwargs,
) -> ImbPipeline:
    """Pipeline hoàn chỉnh: tiền xử lý → lấy mẫu lại → phân loại.

    Luôn trả về pipeline của imblearn, kể cả khi không có bước lấy mẫu lại, để
    cả lưới thí nghiệm đồng nhất về kiểu và ``tests/test_no_leakage.py`` kiểm tra
    được bằng một điều kiện duy nhất.

    ``pos_weight`` suy từ ``y`` nếu không truyền vào; chỉ dùng cho chiến lược
    ``class_weight`` với mô hình boosting.
    """
    if model not in MODEL_NAMES:
        raise ValueError(f"Mô hình không hỗ trợ: {model!r}. Chọn trong {MODEL_NAMES}.")
    if strategy not in STRATEGY_NAMES:
        raise ValueError(f"Chiến lược không hỗ trợ: {strategy!r}. Chọn trong {STRATEGY_NAMES}.")

    if pos_weight is None:
        pos_weight = imbalance_ratio(y) if y is not None else 1.0

    balanced = strategy == "class_weight"
    estimator = _make_estimator(
        model,
        balanced=balanced,
        pos_weight=pos_weight,
        random_state=random_state,
        **model_kwargs,
    )

    steps = [("preprocess", make_preprocessor())]
    resampler = _make_resampler(strategy, random_state)
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


#: Thứ tự cột của bảng kết quả, khớp định dạng bắt buộc ở 04 §3.3
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
    "n_distinct_scores",
    "seconds_per_fold",
    "total_seconds",
]


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
    lệch chuẩn), ngưỡng τ\\* theo tiêu chí chi phí, rồi recall/precision/F1 tại
    τ\\* đó. Cách cũ — ``cross_validate`` để lấy chỉ số rồi ``cross_val_predict``
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
        điểm out-of-fold ghi vào tệp ``.npz`` cùng tên. Với ``resume=True``,
        những tổ hợp đã có trong tệp sẽ được bỏ qua.
    return_scores
        Trả thêm ``dict`` ánh xạ ``"{model}__{strategy}"`` sang mảng điểm
        out-of-fold — đầu vào cho hình PR chồng của T-21.
    """
    cv = cv or make_cv(random_state=random_state)
    pos_weight = imbalance_ratio(y)
    y_array = np.asarray(y)

    da_chay, scores = (_load_checkpoint(checkpoint_path) if resume else (pd.DataFrame(), {}))
    rows = da_chay.to_dict("records") if not da_chay.empty else []
    xong = {(r["model"], r["strategy"]) for r in rows}
    if verbose and xong:
        print(f"Nối tiếp từ điểm lưu: đã có {len(xong)}/{len(models) * len(strategies)} tổ hợp.\n")

    tong = len(models) * len(strategies)
    bat_dau_luoi = time.perf_counter()

    for model in models:
        for strategy in strategies:
            thu_tu = len(rows) + 1
            if (model, strategy) in xong:
                continue

            pipeline = build_pipeline(
                model,
                strategy,
                pos_weight=pos_weight,
                random_state=random_state,
                **(model_params or {}).get(model, {}),
            )

            started = time.perf_counter()
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                oof = cross_val_predict(
                    pipeline, X, y, cv=cv, method="predict_proba"
                )[:, 1]
            elapsed = time.perf_counter() - started

            pr_auc, roc_auc = _fold_metrics(y_array, oof, cv, X)

            nguong = pick_threshold(
                y_array,
                oof,
                criterion,
                cost_fn=cost_fn,
                cost_fp=cost_fp,
                sample_fraction=sample_fraction,
                days=days,
            )
            met = metrics_at_threshold(
                y_array,
                oof,
                nguong,
                cost_fn=cost_fn,
                cost_fp=cost_fp,
                sample_fraction=sample_fraction,
                days=days,
            )

            rows.append({
                "model": model,
                "strategy": strategy,
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
                "n_distinct_scores": int(np.unique(oof).size),
                "seconds_per_fold": elapsed / cv.get_n_splits(),
                "total_seconds": elapsed,
            })
            scores[f"{model}__{strategy}"] = oof
            _save_checkpoint(checkpoint_path, rows, scores)

            if verbose:
                troi = time.perf_counter() - bat_dau_luoi
                con_lai = (tong - thu_tu) * troi / max(thu_tu - len(xong), 1)
                print(
                    f"[{thu_tu:2d}/{tong}] {model:20s} {strategy:12s} "
                    f"PR-AUC {np.mean(pr_auc):.4f} ± {np.std(pr_auc):.4f}  "
                    f"R@τ* {met.recall:.3f}  P@τ* {met.precision:.3f}  "
                    f"({elapsed / 60:5.1f} phút, còn khoảng {con_lai / 60:.0f} phút)"
                )

    table = (
        pd.DataFrame(rows)[GRID_COLUMNS]
        .sort_values("pr_auc_mean", ascending=False)
        .reset_index(drop=True)
    )
    return (table, scores) if return_scores else table


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
