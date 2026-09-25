"""TC-20…TC-21 — chặn rò rỉ dữ liệu ngay ở mức cấu trúc pipeline.

Rò rỉ ở bài toán này không làm chương trình đổ; nó chỉ làm mọi chỉ số đẹp lên
một cách vô nghĩa. Vì vậy phải bắt bằng kiểm thử chứ không thể trông vào việc
đọc lại mã (AC-M4).
"""

import numpy as np
import pandas as pd
import pytest
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.exceptions import NotFittedError
from sklearn.pipeline import Pipeline as SklearnPipeline
from sklearn.utils.validation import check_is_fitted

from src.modeling import (
    DEFAULT_MODELS,
    DEFAULT_STRATEGIES,
    SAMPLING_STRATEGY,
    build_pipeline,
    imbalance_ratio,
    make_cv,
    oof_scores,
)
from src.features import V_COLUMNS, build_features

RESAMPLING_STRATEGIES = ("undersample", "smote", "smote_tomek")


@pytest.fixture
def dataset():
    """600 giao dịch, 30 gian lận — đủ để SMOTE và StratifiedKFold(5) chạy được."""
    rng = np.random.default_rng(11)
    n_rows, n_fraud = 600, 30

    raw = {"Time": rng.uniform(0, 172_000, n_rows)}
    for column in V_COLUMNS:
        raw[column] = rng.normal(size=n_rows)
    raw["Amount"] = rng.lognormal(3.0, 1.2, n_rows)

    y = np.zeros(n_rows, dtype=int)
    fraud_idx = rng.choice(n_rows, size=n_fraud, replace=False)
    y[fraud_idx] = 1

    frame = pd.DataFrame(raw)
    # Cho lớp dương một tín hiệu thật, nếu không mô hình không học được gì
    frame.loc[fraud_idx, "V14"] -= 3.0
    frame.loc[fraud_idx, "V17"] -= 2.0

    return build_features(frame), pd.Series(y)


# --------------------------------------------------------------------------
# TC-20 — resampler bắt buộc nằm trong pipeline của imblearn
# --------------------------------------------------------------------------

@pytest.mark.parametrize("strategy", DEFAULT_STRATEGIES)
def test_tc20_every_pipeline_is_an_imblearn_pipeline(strategy, dataset):
    X, y = dataset
    pipeline = build_pipeline("logistic_regression", strategy, y=y)

    assert isinstance(pipeline, ImbPipeline)
    # sklearn.pipeline.Pipeline KHÔNG bỏ qua bước resample khi transform tập
    # validation, nên dùng nhầm lớp này là rò rỉ ngay
    assert not type(pipeline) is SklearnPipeline


@pytest.mark.parametrize("strategy", RESAMPLING_STRATEGIES)
def test_tc20_resampler_sits_inside_the_pipeline(strategy, dataset):
    X, y = dataset
    pipeline = build_pipeline("logistic_regression", strategy, y=y)

    assert "resample" in pipeline.named_steps
    assert list(pipeline.named_steps) == ["preprocess", "resample", "clf"]


@pytest.mark.parametrize("strategy", ("none", "class_weight"))
def test_tc20_strategies_without_resampling_have_no_resampler(strategy, dataset):
    X, y = dataset
    pipeline = build_pipeline("logistic_regression", strategy, y=y)

    assert "resample" not in pipeline.named_steps


def test_tc20_resamplers_do_not_balance_to_one_to_one(dataset):
    """Cân bằng hoàn toàn làm méo xác suất tiên nghiệm — dừng ở 1:10 (04 §3.1)."""
    X, y = dataset
    for strategy in RESAMPLING_STRATEGIES:
        resampler = build_pipeline("logistic_regression", strategy, y=y).named_steps["resample"]
        assert resampler.sampling_strategy == SAMPLING_STRATEGY


# --------------------------------------------------------------------------
# TC-21 — không bước nào được fit sẵn trước khi vào cross-validation
# --------------------------------------------------------------------------

@pytest.mark.parametrize("strategy", DEFAULT_STRATEGIES)
def test_tc21_pipeline_steps_are_unfitted_before_cv(strategy, dataset):
    X, y = dataset
    pipeline = build_pipeline("logistic_regression", strategy, y=y)

    with pytest.raises(NotFittedError):
        check_is_fitted(pipeline.named_steps["preprocess"])
    with pytest.raises(NotFittedError):
        check_is_fitted(pipeline.named_steps["clf"])


def test_tc21_scaler_statistics_come_only_from_training_data(dataset):
    """Scaler phải học trung vị/IQR của tập huấn luyện, không phải toàn bộ dữ liệu (ML-04)."""
    X, y = dataset
    train_only = build_pipeline("logistic_regression", "none", y=y)
    train_only.fit(X.iloc[:400], y.iloc[:400])

    everything = build_pipeline("logistic_regression", "none", y=y)
    everything.fit(X, y)

    center_train = train_only.named_steps["preprocess"].named_transformers_["amount"].center_
    center_all = everything.named_steps["preprocess"].named_transformers_["amount"].center_

    assert center_train != pytest.approx(center_all)


def test_resampling_does_not_change_prediction_row_count(dataset):
    """Lấy mẫu lại chỉ tác động lúc fit; lúc dự đoán số dòng phải giữ nguyên."""
    X, y = dataset
    pipeline = build_pipeline("logistic_regression", "smote", y=y).fit(X, y)

    assert pipeline.predict_proba(X).shape[0] == len(X)


def test_fitting_does_not_mutate_input_frame(dataset):
    X, y = dataset
    snapshot = X.copy(deep=True)

    build_pipeline("logistic_regression", "smote", y=y).fit(X, y)

    pd.testing.assert_frame_equal(X, snapshot)


# --------------------------------------------------------------------------
# Hành vi của cross-validation
# --------------------------------------------------------------------------

def test_cv_is_stratified_and_every_fold_has_positives(dataset):
    X, y = dataset
    cv = make_cv()

    for _, test_idx in cv.split(X, y):
        assert y.iloc[test_idx].sum() > 0


def test_oof_scores_cover_every_row_exactly_once(dataset):
    X, y = dataset

    scores = oof_scores(build_pipeline("logistic_regression", "none", y=y), X, y)

    assert scores.shape == (len(X),)
    assert np.all((scores >= 0) & (scores <= 1))


def test_imbalance_ratio_matches_definition():
    y = np.array([0] * 578 + [1])
    assert imbalance_ratio(y) == pytest.approx(578.0)


def test_unknown_model_or_strategy_is_rejected(dataset):
    X, y = dataset

    with pytest.raises(ValueError, match="Mô hình không hỗ trợ"):
        build_pipeline("mang_no_ron", "none", y=y)
    with pytest.raises(ValueError, match="Chiến lược không hỗ trợ"):
        build_pipeline("logistic_regression", "xoa_bot_lop_am", y=y)


@pytest.mark.parametrize("model", DEFAULT_MODELS)
def test_every_model_in_the_grid_can_fit_and_score(model, dataset):
    X, y = dataset
    pipeline = build_pipeline(model, "class_weight", y=y).fit(X, y)

    scores = pipeline.predict_proba(X)[:, 1]

    assert scores.shape == (len(X),)
    assert np.isfinite(scores).all()


# --------------------------------------------------------------------------
# Rà soát rò rỉ — docs/08 §3.1, T-28
#
# Bảy ô của danh sách kiểm được chuyển thành kiểm thử để chúng còn đúng cả sau
# ngày 12: đọc lại mã một lần thì chỉ đúng ở thời điểm đọc. Phần lớn là quét mã
# nguồn trong src/, scripts/, app.py và các ô code của notebook.
# --------------------------------------------------------------------------

import json
import re
from pathlib import Path

from src.config import RANDOM_STATE
from src.data import drop_duplicates, split_data

ROOT = Path(__file__).resolve().parents[1]

#: Notebook huấn luyện mô hình — nơi thứ tự nạp → loại trùng → chia → huấn luyện
#: phải được tôn trọng. 01 và 02 chỉ làm EDA và kiểm định thống kê.
TRAINING_NOTEBOOKS = ("03_baseline", "04_imbalance_strategies", "05_advanced_models")


def _notebook_code(path: Path) -> str:
    cells = json.loads(path.read_text(encoding="utf-8"))["cells"]
    return "\n".join("".join(c["source"]) for c in cells if c["cell_type"] == "code")


def _strip_comments(code: str) -> str:
    """Bỏ chú thích ``#`` và docstring để chỉ quét mã thật."""
    code = re.sub(r'"""[\s\S]*?"""', "", code)
    return "\n".join(line.split("#", 1)[0] for line in code.splitlines())


def _sources() -> dict[str, str]:
    files = {}
    for path in [*ROOT.glob("src/*.py"), *ROOT.glob("scripts/*.py"), ROOT / "app.py"]:
        files[path.relative_to(ROOT).as_posix()] = _strip_comments(path.read_text(encoding="utf-8"))
    for path in ROOT.glob("notebooks/*.ipynb"):
        files[path.relative_to(ROOT).as_posix()] = _strip_comments(_notebook_code(path))
    return files


def _hits(pattern: str) -> list[str]:
    found = []
    for name, code in _sources().items():
        for match in re.finditer(pattern, code):
            line = code[: match.start()].count("\n") + 1
            found.append(f"{name}:{line}: {code.splitlines()[line - 1].strip()}")
    return found


def test_leak_1_fit_resample_never_called_outside_a_pipeline():
    """Ô 1 — ``fit_resample`` chỉ được gọi ngầm bởi pipeline của imblearn (ML-01)."""
    assert _hits(r"\.fit_resample\(") == []


@pytest.mark.parametrize("notebook", TRAINING_NOTEBOOKS)
def test_leak_2_and_6_order_is_load_dedupe_split_then_train(notebook):
    """Ô 2 và 6 — nạp qua ``load_prepared`` (đã loại trùng lặp) → chia tập → mới
    tới bất kỳ lệnh huấn luyện hay lấy mẫu lại nào (ML-06, DS-20)."""
    code = _strip_comments(_notebook_code(ROOT / "notebooks" / f"{notebook}.ipynb"))

    load = code.find("load_prepared(")
    splits = [i for i in (code.find("split_data("), code.find("temporal_split(")) if i >= 0]
    assert splits, "notebook huấn luyện mà không chia tập?"
    split = min(splits)
    train_calls = [
        m.start()
        for m in re.finditer(r"\.fit\(|run_grid\(|run_search\(|oof_scores\(|cross_val_predict\(", code)
    ]

    assert load >= 0, "notebook huấn luyện phải nạp dữ liệu qua load_prepared()"
    assert "load_data(" not in code, "load_data() bỏ qua bước loại trùng lặp"
    assert load < split, "phải loại trùng lặp TRƯỚC khi chia tập"
    assert train_calls, "notebook huấn luyện mà không có lệnh huấn luyện nào?"
    assert split < min(train_calls), "có lệnh huấn luyện chạy trước khi chia tập"


def test_leak_3_no_scaler_fitted_on_the_whole_dataset():
    """Ô 3 — không có ``scaler.fit(X)`` đứng riêng; scaler chỉ sống trong pipeline (ML-04)."""
    assert _hits(r"(?i)(scaler\w*|Scaler\(\))\.fit(_transform)?\(") == []


def test_leak_4_every_cross_validation_is_stratified():
    """Ô 4 — không có ``KFold(`` thiếu tiền tố ``Stratified`` (ML-02)."""
    assert _hits(r"(?<!Stratified)KFold\(") == []
    assert _hits(r"cv\s*=\s*\d") == [], "cv=<số> để sklearn tự chọn KFold/StratifiedKFold"


def test_leak_5_threshold_is_never_picked_on_test_labels():
    """Ô 5 — ngưỡng chọn trên điểm out-of-fold hoặc tập huấn luyện (ML-08)."""
    pattern = r"(pick_threshold|threshold_alternatives|sensitivity_analysis)\(\s*\w*test"
    assert _hits(pattern) == []


def test_leak_6_duplicates_removed_before_split_leave_no_shared_rows():
    """Ô 6 — loại trùng lặp rồi mới chia thì không dòng nào có mặt ở cả hai tập."""
    rng = np.random.default_rng(5)
    frame = pd.DataFrame({"Time": rng.uniform(0, 172_000, 300)})
    for column in V_COLUMNS:
        frame[column] = rng.normal(size=300)
    frame["Amount"] = rng.lognormal(3.0, 1.0, 300)
    frame["Class"] = (rng.random(300) < 0.1).astype(int)
    frame = pd.concat([frame, frame.iloc[:40]], ignore_index=True)  # 40 dòng trùng

    X_train, X_test, _, _ = split_data(drop_duplicates(frame, verbose=False), as_features=False)
    shared = pd.merge(X_train, X_test, how="inner")

    assert shared.empty


def test_leak_7_every_literal_random_state_is_the_shared_seed():
    """Ô 7 — mọi ``random_state=`` trong mã là 42 hoặc tên biến trỏ về nó (ML-05)."""
    bad = [
        hit for hit in _hits(r"random_state\s*=\s*(\d+)")
        if not re.search(rf"random_state\s*=\s*{RANDOM_STATE}\b", hit)
    ]
    assert bad == []


@pytest.mark.parametrize("model", DEFAULT_MODELS)
@pytest.mark.parametrize("strategy", DEFAULT_STRATEGIES)
def test_leak_7_every_pipeline_step_carries_the_shared_seed(model, strategy, dataset):
    """Ô 7 — mọi tham số ``*random_state`` trong pipeline thực tế đều bằng 42."""
    _, y = dataset
    params = build_pipeline(model, strategy, y=y).get_params()
    seeds = {k: v for k, v in params.items() if k.endswith("random_state")}

    assert seeds, "pipeline không có bước ngẫu nhiên nào?"
    assert set(seeds.values()) == {RANDOM_STATE}, seeds


def test_leak_7_search_object_is_seeded_and_stratified(dataset):
    """Ô 4 và 7 cho RandomizedSearchCV của T-24."""
    from sklearn.model_selection import StratifiedKFold

    from src.modeling import run_search

    X, y = dataset
    search = run_search(X, y, n_iter=2, n_jobs=1, refit=False, verbose=0)

    assert search.random_state == RANDOM_STATE
    assert isinstance(search.cv, StratifiedKFold)
    assert isinstance(search.estimator, ImbPipeline)
    seeds = {k: v for k, v in search.estimator.get_params().items() if k.endswith("random_state")}
    assert set(seeds.values()) == {RANDOM_STATE}
