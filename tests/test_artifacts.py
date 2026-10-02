"""Hiện vật mô hình — T-35…T-37, AR-02, 06 §6.

Hai phần:

* **Đơn vị** — hàm dựng và hàm kiểm tra trong ``src/artifacts.py``, trên dữ liệu
  tổng hợp. Luôn chạy.
* **Tích hợp** — đọc chính các tệp mà notebook 08 đã ghi ra trong ``models/`` và
  ``data/``. Tự bỏ qua khi chưa có hiện vật (máy sạch chưa chạy notebook). Đây là
  bản thi hành tự động của các điều kiện "xong là khi" ở T-35, T-36, T-37.
"""

import json
import re

import numpy as np
import pandas as pd
import pytest

from src import artifacts
from src.config import (
    EXPLAINER_PATH,
    METRICS_PATH,
    MODEL_PATH,
    SAMPLE_POOL_PATH,
    TEST_SET_PATH,
    THRESHOLD_PATH,
)
from src.features import FEATURE_ORDER, RAW_REQUIRED_COLUMNS, V_COLUMNS, build_features

TAU = 0.05


@pytest.fixture
def scored():
    """2.000 giao dịch thô kèm nhãn và điểm: 60 gian lận, trong đó 25 điểm thấp."""
    rng = np.random.default_rng(3)
    n_rows = 2000
    raw = pd.DataFrame({"Time": rng.uniform(0, 172_000, n_rows)})
    for column in V_COLUMNS:
        raw[column] = rng.normal(size=n_rows)
    raw["Amount"] = rng.lognormal(3.0, 1.2, n_rows).round(2)

    y = np.zeros(n_rows, dtype=int)
    y[:60] = 1
    scores = rng.uniform(0, TAU / 2, n_rows)            # hợp lệ điển hình
    scores[:35] = rng.uniform(0.6, 1.0, 35)              # gian lận dễ
    scores[35:60] = rng.uniform(0.0, 0.4, 25)            # gian lận khó
    scores[60:90] = rng.uniform(TAU, 0.9, 30)            # hợp lệ khó (cảnh báo giả)
    return raw, y, scores


# --------------------------------------------------------------------------
# Tiện ích JSON
# --------------------------------------------------------------------------

def test_to_jsonable_converts_numpy_and_drops_non_finite():
    payload = {
        "a": np.float32(0.25),
        "b": np.int64(7),
        "c": np.array([1.0, np.nan, np.inf]),
        "d": np.bool_(True),
        "e": pd.DataFrame({"x": [1, 2]}),
    }
    out = artifacts.to_jsonable(payload)

    assert out == {"a": 0.25, "b": 7, "c": [1.0, None, None], "d": True, "e": [{"x": 1}, {"x": 2}]}
    json.dumps(out, allow_nan=False)                    # JSON chặt, trình duyệt đọc được


def test_to_jsonable_rejects_unknown_types():
    with pytest.raises(TypeError):
        artifacts.to_jsonable({"x": object()})


def test_write_json_round_trips_float64_exactly(tmp_path):
    """TC-12 cần điểm trong JSON giống hệt điểm trong bộ nhớ, không làm tròn."""
    values = np.random.default_rng(0).random(500).astype("float32").astype("float64")
    path = artifacts.write_json(tmp_path / "x.json", {"v": values})

    assert np.array_equal(np.array(artifacts.read_json(path)["v"]), values)
    assert not (tmp_path / "x.json.tmp").exists()


def test_write_json_turns_nan_into_null(tmp_path):
    path = artifacts.write_json(tmp_path / "x.json", {"v": float("nan")})
    assert artifacts.read_json(path) == {"v": None}


def test_utc_timestamp_format():
    assert artifacts._TIMESTAMP.match(artifacts.utc_timestamp())


def test_score_fingerprint_is_order_and_bit_sensitive():
    scores = np.array([0.1, 0.2, 0.3])
    assert artifacts.score_fingerprint(scores) == artifacts.score_fingerprint(scores.copy())
    assert artifacts.score_fingerprint(scores) != artifacts.score_fingerprint(scores[::-1])
    assert artifacts.score_fingerprint(scores) != artifacts.score_fingerprint(scores + 1e-15)


# --------------------------------------------------------------------------
# threshold.json
# --------------------------------------------------------------------------

def _threshold(**overrides):
    kwargs = dict(
        default_threshold=0.0232,
        alternatives={"min_expected_cost": 0.0232, "max_f1": 0.72, "default_naive": 0.5},
        model_version="xgb_test",
        trained_at="2026-09-27T10:00:00Z",
    )
    kwargs.update(overrides)
    return artifacts.threshold_payload(**kwargs)


def test_threshold_payload_is_valid():
    assert artifacts.validate_threshold(_threshold()) == []


def test_threshold_default_must_equal_the_selected_alternative():
    problems = artifacts.validate_threshold(_threshold(default_threshold=0.03))
    assert any("khác phương án min_expected_cost" in p for p in problems)


@pytest.mark.parametrize("bad", [0.0, 1.0, 1.5])
def test_threshold_out_of_range_is_rejected(bad):
    payload = _threshold(default_threshold=bad, alternatives={"min_expected_cost": bad, "default_naive": 0.5})
    assert artifacts.validate_threshold(payload)


def test_threshold_missing_key_is_reported():
    payload = _threshold()
    del payload["model_version"]
    assert artifacts.validate_threshold(payload) == ["thiếu khóa: model_version"]


# --------------------------------------------------------------------------
# metrics.json
# --------------------------------------------------------------------------

def _metrics(y, scores):
    from src.evaluate import curve_points, headline_metrics
    from src.threshold import confusion_counts

    tp, fp, fn, tn = confusion_counts(y, scores, TAU)
    return artifacts.metrics_payload(
        model_version="xgb_test",
        trained_at="2026-09-27T10:00:00Z",
        dataset={"n_test": len(y), "n_fraud_test": int(y.sum())},
        headline=artifacts.headline_block(headline_metrics(y, scores, TAU, n_boot=20)),
        confusion_at_default={"tp": tp, "fp": fp, "fn": fn, "tn": tn},
        curves=curve_points(y, scores),
        cost_curve=artifacts.cost_curve_points(y, scores),
        y_true=y,
        y_scores=scores,
        grid_results=[{"model": "xgboost", "strategy": "none", "pr_auc_mean": 0.8, "pr_auc_std": 0.03}],
        shap_global=[{"feature": "V14", "mean_abs_shap": 2.7}],
        split_comparison={"random_stratified": {"pr_auc": 0.8}, "temporal": {"pr_auc": 0.7}},
        extra_section={"note": "đi sau các khóa bắt buộc"},
    )


def test_metrics_payload_is_valid_and_ordered(scored):
    _, y, scores = scored
    payload = _metrics(y, scores)

    assert artifacts.validate_metrics(payload) == []
    assert list(payload)[: len(artifacts.METRICS_KEYS)] == list(artifacts.METRICS_KEYS)
    assert list(payload)[-1] == "extra_section"
    assert payload["headline"]["baseline_pr_auc"] == pytest.approx(y.mean())


def test_metrics_test_scores_keep_full_precision(scored):
    _, y, scores = scored
    payload = json.loads(json.dumps(_metrics(y, scores), allow_nan=False))
    assert np.array_equal(np.array(payload["test_scores"]["y_score"]), scores)


def test_metrics_cost_curve_matches_threshold_module(scored):
    from src.threshold import metrics_at_threshold

    _, y, scores = scored
    point = artifacts.cost_curve_points(y, scores)[50]
    expected = metrics_at_threshold(y, scores, point["threshold"])

    assert point["cost"] == pytest.approx(expected.expected_cost)
    assert point["alerts"] == expected.alerts


def test_curve_points_keep_the_high_precision_region():
    """Rút mẫu đường PR không được bỏ mất vùng precision cao (lỗi thấy ở UI-04, giai đoạn 8).

    Với 0,2% lớp dương, gần hết ngưỡng nằm ở vùng precision ≈ 0. Lấy đều theo chỉ số thì
    100 điểm chỉ còn một hai điểm có precision > 0,5 và đường PR vẽ thành đoạn thẳng.
    """
    from src.evaluate import curve_points, thin_curve

    rng = np.random.default_rng(11)
    y = (rng.random(20_000) < 0.002).astype(int)
    scores = np.where(y == 1, rng.beta(6, 2, y.size), rng.beta(1, 60, y.size))
    pr = curve_points(y, scores, max_points=100)["pr_curve"]
    recall, precision = np.array(pr["recall"]), np.array(pr["precision"])

    assert len(recall) == len(precision) == len(pr["thresholds"]) <= 100
    assert (recall[0], recall[-1], precision[-1]) == (1.0, 0.0, 1.0)      # giữ hai đầu
    assert np.all(np.diff(recall) <= 0)                                   # đúng thứ tự
    assert np.count_nonzero(precision > 0.5) >= 20

    idx = thin_curve([0.0, 0.5, 1.0], [0.0, 0.5, 1.0], max_points=10)
    assert idx.tolist() == [0, 1, 2]                                      # ít điểm: giữ nguyên


def test_metrics_inconsistent_counts_are_reported(scored):
    _, y, scores = scored
    payload = _metrics(y, scores)
    payload["dataset"]["n_fraud_test"] += 1
    payload["confusion_at_default"]["tn"] -= 1

    problems = artifacts.validate_metrics(payload)
    assert "dataset.n_fraud_test khác số nhãn 1 trong test_scores" in problems
    assert "confusion_at_default không cộng về đúng n_test" in problems


# --------------------------------------------------------------------------
# sample_pool.json
# --------------------------------------------------------------------------

def _pool(scored, **kwargs):
    raw, y, scores = scored
    return artifacts.select_sample_pool(raw, y, scores, reference_threshold=TAU, **kwargs)


def test_sample_pool_sizes_and_groups(scored):
    items = _pool(scored)
    counts = pd.Series([i["category"] for i in items]).value_counts()

    assert len(items) == 200
    assert counts["fraud_hard"] == 25 and counts["fraud_easy"] == 25      # 50 gian lận
    assert counts["legit_hard"] == 30 and counts["legit_easy"] == 120     # 150 hợp lệ
    payload = artifacts.sample_pool_payload(items, generated_at="2026-09-27T10:00:00Z",
                                            model_version="xgb_test", reference_threshold=TAU)
    assert artifacts.validate_sample_pool(payload) == []


def test_sample_pool_rules_hold_for_every_item(scored):
    for item in _pool(scored):
        score = item["risk_score"]
        if item["category"] == "fraud_hard":
            assert item["label"] == 1 and score < 0.5
        elif item["category"] == "fraud_easy":
            assert item["label"] == 1 and score >= 0.5
        elif item["category"] == "legit_hard":
            assert item["label"] == 0 and score >= TAU
        else:
            assert item["label"] == 0 and score < TAU


def test_sample_pool_text_uses_vietnamese_decimal_comma(scored):
    """Giao diện in nguyên văn mô tả và luật chia nhóm — số phải viết "0,5", không "0.5"."""
    items = _pool(scored)
    payload = artifacts.sample_pool_payload(items, generated_at="2026-09-27T10:00:00Z",
                                            model_version="xgb_test", reference_threshold=TAU)
    texts = [i["description"] for i in items] + [c["rule"] for c in payload["categories"].values()]

    assert not any(re.search(r"\d\.\d", text) for text in texts)
    assert payload["categories"]["legit_hard"]["rule"] == "hợp lệ, điểm ≥ τ* = 0,05000 (cảnh báo giả)"


def test_sample_pool_is_deterministic_and_traceable(scored):
    raw, y, scores = scored
    first, second = _pool(scored), _pool(scored)

    assert first == second
    for item in first:
        row = item["test_row"]
        assert item["risk_score"] == scores[row] and item["label"] == y[row]
        assert list(item["features"]) == RAW_REQUIRED_COLUMNS
        assert item["features"]["V14"] == raw["V14"].iloc[row]
    assert [i["id"] for i in first] == [f"S-{k:03d}" for k in range(1, 201)]


def test_sample_pool_caps_legit_hard_and_spreads_easy_scores(scored):
    items = _pool(scored, max_legit_hard=10)
    legit_easy = [i["risk_score"] for i in items if i["category"] == "legit_easy"]

    assert sum(i["category"] == "legit_hard" for i in items) == 10
    assert len(legit_easy) == 140
    # trải đều theo thứ hạng: có cả điểm thấp nhất lẫn cao nhất của nhóm
    raw, y, scores = scored
    pool_scores = scores[(y == 0) & (scores < TAU)]
    assert min(legit_easy) == pool_scores.min() and max(legit_easy) == pool_scores.max()


def test_validate_sample_pool_flags_thin_hard_group(scored):
    raw, y, scores = scored
    scores = scores.copy()
    scores[35:50] = 0.9                                  # chỉ còn 10 gian lận khó
    items = artifacts.select_sample_pool(raw, y, scores, reference_threshold=TAU)

    problems = artifacts.validate_sample_pool({"items": items})
    assert problems == ["nhóm fraud_hard chỉ có 10 mẫu, cần ít nhất 20"]


def test_select_sample_pool_requires_raw_columns(scored):
    raw, y, scores = scored
    with pytest.raises(ValueError, match="thiếu cột: Time"):
        artifacts.select_sample_pool(raw.drop(columns="Time"), y, scores, reference_threshold=TAU)


# --------------------------------------------------------------------------
# Tích hợp — hiện vật thật do notebook 08 sinh ra
# --------------------------------------------------------------------------

_ALL = (MODEL_PATH, EXPLAINER_PATH, METRICS_PATH, THRESHOLD_PATH, SAMPLE_POOL_PATH, TEST_SET_PATH)
needs_artifacts = pytest.mark.skipif(
    not all(p.exists() for p in _ALL),
    reason="chưa có hiện vật — chạy notebooks/08_export_artifacts.ipynb",
)


@pytest.fixture(scope="module")
def real():
    import joblib

    test_set = pd.read_parquet(TEST_SET_PATH)
    return {
        "model": joblib.load(MODEL_PATH),
        "metrics": artifacts.read_json(METRICS_PATH),
        "threshold": artifacts.read_json(THRESHOLD_PATH),
        "pool": artifacts.read_json(SAMPLE_POOL_PATH),
        "test_set": test_set,
    }


@needs_artifacts
def test_t35_model_is_a_complete_pipeline(real):
    """AR-02 — bước chuẩn hoá nằm TRONG tệp, đã fit, và nhận đúng 31 đặc trưng."""
    model = real["model"]
    assert list(model.named_steps) == ["preprocess", "clf"]
    scaler = model.named_steps["preprocess"].named_transformers_["amount"]
    assert type(scaler).__name__ == "RobustScaler" and hasattr(scaler, "center_")
    assert list(model.feature_names_in_) == FEATURE_ORDER


@needs_artifacts
def test_t35_reloaded_model_reproduces_exported_scores(real):
    """Nạp lại ``model.joblib`` cho đúng từng bit điểm đã xuất (AR-02, T-35)."""
    test_set = real["test_set"]
    scores = real["model"].predict_proba(build_features(test_set))[:, 1]

    assert np.array_equal(scores.astype("float64"), test_set["risk_score"].to_numpy())
    assert np.array_equal(scores.astype("float64"), np.array(real["metrics"]["test_scores"]["y_score"]))
    assert np.array_equal(test_set["Class"].to_numpy(), np.array(real["metrics"]["test_scores"]["y_true"]))


@needs_artifacts
def test_t36_json_artifacts_follow_the_documented_structure(real):
    assert artifacts.validate_threshold(real["threshold"]) == []
    assert artifacts.validate_metrics(real["metrics"]) == []
    assert real["threshold"]["model_version"] == real["metrics"]["model_version"]
    assert real["threshold"]["model_version"] == real["pool"]["model_version"]


@needs_artifacts
def test_t36_headline_matches_exported_scores(real):
    from sklearn.metrics import average_precision_score

    from src.threshold import confusion_counts

    metrics, tau = real["metrics"], real["threshold"]["default_threshold"]
    y, s = metrics["test_scores"]["y_true"], metrics["test_scores"]["y_score"]

    assert metrics["headline"]["pr_auc"]["value"] == pytest.approx(average_precision_score(y, s), abs=1e-12)
    tp, fp, fn, tn = confusion_counts(y, s, tau)
    assert metrics["confusion_at_default"] == {"threshold": tau, "tp": tp, "fp": fp, "fn": fn, "tn": tn}
    assert metrics["headline"]["pr_auc"]["value"] >= 0.75            # AC-M1
    assert metrics["headline"]["recall"]["value"] >= 0.75            # AC-M2


@needs_artifacts
def test_t36_pr_curves_show_the_high_precision_region(real):
    """UI-04 vẽ thẳng các đường này: vùng precision cao phải có đủ điểm."""
    metrics = real["metrics"]
    curves = [metrics["pr_curve"]]
    if metrics.get("strategy_pr_curves"):
        curves += metrics["strategy_pr_curves"]["curves"]
    for curve in curves:
        precision = np.array(curve["precision"])
        assert np.count_nonzero(precision >= 0.5) >= 50, curve.get("strategy", "pr_curve")


@needs_artifacts
def test_t36_explainer_is_additive_on_the_exported_model(real):
    """``base + ΣSHAP`` = logit của điểm — tính chất mà biểu đồ thác nước dựa vào."""
    import joblib

    explainer = joblib.load(EXPLAINER_PATH)
    model = real["model"]
    # Lấy cả các vụ gian lận: điểm của chúng sát 1, nơi phép cộng dễ lệch nhất
    test_set = real["test_set"]
    rows = np.concatenate([np.flatnonzero(test_set["Class"] == 1), np.arange(200)])
    Z = model[:-1].transform(build_features(test_set.iloc[rows]))
    shap_values = explainer.shap_values(Z)
    base = float(np.ravel(explainer.expected_value)[0])
    # So với margin của booster chứ không với logit(xác suất): xác suất float32
    # sát 1 đã mất độ chính xác, logit của nó lệch hàng chục phần trăm
    margin = model[-1].predict(Z, output_margin=True)

    assert shap_values.shape == (rows.size, len(FEATURE_ORDER))
    assert np.abs(base + shap_values.sum(axis=1) - margin).max() < 1e-3


@needs_artifacts
def test_t37_sample_pool_meets_the_task_conditions(real):
    pool = real["pool"]
    assert artifacts.validate_sample_pool(pool) == []
    assert 180 <= len(pool["items"]) <= 220


@needs_artifacts
def test_t37_sample_pool_scores_are_what_the_model_returns(real):
    """Chấm lại 30 cột thô của từng mẫu — đúng đường đi của API — ra đúng điểm đã lưu."""
    items = real["pool"]["items"]
    raw = pd.DataFrame([item["features"] for item in items])
    scores = real["model"].predict_proba(build_features(raw))[:, 1].astype("float64")

    assert np.array_equal(scores, np.array([item["risk_score"] for item in items]))
    labels = real["test_set"]["Class"].to_numpy()[[item["test_row"] for item in items]]
    assert np.array_equal(labels, [item["label"] for item in items])
