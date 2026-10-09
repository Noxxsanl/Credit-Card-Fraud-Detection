"""Nạp hiện vật một lần lúc khởi động (AR-01, docs/03 §3.4).

API không huấn luyện, không đọc ``creditcard.csv``. Nó chỉ nạp các tệp mà notebook 08
đã xuất, kiểm tra chúng bằng chính các hàm ``validate_*`` của ``src/artifacts.py``,
rồi giữ trong bộ nhớ suốt vòng đời tiến trình.

Sáu tệp: ``model.joblib``, ``metrics.json``, ``threshold.json``, ``oof_scores.npz``,
``test_set.parquet``, ``sample_pool.json``. ``explainer.joblib`` không còn cần: SHAP tính bằng
``pred_contribs`` của XGBoost (``api/serving.py``), trùng từng bit với tệp đó.

Hiện vật thiếu hay sai thì ``load_artifacts`` ném ``ArtifactError``. ``main.py`` bắt
lỗi đó để tiến trình vẫn sống và ``/health`` trả 503 ``MODEL_NOT_LOADED`` thay vì
sập (TC-42).
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from importlib import metadata

import joblib
import numpy as np
import pandas as pd

from src import artifacts as art
from src.features import RAW_REQUIRED_COLUMNS, build_features
from src.threshold import pick_threshold

from .config import BLOCK_MIN_PRECISION, Settings
from .serving import FastScorer, UnsupportedPipeline

log = logging.getLogger("api")

#: Ngày thứ hai của dữ liệu bắt đầu ở giây này — chế độ phát lại đọc từ đây (UI-05)
DAY_TWO_START = 86_400.0

#: Số luồng XGBoost khi chấm điểm lúc phục vụ (xem load_artifacts)
SERVING_THREADS = 1

#: Thư viện mà API thật sự nạp lúc chạy — chỉ những gói này mới cần trùng phiên bản lúc xuất
#: hiện vật. shap có trong metrics.json (notebook dùng) nhưng ảnh api không cài.
SERVING_PACKAGES = ("numpy", "pandas", "scikit-learn", "imbalanced-learn", "xgboost", "joblib")


class ArtifactError(RuntimeError):
    pass


@dataclass
class Artifacts:
    model: object
    scorer: FastScorer
    metrics: dict
    threshold: dict
    oof: dict
    sample_pool: dict
    test_set: pd.DataFrame
    loaded_at: datetime
    # Suy ra lúc nạp, dùng lại ở mọi yêu cầu
    y_test: np.ndarray = field(repr=False, default=None)
    s_test: np.ndarray = field(repr=False, default=None)
    oof_candidates: np.ndarray = field(repr=False, default=None)
    amounts_sorted: np.ndarray = field(repr=False, default=None)
    samples_by_id: dict = field(repr=False, default_factory=dict)
    metrics_json: bytes = field(repr=False, default=b"")
    #: Từ ngưỡng này trở lên API đề xuất chặn (05 §2) — chọn trên out-of-fold, xem block_threshold_from
    block_threshold: float = 1.0

    @property
    def model_version(self) -> str:
        return self.threshold["model_version"]

    @property
    def trained_at(self) -> str:
        return self.threshold["trained_at"]

    @property
    def default_threshold(self) -> float:
        return float(self.threshold["default_threshold"])

    @property
    def test_fraction(self) -> float:
        return float(self.metrics["dataset"]["test_fraction"])

    @property
    def days(self) -> float:
        return float(self.metrics["dataset"]["days"])

    @property
    def model_feature_names(self) -> list[str]:
        return list(self.metrics["training"]["model_feature_names"])

    def metrics_section(self, name: str) -> bytes:
        return json.dumps({"model_version": self.model_version, name: self.metrics[name]},
                          ensure_ascii=False, allow_nan=False).encode("utf-8")


def _require(problems: list[str], what: str) -> None:
    if problems:
        raise ArtifactError(f"{what} sai cấu trúc: " + "; ".join(problems[:5]))


def load_artifacts(settings: Settings) -> Artifacts:
    models = settings.models_dir
    paths = {
        "model": models / "model.joblib",
        "metrics": models / "metrics.json",
        "threshold": models / "threshold.json",
        "oof": models / "oof_scores.npz",
        "sample_pool": settings.sample_pool_path,
        "test_set": settings.test_set_path,
    }
    missing = [str(p) for p in paths.values() if not p.exists()]
    if missing:
        raise ArtifactError("Thiếu hiện vật — chạy notebooks/08_export_artifacts.ipynb: " + ", ".join(missing))

    metrics = art.read_json(paths["metrics"])
    threshold = art.read_json(paths["threshold"])
    sample_pool = art.read_json(paths["sample_pool"])
    _require(art.validate_metrics(metrics), "metrics.json")
    _require(art.validate_threshold(threshold), "threshold.json")
    _require(art.validate_sample_pool(sample_pool), "sample_pool.json")
    versions = {threshold["model_version"], metrics["model_version"], sample_pool["model_version"]}
    if len(versions) != 1:
        raise ArtifactError(f"Hiện vật đến từ nhiều lần xuất khác nhau: {sorted(versions)}")
    if "training" not in metrics:
        raise ArtifactError("metrics.json thiếu khóa training — xuất lại bằng notebook 08")

    oof = art.read_oof_scores(paths["oof"])
    _require(art.validate_oof_scores(oof, fingerprint=metrics.get("fingerprint", {}).get("oof_scores_sha256")),
             "oof_scores.npz")

    _warn_on_version_drift(metrics.get("environment", {}).get("packages", {}))
    model = joblib.load(paths["model"])
    # Phục vụ bằng một luồng. Dự đoán của XGBoost không phụ thuộc số luồng (trùng từng bit trên
    # 56.746 giao dịch; chỉ huấn luyện mới phụ thuộc — docs/10 §6), còn 12 luồng cho một giao dịch
    # chỉ tốn chi phí đồng bộ và tranh CPU với các yêu cầu khác: p95 của /score giảm khoảng một nửa.
    model[-1].set_params(n_jobs=SERVING_THREADS)
    try:
        scorer = FastScorer(model)
    except UnsupportedPipeline as exc:
        raise ArtifactError(f"model.joblib không đúng dạng pipeline mà API phục vụ được: {exc}") from exc
    if scorer.feature_names != list(metrics["training"]["model_feature_names"]):
        raise ArtifactError("thứ tự đặc trưng của model.joblib khác metrics.json → training.model_feature_names")

    test_set = pd.read_parquet(paths["test_set"])
    missing_columns = [c for c in (*RAW_REQUIRED_COLUMNS, "Class") if c not in test_set.columns]
    if missing_columns:
        raise ArtifactError("test_set.parquet thiếu cột: " + ", ".join(missing_columns))

    y_test = np.asarray(metrics["test_scores"]["y_true"], dtype="int8")
    s_test = np.asarray(metrics["test_scores"]["y_score"], dtype="float64")
    if len(test_set) != s_test.size:
        raise ArtifactError("test_set.parquet và metrics.json khác số dòng — xuất lại bằng notebook 08")
    if "risk_score" not in test_set.columns:
        # Notebook 03 chạy lại sau notebook 08 thì mất cột này (docs/10 §3)
        log.warning("test_set.parquet thiếu cột risk_score — chấm lại cả tập lúc khởi động")
        test_set["risk_score"] = model.predict_proba(build_features(test_set))[:, 1].astype("float64")
    if not np.array_equal(test_set["risk_score"].to_numpy(dtype="float64"), s_test):
        raise ArtifactError("risk_score trong test_set.parquet khác test_scores của metrics.json")

    # Unpickle thành công chưa đủ: lệch phiên bản thư viện có thể cho điểm khác mà không báo lỗi
    probe = test_set.iloc[:: max(1, len(test_set) // 50)]
    probe_features = build_features(probe)
    rescored = model.predict_proba(probe_features)[:, 1].astype("float64")
    if not np.allclose(rescored, probe["risk_score"].to_numpy(), rtol=0, atol=1e-6):
        raise ArtifactError("model.joblib chấm lại tập kiểm thử ra điểm khác — lệch phiên bản thư viện?")
    # Đường tắt phải là CÙNG phép tính, không phải một phép gần đúng: so khớp từng bit
    if not np.array_equal(scorer.predict(probe_features), rescored):
        raise ArtifactError("đường chấm nhanh (api/serving.py) lệch pipeline đầy đủ — không phục vụ")
    shap_values, bias = scorer.contributions(probe_features.iloc[:5])
    if not np.allclose(bias, metrics["training"]["shap_base_value"], rtol=0, atol=1e-5):
        raise ArtifactError("giá trị cơ sở SHAP khác metrics.json → training.shap_base_value")
    if not np.allclose(shap_values.sum(axis=1) + bias, scorer.margin(probe_features.iloc[:5]), rtol=0, atol=1e-3):
        raise ArtifactError("SHAP không cộng đủ thành margin của mô hình")

    oof_candidates = np.unique(oof["y_score"])
    loaded = Artifacts(
        model=model,
        scorer=scorer,
        metrics=metrics,
        threshold=threshold,
        oof=oof,
        sample_pool=sample_pool,
        test_set=test_set,
        loaded_at=datetime.now(timezone.utc),
        y_test=y_test,
        s_test=s_test,
        oof_candidates=oof_candidates,
        amounts_sorted=np.sort(test_set["Amount"].to_numpy(dtype="float64")),
        samples_by_id={item["id"]: item for item in sample_pool["items"]},
        metrics_json=json.dumps(metrics, ensure_ascii=False, allow_nan=False).encode("utf-8"),
        block_threshold=block_threshold_from(oof, oof_candidates),
    )
    log.info("Đã nạp hiện vật %s (huấn luyện %s), τ* = %.6f, ngưỡng đề xuất chặn = %.4f (precision ≥ %.0f%% "
             "trên out-of-fold)", loaded.model_version, loaded.trained_at, loaded.default_threshold,
             loaded.block_threshold, BLOCK_MIN_PRECISION * 100)
    return loaded


def block_threshold_from(oof: dict, candidates: np.ndarray, min_precision: float = BLOCK_MIN_PRECISION) -> float:
    """Ngưỡng "đề xuất chặn": nhỏ nhất mà mọi ngưỡng từ đó trở lên có precision ≥ ``min_precision``
    trên out-of-fold (05 §2).

    Chặn tự động một khách hợp lệ đắt hơn nhiều một cảnh báo giả phải thẩm định, nên mức chặn
    đặt theo precision mong muốn chứ không theo bội số của τ. Chọn trên out-of-fold như mọi
    ngưỡng khác (ML-08); không phụ thuộc τ người dùng đặt.
    """
    return pick_threshold(oof["y_true"], oof["y_score"], "min_precision", value=min_precision,
                          thresholds=candidates, sample_fraction=oof["train_fraction"], days=oof["days"])


def _warn_on_version_drift(expected: dict) -> None:
    for package, version in expected.items():
        if package not in SERVING_PACKAGES:
            continue
        try:
            installed = metadata.version(package)
        except metadata.PackageNotFoundError:
            installed = None
        if version and installed != version:
            log.warning("Phiên bản %s đang cài (%s) khác lúc xuất hiện vật (%s)", package, installed, version)
