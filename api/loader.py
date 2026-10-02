"""Nạp hiện vật một lần lúc khởi động (AR-01, docs/03 §3.4).

API không huấn luyện, không đọc ``creditcard.csv``. Nó chỉ nạp các tệp mà notebook 08
đã xuất, kiểm tra chúng bằng chính các hàm ``validate_*`` của ``src/artifacts.py``,
rồi giữ trong bộ nhớ suốt vòng đời tiến trình.

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

from .config import Settings

log = logging.getLogger("api")

#: Ngày thứ hai của dữ liệu bắt đầu ở giây này — chế độ phát lại đọc từ đây (UI-05)
DAY_TWO_START = 86_400.0

#: Số luồng XGBoost khi chấm điểm lúc phục vụ (xem load_artifacts)
SERVING_THREADS = 1


class ArtifactError(RuntimeError):
    pass


@dataclass
class Artifacts:
    model: object
    explainer: object
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
    def base_value(self) -> float:
        return float(np.ravel(self.explainer.expected_value)[0])

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
        "explainer": models / "explainer.joblib",
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
    explainer = joblib.load(paths["explainer"])

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
    rescored = model.predict_proba(build_features(probe))[:, 1].astype("float64")
    if not np.allclose(rescored, probe["risk_score"].to_numpy(), rtol=0, atol=1e-6):
        raise ArtifactError("model.joblib chấm lại tập kiểm thử ra điểm khác — lệch phiên bản thư viện?")

    loaded = Artifacts(
        model=model,
        explainer=explainer,
        metrics=metrics,
        threshold=threshold,
        oof=oof,
        sample_pool=sample_pool,
        test_set=test_set,
        loaded_at=datetime.now(timezone.utc),
        y_test=y_test,
        s_test=s_test,
        oof_candidates=np.unique(oof["y_score"]),
        amounts_sorted=np.sort(test_set["Amount"].to_numpy(dtype="float64")),
        samples_by_id={item["id"]: item for item in sample_pool["items"]},
        metrics_json=json.dumps(metrics, ensure_ascii=False, allow_nan=False).encode("utf-8"),
    )
    log.info("Đã nạp hiện vật %s (huấn luyện %s), τ* = %.6f", loaded.model_version, loaded.trained_at,
             loaded.default_threshold)
    return loaded


def _warn_on_version_drift(expected: dict) -> None:
    for package, version in expected.items():
        try:
            installed = metadata.version(package)
        except metadata.PackageNotFoundError:
            installed = None
        if version and installed != version:
            log.warning("Phiên bản %s đang cài (%s) khác lúc xuất hiện vật (%s)", package, installed, version)
