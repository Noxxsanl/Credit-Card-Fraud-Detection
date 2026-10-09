"""Tầng dịch vụ chấm điểm — TC-50…TC-55 (docs/08 §2.5).

Gọi thẳng ``api/services`` không qua HTTP: logic ngưỡng và chấm điểm phải kiểm thử được
mà không cần dựng máy chủ (03 §3.2).
"""

from __future__ import annotations

import time
from decimal import Decimal

import numpy as np
import pandas as pd
import pytest
from sqlalchemy import text
from sqlalchemy.orm import Session

from api.services import scoring, thresholding
from api.services.transactions import raw_features
from api.models_orm import Transaction
from src.features import RAW_REQUIRED_COLUMNS


# --------------------------------------------------------------------------
# TC-51 — dải rủi ro và quyết định, không cần hiện vật
# --------------------------------------------------------------------------

@pytest.mark.parametrize("score, band, decision", [
    (0.0, "low", "allow"), (0.0249, "low", "allow"),
    (0.025, "medium", "allow"), (0.0499, "medium", "allow"),
    (0.05, "high", "review"), (0.15, "high", "review"), (0.8999, "high", "review"),
    (0.9, "critical", "block"), (1.0, "critical", "block"),
])
def test_tc51_risk_band_follows_the_contract(score, band, decision):
    """TC-51: bốn dải với τ = 0,05 và ngưỡng đề xuất chặn 0,9 theo bảng ở 05 §2, kể cả đúng
    tại biên. Mức chặn là một ngưỡng riêng chọn theo precision, không còn là 3τ."""
    assert scoring.risk_band(score, 0.05, 0.9) == band
    assert scoring.decision(score, 0.05, 0.9) == decision


def test_block_never_starts_below_the_alert_threshold():
    """Người dùng nâng τ vượt mức chặn: mọi cảnh báo đều là đề xuất chặn, không còn dải review."""
    assert scoring.decision(0.5, 0.6, 0.4) == "allow" and scoring.risk_band(0.5, 0.6, 0.4) == "medium"
    assert scoring.decision(0.6, 0.6, 0.4) == "block" and scoring.risk_band(0.6, 0.6, 0.4) == "critical"


def test_bands_are_vectorised_consistently():
    scores = np.linspace(0, 1, 1001)
    bands = scoring.risk_bands(scores, 0.05, 0.9)
    assert [scoring.risk_band(s, 0.05, 0.9) for s in scores] == list(bands)
    counts = scoring.distribution(bands)
    assert counts == {b: int((bands == b).sum()) for b in scoring.BANDS} and sum(counts.values()) == 1001
    assert counts["low"] == 25 and counts["medium"] == 25


def test_normalize_rounds_amount_to_cents_and_orders_columns():
    frame = pd.DataFrame([{**{c: 1.0 for c in RAW_REQUIRED_COLUMNS}, "Amount": 12.3456, "extra": 5}])
    out = scoring.normalize(frame.iloc[:, ::-1])
    assert list(out.columns) == RAW_REQUIRED_COLUMNS and out["Amount"].iloc[0] == 12.35


def test_parse_csv_reports_rows_one_based(tmp_path):
    frame = pd.DataFrame([{c: 1.0 for c in RAW_REQUIRED_COLUMNS}] * 3).astype(object)
    frame.loc[1, "Amount"] = "-2"
    parsed = scoring.parse_csv(frame.to_csv(index=False).encode())
    assert parsed.rows_read == 3 and len(parsed.frame) == 2
    assert parsed.rejected == [(2, "Amount âm")] and parsed.labels is None


# --------------------------------------------------------------------------
# Cần hiện vật
# --------------------------------------------------------------------------

@pytest.fixture(scope="module")
def sample(loaded):
    return loaded.test_set.iloc[:10_000]


def test_tc50_one_by_one_equals_batch(loaded, sample):
    """TC-50 (tầng dịch vụ): 100 giao dịch chấm từng cái và chấm một lô — sai số < 1e-12."""
    frame = scoring.normalize(sample.iloc[:100])
    batch = scoring.score_frame(loaded, frame)
    single = np.array([scoring.score_frame(loaded, frame.iloc[[i]])[0] for i in range(100)])
    assert np.abs(batch - single).max() < 1e-12


def test_tc53_scoring_10000_is_well_under_30_seconds(loaded, sample):
    """TC-53 / NFR-02: chấm 10.000 giao dịch."""
    frame = scoring.normalize(sample)
    started = time.perf_counter()
    scores = scoring.score_frame(loaded, frame)
    elapsed = time.perf_counter() - started
    print(f"\nTC-53: chấm 10.000 giao dịch trong {elapsed * 1000:.0f} ms")
    assert elapsed < 30
    assert np.array_equal(scores, loaded.s_test[:10_000])       # đúng từng bit với metrics.json


def test_fast_path_equals_the_full_pipeline_bit_for_bit(loaded):
    """``FastScorer`` (api/serving.py) phải là CÙNG phép tính với ``model.predict_proba`` trên cả
    tập kiểm thử, không phải phép gần đúng — loader chỉ đối chiếu 1/50 lúc khởi động."""
    from src.features import build_features

    features = build_features(loaded.test_set)
    full = loaded.model.predict_proba(features)[:, 1].astype("float64")
    assert np.array_equal(loaded.scorer.predict(features), full)
    margin = loaded.model[-1].predict(loaded.model[:-1].transform(features), output_margin=True)
    assert np.array_equal(loaded.scorer.margin(features), margin.astype("float64"))


def test_block_threshold_is_chosen_by_precision_on_out_of_fold(loaded):
    """Ngưỡng đề xuất chặn: mọi ngưỡng từ đó trở lên có precision ≥ 95% trên out-of-fold."""
    from api.config import BLOCK_MIN_PRECISION
    from src.threshold import cost_curve

    tau_block = loaded.block_threshold
    table = cost_curve(loaded.oof["y_true"], loaded.oof["y_score"], thresholds=loaded.oof_candidates)
    tail = table[(table["threshold"] >= tau_block) & (table["alerts"] > 0)]
    assert len(tail) and (tail["precision"] >= BLOCK_MIN_PRECISION).all()
    assert loaded.default_threshold < tau_block < 1
    # Trên tập kiểm thử, ở τ*: dải review còn chứa gian lận thật để người thẩm định bắt (luật cũ
    # 3τ để dải này rỗng gian lận), và chặn tự động rất ít khách hợp lệ
    y, s, tau = loaded.y_test, loaded.s_test, loaded.default_threshold
    decided = scoring.decisions(s, tau, tau_block)
    assert y[decided == "review"].sum() > 0
    blocked = decided == "block"
    assert y[blocked].mean() >= 0.95


def _rows(loaded, frame, prefix):
    scores = scoring.score_frame(loaded, frame)
    ids = [f"{prefix}-{i}" for i in range(len(frame))]
    return scoring.transaction_rows(frame, scores, ids, model_version=loaded.model_version,
                                    source="upload", batch_id=prefix, labels=None)


def test_tc54_copy_10000_rows_under_3_seconds(loaded, sample, clean_db):
    """TC-54: ghi 10.000 dòng bằng COPY dưới 3 giây; đo kèm INSERT từng dòng để đưa vào báo cáo."""
    frame = scoring.normalize(sample)
    copy_rows, insert_rows = _rows(loaded, frame, "C"), _rows(loaded, frame.iloc[:1000], "I")

    with Session(clean_db) as session:
        started = time.perf_counter()
        scoring.copy_rows(session, copy_rows)
        session.commit()
        copy_seconds = time.perf_counter() - started

        started = time.perf_counter()
        for row in insert_rows:                                   # đúng kiểu "INSERT từng dòng qua ORM"
            scoring.insert_rows(session, [row])
            session.commit()
        insert_seconds = time.perf_counter() - started

    with clean_db.connect() as conn:
        assert conn.execute(text("SELECT count(*) FROM transactions WHERE batch_id = 'C'")).scalar() == 10_000
    copy_rate, insert_rate = 10_000 / copy_seconds, 1000 / insert_seconds
    print(f"\nTC-54: COPY 10.000 dòng {copy_seconds:.2f}s ({copy_rate:,.0f} dòng/s); "
          f"INSERT từng dòng {insert_rate:,.0f} dòng/s — COPY nhanh gấp {copy_rate / insert_rate:.0f} lần")
    assert copy_seconds < 3


def test_tc44_insert_path_ignores_duplicate_ids(loaded, sample, clean_db):
    """TC-44: ghi trùng transactions.id không tạo bản ghi thứ hai, không ném lỗi."""
    rows = _rows(loaded, scoring.normalize(sample.iloc[:5]), "D")
    with Session(clean_db) as session:
        scoring.insert_rows(session, rows)
        scoring.insert_rows(session, rows)
        session.commit()
    with clean_db.connect() as conn:
        assert conn.execute(text("SELECT count(*) FROM transactions")).scalar() == 5


def test_tc55_amount_comes_back_as_decimal_and_is_converted(loaded, sample, clean_db):
    """TC-55: NUMERIC trả Decimal; phải ép về float trước build_features (ST-07)."""
    frame = scoring.normalize(sample.iloc[:50])
    with Session(clean_db) as session:
        scoring.copy_rows(session, _rows(loaded, frame, "N"))
        session.commit()
        stored = session.query(Transaction).order_by(Transaction.time_offset, Transaction.id).all()
        assert isinstance(stored[0].amount, Decimal)
        raw = pd.DataFrame([raw_features(tx) for tx in stored])
    assert all(isinstance(v, float) for v in raw.iloc[0])
    rescored = scoring.score_frame(loaded, raw)
    assert np.array_equal(np.sort(rescored), np.sort([tx.risk_score for tx in stored]))


def test_reserve_ids_are_unique_and_prefixed(clean_db):
    with Session(clean_db) as session:
        ids = scoring.reserve_ids(session, 500) + scoring.reserve_ids(session, 500)
    assert len(set(ids)) == 1000 and all(i.startswith("TX-") for i in ids)


# --------------------------------------------------------------------------
# Ngưỡng — T-47 ở tầng dịch vụ
# --------------------------------------------------------------------------

def test_t47_preview_service_is_a_few_milliseconds(loaded):
    """T-47: phần tính của /threshold/preview — không chạy mô hình, không đụng ổ đĩa."""
    timings = []
    for value in np.linspace(0.001, 0.999, 200):
        started = time.perf_counter()
        thresholding.preview(loaded, value, 122.21, 5.0)
        timings.append((time.perf_counter() - started) * 1000)
    print(f"\nT-47: tính preview trung vị {np.median(timings):.2f} ms, p95 {np.percentile(timings, 95):.2f} ms")
    assert np.median(timings) < 10


def test_optimize_matches_the_notebook_alternatives(loaded):
    """Chọn trên out-of-fold như notebook 06: chi phí mặc định ra đúng τ*, và ràng buộc recall
    chọn ra đúng ngưỡng recall ≥ 90% nếu chi phí cực tiểu trong vùng khả thi nằm ở biên."""
    result = thresholding.optimize(loaded, 122.21, 5.0)
    assert result["optimal_threshold"] == loaded.default_threshold
    constrained = thresholding.optimize(loaded, 122.21, 5.0, "min_recall", 0.9)
    assert constrained["optimal_threshold"] == loaded.threshold["alternatives"]["recall_at_least_90"]
    assert constrained["metrics_at_optimal_oof"]["recall"] >= 0.9


def test_optimize_reports_infeasible_constraints(loaded):
    result = thresholding.optimize(loaded, 122.21, 5.0, "max_alerts_per_day", 0.001)
    assert result["constraint_satisfied"] is False and result["constraint_binding"] is True
