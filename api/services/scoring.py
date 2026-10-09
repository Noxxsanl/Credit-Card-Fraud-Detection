"""Chấm điểm một giao dịch, một lô, một tệp CSV — và ghi kết quả (docs/03 §5.1, 06 §4.3).

Ba quy tắc giữ cho số của API khớp số của báo cáo:

- Đặc trưng chỉ sinh bằng ``src.features.build_features`` — cùng hàm với lúc huấn luyện.
- Một lời gọi ``predict_proba`` cho cả lô, không lặp từng dòng (NFR-02).
- Điểm rủi ro là dữ liệu bất biến; ``decision`` và ``risk_band`` là dẫn xuất từ ngưỡng
  hiện hành, tính lại mỗi lần trả về và không bao giờ lưu (AR-03, ST-01).
"""

from __future__ import annotations

import io
import json
import secrets
from dataclasses import dataclass
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score
from sqlalchemy import text
from sqlalchemy.orm import Session

from src.features import AMOUNT_COLUMN, RAW_REQUIRED_COLUMNS, TIME_COLUMN, V_COLUMNS, build_features, hour_of_day
from src.threshold import confusion_counts

from ..config import MAX_REPORTED_REJECTIONS, MAX_UPLOAD_RESULTS
from ..errors import ApiError
from ..loader import Artifacts
from ..schemas import MAX_AMOUNT

BANDS = ("low", "medium", "high", "critical")
TARGET_COLUMN = "Class"

COPY_COLUMNS = ("id", "time_offset", "hour", "amount", "features", "risk_score",
                "model_version", "true_label", "source", "batch_id")

_INSERT = text(
    "INSERT INTO transactions (id, time_offset, hour, amount, features, risk_score, model_version, "
    "true_label, source, batch_id) VALUES (:id, :time_offset, :hour, :amount, CAST(:features AS JSONB), "
    ":risk_score, :model_version, :true_label, :source, :batch_id) ON CONFLICT (id) DO NOTHING"
)


# --------------------------------------------------------------------------
# Dẫn xuất từ điểm và ngưỡng — 05 §2
# --------------------------------------------------------------------------

def block_cut(tau: float, tau_block: float) -> float:
    """Biên của dải ``critical``/``block``: ngưỡng đề xuất chặn, nhưng không bao giờ dưới τ —
    người dùng nâng τ vượt mức chặn thì mọi cảnh báo đều là đề xuất chặn."""
    return max(tau, tau_block)


def risk_bands(scores, tau: float, tau_block: float) -> np.ndarray:
    """``low`` < τ/2 ≤ ``medium`` < τ ≤ ``high`` < τ_chặn ≤ ``critical`` (05 §2)."""
    s = np.asarray(scores, dtype="float64")
    return np.select([s < tau / 2, s < tau, s < block_cut(tau, tau_block)], ["low", "medium", "high"],
                     default="critical")


def decisions(scores, tau: float, tau_block: float) -> np.ndarray:
    """``allow`` dưới τ; ``review`` từ τ tới dưới τ_chặn; ``block`` từ τ_chặn (trùng dải critical).

    τ_chặn chọn theo precision trên out-of-fold (``api.loader.block_threshold_from``), không theo
    bội số của τ: luật cũ "≥ 3τ" chặn tự động 19/38 cảnh báo giả của tập kiểm thử, trong khi dải
    "review" của nó không chứa vụ gian lận nào.
    """
    s = np.asarray(scores, dtype="float64")
    return np.select([s < tau, s < block_cut(tau, tau_block)], ["allow", "review"], default="block")


def risk_band(score: float, tau: float, tau_block: float) -> str:
    return str(risk_bands([score], tau, tau_block)[0])


def decision(score: float, tau: float, tau_block: float) -> str:
    return str(decisions([score], tau, tau_block)[0])


def distribution(bands: np.ndarray) -> dict[str, int]:
    return {band: int(np.count_nonzero(bands == band)) for band in BANDS}


# --------------------------------------------------------------------------
# Chấm điểm
# --------------------------------------------------------------------------

def normalize(frame: pd.DataFrame) -> pd.DataFrame:
    """30 cột thô, float64, ``Amount`` làm tròn tới xu — đúng giá trị sẽ nằm trong
    cột ``NUMERIC(12, 2)`` (ST-07), nên chấm lại từ cơ sở dữ liệu ra cùng điểm."""
    out = frame[RAW_REQUIRED_COLUMNS].astype("float64").reset_index(drop=True)
    out[AMOUNT_COLUMN] = out[AMOUNT_COLUMN].round(2)
    return out


def score_frame(loaded: Artifacts, frame: pd.DataFrame) -> np.ndarray:
    """Điểm rủi ro. Đặc trưng qua ``build_features`` như lúc huấn luyện; bước tiền xử lý và dự
    đoán qua ``FastScorer`` — trùng từng bit với ``model.predict_proba`` (kiểm lúc nạp)."""
    return loaded.scorer.predict(build_features(frame))


def frame_from_inputs(transactions) -> pd.DataFrame:
    return normalize(pd.DataFrame([t.model_dump() for t in transactions], columns=RAW_REQUIRED_COLUMNS))


def new_batch_id() -> str:
    return f"b-{datetime.now(timezone.utc):%Y%m%d-%H%M%S}-{secrets.token_hex(2)}"


# --------------------------------------------------------------------------
# Ghi xuống PostgreSQL
# --------------------------------------------------------------------------

def reserve_ids(session: Session, n: int) -> list[str]:
    """Lấy trước ``n`` mã ``TX-<số>`` từ dãy của PostgreSQL — mỗi ``nextval`` là nguyên tử,
    nên hai yêu cầu đồng thời không bao giờ nhận trùng mã."""
    if n == 0:
        return []
    values = session.execute(
        text("SELECT nextval('transaction_id_seq') FROM generate_series(1, :n)"), {"n": n}
    ).scalars().all()
    return [f"TX-{v}" for v in values]


def transaction_rows(frame: pd.DataFrame, scores: np.ndarray, ids: list[str], *, model_version: str,
                     source: str, batch_id: str | None, labels=None) -> list[tuple]:
    """Các bộ giá trị theo ``COPY_COLUMNS``. ``features`` là 28 giá trị V dạng đối tượng
    JSON có tên khóa — không lệch thứ tự được giữa lúc ghi và lúc đọc (ST-03)."""
    v_values = frame[V_COLUMNS].to_numpy(dtype="float64")
    times = frame[TIME_COLUMN].to_numpy(dtype="float64")
    amounts = frame[AMOUNT_COLUMN].to_numpy(dtype="float64")
    hours = hour_of_day(times).astype(int)
    rows = []
    for i, tx_id in enumerate(ids):
        label = None
        if labels is not None and not pd.isna(labels[i]):
            label = int(labels[i])
        rows.append((
            tx_id, float(times[i]), int(hours[i]), float(amounts[i]),
            json.dumps(dict(zip(V_COLUMNS, v_values[i].tolist()))),
            float(scores[i]), model_version, label, source, batch_id,
        ))
    return rows


def copy_rows(session: Session, rows: list[tuple]) -> None:
    """``COPY … FROM STDIN`` của psycopg 3, trong giao dịch hiện tại của phiên (06 §4.3).

    Nhanh gấp khoảng 50 lần ``INSERT`` từng dòng — điều kiện để đạt NFR-02. Chỉ dùng cho
    mã vừa lấy từ dãy, vì ``COPY`` không có ``ON CONFLICT``.
    """
    driver = session.connection().connection.driver_connection
    with driver.cursor() as cursor:
        with cursor.copy(f"COPY transactions ({', '.join(COPY_COLUMNS)}) FROM STDIN") as copy:
            for row in rows:
                copy.write_row(row)


def insert_rows(session: Session, rows: list[tuple]) -> None:
    """``INSERT … ON CONFLICT (id) DO NOTHING`` — cho mã tất định (phát lại) và từng giao
    dịch lẻ: ghi trùng mã không tạo dòng thứ hai và không ném lỗi (TC-44)."""
    if rows:
        session.execute(_INSERT, [dict(zip(COPY_COLUMNS, row)) for row in rows])


# --------------------------------------------------------------------------
# Ba đường vào
# --------------------------------------------------------------------------

def score_one(loaded: Artifacts, session: Session, frame: pd.DataFrame, tau: float, *,
              persist: bool, source: str = "manual", label: int | None = None) -> dict:
    score = float(score_frame(loaded, frame)[0])
    tx_id = None
    if persist:
        tx_id = reserve_ids(session, 1)[0]
        insert_rows(session, transaction_rows(frame, [score], [tx_id], model_version=loaded.model_version,
                                              source=source, batch_id=None, labels=[label]))
        session.commit()
    return {
        "transaction_id": tx_id,
        "risk_score": score,
        "threshold": tau,
        "decision": decision(score, tau, loaded.block_threshold),
        "risk_band": risk_band(score, tau, loaded.block_threshold),
        "model_version": loaded.model_version,
    }


def score_many(loaded: Artifacts, session: Session, frame: pd.DataFrame, tau: float, *, persist: bool,
               batch_id: str | None, source: str, labels=None) -> dict:
    """Chấm cả lô bằng một lời gọi, ghi bằng ``COPY`` trong **một** giao dịch — hoặc vào
    hết, hoặc không dòng nào (06 §4.2)."""
    batch_id = batch_id or new_batch_id()
    scores = score_frame(loaded, frame) if len(frame) else np.array([], dtype="float64")
    ids: list[str | None] = [None] * len(frame)
    if persist and len(frame):
        ids = reserve_ids(session, len(frame))
        copy_rows(session, transaction_rows(frame, scores, ids, model_version=loaded.model_version,
                                            source=source, batch_id=batch_id, labels=labels))
        session.commit()
    bands = risk_bands(scores, tau, loaded.block_threshold)
    return {
        "batch_id": batch_id,
        "count": int(len(frame)),
        "alerts": int(np.count_nonzero(scores >= tau)),
        "threshold": tau,
        "score_distribution": distribution(bands),
        "model_version": loaded.model_version,
        "scores": scores,
        "ids": ids,
        "bands": bands,
        "decisions": decisions(scores, tau, loaded.block_threshold),
    }


def results(scored: dict, order=None) -> list[dict]:
    order = range(len(scored["ids"])) if order is None else order
    return [
        {"transaction_id": scored["ids"][i], "risk_score": float(scored["scores"][i]),
         "risk_band": str(scored["bands"][i]), "decision": str(scored["decisions"][i])}
        for i in order
    ]


# --------------------------------------------------------------------------
# CSV — API-04
# --------------------------------------------------------------------------

@dataclass
class ParsedUpload:
    frame: pd.DataFrame
    labels: np.ndarray | None
    rejected: list[tuple[int, str]]
    rows_read: int


def parse_csv(content: bytes) -> ParsedUpload:
    """Đọc CSV, loại từng dòng lỗi kèm lý do thay vì bỏ cả tệp (05 API-04).

    Thiếu cột bắt buộc thì cả tệp vô nghĩa → 422 ``MISSING_FEATURES`` (DS-10). Cột thừa bị
    bỏ qua (DS-14); ``Class`` nếu có thì thành ``true_label``.
    """
    try:
        raw = pd.read_csv(io.BytesIO(content), low_memory=False)
    except (pd.errors.EmptyDataError, pd.errors.ParserError, UnicodeDecodeError) as exc:
        raise ApiError(422, "INVALID_BODY", "Không đọc được tệp CSV", {"reason": str(exc)[:200]}) from exc

    missing = [c for c in RAW_REQUIRED_COLUMNS if c not in raw.columns]
    if missing:
        raise ApiError(422, "MISSING_FEATURES", "Thiếu cột bắt buộc trong dữ liệu vào", {"missing": missing})

    numeric = raw[RAW_REQUIRED_COLUMNS].apply(pd.to_numeric, errors="coerce")
    reason = pd.Series(pd.NA, index=raw.index, dtype="object")

    def flag(mask: pd.Series, message: str) -> None:
        reason.loc[mask & reason.isna()] = message

    for column in RAW_REQUIRED_COLUMNS:
        values = numeric[column]
        flag(values.isna() & raw[column].notna(), f"{column} không phải số")
        flag(values.isna(), f"{column} bị trống")
        flag(np.isinf(values), f"{column} là vô cực")
    flag(numeric[TIME_COLUMN] < 0, f"{TIME_COLUMN} âm")
    flag(numeric[AMOUNT_COLUMN] < 0, f"{AMOUNT_COLUMN} âm")
    flag(numeric[AMOUNT_COLUMN] > MAX_AMOUNT, f"{AMOUNT_COLUMN} vượt {MAX_AMOUNT:,.2f}")

    labels = None
    if TARGET_COLUMN in raw.columns:
        labels = pd.to_numeric(raw[TARGET_COLUMN], errors="coerce")
        flag(labels.notna() & ~labels.isin([0, 1]), f"{TARGET_COLUMN} phải là 0 hoặc 1")
        flag(labels.isna() & raw[TARGET_COLUMN].notna(), f"{TARGET_COLUMN} phải là 0 hoặc 1")

    bad = reason.notna()
    rejected = [(int(i) + 1, str(r)) for i, r in reason[bad].items()]
    frame = normalize(numeric[~bad])
    kept_labels = labels[~bad].to_numpy(dtype="float64") if labels is not None else None
    return ParsedUpload(frame=frame, labels=kept_labels, rejected=rejected, rows_read=int(len(raw)))


def score_upload(loaded: Artifacts, session: Session, content: bytes, tau: float) -> dict:
    parsed = parse_csv(content)
    scored = score_many(loaded, session, parsed.frame, tau, persist=True, batch_id=None,
                        source="upload", labels=parsed.labels)
    scores = scored["scores"]

    has_labels = parsed.labels is not None and bool(np.isfinite(parsed.labels).any())
    actual = None
    if has_labels:
        known = np.isfinite(parsed.labels)
        y, s = parsed.labels[known].astype(int), scores[known]
        tp, fp, fn, _ = confusion_counts(y, s, tau)
        actual = {
            "precision": tp / (tp + fp) if tp + fp else None,
            "recall": tp / (tp + fn) if tp + fn else None,
            "pr_auc": float(average_precision_score(y, s)) if 0 < y.sum() < y.size else None,
        }

    top = np.argsort(-scores, kind="stable")[:MAX_UPLOAD_RESULTS]
    return {
        **scored,
        "results": results(scored, top),
        "results_truncated": len(scores) > MAX_UPLOAD_RESULTS,
        "rows_read": parsed.rows_read,
        "rows_rejected": len(parsed.rejected),
        "rejection_reasons": [{"row": r, "reason": m} for r, m in parsed.rejected[:MAX_REPORTED_REJECTIONS]],
        "rejection_reasons_truncated": len(parsed.rejected) > MAX_REPORTED_REJECTIONS,
        "has_labels": has_labels,
        "actual_metrics": actual,
    }
