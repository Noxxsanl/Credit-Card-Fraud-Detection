"""API-02 … API-05: chấm điểm một giao dịch, một lô, một tệp CSV, và giải thích."""

from __future__ import annotations

import time
from datetime import datetime, timezone

import pandas as pd
from fastapi import APIRouter, Depends, File, UploadFile
from sqlalchemy.orm import Session

from ..config import API_PREFIX, MAX_BATCH_SIZE, MAX_UPLOAD_BYTES
from ..deps import get_artifacts, get_session
from ..errors import ApiError
from ..loader import Artifacts
from ..schemas import (
    BatchRequest,
    BatchResponse,
    ExplainRequest,
    ExplainResponse,
    ScoreRequest,
    ScoreResponse,
    UploadResponse,
)
from ..services import explaining, runtime_settings, scoring, transactions

router = APIRouter(prefix=API_PREFIX, tags=["chấm điểm"])


def _ms(started: float) -> float:
    return round((time.perf_counter() - started) * 1000, 2)


@router.post("/score", response_model=ScoreResponse, summary="API-02 — chấm điểm một giao dịch")
def score(body: ScoreRequest, loaded: Artifacts = Depends(get_artifacts), session: Session = Depends(get_session)):
    started = time.perf_counter()
    source, label = "manual", None
    if body.sample_id is not None:
        sample = loaded.samples_by_id.get(body.sample_id)
        if sample is None:
            raise ApiError(404, "NOT_FOUND", f"Không có mẫu với mã {body.sample_id}", {"id": body.sample_id})
        source, label = "sample", sample["label"]
    tau, _ = runtime_settings.current_threshold(session, loaded)
    frame = scoring.frame_from_inputs([body.transaction])
    result = scoring.score_one(loaded, session, frame, tau, persist=body.persist, source=source, label=label)
    return {**result, "scored_at": datetime.now(timezone.utc), "latency_ms": _ms(started)}


@router.post("/score/batch", response_model=BatchResponse, summary="API-03 — chấm điểm tối đa 50.000 giao dịch")
def score_batch(body: BatchRequest, loaded: Artifacts = Depends(get_artifacts),
                session: Session = Depends(get_session)):
    started = time.perf_counter()
    if len(body.transactions) > MAX_BATCH_SIZE:
        raise ApiError(413, "PAYLOAD_TOO_LARGE", f"Lô vượt {MAX_BATCH_SIZE:,} giao dịch (DS-15)",
                       {"count": len(body.transactions), "limit": MAX_BATCH_SIZE})
    tau, _ = runtime_settings.current_threshold(session, loaded)
    scored = scoring.score_many(loaded, session, scoring.frame_from_inputs(body.transactions), tau,
                                persist=body.persist, batch_id=body.batch_id, source="manual")
    return {**scored, "results": scoring.results(scored), "elapsed_ms": _ms(started)}


@router.post("/score/upload", response_model=UploadResponse, summary="API-04 — tải CSV, chấm điểm, lưu")
def score_upload(file: UploadFile = File(..., description="CSV đủ 30 cột bắt buộc; cột Class nếu có thành nhãn thật"),
                 loaded: Artifacts = Depends(get_artifacts), session: Session = Depends(get_session)):
    started = time.perf_counter()
    content = file.file.read(MAX_UPLOAD_BYTES + 1)
    if len(content) > MAX_UPLOAD_BYTES:
        raise ApiError(413, "PAYLOAD_TOO_LARGE", "Tệp vượt 100 MB (NFR-06)", {"limit_bytes": MAX_UPLOAD_BYTES})
    tau, _ = runtime_settings.current_threshold(session, loaded)
    result = scoring.score_upload(loaded, session, content, tau)
    return {**result, "elapsed_ms": _ms(started)}


@router.post("/explain", response_model=ExplainResponse, summary="API-05 — giá trị SHAP của một giao dịch")
def explain(body: ExplainRequest, loaded: Artifacts = Depends(get_artifacts), session: Session = Depends(get_session)):
    started = time.perf_counter()
    if body.transaction_id is not None:
        tx = transactions.get_transaction(session, body.transaction_id)
        frame = pd.DataFrame([transactions.raw_features(tx)])
    else:
        frame = scoring.frame_from_inputs([body.transaction])
    result = explaining.explain(loaded, frame)
    return {**result, "transaction_id": body.transaction_id, "elapsed_ms": _ms(started)}
