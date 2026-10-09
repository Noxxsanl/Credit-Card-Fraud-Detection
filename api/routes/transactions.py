"""API-06 … API-08: hàng đợi thẩm định, chi tiết giao dịch, kết luận của người thẩm định."""

from __future__ import annotations

from typing import Literal

from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session

from ..config import API_PREFIX, DEFAULT_PAGE_SIZE, MAX_PAGE_SIZE
from ..deps import get_artifacts, get_session
from ..loader import Artifacts
from ..schemas import ReviewRequest, ReviewResponse, RiskBand, Source, TransactionDetail, TransactionPage
from ..services import runtime_settings, transactions

router = APIRouter(prefix=API_PREFIX, tags=["giao dịch"])

SortKey = Literal["risk_score", "-risk_score", "amount", "-amount", "created_at", "-created_at", "hour", "-hour"]


@router.get("/transactions", response_model=TransactionPage, summary="API-06 — danh sách có lọc và phân trang")
def list_transactions(
    min_score: float | None = Query(None, ge=0, le=1, description="Mặc định: ngưỡng hiện hành"),
    max_score: float = Query(1.0, ge=0, le=1),
    band: RiskBand | None = None,
    reviewed: bool | None = None,
    review_status: Literal["pending", "confirmed_fraud", "false_alarm"] | None = None,
    batch_id: str | None = None,
    source: Source | None = None,
    sort: SortKey = "-risk_score",
    page: int = Query(1, ge=1),
    page_size: int = Query(DEFAULT_PAGE_SIZE, ge=1, le=MAX_PAGE_SIZE),
    loaded: Artifacts = Depends(get_artifacts),
    session: Session = Depends(get_session),
):
    tau, _ = runtime_settings.current_threshold(session, loaded)
    return transactions.list_transactions(
        session, tau, loaded.block_threshold, min_score=min_score, max_score=max_score, band=band, reviewed=reviewed,
        review_status=review_status, batch_id=batch_id, source=source, sort=sort, page=page, page_size=page_size,
    )


@router.get("/transactions/{tx_id}", response_model=TransactionDetail, summary="API-07 — chi tiết một giao dịch")
def get_transaction(tx_id: str, loaded: Artifacts = Depends(get_artifacts), session: Session = Depends(get_session)):
    tau, _ = runtime_settings.current_threshold(session, loaded)
    return transactions.detail(session, loaded, tx_id, tau)


@router.post("/reviews", response_model=ReviewResponse, summary="API-08 — ghi kết quả thẩm định")
def post_review(body: ReviewRequest, loaded: Artifacts = Depends(get_artifacts),
                session: Session = Depends(get_session)):
    # Máy chủ tự ghi threshold_used bằng ngưỡng tại thời điểm gửi (ST-02, TC-39)
    tau, _ = runtime_settings.current_threshold(session, loaded)
    return transactions.upsert_review(session, body.transaction_id, body.decision, body.note, tau)
