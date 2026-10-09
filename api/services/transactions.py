"""Hàng đợi thẩm định, chi tiết giao dịch và kết luận của con người — API-06…API-08.

``decision`` và ``risk_band`` không nằm trong bảng: chúng tính lại từ ``risk_score`` và
ngưỡng hiện hành ở mỗi lần truy vấn (ST-01). Ngược lại, ``reviews.threshold_used`` lưu
ngưỡng tại thời điểm con người quyết định, vì chính sách lúc đó là dữ kiện lịch sử (ST-02).
"""

from __future__ import annotations

import numpy as np
from sqlalchemy import func, select, text
from sqlalchemy.orm import Session

from src.features import AMOUNT_COLUMN, RAW_REQUIRED_COLUMNS, TIME_COLUMN, V_COLUMNS

from ..errors import not_found
from ..loader import Artifacts
from ..models_orm import Review, Transaction
from .scoring import block_cut, decision, risk_band

SORT_COLUMNS = {
    "risk_score": Transaction.risk_score,
    "amount": Transaction.amount,
    "created_at": Transaction.created_at,
    "hour": Transaction.hour,
}


def band_range(band: str, tau: float, tau_block: float) -> tuple[float, float]:
    """Khoảng điểm [thấp, cao) của một dải rủi ro — cùng bảng với ``scoring.risk_bands``."""
    cut = block_cut(tau, tau_block)
    edges = {"low": (0.0, tau / 2), "medium": (tau / 2, tau), "high": (tau, cut), "critical": (cut, np.inf)}
    return edges[band]


def list_transactions(session: Session, tau: float, tau_block: float, *, min_score: float | None, max_score: float,
                      band: str | None, reviewed: bool | None, review_status: str | None,
                      batch_id: str | None, source: str | None, sort: str, page: int, page_size: int) -> dict:
    """Mặc định chỉ lấy giao dịch vượt ngưỡng hiện hành, điểm giảm dần (FR-40, AC-A2).

    Khi lọc theo ``band`` mà không truyền ``min_score`` thì bỏ mặc định "≥ τ" — nếu không,
    lọc dải ``low`` hay ``medium`` sẽ luôn rỗng.
    """
    conditions = [Transaction.risk_score <= max_score]
    if min_score is None and band is None:
        min_score = tau
    if min_score is not None:
        conditions.append(Transaction.risk_score >= min_score)
    if band is not None:
        low, high = band_range(band, tau, tau_block)
        conditions.append(Transaction.risk_score >= low)
        if np.isfinite(high):
            conditions.append(Transaction.risk_score < high)
    if reviewed is not None:
        conditions.append(Review.id.is_not(None) if reviewed else Review.id.is_(None))
    if review_status == "pending":
        conditions.append(Review.id.is_(None))
    elif review_status is not None:
        conditions.append(Review.decision == review_status)
    if batch_id is not None:
        conditions.append(Transaction.batch_id == batch_id)
    if source is not None:
        conditions.append(Transaction.source == source)

    column = SORT_COLUMNS[sort.lstrip("-")]
    order = column.desc() if sort.startswith("-") else column.asc()
    stmt = (
        select(Transaction, Review.decision, func.count().over().label("total"))
        .outerjoin(Review, Review.transaction_id == Transaction.id)
        .where(*conditions)
        .order_by(order, Transaction.id)          # thêm id để phân trang ổn định khi điểm trùng nhau
        .limit(page_size)
        .offset((page - 1) * page_size)
    )
    rows = session.execute(stmt).all()
    total = rows[0].total if rows else _count(session, conditions)
    return {
        "items": [_item(tx, review_decision, tau, tau_block) for tx, review_decision, _ in rows],
        "page": page,
        "page_size": page_size,
        "total": int(total),
        "threshold": tau,
    }


def _count(session: Session, conditions) -> int:
    """``COUNT(*) OVER ()`` không có dòng nào để mang về khi trang vượt quá cuối danh sách."""
    stmt = (select(func.count()).select_from(Transaction)
            .outerjoin(Review, Review.transaction_id == Transaction.id).where(*conditions))
    return int(session.execute(stmt).scalar_one())


def _item(tx: Transaction, review_decision: str | None, tau: float, tau_block: float) -> dict:
    return {
        "id": tx.id,
        "risk_score": tx.risk_score,
        "risk_band": risk_band(tx.risk_score, tau, tau_block),
        "decision": decision(tx.risk_score, tau, tau_block),
        "amount": float(tx.amount),            # NUMERIC → Decimal, ép về float (ST-07)
        "hour": tx.hour,
        "true_label": tx.true_label,
        "reviewed": review_decision is not None,
        "review_decision": review_decision,
        "source": tx.source,
        "batch_id": tx.batch_id,
        "model_version": tx.model_version,
        "created_at": tx.created_at,
    }


def raw_features(tx: Transaction) -> dict[str, float]:
    """30 cột thô theo đúng thứ tự hợp đồng dữ liệu, dựng lại từ các cột của bảng."""
    values = {TIME_COLUMN: float(tx.time_offset)}
    values.update({name: float(tx.features[name]) for name in V_COLUMNS})
    values[AMOUNT_COLUMN] = float(tx.amount)
    return {name: values[name] for name in RAW_REQUIRED_COLUMNS}


def get_transaction(session: Session, tx_id: str) -> Transaction:
    tx = session.get(Transaction, tx_id)
    if tx is None:
        raise not_found("giao dịch", tx_id)
    return tx


def detail(session: Session, loaded: Artifacts, tx_id: str, tau: float) -> dict:
    tx = get_transaction(session, tx_id)
    amount = float(tx.amount)
    review = tx.review
    return {
        "id": tx.id,
        "features": raw_features(tx),
        "risk_score": tx.risk_score,
        "risk_band": risk_band(tx.risk_score, tau, loaded.block_threshold),
        "decision": decision(tx.risk_score, tau, loaded.block_threshold),
        "threshold": tau,
        "amount": amount,
        # So với phân bố của tập kiểm thử, không với bảng transactions: bảng chỉ chứa những gì
        # người dùng đã nạp, vài chục dòng thì phân vị vô nghĩa (US-04)
        "amount_percentile": float(np.searchsorted(loaded.amounts_sorted, amount, side="left")
                                   / loaded.amounts_sorted.size),
        "hour": tx.hour,
        "true_label": tx.true_label,
        "review": None if review is None else {
            "decision": review.decision, "threshold_used": review.threshold_used,
            "reviewed_at": review.reviewed_at, "note": review.note,
        },
        "source": tx.source,
        "batch_id": tx.batch_id,
        "model_version": tx.model_version,
        "model_version_current": tx.model_version == loaded.model_version,
        "created_at": tx.created_at,
    }


_UPSERT_REVIEW = text(
    "INSERT INTO reviews (transaction_id, decision, threshold_used, note) "
    "VALUES (:tx_id, :decision, :threshold, :note) "
    "ON CONFLICT (transaction_id) DO UPDATE "
    "   SET decision = EXCLUDED.decision, threshold_used = EXCLUDED.threshold_used, "
    "       note = EXCLUDED.note, reviewed_at = now() "
    "RETURNING decision, threshold_used, reviewed_at, note"
)


def upsert_review(session: Session, tx_id: str, decision_value: str, note: str | None, tau: float) -> dict:
    """Một giao dịch một kết luận; gửi lại là ghi đè, nguyên tử nhờ ``ON CONFLICT`` (ST-04)."""
    tx = get_transaction(session, tx_id)
    row = session.execute(_UPSERT_REVIEW, {"tx_id": tx_id, "decision": decision_value,
                                           "threshold": tau, "note": note}).one()
    session.commit()
    matches = None
    if tx.true_label is not None:
        matches = (row.decision == "confirmed_fraud") == (tx.true_label == 1)
    return {
        "transaction_id": tx_id,
        "decision": row.decision,
        "threshold_used": row.threshold_used,
        "reviewed_at": row.reviewed_at,
        "note": row.note,
        "true_label": tx.true_label,
        "matches_label": matches,
    }
