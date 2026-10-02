"""Ba bảng SQLAlchemy — bản ánh xạ của lược đồ ở docs/06 §2.

Lược đồ **thật** do Alembic dựng (``api/migrations/versions``), không do
``Base.metadata.create_all()``. Tệp này chỉ để truy vấn; sửa ở đây thì phải có
migration đi kèm (docs/10 §8).
"""

from __future__ import annotations

from datetime import datetime
from decimal import Decimal

from sqlalchemy import (
    BigInteger,
    CheckConstraint,
    Double,
    ForeignKey,
    Identity,
    Numeric,
    SmallInteger,
    Text,
    func,
)
from sqlalchemy.dialects.postgresql import JSONB, TIMESTAMP
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship

SOURCES = ("upload", "sample", "replay", "manual")
REVIEW_DECISIONS = ("confirmed_fraud", "false_alarm")


class Base(DeclarativeBase):
    pass


class Transaction(Base):
    __tablename__ = "transactions"
    __table_args__ = (
        CheckConstraint("time_offset >= 0", name="ck_tx_time"),
        CheckConstraint("hour BETWEEN 0 AND 23", name="ck_tx_hour"),
        CheckConstraint("amount >= 0", name="ck_tx_amount"),
        CheckConstraint("risk_score >= 0 AND risk_score <= 1", name="ck_tx_score"),
        CheckConstraint("true_label IS NULL OR true_label IN (0, 1)", name="ck_tx_label"),
        CheckConstraint("source IN ('upload', 'sample', 'replay', 'manual')", name="ck_tx_source"),
    )

    id: Mapped[str] = mapped_column(Text, primary_key=True)
    time_offset: Mapped[float] = mapped_column(Double, nullable=False)
    hour: Mapped[int] = mapped_column(SmallInteger, nullable=False)
    amount: Mapped[Decimal] = mapped_column(Numeric(12, 2), nullable=False)
    features: Mapped[dict] = mapped_column(JSONB, nullable=False)
    risk_score: Mapped[float] = mapped_column(Double, nullable=False)
    model_version: Mapped[str] = mapped_column(Text, nullable=False)
    true_label: Mapped[int | None] = mapped_column(SmallInteger)
    source: Mapped[str] = mapped_column(Text, nullable=False)
    batch_id: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(TIMESTAMP(timezone=True), nullable=False, server_default=func.now())

    review: Mapped["Review | None"] = relationship(back_populates="transaction", uselist=False)


class Review(Base):
    __tablename__ = "reviews"
    __table_args__ = (
        CheckConstraint("decision IN ('confirmed_fraud', 'false_alarm')", name="ck_review_decision"),
        CheckConstraint("threshold_used > 0 AND threshold_used < 1", name="ck_review_threshold"),
    )

    id: Mapped[int] = mapped_column(BigInteger, Identity(always=True), primary_key=True)
    transaction_id: Mapped[str] = mapped_column(
        Text, ForeignKey("transactions.id", ondelete="CASCADE"), nullable=False, unique=True
    )
    decision: Mapped[str] = mapped_column(Text, nullable=False)
    threshold_used: Mapped[float] = mapped_column(Double, nullable=False)
    reviewed_at: Mapped[datetime] = mapped_column(TIMESTAMP(timezone=True), nullable=False, server_default=func.now())
    note: Mapped[str | None] = mapped_column(Text)

    transaction: Mapped[Transaction] = relationship(back_populates="review")


class Setting(Base):
    __tablename__ = "settings"

    key: Mapped[str] = mapped_column(Text, primary_key=True)
    value: Mapped[object] = mapped_column(JSONB, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(TIMESTAMP(timezone=True), nullable=False, server_default=func.now())
