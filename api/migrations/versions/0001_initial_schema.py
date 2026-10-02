"""Ba bảng transactions, reviews, settings cùng chỉ mục và ràng buộc (docs/06 §2).

Viết tay, không autogenerate, để khớp từng ràng buộc ``CHECK`` và chỉ mục một phần
với tài liệu. Thêm so với 06 §2: dãy ``transaction_id_seq`` cấp mã ``TX-<số>`` cho
giao dịch chấm qua API — lấy trước N mã rồi ghi bằng ``COPY`` (06 §4.3).

Revision ID: 0001
Revises:
Create Date: 2026-09-28
"""

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import JSONB

revision = "0001"
down_revision = None
branch_labels = None
depends_on = None

TIMESTAMPTZ = sa.TIMESTAMP(timezone=True)


def upgrade() -> None:
    op.execute("CREATE SEQUENCE transaction_id_seq")

    op.create_table(
        "transactions",
        sa.Column("id", sa.Text, primary_key=True),                       # 'TX-8841'
        sa.Column("time_offset", sa.Double, nullable=False),              # cột Time gốc, giây
        sa.Column("hour", sa.SmallInteger, nullable=False),               # dẫn xuất, 0..23
        sa.Column("amount", sa.Numeric(12, 2), nullable=False),
        sa.Column("features", JSONB, nullable=False),                     # 28 giá trị V
        sa.Column("risk_score", sa.Double, nullable=False),
        sa.Column("model_version", sa.Text, nullable=False),
        sa.Column("true_label", sa.SmallInteger),                         # NULL nếu không có nhãn
        sa.Column("source", sa.Text, nullable=False),
        sa.Column("batch_id", sa.Text),
        sa.Column("created_at", TIMESTAMPTZ, nullable=False, server_default=sa.text("now()")),
        sa.CheckConstraint("time_offset >= 0", name="ck_tx_time"),
        sa.CheckConstraint("hour BETWEEN 0 AND 23", name="ck_tx_hour"),
        sa.CheckConstraint("amount >= 0", name="ck_tx_amount"),
        sa.CheckConstraint("risk_score >= 0 AND risk_score <= 1", name="ck_tx_score"),
        sa.CheckConstraint("true_label IS NULL OR true_label IN (0, 1)", name="ck_tx_label"),
        sa.CheckConstraint("source IN ('upload', 'sample', 'replay', 'manual')", name="ck_tx_source"),
    )
    op.create_index("idx_tx_risk", "transactions", [sa.text("risk_score DESC")])
    op.create_index("idx_tx_created", "transactions", [sa.text("created_at DESC")])
    op.create_index("idx_tx_batch", "transactions", ["batch_id"],
                    postgresql_where=sa.text("batch_id IS NOT NULL"))

    op.create_table(
        "reviews",
        sa.Column("id", sa.BigInteger, sa.Identity(always=True), primary_key=True),
        sa.Column("transaction_id", sa.Text,
                  sa.ForeignKey("transactions.id", ondelete="CASCADE", name="fk_reviews_transaction"),
                  nullable=False),
        sa.Column("decision", sa.Text, nullable=False),
        sa.Column("threshold_used", sa.Double, nullable=False),
        sa.Column("reviewed_at", TIMESTAMPTZ, nullable=False, server_default=sa.text("now()")),
        sa.Column("note", sa.Text),
        sa.UniqueConstraint("transaction_id", name="uq_reviews_transaction"),
        sa.CheckConstraint("decision IN ('confirmed_fraud', 'false_alarm')", name="ck_review_decision"),
        sa.CheckConstraint("threshold_used > 0 AND threshold_used < 1", name="ck_review_threshold"),
    )

    op.create_table(
        "settings",
        sa.Column("key", sa.Text, primary_key=True),
        sa.Column("value", JSONB, nullable=False),
        sa.Column("updated_at", TIMESTAMPTZ, nullable=False, server_default=sa.text("now()")),
    )


def downgrade() -> None:
    op.drop_table("settings")
    op.drop_table("reviews")
    op.drop_index("idx_tx_batch", table_name="transactions")
    op.drop_index("idx_tx_created", table_name="transactions")
    op.drop_index("idx_tx_risk", table_name="transactions")
    op.drop_table("transactions")
    op.execute("DROP SEQUENCE transaction_id_seq")
