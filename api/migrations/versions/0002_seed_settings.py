"""Giá trị mặc định của bảng settings (docs/06 §2).

Chỉ nạp ba khóa ``cost_fn``, ``cost_fp``, ``replay_speed``. Khóa ``threshold`` cố ý
**không** nạp: ngưỡng mặc định là τ* trong ``models/threshold.json`` của mô hình đang
chạy, còn dòng ``threshold`` trong bảng chỉ xuất hiện khi người dùng tự đặt
(``PUT /threshold``). Nạp sẵn một con số ở đây thì sau khi huấn luyện lại, bảng vẫn
giữ τ* của mô hình cũ mà không ai biết.

Giá trị lấy từ ``src/config.py`` (DEFAULT_COST_FN, DEFAULT_COST_FP) nhưng chép cứng:
migration phải cho cùng kết quả dù mã nguồn về sau có đổi.

Revision ID: 0002
Revises: 0001
Create Date: 2026-09-28
"""

import sqlalchemy as sa
from alembic import op

revision = "0002"
down_revision = "0001"
branch_labels = None
depends_on = None

DEFAULTS = {"cost_fn": "122.21", "cost_fp": "5.0", "replay_speed": "60"}


def upgrade() -> None:
    for key, value in DEFAULTS.items():
        op.execute(
            sa.text("INSERT INTO settings (key, value) VALUES (:key, CAST(:value AS JSONB)) "
                    "ON CONFLICT (key) DO NOTHING").bindparams(key=key, value=value)
        )


def downgrade() -> None:
    op.execute(sa.text("DELETE FROM settings WHERE key IN ('cost_fn', 'cost_fp', 'replay_speed')"))
