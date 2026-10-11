"""Đổi giá trị mặc định của ``cost_fn`` từ 122,21 sang 115,94 EUR (docs/11 m3).

122,21 là trung bình ``Amount`` của gian lận trên dữ liệu **thô** — gồm cả tập kiểm thử và dòng
trùng lặp. Tham số nào của hệ thống cũng chỉ được ước lượng trên tập huấn luyện; trung bình chỉ trên
tập huấn luyện là 115,939… EUR (``src/config.py → DEFAULT_COST_FN``).

Chỉ đổi khi dòng ``cost_fn`` vẫn mang đúng giá trị mặc định cũ mà migration 0002 đã nạp. Người dùng
đã tự đặt chi phí qua ``PUT /threshold`` thì giữ nguyên lựa chọn của họ.

Giá trị chép cứng, như 0002: migration phải cho cùng kết quả dù mã nguồn về sau có đổi.

Revision ID: 0003
Revises: 0002
Create Date: 2026-10-10
"""

import sqlalchemy as sa
from alembic import op

revision = "0003"
down_revision = "0002"
branch_labels = None
depends_on = None

OLD_COST_FN = "122.21"
NEW_COST_FN = "115.94"

_SWAP = ("UPDATE settings SET value = CAST(:new AS JSONB), updated_at = now() "
         "WHERE key = 'cost_fn' AND value = CAST(:old AS JSONB)")


def upgrade() -> None:
    op.execute(sa.text(_SWAP).bindparams(new=NEW_COST_FN, old=OLD_COST_FN))


def downgrade() -> None:
    op.execute(sa.text(_SWAP).bindparams(new=OLD_COST_FN, old=NEW_COST_FN))
