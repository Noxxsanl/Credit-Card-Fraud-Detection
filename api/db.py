"""Engine SQLAlchemy, pool kết nối và phiên làm việc với PostgreSQL (docs/06 §4.1).

``pool_pre_ping`` là bắt buộc trong Compose: khi container ``db`` khởi động lại, kết
nối trong pool thành rác và yêu cầu đầu tiên sau đó sẽ lỗi nếu không thăm dò trước.
"""

from __future__ import annotations

from pathlib import Path

from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

from src.config import PROJECT_ROOT

ALEMBIC_INI = PROJECT_ROOT / "alembic.ini"


def make_engine(url: str) -> Engine:
    return create_engine(
        url,
        pool_size=5,            # đủ cho uvicorn một worker
        max_overflow=10,
        pool_pre_ping=True,     # tự phát hiện kết nối chết sau khi db khởi động lại
        pool_recycle=1800,
        # Không đợi vô hạn khi db mất: /health phải trả 503 trong vài giây (TC-42)
        connect_args={"connect_timeout": 3},
    )


def make_session_factory(engine: Engine) -> sessionmaker[Session]:
    return sessionmaker(bind=engine, expire_on_commit=False)


def ping(engine: Engine) -> bool:
    """Truy vấn thăm dò ``SELECT 1`` — nguồn của trường ``db`` trong /health."""
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        return True
    except Exception:  # noqa: BLE001 — mọi lỗi kết nối đều nghĩa là "down"
        return False


def alembic_config(url: str):
    """Cấu hình Alembic trỏ vào ``url`` — dùng cho kiểm thử và entrypoint."""
    from alembic.config import Config

    config = Config(str(ALEMBIC_INI))
    config.attributes["url"] = url
    return config


def upgrade(url: str, revision: str = "head") -> None:
    from alembic import command

    command.upgrade(alembic_config(url), revision)


def downgrade(url: str, revision: str = "base") -> None:
    from alembic import command

    command.downgrade(alembic_config(url), revision)


def migrations_dir() -> Path:
    return Path(__file__).resolve().parent / "migrations"
