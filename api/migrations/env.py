"""Môi trường Alembic.

URL lấy theo thứ tự: ``config.attributes["url"]`` (kiểm thử và mã gọi trực tiếp, xem
``api.db.upgrade``) → biến môi trường ``DATABASE_URL`` (kể cả từ ``.env``).
"""

import os
from logging.config import fileConfig

from alembic import context
from dotenv import load_dotenv
from sqlalchemy import create_engine, pool

from api.models_orm import Base
from src.config import PROJECT_ROOT

load_dotenv(PROJECT_ROOT / ".env", override=False)

config = context.config
if config.config_file_name is not None and not config.attributes.get("url"):
    fileConfig(config.config_file_name, disable_existing_loggers=False)

target_metadata = Base.metadata


def database_url() -> str:
    url = config.attributes.get("url") or os.environ.get("DATABASE_URL")
    if not url:
        raise RuntimeError("Chưa có DATABASE_URL — sao chép .env.example thành .env (docs/10 §2.2)")
    return url


def run_migrations_offline() -> None:
    context.configure(url=database_url(), target_metadata=target_metadata, literal_binds=True,
                      dialect_opts={"paramstyle": "named"})
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    engine = create_engine(database_url(), poolclass=pool.NullPool)
    with engine.connect() as connection:
        context.configure(connection=connection, target_metadata=target_metadata)
        with context.begin_transaction():
            context.run_migrations()
    engine.dispose()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
