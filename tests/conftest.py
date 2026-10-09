"""Fixture dùng chung — cơ sở dữ liệu kiểm thử dùng một lần (docs/08 §1.1).

Kiểm thử **không bao giờ** chạy trên cơ sở dữ liệu phát triển. Phiên pytest tạo
``fraud_test`` (theo ``TEST_DATABASE_URL``), dựng lược đồ bằng **chính Alembic** — không
bằng ``create_all()``, để kiểm thử xanh trên đúng lược đồ mà môi trường thật dùng — rồi
xoá khi xong.

Máy không có PostgreSQL thì các ca cần cơ sở dữ liệu tự bỏ qua, phần còn lại vẫn chạy:
``docker compose up -d db`` để bật.

Lệch khỏi 08 §1.1: giữa các ca, bảng được dọn bằng ``TRUNCATE`` thay vì rollback. Ca
kiểm thử API đi qua máy chủ thật, và đường ghi ``COPY`` cùng chế độ phát lại tự mở
giao dịch riêng, nên không bọc được trong một giao dịch ngoài để rollback.
"""

from __future__ import annotations

import os

import pytest
from sqlalchemy import create_engine, text
from sqlalchemy.engine import make_url

from api.config import Settings
from src.config import MODELS_DIR, SAMPLE_POOL_PATH, TEST_SET_PATH

DEFAULT_TEST_URL = "postgresql+psycopg://fraud:fraud@127.0.0.1:5432/fraud_test"

ARTIFACT_FILES = (
    MODELS_DIR / "model.joblib",
    MODELS_DIR / "explainer.joblib",
    MODELS_DIR / "metrics.json",
    MODELS_DIR / "threshold.json",
    MODELS_DIR / "oof_scores.npz",
    SAMPLE_POOL_PATH,
    TEST_SET_PATH,
)


def resolve_test_database_url() -> str:
    import api.config  # noqa: F401 — nạp .env

    return os.environ.get("TEST_DATABASE_URL", DEFAULT_TEST_URL)



def admin_engine(url: str, database: str = "postgres"):
    return create_engine(make_url(url).set(database=database), isolation_level="AUTOCOMMIT",
                         connect_args={"connect_timeout": 3})


def create_database(url: str) -> None:
    name = make_url(url).database
    engine = admin_engine(url)
    with engine.connect() as conn:
        conn.execute(text(f'DROP DATABASE IF EXISTS "{name}" WITH (FORCE)'))
        conn.execute(text(f'CREATE DATABASE "{name}"'))
    engine.dispose()


def drop_database(url: str) -> None:
    engine = admin_engine(url)
    with engine.connect() as conn:
        conn.execute(text(f'DROP DATABASE IF EXISTS "{make_url(url).database}" WITH (FORCE)'))
    engine.dispose()


def postgres_available(url: str) -> bool:
    try:
        engine = admin_engine(url)
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        engine.dispose()
        return True
    except Exception:  # noqa: BLE001
        return False


@pytest.fixture(scope="session")
def database_url():
    url = resolve_test_database_url()
    if not postgres_available(url):
        pytest.skip("PostgreSQL không chạy — docker compose up -d db (docs/10 §5)")
    from api.db import upgrade

    create_database(url)
    upgrade(url)
    yield url
    drop_database(url)


@pytest.fixture(scope="session")
def engine(database_url):
    from api.db import make_engine

    eng = make_engine(database_url)
    yield eng
    eng.dispose()


#: Trạng thái bảng settings ngay sau migration 0002 — mỗi ca bắt đầu từ đây
SETTINGS_DEFAULTS = {"cost_fn": "122.21", "cost_fp": "5.0", "replay_speed": "60"}


def truncate(eng) -> None:
    with eng.begin() as conn:
        conn.execute(text("TRUNCATE transactions, reviews, settings RESTART IDENTITY CASCADE"))
        for key, value in SETTINGS_DEFAULTS.items():
            conn.execute(text("INSERT INTO settings (key, value) VALUES (:k, CAST(:v AS JSONB))"),
                         {"k": key, "v": value})


@pytest.fixture
def clean_db(engine):
    truncate(engine)
    yield engine
    truncate(engine)


@pytest.fixture(scope="session")
def artifacts_present():
    missing = [str(p) for p in ARTIFACT_FILES if not p.exists()]
    if missing:
        pytest.skip("chưa có hiện vật — chạy notebooks/08_export_artifacts.ipynb")


@pytest.fixture(scope="session")
def client(database_url, artifacts_present):
    """Máy chủ thật trên cơ sở dữ liệu kiểm thử, nạp hiện vật một lần cho cả phiên."""
    from fastapi.testclient import TestClient

    from api.main import create_app

    app = create_app(Settings.from_env(database_url=database_url))
    with TestClient(app, raise_server_exceptions=False) as test_client:
        yield test_client


@pytest.fixture(scope="session")
def loaded(client):
    return client.app.state.artifacts
