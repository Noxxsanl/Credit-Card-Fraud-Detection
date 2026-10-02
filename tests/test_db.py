"""Tầng cơ sở dữ liệu — T-41, T-42, TC-45…TC-47 (docs/08 §2.4).

Ràng buộc toàn vẹn là lưới an toàn cuối cùng: tầng API đã kiểm tra dữ liệu vào, nhưng
nếu một đường ghi nào đó lọt lưới thì PostgreSQL phải từ chối, không âm thầm nhận rác.
"""

from __future__ import annotations

import json

import pytest
from sqlalchemy import inspect, text
from sqlalchemy.exc import IntegrityError

from api.db import downgrade, make_engine, ping, upgrade

from conftest import create_database, drop_database

GOOD_ROW = {
    "id": "TX-T1", "time_offset": 7834.0, "hour": 2, "amount": 149.62,
    "features": json.dumps({f"V{i}": 0.0 for i in range(1, 29)}), "risk_score": 0.5,
    "model_version": "xgb_test", "true_label": None, "source": "manual", "batch_id": None,
}

INSERT = text(
    "INSERT INTO transactions (id, time_offset, hour, amount, features, risk_score, model_version, true_label, "
    "source, batch_id) VALUES (:id, :time_offset, :hour, :amount, CAST(:features AS JSONB), :risk_score, "
    ":model_version, :true_label, :source, :batch_id)"
)


def test_t41_engine_connects_with_pool_pre_ping(engine):
    """T-41: kết nối được, và engine bật pool_pre_ping (06 §4.1)."""
    assert ping(engine)
    assert engine.pool._pre_ping


def test_t41_ping_reports_down_instead_of_raising():
    dead = make_engine("postgresql+psycopg://fraud:fraud@127.0.0.1:1/fraud")
    assert ping(dead) is False
    dead.dispose()


def _schema(url: str) -> dict:
    eng = make_engine(url)
    insp = inspect(eng)
    tables = sorted(t for t in insp.get_table_names() if t != "alembic_version")
    snapshot = {
        table: {
            "columns": [(c["name"], str(c["type"]), c["nullable"]) for c in insp.get_columns(table)],
            "indexes": sorted((i["name"], tuple(i["column_names"] or ())) for i in insp.get_indexes(table)),
            "checks": sorted((c["name"], c["sqltext"]) for c in insp.get_check_constraints(table)),
            "uniques": sorted(u["name"] for u in insp.get_unique_constraints(table)),
            "fks": sorted(f["name"] for f in insp.get_foreign_keys(table)),
        }
        for table in tables
    }
    with eng.connect() as conn:
        snapshot["_settings_rows"] = sorted(conn.execute(text("SELECT key, value::text FROM settings")).all()) \
            if "settings" in tables else []
        snapshot["_sequences"] = sorted(insp.get_sequence_names())
    eng.dispose()
    return snapshot


def test_tc47_upgrade_downgrade_upgrade_round_trip(database_url):
    """TC-47 / T-42: upgrade head → downgrade base → upgrade head chạy trọn, lược đồ cuối như đầu.

    Chạy trên một cơ sở dữ liệu riêng để không đụng tới fraud_test mà các ca khác đang dùng.
    """
    url = database_url.rsplit("/", 1)[0] + "/fraud_test_migrations"
    create_database(url)
    try:
        upgrade(url)
        first = _schema(url)
        assert set(first) >= {"transactions", "reviews", "settings"}
        assert first["_sequences"] == ["reviews_id_seq", "transaction_id_seq"]
        assert [k for k, _ in first["_settings_rows"]] == ["cost_fn", "cost_fp", "replay_speed"]

        downgrade(url, "base")
        empty = _schema(url)
        assert not {"transactions", "reviews", "settings"} & set(empty)
        assert empty["_sequences"] == []

        upgrade(url)
        assert _schema(url) == first
    finally:
        drop_database(url)


def test_schema_matches_the_storage_design(engine):
    """Chỉ mục và ràng buộc ở 06 §2 có mặt đúng tên."""
    insp = inspect(engine)
    assert {i["name"] for i in insp.get_indexes("transactions")} >= {"idx_tx_risk", "idx_tx_created", "idx_tx_batch"}
    checks = {c["name"] for c in insp.get_check_constraints("transactions")}
    assert checks == {"ck_tx_time", "ck_tx_hour", "ck_tx_amount", "ck_tx_score", "ck_tx_label", "ck_tx_source"}
    assert {c["name"] for c in insp.get_check_constraints("reviews")} == {"ck_review_decision", "ck_review_threshold"}
    assert [u["column_names"] for u in insp.get_unique_constraints("reviews")] == [["transaction_id"]]
    fk = insp.get_foreign_keys("reviews")[0]
    assert fk["referred_table"] == "transactions" and fk["options"].get("ondelete") == "CASCADE"


@pytest.mark.parametrize("field, value", [
    ("risk_score", 1.7), ("risk_score", -0.1), ("hour", 25), ("amount", -1.0),
    ("time_offset", -5.0), ("true_label", 2), ("source", "internet"),
])
def test_tc45_check_constraints_reject_garbage(clean_db, field, value):
    """TC-45: CHECK chặn dữ liệu rác ngay tại cơ sở dữ liệu."""
    with pytest.raises(IntegrityError, match="ck_tx_"):
        with clean_db.begin() as conn:
            conn.execute(INSERT, {**GOOD_ROW, field: value})


def test_tc45_valid_row_is_accepted(clean_db):
    with clean_db.begin() as conn:
        conn.execute(INSERT, GOOD_ROW)
        assert conn.execute(text("SELECT count(*) FROM transactions")).scalar() == 1


def test_tc46_review_for_unknown_transaction_is_rejected(clean_db):
    """TC-46: khóa ngoại reviews.transaction_id."""
    with pytest.raises(IntegrityError, match="fk_reviews_transaction"):
        with clean_db.begin() as conn:
            conn.execute(text("INSERT INTO reviews (transaction_id, decision, threshold_used) "
                              "VALUES ('TX-NOPE', 'false_alarm', 0.05)"))


@pytest.mark.parametrize("decision, threshold", [("approved", 0.05), ("false_alarm", 0.0), ("false_alarm", 1.0)])
def test_review_checks(clean_db, decision, threshold):
    with clean_db.begin() as conn:
        conn.execute(INSERT, GOOD_ROW)
    with pytest.raises(IntegrityError, match="ck_review_"):
        with clean_db.begin() as conn:
            conn.execute(text("INSERT INTO reviews (transaction_id, decision, threshold_used) "
                              "VALUES ('TX-T1', :d, :t)"), {"d": decision, "t": threshold})


def test_deleting_a_transaction_cascades_to_its_review(clean_db):
    with clean_db.begin() as conn:
        conn.execute(INSERT, GOOD_ROW)
        conn.execute(text("INSERT INTO reviews (transaction_id, decision, threshold_used) "
                          "VALUES ('TX-T1', 'false_alarm', 0.05)"))
        conn.execute(text("DELETE FROM transactions WHERE id = 'TX-T1'"))
        assert conn.execute(text("SELECT count(*) FROM reviews")).scalar() == 0


def test_timestamps_are_utc(engine):
    """ST-08: máy chủ đặt timezone = UTC."""
    with engine.connect() as conn:
        assert conn.execute(text("SHOW TimeZone")).scalar() in ("UTC", "Etc/UTC")
