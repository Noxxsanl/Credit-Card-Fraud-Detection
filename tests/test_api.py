"""Kiểm thử tích hợp API — TC-30…TC-44 (docs/08 §2.4) và điều kiện của T-44…T-50.

Chạy máy chủ thật (``TestClient``) trên cơ sở dữ liệu ``fraud_test`` dựng bằng Alembic,
với hiện vật thật trong ``models/``. Thiếu PostgreSQL hoặc thiếu hiện vật thì bỏ qua.
"""

from __future__ import annotations

import json
import math
import time
from datetime import datetime

import numpy as np
import pandas as pd
import pytest
from sqlalchemy import text

from api.config import Settings
from src.features import RAW_REQUIRED_COLUMNS, build_features

API = "/api/v1"

pytestmark = pytest.mark.usefixtures("clean_db")


@pytest.fixture(scope="module")
def pool(client):
    return client.get(f"{API}/samples").json()["items"]


@pytest.fixture(scope="module")
def tx(pool):
    """Một giao dịch gian lận điển hình của thư viện mẫu."""
    return dict(pool[0]["features"])


def count(engine, sql: str = "SELECT count(*) FROM transactions", **params) -> int:
    with engine.connect() as conn:
        return conn.execute(text(sql), params).scalar()


def score(client, transaction, **extra):
    return client.post(f"{API}/score", json={"transaction": transaction, **extra})


# --------------------------------------------------------------------------
# API-01 — T-44
# --------------------------------------------------------------------------

def test_tc30_health_reports_model_version(client, loaded):
    body = client.get(f"{API}/health").json()
    assert body["status"] == "ok" and body["db"] == "ok"
    assert body["model_version"] == loaded.model_version
    assert body["threshold"] == loaded.default_threshold


def test_tc42_health_is_503_when_database_is_down(artifacts_present):
    """TC-42 / T-44: mất cơ sở dữ liệu → 503 DATABASE_UNAVAILABLE, tiến trình không sập."""
    from fastapi.testclient import TestClient

    from api.main import create_app

    app = create_app(Settings.from_env(database_url="postgresql+psycopg://fraud:fraud@127.0.0.1:1/fraud"))
    with TestClient(app, raise_server_exceptions=False) as dead:
        started = time.perf_counter()
        response = dead.get(f"{API}/health")
        assert response.status_code == 503
        assert response.json()["error"]["code"] == "DATABASE_UNAVAILABLE"
        assert response.json()["error"]["details"]["db"] == "down"
        assert time.perf_counter() - started < 10
        # Endpoint cần cơ sở dữ liệu cũng trả 503, không phải 500; endpoint không cần thì vẫn chạy
        assert dead.get(f"{API}/threshold").json()["error"]["code"] == "DATABASE_UNAVAILABLE"
        assert dead.post(f"{API}/threshold/optimize", json={}).status_code == 200
        assert dead.get(f"{API}/health").status_code == 503       # vẫn sống


def test_tc42_health_is_503_when_model_is_missing(database_url, tmp_path):
    """T-44: thiếu hiện vật → 503 MODEL_NOT_LOADED; endpoint cần mô hình cũng 503."""
    from fastapi.testclient import TestClient

    from api.main import create_app

    app = create_app(Settings.from_env(database_url=database_url, models_dir=tmp_path))
    with TestClient(app, raise_server_exceptions=False) as empty:
        response = empty.get(f"{API}/health")
        assert response.status_code == 503
        error = response.json()["error"]
        assert error["code"] == "MODEL_NOT_LOADED" and "model.joblib" in error["details"]["reason"]
        assert empty.post(f"{API}/score", json={"transaction": {}}).status_code in (422, 503)
        assert empty.get(f"{API}/metrics").json()["error"]["code"] == "MODEL_NOT_LOADED"


def test_tc42_corrupted_artifact_is_503_not_a_crash(database_url, tmp_path):
    import shutil

    from fastapi.testclient import TestClient

    from api.main import create_app
    from src.config import MODELS_DIR

    for path in MODELS_DIR.glob("*.*"):
        shutil.copy(path, tmp_path / path.name)
    (tmp_path / "model.joblib").write_bytes(b"not a pickle")
    app = create_app(Settings.from_env(database_url=database_url, models_dir=tmp_path))
    with TestClient(app, raise_server_exceptions=False) as broken:
        assert broken.get(f"{API}/health").json()["error"]["code"] == "MODEL_NOT_LOADED"


# --------------------------------------------------------------------------
# API-02 — T-45, TC-31…TC-34
# --------------------------------------------------------------------------

def test_tc31_score_returns_the_full_contract(client, tx, clean_db, loaded):
    response = score(client, tx)
    body = response.json()
    assert response.status_code == 200
    assert 0 <= body["risk_score"] <= 1
    assert body["decision"] in ("allow", "review", "block") and body["risk_band"] in ("low", "medium", "high", "critical")
    assert body["model_version"] == loaded.model_version                       # TC-52
    assert body["scored_at"].endswith("Z")                                     # 05 §5
    assert body["transaction_id"].startswith("TX-") and count(clean_db) == 1


def test_score_matches_the_exported_model_bit_for_bit(client, pool):
    for item in pool[::20]:
        body = score(client, item["features"], persist=False).json()
        assert body["risk_score"] == item["risk_score"] and body["transaction_id"] is None


def test_score_with_sample_id_keeps_the_true_label(client, pool, clean_db):
    item = next(i for i in pool if i["category"] == "fraud_hard")
    body = score(client, item["features"], sample_id=item["id"]).json()
    assert count(clean_db, "SELECT true_label FROM transactions WHERE id = :i", i=body["transaction_id"]) == 1
    assert count(clean_db, "SELECT count(*) FROM transactions WHERE source = 'sample'") == 1
    assert score(client, item["features"], sample_id="S-999").status_code == 404


def test_tc32_missing_column_names_the_column(client, tx):
    response = score(client, {k: v for k, v in tx.items() if k != "V13"})
    assert response.status_code == 422
    assert response.json()["error"] == {"code": "MISSING_FEATURES", "message": "Thiếu cột bắt buộc trong dữ liệu vào",
                                        "details": {"missing": ["V13"]}}


def test_tc33_negative_amount(client, tx):
    response = score(client, {**tx, "Amount": -1})
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "INVALID_FEATURE_VALUE"
    assert response.json()["error"]["details"]["errors"][0]["field"] == "Amount"


@pytest.mark.parametrize("bad", [math.nan, math.inf, "abc", None])
def test_tc34_nan_inf_and_non_numbers(client, tx, bad):
    content = json.dumps({"transaction": {**tx, "V5": bad}})       # json.dumps ghi được NaN/Infinity
    response = client.post(f"{API}/score", content=content, headers={"content-type": "application/json"})
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "INVALID_FEATURE_VALUE"


def test_extra_columns_are_ignored(client, tx):
    """DS-14: cột thừa (kể cả Class) bị bỏ qua, không làm lỗi."""
    assert score(client, {**tx, "Class": 1, "merchant": "x"}, persist=False).status_code == 200


# --------------------------------------------------------------------------
# API-03 — T-45, TC-35, TC-50
# --------------------------------------------------------------------------

def test_t45_single_and_batch_scoring_agree(client, loaded):
    """TC-50 / T-45: chấm từng giao dịch và chấm cả lô cho cùng kết quả trên 100 giao dịch."""
    rows = loaded.test_set.iloc[::567][RAW_REQUIRED_COLUMNS].head(100).to_dict("records")
    batch = client.post(f"{API}/score/batch", json={"transactions": rows, "persist": False}).json()
    single = [score(client, r, persist=False).json()["risk_score"] for r in rows]
    assert batch["count"] == 100
    assert np.abs(np.array(single) - [r["risk_score"] for r in batch["results"]]).max() < 1e-12
    assert batch["model_version"] == loaded.model_version


def test_batch_persists_in_one_go(client, loaded, clean_db):
    rows = loaded.test_set.iloc[:500][RAW_REQUIRED_COLUMNS].to_dict("records")
    body = client.post(f"{API}/score/batch", json={"transactions": rows, "batch_id": "b-test"}).json()
    assert body["batch_id"] == "b-test"
    assert count(clean_db, "SELECT count(*) FROM transactions WHERE batch_id = 'b-test'") == 500
    assert sum(body["score_distribution"].values()) == 500
    assert body["alerts"] == sum(r["decision"] != "allow" for r in body["results"])


def test_tc35_batch_over_limit_is_413(client, tx):
    response = client.post(f"{API}/score/batch", json={"transactions": [tx] * 50_001, "persist": False})
    assert response.status_code == 413
    assert response.json()["error"]["code"] == "PAYLOAD_TOO_LARGE"


def test_batch_with_one_bad_row_names_the_row(client, tx):
    response = client.post(f"{API}/score/batch", json={"transactions": [tx, {**tx, "Amount": -3}], "persist": False})
    assert response.status_code == 422
    assert response.json()["error"]["details"]["errors"][0] == {
        "field": "Amount", "reason": "Input should be greater than or equal to 0", "row": 1}


# --------------------------------------------------------------------------
# API-04 — T-46, TC-43, AC-A1
# --------------------------------------------------------------------------

def csv_bytes(frame: pd.DataFrame) -> bytes:
    return frame.to_csv(index=False).encode()


def upload(client, content: bytes, name="giao-dich.csv"):
    return client.post(f"{API}/score/upload", files={"file": (name, content, "text/csv")})


def test_tc43_bad_rows_are_dropped_and_the_rest_written_together(client, loaded, clean_db):
    frame = loaded.test_set.iloc[:50][[*RAW_REQUIRED_COLUMNS, "Class"]].astype(object)
    frame.loc[10, "Amount"] = -5
    frame.loc[20, "V13"] = "abc"
    frame.loc[30, "Time"] = None
    frame.loc[40, "Class"] = 7
    body = upload(client, csv_bytes(frame)).json()

    assert body["rows_read"] == 50 and body["rows_rejected"] == 4 and body["count"] == 46
    assert body["rejection_reasons"] == [
        {"row": 11, "reason": "Amount âm"}, {"row": 21, "reason": "V13 không phải số"},
        {"row": 31, "reason": "Time bị trống"}, {"row": 41, "reason": "Class phải là 0 hoặc 1"},
    ]
    assert count(clean_db, "SELECT count(*) FROM transactions WHERE batch_id = :b", b=body["batch_id"]) == 46
    assert count(clean_db, "SELECT count(*) FROM transactions WHERE source = 'upload' AND true_label IS NOT NULL") == 46
    assert body["has_labels"] is True


def test_upload_missing_column_rejects_the_whole_file(client, loaded, clean_db):
    frame = loaded.test_set.iloc[:5][RAW_REQUIRED_COLUMNS].drop(columns=["V2", "V27"])
    response = upload(client, csv_bytes(frame))
    assert response.status_code == 422
    assert response.json()["error"]["details"] == {"missing": ["V2", "V27"]}
    assert count(clean_db) == 0


def test_upload_garbage_is_422_not_500(client):
    assert upload(client, b"").status_code == 422
    assert upload(client, b"\x00\x01\x02 not a csv").status_code == 422


def test_t46_upload_10000_rows_under_30_seconds(client, loaded, clean_db):
    """T-46 / AC-A1 / NFR-02: 10.000 dòng chấm điểm và ghi xong dưới 30 giây."""
    frame = loaded.test_set.iloc[:10_000][[*RAW_REQUIRED_COLUMNS, "Class"]]
    content = csv_bytes(frame)
    started = time.perf_counter()
    body = upload(client, content).json()
    elapsed = time.perf_counter() - started
    print(f"\nT-46: 10.000 dòng ({len(content) / 1024**2:.1f} MB) chấm và ghi trong {elapsed:.2f}s "
          f"(máy chủ báo {body['elapsed_ms']:.0f} ms)")

    assert elapsed < 30
    assert body["count"] == 10_000 and body["rows_rejected"] == 0
    assert count(clean_db) == 10_000
    # Điểm ghi xuống trùng từng bit với điểm của mô hình trên chính các dòng đó. Dữ liệu gốc đã có
    # đúng 2 chữ số thập phân, nên làm tròn Amount tới xu (ST-07) không đổi đầu vào của mô hình
    assert (frame["Amount"].round(2) == frame["Amount"]).all()
    with clean_db.connect() as conn:
        stored = np.sort(conn.execute(text("SELECT risk_score FROM transactions")).scalars().all())
    assert np.array_equal(stored, np.sort(loaded.s_test[:10_000]))
    assert body["actual_metrics"]["recall"] is not None and len(body["results"]) == 200


# --------------------------------------------------------------------------
# API-05 — T-48, AC-A4
# --------------------------------------------------------------------------

def test_t48_explain_returns_5_up_3_down_matching_shap(client, pool, loaded):
    """T-48 / AC-A4: 5 yếu tố dương, 3 yếu tố âm, khớp SHAP tính trực tiếp bằng explainer.

    "Dương" nghĩa là SHAP > 0 thật. Với giao dịch hợp lệ điểm thấp, mô hình thường chỉ có 1–4 đặc
    trưng đẩy điểm lên (77% tập kiểm thử); khi đó danh sách ngắn hơn chứ không mượn một đóng góp âm
    để cho đủ 5. Trong hàng đợi (điểm ≥ τ*) 114/115 giao dịch có đủ 5.
    """
    tau = loaded.default_threshold
    full = 0
    for item in pool[::5]:
        body = client.post(f"{API}/explain", json={"transaction": item["features"]}).json()
        features = build_features(pd.DataFrame([item["features"]]))
        expected = loaded.explainer.shap_values(loaded.model[:-1].transform(features))[0].astype("float64")
        names = loaded.model_feature_names

        assert len(body["top_positive"]) == min(5, int((expected > 0).sum()))
        assert len(body["top_negative"]) == min(3, int((expected < 0).sum())) == 3
        assert all(c["shap"] > 0 for c in body["top_positive"]) and all(c["shap"] < 0 for c in body["top_negative"])
        if item["risk_score"] >= tau:
            full += len(body["top_positive"]) == 5

        got = {c["feature"]: c["shap"] for c in body["contributions"]}
        assert np.array_equal([got[n] for n in names], expected)
        ranked = [n for n, v in sorted(zip(names, expected), key=lambda p: -p[1]) if v > 0][:5]
        assert [c["feature"] for c in body["top_positive"]] == ranked
        assert abs(body["base_value"] + sum(got.values()) - body["margin"]) < 1e-3
        assert body["risk_score"] == item["risk_score"]
    alerts = sum(i["risk_score"] >= tau for i in pool[::5])
    assert alerts > 0 and full == alerts        # mọi mẫu vượt ngưỡng trong lượt này đều đủ 5 yếu tố dương


def test_explain_by_transaction_id_round_trips_through_the_database(client, tx):
    """TC-55: amount đọc ra là Decimal, phải ép về float — chấm lại ra đúng điểm đã lưu."""
    scored = score(client, tx).json()
    body = client.post(f"{API}/explain", json={"transaction_id": scored["transaction_id"]}).json()
    assert body["risk_score"] == scored["risk_score"]
    assert client.post(f"{API}/explain", json={"transaction_id": "TX-0"}).status_code == 404
    assert client.post(f"{API}/explain", json={}).status_code == 422
    assert client.post(f"{API}/explain", json={"transaction_id": "TX-1", "transaction": tx}).status_code == 422


# --------------------------------------------------------------------------
# API-06…API-08 — T-49, TC-37…TC-40
# --------------------------------------------------------------------------

def load_test_rows(client, loaded, n=400):
    rows = loaded.test_set.iloc[:n][RAW_REQUIRED_COLUMNS]
    fraud = loaded.test_set[loaded.test_set["Class"] == 1][RAW_REQUIRED_COLUMNS].head(30)
    frame = pd.concat([rows, fraud])
    return upload(client, csv_bytes(frame)).json()


def test_transactions_are_sorted_by_score_descending(client, loaded):
    """AC-A2: hàng đợi sắp đúng theo điểm giảm dần; mặc định chỉ lấy giao dịch ≥ τ."""
    load_test_rows(client, loaded)
    body = client.get(f"{API}/transactions", params={"page_size": 200}).json()
    scores = [i["risk_score"] for i in body["items"]]
    assert scores == sorted(scores, reverse=True)
    assert all(s >= body["threshold"] for s in scores) and body["total"] == len(scores)


def test_pagination_and_filters(client, loaded):
    load_test_rows(client, loaded)
    everything = client.get(f"{API}/transactions", params={"min_score": 0, "page_size": 200}).json()
    assert everything["total"] == 430
    page2 = client.get(f"{API}/transactions", params={"min_score": 0, "page_size": 50, "page": 2}).json()
    assert [i["id"] for i in page2["items"]] == [i["id"] for i in everything["items"][50:100]]
    low = client.get(f"{API}/transactions", params={"band": "low", "page_size": 200}).json()
    assert low["total"] > 0 and all(i["risk_band"] == "low" for i in low["items"])
    by_amount = client.get(f"{API}/transactions", params={"min_score": 0, "sort": "amount"}).json()["items"]
    assert [i["amount"] for i in by_amount] == sorted(i["amount"] for i in by_amount)
    beyond = client.get(f"{API}/transactions", params={"min_score": 0, "page": 99}).json()
    assert beyond["items"] == [] and beyond["total"] == 430
    for bad in ({"page": 0}, {"page_size": 201}, {"sort": "id"}, {"band": "extreme"}, {"min_score": 2}):
        response = client.get(f"{API}/transactions", params=bad)
        assert response.status_code == 400 and response.json()["error"]["code"] == "INVALID_REQUEST", bad


def test_tc37_unknown_transaction_is_404(client):
    response = client.get(f"{API}/transactions/TX-404")
    assert response.status_code == 404 and response.json()["error"]["code"] == "NOT_FOUND"


def test_transaction_detail(client, tx, loaded):
    scored = score(client, tx).json()
    body = client.get(f"{API}/transactions/{scored['transaction_id']}").json()
    assert list(body["features"]) == RAW_REQUIRED_COLUMNS
    assert body["features"] == pytest.approx(tx)
    assert 0 <= body["amount_percentile"] <= 1 and body["model_version_current"] is True
    assert body["review"] is None


def test_tc38_reviewing_twice_overwrites(client, tx, clean_db):
    """TC-38 / T-49: gửi thẩm định hai lần không tạo dòng thứ hai."""
    tx_id = score(client, tx).json()["transaction_id"]
    first = client.post(f"{API}/reviews", json={"transaction_id": tx_id, "decision": "false_alarm"}).json()
    second = client.post(f"{API}/reviews", json={"transaction_id": tx_id, "decision": "confirmed_fraud",
                                                 "note": "đã gọi chủ thẻ"}).json()
    assert count(clean_db, "SELECT count(*) FROM reviews") == 1
    assert second["decision"] == "confirmed_fraud" and second["note"] == "đã gọi chủ thẻ"
    assert datetime.fromisoformat(second["reviewed_at"]) >= datetime.fromisoformat(first["reviewed_at"])
    detail = client.get(f"{API}/transactions/{tx_id}").json()
    assert detail["review"]["decision"] == "confirmed_fraud"


def test_tc39_review_records_the_threshold_in_force(client, tx, clean_db):
    """TC-39 / T-49: threshold_used là ngưỡng hiện hành lúc gửi, không phải τ* mặc định."""
    tx_id = score(client, tx).json()["transaction_id"]
    assert client.put(f"{API}/threshold", json={"value": 0.3}).status_code == 200
    body = client.post(f"{API}/reviews", json={"transaction_id": tx_id, "decision": "false_alarm"}).json()
    assert body["threshold_used"] == 0.3
    assert count(clean_db, "SELECT threshold_used FROM reviews") == 0.3


def test_review_reveals_the_label_only_after_deciding(client, pool):
    item = next(i for i in pool if i["category"] == "legit_hard")
    tx_id = score(client, item["features"], sample_id=item["id"]).json()["transaction_id"]
    body = client.post(f"{API}/reviews", json={"transaction_id": tx_id, "decision": "false_alarm"}).json()
    assert body["true_label"] == 0 and body["matches_label"] is True


def test_review_errors(client, tx):
    assert client.post(f"{API}/reviews", json={"transaction_id": "TX-0", "decision": "false_alarm"}).status_code == 404
    tx_id = score(client, tx).json()["transaction_id"]
    response = client.post(f"{API}/reviews", json={"transaction_id": tx_id, "decision": "approved"})
    assert response.status_code == 422 and response.json()["error"]["code"] == "INVALID_BODY"


def test_tc40_changing_the_threshold_changes_the_queue(client, pool):
    """TC-40: đổi ngưỡng rồi gọi lại /transactions — số dòng đổi theo.

    Dùng thư viện mẫu: 34 mẫu legit_hard có điểm trong [τ*, 0,5), chắc chắn rời hàng đợi khi τ = 0,9.
    """
    client.post(f"{API}/score/batch", json={"transactions": [i["features"] for i in pool]})
    before = client.get(f"{API}/transactions").json()["total"]
    client.put(f"{API}/threshold", json={"value": 0.9})
    after = client.get(f"{API}/transactions").json()
    assert after["threshold"] == 0.9 and after["total"] < before


# --------------------------------------------------------------------------
# API-09…API-12 — T-47, TC-36
# --------------------------------------------------------------------------

def test_threshold_state_and_update(client, loaded):
    state = client.get(f"{API}/threshold").json()
    assert state["source"] == "artifact" and state["current"] == loaded.default_threshold
    assert state["alternatives"] == loaded.threshold["alternatives"]
    updated = client.put(f"{API}/threshold", json={"value": 0.05, "cost_fn": 200, "cost_fp": 3}).json()
    assert updated["source"] == "user" and updated["current"] == 0.05
    assert (updated["cost_fn"], updated["cost_fp"]) == (200, 3)
    assert client.get(f"{API}/health").json()["threshold"] == 0.05


def test_user_threshold_is_ignored_for_another_model(client, loaded, clean_db):
    """Ngưỡng đặt cho mô hình cũ không có hiệu lực với mô hình mới (06 §2)."""
    with clean_db.begin() as conn:
        conn.execute(text("INSERT INTO settings (key, value) VALUES ('threshold', "
                          "'{\"value\": 0.4, \"model_version\": \"xgb_cu\"}')"))
    state = client.get(f"{API}/threshold").json()
    assert state["source"] == "artifact" and state["current"] == loaded.default_threshold


@pytest.mark.parametrize("value", [1.5, 0, 1, -0.2])
def test_tc36_threshold_outside_0_1_is_rejected(client, value):
    response = client.put(f"{API}/threshold", json={"value": value})
    assert response.status_code == 422 and response.json()["error"]["code"] == "INVALID_THRESHOLD"


def test_t47_preview_is_fast_and_matches_the_threshold_module(client, loaded):
    """T-47: /threshold/preview trả trong khoảng 10 ms, số khớp src/threshold.py."""
    from src.threshold import metrics_at_threshold

    timings = []
    for value in np.linspace(0.001, 0.999, 50):
        started = time.perf_counter()
        body = client.get(f"{API}/threshold/preview", params={"value": value}).json()
        timings.append((time.perf_counter() - started) * 1000)
    median = float(np.median(timings))
    print(f"\nT-47: /threshold/preview trung vị {median:.1f} ms, p95 {np.percentile(timings, 95):.1f} ms (qua HTTP)")
    assert median < 25
    expected = metrics_at_threshold(loaded.y_test, loaded.s_test, 0.999, sample_fraction=loaded.test_fraction)
    assert body == pytest.approx(expected.as_dict())
    assert client.get(f"{API}/threshold/preview", params={"value": 1.2}).json()["error"]["code"] == "INVALID_THRESHOLD"


def test_t47_optimize_with_default_costs_reproduces_threshold_json(client, loaded):
    """ML-08: chọn trên out-of-fold → đúng τ* của threshold.json, không phải τ tốt nhất trên test."""
    body = client.post(f"{API}/threshold/optimize", json={}).json()
    assert body["optimal_threshold"] == loaded.default_threshold
    assert body["selected_on"] == "out_of_fold" and body["constraint_binding"] is False
    assert body["metrics_at_optimal"]["tp"] == 77 and len(body["curve"]) >= 150


@pytest.mark.parametrize("constraint, binding", [
    ({"type": "none"}, False),
    ({"type": "max_alerts_per_day", "value": 200}, True),      # τ* sinh 287 cảnh báo/ngày
    ({"type": "max_alerts_per_day", "value": 1000}, False),
    ({"type": "min_recall", "value": 0.5}, False),
    ({"type": "min_recall", "value": 0.9}, True),              # τ* chỉ có recall OOF 0,854
])
def test_t47_constraint_binding(client, constraint, binding):
    """T-47: /threshold/optimize báo đúng constraint_binding."""
    body = client.post(f"{API}/threshold/optimize", json={"constraint": constraint}).json()
    assert body["constraint_binding"] is binding and body["constraint_satisfied"] is True
    oof = body["metrics_at_optimal_oof"]
    if constraint["type"] == "max_alerts_per_day":
        assert oof["alerts_per_day"] <= constraint["value"]
    if constraint["type"] == "min_recall":
        assert oof["recall"] >= constraint["value"]


def test_ac_a5_cost_parameters_move_the_optimum(client):
    """AC-A5: đổi tham số chi phí làm dịch chuyển ngưỡng tối ưu (bậc 5:1 của T-31)."""
    cheap = client.post(f"{API}/threshold/optimize", json={"cost_fn": 25, "cost_fp": 5}).json()
    default = client.post(f"{API}/threshold/optimize", json={}).json()
    assert cheap["optimal_threshold"] > default["optimal_threshold"]
    assert cheap["optimal_threshold"] == pytest.approx(0.720512330532074)


def test_optimize_rejects_bad_input(client):
    for body in ({"cost_fn": 0}, {"cost_fp": -1}, {"constraint": {"type": "min_recall"}},
                 {"constraint": {"type": "min_recall", "value": 1.5}}):
        assert client.post(f"{API}/threshold/optimize", json=body).status_code == 422, body


# --------------------------------------------------------------------------
# API-13…API-15 — T-50
# --------------------------------------------------------------------------

def test_t50_metrics_full_and_by_section(client, loaded):
    full = client.get(f"{API}/metrics")
    assert full.headers["content-type"] == "application/json"
    assert full.json()["headline"] == loaded.metrics["headline"]
    section = client.get(f"{API}/metrics", params={"section": "test_scores"}).json()
    assert section["model_version"] == loaded.model_version
    assert np.array_equal(section["test_scores"]["y_score"], loaded.s_test)
    bad = client.get(f"{API}/metrics", params={"section": "nope"})
    assert bad.status_code == 400 and "test_scores" in bad.json()["error"]["details"]["available"]


def test_t50_samples(client):
    body = client.get(f"{API}/samples").json()
    assert len(body["items"]) == 200
    assert body["categories"]["fraud_hard"]["count"] >= 20 and body["categories"]["legit_hard"]["count"] >= 20
    hard = client.get(f"{API}/samples", params={"category": "fraud_hard"}).json()["items"]
    assert {i["category"] for i in hard} == {"fraud_hard"}
    assert all(i["amount"] == i["features"]["Amount"] for i in hard)


#: Giờ cuối của ngày 2. TestClient đọc hết thân phản hồi rồi mới trả (không stream thật), nên ca
#: kiểm thử phát một giờ ở tốc độ 3600 — khoảng một giây. AC-A6 (3 phút liên tục) kiểm trên uvicorn.
LAST_HOUR = 23 * 3600


def read_events(client, params):
    response = client.get(f"{API}/replay/stream", params=params)
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    events, name = [], None
    for line in response.text.splitlines():
        if line.startswith("event: "):
            name = line[7:]
        elif line.startswith("data: "):
            events.append((name, json.loads(line[6:])))
    return events


def test_t50_replay_streams_day_two_and_writes_to_the_queue(client, clean_db, loaded):
    events = read_events(client, {"speed": 3600, "start": LAST_HOUR})
    kinds = [k for k, _ in events]
    start = events[0][1]
    assert kinds[0] == "start" and start["start"] == LAST_HOUR and kinds[-1] == "end"
    txs = [d for k, d in events if k in ("transaction", "alert")]
    expected = int((loaded.test_set["Time"] >= 86_400 + LAST_HOUR).sum())
    assert len(txs) == start["total"] == expected == events[-1][1]["processed"]
    assert all(d["sim_seconds"] >= LAST_HOUR for d in txs)
    assert [d["sim_seconds"] for d in txs] == sorted(d["sim_seconds"] for d in txs)
    assert all((k == "alert") == (d["risk_score"] >= start["threshold"]) for k, d in events if k in ("transaction", "alert"))
    assert "stats" in kinds
    # Mọi giao dịch đã phát đều nằm trong bảng, với nhãn thật của tập kiểm thử
    assert count(clean_db, "SELECT count(*) FROM transactions WHERE source = 'replay'") == expected
    assert count(clean_db, "SELECT count(*) FROM transactions WHERE source = 'replay' AND true_label IS NULL") == 0


def test_tc44_replaying_twice_does_not_duplicate(client, clean_db):
    """TC-44: mã phát lại tất định + ON CONFLICT DO NOTHING — phát lại lần hai không nhân đôi."""
    read_events(client, {"speed": 3600, "start": LAST_HOUR})
    first = count(clean_db, "SELECT count(*) FROM transactions WHERE source = 'replay'")
    read_events(client, {"speed": 3600, "start": LAST_HOUR})
    assert count(clean_db, "SELECT count(*) FROM transactions WHERE source = 'replay'") == first > 0


def test_replay_rejects_bad_speed(client):
    assert client.get(f"{API}/replay/stream", params={"speed": 0}).status_code == 400
    assert client.get(f"{API}/replay/stream", params={"start": 90_000}).status_code == 400


# --------------------------------------------------------------------------
# AC-A10 / TC-41, NFR-01
# --------------------------------------------------------------------------

@pytest.mark.parametrize("method, path", [
    ("post", "/score"), ("post", "/score/batch"), ("post", "/score/upload"), ("post", "/explain"),
    ("post", "/reviews"), ("put", "/threshold"), ("post", "/threshold/optimize"),
])
def test_tc41_empty_body_is_422_never_500(client, method, path):
    response = getattr(client, method)(f"{API}{path}")
    assert response.status_code == 422, response.text
    assert response.json()["error"]["code"] in ("INVALID_BODY", "MISSING_FEATURES")


def test_unknown_route_uses_the_error_envelope(client):
    response = client.get(f"{API}/khong-co")
    assert response.status_code == 404 and response.json()["error"]["code"] == "NOT_FOUND"


def test_nfr01_single_score_latency(client, tx):
    """NFR-01: p95 < 50 ms trên 1.000 lời gọi liên tiếp (chấm thử, không ghi)."""
    server, wall = [], []
    for _ in range(1000):
        started = time.perf_counter()
        body = score(client, tx, persist=False).json()
        wall.append((time.perf_counter() - started) * 1000)
        server.append(body["latency_ms"])
    p95_server, p95_wall = np.percentile(server, 95), np.percentile(wall, 95)
    print(f"\nNFR-01: p95 máy chủ {p95_server:.1f} ms, p95 tính cả HTTP {p95_wall:.1f} ms "
          f"(trung vị {np.median(server):.1f} / {np.median(wall):.1f} ms)")
    assert p95_server < 50
