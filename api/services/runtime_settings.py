"""Cấu hình thay đổi được lúc chạy — bảng ``settings`` (docs/06 §2).

Ngưỡng hiện hành là cấu hình chứ không phải một phần của mô hình (AR-03):

- Người dùng chưa đặt gì → ngưỡng là τ* của ``threshold.json``, ``source = "artifact"``.
- Người dùng đặt qua ``PUT /threshold`` → dòng ``threshold`` lưu ``{"value", "model_version"}``,
  ``source = "user"``. Dòng này chỉ có hiệu lực với đúng mô hình lúc đặt: huấn luyện lại
  thì phân bố điểm đổi, một con số đặt cho mô hình cũ không còn nghĩa gì.
"""

from __future__ import annotations

import json
import logging

from sqlalchemy import text
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session

from ..loader import Artifacts

log = logging.getLogger("api")

THRESHOLD, COST_FN, COST_FP, REPLAY_SPEED = "threshold", "cost_fn", "cost_fp", "replay_speed"
DEFAULT_REPLAY_SPEED = 60

_UPSERT = text(
    "INSERT INTO settings (key, value, updated_at) VALUES (:key, CAST(:value AS JSONB), now()) "
    "ON CONFLICT (key) DO UPDATE SET value = EXCLUDED.value, updated_at = now()"
)


def ensure_defaults(engine: Engine, loaded: Artifacts) -> None:
    """Chèn khóa còn thiếu, không ghi đè khóa đã có (06 §2). Gọi lúc khởi động."""
    defaults = {
        COST_FN: loaded.threshold["cost_false_negative"],
        COST_FP: loaded.threshold["cost_false_positive"],
        REPLAY_SPEED: DEFAULT_REPLAY_SPEED,
    }
    with engine.begin() as conn:
        for key, value in defaults.items():
            conn.execute(
                text("INSERT INTO settings (key, value) VALUES (:key, CAST(:value AS JSONB)) "
                     "ON CONFLICT (key) DO NOTHING"),
                {"key": key, "value": json.dumps(value)},
            )


def _all(session: Session) -> dict:
    return dict(session.execute(text("SELECT key, value FROM settings")).all())


def current_threshold(session: Session, loaded: Artifacts) -> tuple[float, str]:
    """Ngưỡng hiện hành và nguồn của nó (``artifact`` hoặc ``user``)."""
    row = session.execute(text("SELECT value FROM settings WHERE key = :k"), {"k": THRESHOLD}).scalar()
    if isinstance(row, dict) and row.get("model_version") == loaded.model_version:
        return float(row["value"]), "user"
    return loaded.default_threshold, "artifact"


def costs(session: Session, loaded: Artifacts) -> tuple[float, float]:
    values = _all(session)
    return (
        float(values.get(COST_FN, loaded.threshold["cost_false_negative"])),
        float(values.get(COST_FP, loaded.threshold["cost_false_positive"])),
    )


def replay_speed(session: Session) -> float:
    return float(_all(session).get(REPLAY_SPEED, DEFAULT_REPLAY_SPEED))


def set_threshold(session: Session, loaded: Artifacts, value: float,
                  cost_fn: float | None = None, cost_fp: float | None = None) -> None:
    """Ghi ngưỡng (và tham số chi phí nếu có) trong một giao dịch — UPSERT (06 §4.2)."""
    session.execute(_UPSERT, {"key": THRESHOLD,
                              "value": json.dumps({"value": value, "model_version": loaded.model_version})})
    if cost_fn is not None:
        session.execute(_UPSERT, {"key": COST_FN, "value": json.dumps(cost_fn)})
    if cost_fp is not None:
        session.execute(_UPSERT, {"key": COST_FP, "value": json.dumps(cost_fp)})
    session.commit()
