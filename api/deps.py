"""Dependency của FastAPI: hiện vật trong bộ nhớ và phiên cơ sở dữ liệu."""

from __future__ import annotations

from collections.abc import Iterator

from fastapi import Request
from sqlalchemy.orm import Session

from .errors import ApiError
from .loader import Artifacts


def get_artifacts(request: Request) -> Artifacts:
    loaded = request.app.state.artifacts
    if loaded is None:
        raise ApiError(503, "MODEL_NOT_LOADED", "Chưa nạp được hiện vật mô hình",
                       {"reason": request.app.state.artifact_error})
    return loaded


def get_session(request: Request) -> Iterator[Session]:
    session = request.app.state.session_factory()
    try:
        yield session
    finally:
        session.close()
