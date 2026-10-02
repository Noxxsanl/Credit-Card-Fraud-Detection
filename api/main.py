"""Khởi tạo FastAPI: nạp hiện vật lúc khởi động, CORS, mô hình lỗi, API-01 /health.

Chạy trên máy:

    docker compose up -d db && alembic upgrade head
    uvicorn api.main:app --reload --port 8000        # tài liệu: http://localhost:8000/docs

Thứ tự khởi động (06 §8): nạp hiện vật vào bộ nhớ → chèn khóa ``settings`` còn thiếu.
Lược đồ do Alembic dựng từ trước (entrypoint của container, hoặc tay khi phát triển).
Thiếu hiện vật hay mất cơ sở dữ liệu thì tiến trình **vẫn sống**; ``/health`` trả 503.
"""

from __future__ import annotations

import logging
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware

from . import errors
from .config import API_PREFIX, Settings
from .db import make_engine, make_session_factory, ping
from .loader import ArtifactError, load_artifacts
from .routes import metrics, replay, scoring, threshold, transactions
from .schemas import ErrorResponse, HealthResponse
from .services import runtime_settings

log = logging.getLogger("api")

_ERRORS = {code: {"model": ErrorResponse} for code in (400, 404, 413, 422, 500, 503)}


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or Settings.from_env()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.settings = settings
        app.state.started_at = time.monotonic()
        app.state.engine = make_engine(settings.database_url)
        app.state.session_factory = make_session_factory(app.state.engine)
        app.state.artifacts, app.state.artifact_error = None, None
        try:
            app.state.artifacts = load_artifacts(settings)
        except ArtifactError as exc:
            app.state.artifact_error = str(exc)
            log.error("Không nạp được hiện vật: %s", exc)
        except Exception as exc:  # noqa: BLE001 — pickle hỏng, lệch phiên bản…: vẫn phải sống để báo 503
            app.state.artifact_error = f"{type(exc).__name__}: {exc}"
            log.exception("Không nạp được hiện vật")
        if app.state.artifacts is not None:
            try:
                runtime_settings.ensure_defaults(app.state.engine, app.state.artifacts)
            except Exception as exc:  # noqa: BLE001
                log.warning("Chưa chèn được giá trị mặc định vào settings (db chưa sẵn sàng?): %s", exc)
        yield
        app.state.engine.dispose()

    app = FastAPI(
        title="Fraud Console API",
        version="1.0.0",
        summary="Chấm điểm rủi ro gian lận thẻ tín dụng — hợp đồng ở docs/05",
        lifespan=lifespan,
        responses=_ERRORS,
    )
    app.add_middleware(CORSMiddleware, allow_origins=list(settings.cors_origins),
                       allow_methods=["*"], allow_headers=["*"])
    errors.install(app)

    @app.middleware("http")
    async def access_log(request: Request, call_next):
        started = time.perf_counter()
        response = await call_next(request)
        log.info("%s %s %s %.1fms", request.method, request.url.path, response.status_code,
                 (time.perf_counter() - started) * 1000)
        return response

    @app.get(f"{API_PREFIX}/health", response_model=HealthResponse, tags=["hệ thống"],
             summary="API-01 — tình trạng và phiên bản mô hình")
    def health(request: Request):
        state = request.app.state
        db_ok = ping(state.engine)
        loaded = state.artifacts
        body = {
            "status": "ok",
            "model_version": loaded.model_version if loaded else None,
            "model_loaded_at": loaded.loaded_at if loaded else None,
            "threshold": None,
            "db": "ok" if db_ok else "down",
            "uptime_seconds": round(time.monotonic() - state.started_at, 1),
        }
        if loaded is None:
            raise errors.ApiError(503, "MODEL_NOT_LOADED", "Chưa nạp được hiện vật mô hình",
                                  _jsonable({**body, "status": "unavailable", "reason": state.artifact_error}))
        if not db_ok:
            raise errors.ApiError(503, "DATABASE_UNAVAILABLE", "Không kết nối được PostgreSQL",
                                  _jsonable({**body, "status": "unavailable"}))
        with state.session_factory() as session:
            body["threshold"] = runtime_settings.current_threshold(session, loaded)[0]
        return body

    for module in (scoring, transactions, threshold, metrics, replay):
        app.include_router(module.router)
    return app


def _jsonable(body: dict) -> dict:
    return {k: (v.isoformat().replace("+00:00", "Z") if hasattr(v, "isoformat") else v) for k, v in body.items()}


logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s [%(name)s] %(message)s")
app = create_app()
