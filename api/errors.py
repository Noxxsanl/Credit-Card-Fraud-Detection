"""Mô hình lỗi thống nhất (docs/05 §4).

Mọi lỗi trả về ``{"error": {"code", "message", "details"}}``. Hai nguyên tắc: không
bao giờ trả 200 kèm thông báo lỗi, và không bao giờ để ngoại lệ chưa bắt làm sập
tiến trình (AC-A10).
"""

from __future__ import annotations

import logging
import uuid

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from sqlalchemy.exc import DBAPIError, OperationalError
from starlette.exceptions import HTTPException as StarletteHTTPException

from src.features import RAW_REQUIRED_COLUMNS

log = logging.getLogger("api")

_FEATURES = set(RAW_REQUIRED_COLUMNS)


class ApiError(Exception):
    def __init__(self, status: int, code: str, message: str, details: dict | None = None):
        super().__init__(message)
        self.status, self.code, self.message, self.details = status, code, message, details


def error_response(status: int, code: str, message: str, details: dict | None = None) -> JSONResponse:
    body = {"error": {"code": code, "message": message, "details": details}}
    return JSONResponse(status_code=status, content=body)


def not_found(what: str, key: str) -> ApiError:
    return ApiError(404, "NOT_FOUND", f"Không có {what} với mã {key}", {"id": key})


def _feature_of(loc: tuple) -> tuple[str | None, int | None]:
    """Tên đặc trưng và chỉ số phần tử (trong lô) mà một lỗi Pydantic trỏ tới."""
    feature = next((p for p in reversed(loc) if isinstance(p, str) and p in _FEATURES), None)
    row = next((p for p in loc if isinstance(p, int)), None)
    return feature, row


def classify_validation(exc: RequestValidationError) -> tuple[int, str, str, dict]:
    """Đổi lỗi Pydantic sang mã lỗi của hợp đồng.

    - Thiếu cột bắt buộc → 422 ``MISSING_FEATURES`` kèm danh sách cột (DS-10, TC-32).
    - Giá trị đặc trưng sai (không phải số, NaN, Inf, Amount âm…) → 422 ``INVALID_FEATURE_VALUE``.
    - Thân yêu cầu sai cấu trúc → 422 ``INVALID_BODY``; tham số truy vấn sai → 400 ``INVALID_REQUEST``.
    """
    errors = exc.errors()
    missing, invalid, other = [], [], []
    for err in errors:
        loc = tuple(err.get("loc", ()))
        feature, row = _feature_of(loc)
        if feature and err.get("type") == "missing":
            missing.append((feature, row))
        elif feature:
            item = {"field": feature, "reason": err.get("msg", "")}
            if row is not None:
                item["row"] = row
            invalid.append(item)
        else:
            other.append({"loc": [str(p) for p in loc], "reason": err.get("msg", "")})

    if missing:
        names = sorted({f for f, _ in missing}, key=RAW_REQUIRED_COLUMNS.index)
        details = {"missing": names}
        rows = sorted({r for _, r in missing if r is not None})
        if rows:
            details["rows"] = rows[:100]
        return 422, "MISSING_FEATURES", "Thiếu cột bắt buộc trong dữ liệu vào", details
    if invalid:
        return 422, "INVALID_FEATURE_VALUE", "Giá trị đặc trưng không hợp lệ", {"errors": invalid[:100]}
    if other and all(o["loc"] and o["loc"][0] in ("query", "path") for o in other):
        return 400, "INVALID_REQUEST", "Tham số truy vấn sai định dạng", {"errors": other[:100]}
    return 422, "INVALID_BODY", "Thân yêu cầu sai cấu trúc", {"errors": other[:100]}


def install(app: FastAPI) -> None:
    @app.exception_handler(ApiError)
    async def _api_error(_: Request, exc: ApiError):
        return error_response(exc.status, exc.code, exc.message, exc.details)

    @app.exception_handler(RequestValidationError)
    async def _validation(_: Request, exc: RequestValidationError):
        status, code, message, details = classify_validation(exc)
        return error_response(status, code, message, details)

    @app.exception_handler(StarletteHTTPException)
    async def _http(_: Request, exc: StarletteHTTPException):
        code = {404: "NOT_FOUND", 405: "METHOD_NOT_ALLOWED", 413: "PAYLOAD_TOO_LARGE"}.get(exc.status_code, "INVALID_REQUEST")
        return error_response(exc.status_code, code, str(exc.detail))

    @app.exception_handler(OperationalError)
    @app.exception_handler(DBAPIError)
    async def _database(_: Request, exc: DBAPIError):
        if isinstance(exc, OperationalError) or exc.connection_invalidated:
            log.warning("Mất kết nối PostgreSQL: %s", exc.orig)
            return error_response(503, "DATABASE_UNAVAILABLE", "Không kết nối được PostgreSQL")
        return _internal(exc)

    @app.exception_handler(Exception)
    async def _unexpected(_: Request, exc: Exception):
        return _internal(exc)


def _internal(exc: Exception) -> JSONResponse:
    trace_id = uuid.uuid4().hex[:12]
    log.exception("Lỗi không lường trước [%s]", trace_id, exc_info=exc)
    return error_response(500, "INTERNAL_ERROR", "Lỗi không lường trước", {"trace_id": trace_id})
