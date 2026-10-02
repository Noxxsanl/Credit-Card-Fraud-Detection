"""API-09 … API-12: ngưỡng hiện hành, đặt ngưỡng, xem trước chỉ số, tối ưu theo chi phí."""

from __future__ import annotations

import math

from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session

from ..config import API_PREFIX
from ..deps import get_artifacts, get_session
from ..errors import ApiError
from ..loader import Artifacts
from ..schemas import OptimizeRequest, OptimizeResponse, ThresholdMetricsOut, ThresholdState, ThresholdUpdate
from ..services import runtime_settings, thresholding

router = APIRouter(prefix=API_PREFIX, tags=["ngưỡng"])


def _check_threshold(value: float) -> None:
    if not (math.isfinite(value) and 0 < value < 1):
        raise ApiError(422, "INVALID_THRESHOLD", "Ngưỡng phải nằm trong khoảng (0, 1)", {"value": value})


def _state(session: Session, loaded: Artifacts) -> dict:
    current, source = runtime_settings.current_threshold(session, loaded)
    cost_fn, cost_fp = runtime_settings.costs(session, loaded)
    return {
        "current": current, "source": source, "default": loaded.default_threshold,
        "cost_fn": cost_fn, "cost_fp": cost_fp,
        "alternatives": loaded.threshold["alternatives"], "model_version": loaded.model_version,
    }


@router.get("/threshold", response_model=ThresholdState, summary="API-09 — ngưỡng hiện hành và các phương án")
def get_threshold(loaded: Artifacts = Depends(get_artifacts), session: Session = Depends(get_session)):
    return _state(session, loaded)


@router.put("/threshold", response_model=ThresholdState, summary="API-10 — đặt ngưỡng mới")
def put_threshold(body: ThresholdUpdate, loaded: Artifacts = Depends(get_artifacts),
                  session: Session = Depends(get_session)):
    _check_threshold(body.value)
    runtime_settings.set_threshold(session, loaded, body.value, body.cost_fn, body.cost_fp)
    return _state(session, loaded)


@router.get("/threshold/preview", response_model=ThresholdMetricsOut,
            summary="API-11 — chỉ số trên tập kiểm thử ứng với một ngưỡng giả định")
def preview(value: float = Query(..., description="Ngưỡng giả định, trong (0, 1)"),
            cost_fn: float | None = Query(None, gt=0), cost_fp: float | None = Query(None, ge=0),
            loaded: Artifacts = Depends(get_artifacts), session: Session = Depends(get_session)):
    _check_threshold(value)
    if cost_fn is None or cost_fp is None:
        saved_fn, saved_fp = runtime_settings.costs(session, loaded)
        cost_fn = saved_fn if cost_fn is None else cost_fn
        cost_fp = saved_fp if cost_fp is None else cost_fp
    return thresholding.preview(loaded, value, cost_fn, cost_fp)


@router.post("/threshold/optimize", response_model=OptimizeResponse,
             summary="API-12 — ngưỡng tối ưu theo tham số chi phí, chọn trên out-of-fold")
def optimize(body: OptimizeRequest, loaded: Artifacts = Depends(get_artifacts)):
    return thresholding.optimize(loaded, body.cost_fn, body.cost_fp, body.constraint.type, body.constraint.value)
