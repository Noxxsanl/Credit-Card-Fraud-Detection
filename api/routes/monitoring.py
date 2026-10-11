"""API-16: theo dõi vận hành — cảnh báo và precision theo thời gian, so với kỳ vọng (docs/11 M2)."""

from __future__ import annotations

from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session

from ..config import API_PREFIX
from ..deps import get_artifacts, get_session
from ..loader import Artifacts
from ..schemas import MonitoringReport
from ..services import monitoring, runtime_settings

router = APIRouter(prefix=API_PREFIX, tags=["theo dõi"])


@router.get("/monitoring", response_model=MonitoringReport,
            summary="API-16 — tỷ lệ cảnh báo, precision thẩm định và recall theo thời gian, so với kỳ vọng")
def report(bucket_hours: float = Query(24.0, gt=0, le=48, description="Độ rộng mỗi khoảng, theo giờ của dữ liệu"),
           loaded: Artifacts = Depends(get_artifacts), session: Session = Depends(get_session)):
    tau, _ = runtime_settings.current_threshold(session, loaded)
    return monitoring.report(session, loaded, tau, bucket_hours=bucket_hours)
