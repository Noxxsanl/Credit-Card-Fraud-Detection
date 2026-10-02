"""API-15: Server-Sent Events cho chế độ phát lại."""

from __future__ import annotations

from fastapi import APIRouter, Depends, Query, Request
from fastapi.responses import StreamingResponse

from ..config import API_PREFIX
from ..deps import get_artifacts
from ..loader import Artifacts
from ..services import replay, runtime_settings
from ..services.scoring import insert_rows

router = APIRouter(prefix=API_PREFIX, tags=["phát lại"])


@router.get("/replay/stream", summary="API-15 — phát lại ngày 2 của tập kiểm thử (SSE)",
            response_description="text/event-stream: start, transaction, alert, stats, end")
def replay_stream(
    request: Request,
    speed: float | None = Query(None, ge=1, le=3600, description="Nén thời gian; mặc định lấy từ settings (60)"),
    start: float = Query(0, ge=0, lt=86_400, description="Giây mô phỏng tính từ đầu ngày 2 — để tiếp tục sau khi tạm dừng"),
    loaded: Artifacts = Depends(get_artifacts),
):
    factory = request.app.state.session_factory

    def read_threshold() -> float:
        with factory() as session:
            return runtime_settings.current_threshold(session, loaded)[0]

    def write(rows: list[tuple]) -> None:
        with factory() as session:
            insert_rows(session, rows)
            session.commit()

    if speed is None:
        with factory() as session:
            speed = runtime_settings.replay_speed(session)

    events = replay.stream(loaded, speed=speed, start=start, read_threshold=read_threshold, write=write,
                           is_disconnected=request.is_disconnected)
    return StreamingResponse(events, media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})
