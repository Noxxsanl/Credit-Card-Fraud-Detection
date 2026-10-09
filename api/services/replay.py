"""Chế độ phát lại — API-15, UI-05.

Phát các giao dịch **ngày thứ hai** của ``test_set.parquet`` theo thứ tự ``Time``, thời
gian nén ``speed`` lần (60: một phút dữ liệu mỗi giây). Máy chủ không giữ trạng thái
nào ngoài con trỏ của chính kết nối; tạm dừng là ngắt kết nối, tiếp tục là mở lại với
``start`` bằng đồng hồ mô phỏng lúc dừng.

Điểm rủi ro do **mô hình chấm lúc giao dịch "đến"**, cùng đường với ``/score``
(``build_features`` → ``FastScorer``), theo từng mẻ nhỏ gồm những giao dịch đã tới giờ trên
đồng hồ mô phỏng — không đọc cột ``risk_score`` tính sẵn trong tệp parquet. Cột đó chỉ còn
dùng để ``api/loader.py`` kiểm mô hình lúc khởi động.

Mỗi giao dịch được ghi vào bảng ``transactions`` (``source = 'replay'``) theo mẻ 100
dòng, không giữ một giao dịch cơ sở dữ liệu mở suốt phiên (06 §4.2). Mã giao dịch tất
định ``RP-<dòng>`` và ghi bằng ``ON CONFLICT DO NOTHING``: phát lại lần hai không nhân
đôi hàng đợi.
"""

from __future__ import annotations

import asyncio
import json
import logging
import secrets
import time
from collections.abc import AsyncIterator, Callable

import numpy as np
import pandas as pd
from starlette.concurrency import run_in_threadpool

from src.features import RAW_REQUIRED_COLUMNS

from ..config import REPLAY_WRITE_BATCH
from ..loader import DAY_TWO_START, Artifacts
from .scoring import insert_rows, risk_band, score_frame, transaction_rows

log = logging.getLogger("api")

#: Nhịp gửi sự kiện stats — cũng là nhịp "còn sống" khi lâu không có giao dịch
STATS_EVERY_SECONDS = 1.0
#: Đọc lại ngưỡng hiện hành theo nhịp này, để đổi ngưỡng giữa chừng có tác dụng ngay
THRESHOLD_REFRESH_SECONDS = 5.0
DAY_SECONDS = 86_400
#: Chấm tối đa chừng này giao dịch một lần gọi mô hình — ở tốc độ cao không dồn cả giờ vào một mẻ
SCORE_BATCH_LIMIT = 500


def day_two(loaded: Artifacts) -> pd.DataFrame:
    frame = loaded.test_set
    part = frame[frame["Time"] >= DAY_TWO_START]
    return part.assign(offset=part["Time"] - DAY_TWO_START, test_row=part.index.to_numpy())


def clock(seconds: float) -> str:
    seconds = int(seconds) % DAY_SECONDS
    return f"{seconds // 3600:02d}:{seconds % 3600 // 60:02d}:{seconds % 60:02d}"


def sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


async def stream(loaded: Artifacts, *, speed: float, start: float, read_threshold: Callable[[], float],
                 write: Callable[[list[tuple]], None], is_disconnected: Callable) -> AsyncIterator[str]:
    """Sinh chuỗi sự kiện SSE: ``start``, rồi ``transaction``/``alert``, ``stats`` mỗi giây, ``end``."""
    rows = day_two(loaded)
    rows = rows[rows["offset"] >= start]
    total = len(rows)
    batch_id = f"replay-{secrets.token_hex(3)}"
    tau = await run_in_threadpool(read_threshold)
    offsets = rows["offset"].to_numpy(dtype="float64")
    scores = np.empty(total, dtype="float64")       # điền dần khi giao dịch tới giờ
    amounts = rows["Amount"].to_numpy(dtype="float64")
    test_rows = rows["test_row"].to_numpy()
    raw = rows[RAW_REQUIRED_COLUMNS].reset_index(drop=True)
    labels = rows["Class"].to_numpy()

    started = time.monotonic()
    last_stats = last_refresh = started
    processed = alerts = scored = 0
    pending: list[int] = []

    def sim_now() -> float:
        return start + (time.monotonic() - started) * speed

    def stats() -> str:
        return sse("stats", {"processed": processed, "alerts": alerts, "remaining": total - processed,
                             "elapsed_sim_seconds": round(sim_now() - start, 1),
                             "sim_time": clock(min(sim_now(), DAY_SECONDS)), "threshold": tau})

    async def flush() -> None:
        nonlocal pending
        if not pending:
            return
        chunk, pending = pending, []
        await run_in_threadpool(write, _rows(chunk))

    def score_due(first: int) -> int:
        """Chấm mọi giao dịch từ ``first`` đã tới giờ trên đồng hồ mô phỏng (ít nhất một)."""
        due = int(np.searchsorted(offsets, sim_now(), side="right"))
        last = min(max(due, first + 1), first + SCORE_BATCH_LIMIT, total)
        scores[first:last] = score_frame(loaded, raw.iloc[first:last])
        return last

    def _rows(positions: list[int]) -> list[tuple]:
        idx = np.asarray(positions)
        return transaction_rows(raw.iloc[idx], scores[idx], [f"RP-{test_rows[i]:05d}" for i in idx],
                                model_version=loaded.model_version, source="replay", batch_id=batch_id,
                                labels=labels[idx])

    yield "retry: 3000\n\n"
    yield sse("start", {"total": total, "start": start, "speed": speed, "threshold": tau,
                        "sim_time": clock(start), "model_version": loaded.model_version, "batch_id": batch_id})
    try:
        for i in range(total):
            # Đợi tới lúc đồng hồ mô phỏng chạm giao dịch này; trong lúc đợi vẫn gửi stats
            while (wait := (offsets[i] - sim_now()) / speed) > 0:
                await asyncio.sleep(min(wait, STATS_EVERY_SECONDS))
                if time.monotonic() - last_stats >= STATS_EVERY_SECONDS:
                    last_stats = time.monotonic()
                    yield stats()
                if await is_disconnected():
                    return
            if time.monotonic() - last_refresh >= THRESHOLD_REFRESH_SECONDS:
                last_refresh = time.monotonic()
                tau = await run_in_threadpool(read_threshold)
            if i >= scored:
                scored = await run_in_threadpool(score_due, i)

            score = float(scores[i])
            is_alert = score >= tau
            processed += 1
            alerts += int(is_alert)
            yield sse("alert" if is_alert else "transaction", {
                "id": f"RP-{test_rows[i]:05d}", "risk_score": score,
                "risk_band": risk_band(score, tau, loaded.block_threshold),
                "amount": float(amounts[i]), "sim_time": clock(offsets[i]), "sim_seconds": float(offsets[i]),
            })
            pending.append(i)
            if len(pending) >= REPLAY_WRITE_BATCH:
                await flush()
            if time.monotonic() - last_stats >= STATS_EVERY_SECONDS:
                last_stats = time.monotonic()
                yield stats()
                if await is_disconnected():
                    return
            elif processed % 200 == 0:
                await asyncio.sleep(0)       # tốc độ cao: nhường vòng lặp sự kiện
        await flush()
        yield stats()
        yield sse("end", {"processed": processed, "alerts": alerts})
    finally:
        # Ngắt kết nối giữa chừng: ghi nốt những gì đã phát, đồng bộ vì vòng lặp có thể đang đóng
        if pending:
            try:
                write(_rows(pending))
            except Exception:  # noqa: BLE001
                log.exception("Không ghi được mẻ cuối của phiên phát lại %s", batch_id)
