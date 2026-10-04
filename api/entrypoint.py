"""Điểm vào của container ``api`` (docs/03 §6.3, docs/06 §8).

    python -m api.entrypoint

1. ``alembic upgrade head`` — lần đầu dựng lược đồ và hạt giống ``settings``; các lần sau
   lược đồ đã mới nhất nên không làm gì. Compose chỉ khởi động container này khi
   healthcheck của ``db`` báo khỏe, nhưng vẫn thử lại vài lần cho chắc: kết nối đầu tiên
   ngay sau ``initdb`` đôi khi bị từ chối.
2. ``uvicorn api.main:app`` ở ``0.0.0.0:8000``, **một** worker: hiện vật và trạng thái phát
   lại nằm trong bộ nhớ tiến trình.

Viết bằng Python thay vì shell để không vỡ khi git trên Windows đổi xuống dòng thành CRLF.
"""

from __future__ import annotations

import logging
import os
import sys
import time

import uvicorn
from sqlalchemy.exc import OperationalError

from .config import Settings
from .db import upgrade

log = logging.getLogger("api.entrypoint")

MIGRATE_ATTEMPTS = 30
MIGRATE_DELAY_SECONDS = 1.0


def migrate(url: str) -> None:
    started = time.perf_counter()
    for attempt in range(1, MIGRATE_ATTEMPTS + 1):
        try:
            upgrade(url)
            break
        except OperationalError as exc:
            if attempt == MIGRATE_ATTEMPTS:
                raise
            log.warning("PostgreSQL chưa nhận kết nối (lần %d/%d): %s", attempt, MIGRATE_ATTEMPTS,
                        str(exc.orig).strip().splitlines()[0])
            time.sleep(MIGRATE_DELAY_SECONDS)
    log.info("alembic upgrade head xong sau %.1f giây", time.perf_counter() - started)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s [%(name)s] %(message)s")
    try:
        migrate(Settings.from_env().database_url)
    except Exception:
        log.exception("Không dựng được lược đồ — dừng container")
        sys.exit(1)
    uvicorn.run(
        "api.main:app",
        host="0.0.0.0",
        port=int(os.environ.get("PORT", "8000")),
        workers=1,
        # Luồng SSE của chế độ phát lại không tự đóng: không giới hạn thì `docker compose
        # restart api` phải chờ hết 10 giây ân hạn của Docker rồi mới giết tiến trình.
        timeout_graceful_shutdown=3,
        access_log=False,  # api/main.py đã ghi một dòng cho mỗi yêu cầu, kèm thời gian xử lý
    )


if __name__ == "__main__":
    main()
