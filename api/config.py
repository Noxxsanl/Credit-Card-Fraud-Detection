"""Cấu hình tầng API, đọc từ biến môi trường.

Khi chạy trên máy, ``.env`` ở gốc repo được nạp trước (không ghi đè biến đã đặt sẵn).
Trong Docker Compose, biến đến từ ``docker-compose.yml``.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

from src.config import MODELS_DIR, PROJECT_ROOT, SAMPLE_POOL_PATH, TEST_SET_PATH

load_dotenv(PROJECT_ROOT / ".env", override=False)

API_PREFIX = "/api/v1"

#: DS-15 — số giao dịch tối đa trong một lời gọi /score/batch
MAX_BATCH_SIZE = 50_000

#: NFR-06 — kích thước tệp CSV tối đa
MAX_UPLOAD_BYTES = 100 * 1024 * 1024

#: 05 API-06
DEFAULT_PAGE_SIZE = 50
MAX_PAGE_SIZE = 200

#: 06 §4.2 — phát lại ghi theo từng mẻ, không giữ một giao dịch mở suốt phiên
REPLAY_WRITE_BATCH = 100

#: Số dòng lỗi và số kết quả tối đa trả về trong phản hồi /score/upload
MAX_REPORTED_REJECTIONS = 100
MAX_UPLOAD_RESULTS = 200

#: 05 §2 — chỉ "đề xuất chặn" khi trên out-of-fold, mọi ngưỡng từ mức đó trở lên đều có ít nhất
#: 95% cảnh báo là gian lận thật. Với mô hình hiện tại: ngưỡng ≈ 0,9735; trên tập kiểm thử chặn
#: 70 giao dịch, 69 là gian lận (luật cũ "≥ 3τ" chặn 96, trong đó 19 khách hợp lệ).
BLOCK_MIN_PRECISION = 0.95

# 127.0.0.1 chứ không phải localhost: trên Windows "localhost" thử ::1 trước, và cổng chỉ mở trên
# 127.0.0.1 (docker-compose.yml) thì mỗi kết nối mới chờ thêm 2–5 giây trước khi quay về IPv4
DEFAULT_DATABASE_URL = "postgresql+psycopg://fraud:fraud@127.0.0.1:5432/fraud"


@dataclass(frozen=True)
class Settings:
    database_url: str
    models_dir: Path
    test_set_path: Path
    sample_pool_path: Path
    cors_origins: tuple[str, ...]

    @classmethod
    def from_env(cls, **overrides) -> "Settings":
        values = dict(
            database_url=os.environ.get("DATABASE_URL", DEFAULT_DATABASE_URL),
            models_dir=Path(os.environ.get("MODELS_DIR", MODELS_DIR)),
            test_set_path=Path(os.environ.get("TEST_SET_PATH", TEST_SET_PATH)),
            sample_pool_path=Path(os.environ.get("SAMPLE_POOL_PATH", SAMPLE_POOL_PATH)),
            cors_origins=tuple(
                o.strip()
                for o in os.environ.get("CORS_ORIGINS", "http://localhost:3000,http://127.0.0.1:3000").split(",")
                if o.strip()
            ),
        )
        values.update(overrides)
        return cls(**values)
