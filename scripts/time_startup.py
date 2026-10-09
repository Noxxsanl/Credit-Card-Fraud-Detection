"""Bấm giờ khởi động cả hệ thống — NFR-04, AC-A8 (T-59).

    python scripts/time_startup.py                 # warm: down (GIỮ volume) → up → /health 200
    python scripts/time_startup.py --cold --yes    # cold: down -v (XÓA dữ liệu) → up → /health 200
    python scripts/time_startup.py --runs 3        # lặp lại, in từng lần

Đo từ lúc gọi ``docker compose up -d`` tới khi ``GET /api/v1/health`` trả 200 (cách đo của
NFR-04, docs/01), thăm mỗi 0,2 giây; kèm mốc ``db`` khỏe và trang giao diện trả 200. Ảnh phải build
sẵn (``docker compose build``) — thời gian build không tính vào NFR-04.

Chỉ dùng thư viện chuẩn: chạy được bằng bất kỳ Python 3 nào, không cần .venv. Cổng và tên dự án
Compose theo đúng biến môi trường mà ``docker compose`` đọc (``API_HOST_PORT``, ``WEB_HOST_PORT``,
``COMPOSE_PROJECT_NAME``, hoặc tệp ``.env``).
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LIMITS = {"cold": 45.0, "warm": 15.0}  # NFR-04


def env_port(name: str, default: int) -> int:
    if name in os.environ:
        return int(os.environ[name])
    env_file = ROOT / ".env"
    if env_file.exists():
        for line in env_file.read_text(encoding="utf-8").splitlines():
            key, _, value = line.partition("=")
            if key.strip() == name and value.strip():
                return int(value.strip())
    return default


def compose(*args: str) -> None:
    subprocess.run(["docker", "compose", *args], cwd=ROOT, check=True,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def ok(url: str) -> bool:
    try:
        with urllib.request.urlopen(url, timeout=2) as response:
            return response.status == 200
    except Exception:  # noqa: BLE001 — chưa lên, 502, 503… đều là "chưa"
        return False


def db_healthy() -> bool:
    out = subprocess.run(["docker", "compose", "ps", "--format", "{{.Service}} {{.Health}}"], cwd=ROOT,
                         capture_output=True, text=True).stdout
    return any(line.split() == ["db", "healthy"] for line in out.splitlines())


def measure(mode: str, timeout: float) -> dict[str, float | None]:
    compose("down", "-v") if mode == "cold" else compose("down")
    # 127.0.0.1: trên Windows "localhost" thử ::1 trước và chờ ~2 giây mỗi lần — sai số đo khởi động
    health = f"http://127.0.0.1:{env_port('API_HOST_PORT', 8000)}/api/v1/health"
    page = f"http://127.0.0.1:{env_port('WEB_HOST_PORT', 3000)}/"
    marks: dict[str, float | None] = {"db": None, "web": None, "health": None}

    started = time.perf_counter()
    up = subprocess.Popen(["docker", "compose", "up", "-d"], cwd=ROOT,
                          stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    while time.perf_counter() - started < timeout:
        now = time.perf_counter() - started
        if marks["db"] is None and db_healthy():
            marks["db"] = now
        if marks["web"] is None and ok(page):
            marks["web"] = now
        if ok(health):
            marks["health"] = time.perf_counter() - started
            break
        time.sleep(0.2)
    up.wait()
    return marks


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--cold", action="store_true", help="xóa volume pgdata trước khi đo (mất dữ liệu ứng dụng)")
    parser.add_argument("--yes", action="store_true", help="đồng ý xóa volume khi --cold")
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument("--timeout", type=float, default=180)
    args = parser.parse_args()

    mode = "cold" if args.cold else "warm"
    if args.cold and not args.yes:
        sys.exit("--cold chạy `docker compose down -v`: XÓA toàn bộ giao dịch và thẩm định trong volume pgdata.\n"
                 "Sao lưu trước nếu cần (python scripts/demo_db.py dump), rồi chạy lại kèm --yes.")

    print(f"{mode}: giới hạn NFR-04 {LIMITS[mode]:.0f} giây")
    worst = 0.0
    for run in range(1, args.runs + 1):
        m = measure(mode, args.timeout)
        fmt = lambda v: "—" if v is None else f"{v:5.1f} s"  # noqa: E731
        print(f"  lần {run}: db khỏe {fmt(m['db'])} · giao diện 200 {fmt(m['web'])} · /health 200 {fmt(m['health'])}")
        if m["health"] is None:
            sys.exit(f"/health chưa trả 200 sau {args.timeout:.0f} giây — xem `docker compose logs api`")
        worst = max(worst, m["health"])
    verdict = "ĐẠT" if worst < LIMITS[mode] else "KHÔNG ĐẠT"
    print(f"Lâu nhất {worst:.1f} s / giới hạn {LIMITS[mode]:.0f} s → {verdict}")


if __name__ == "__main__":
    main()
