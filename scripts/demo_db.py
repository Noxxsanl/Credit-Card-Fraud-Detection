"""Dữ liệu demo cho buổi bảo vệ: nạp, sao lưu, khôi phục (T-60, docs/06 §7.3–7.4).

    python scripts/demo_db.py seed       # nạp dữ liệu demo qua API (cơ sở dữ liệu phải đang rỗng)
    python scripts/demo_db.py dump       # pg_dump → backup/fraud-demo.dump
    python scripts/demo_db.py restore    # khôi phục bản dump, bấm giờ
    python scripts/demo_db.py reset      # xóa giao dịch, thẩm định và ngưỡng người dùng đặt
    python scripts/demo_db.py status     # đếm dòng trong ba bảng

``seed`` gọi API (mặc định http://127.0.0.1:8000/api/v1, đổi bằng ``--api``) nên đi đúng đường
của giao diện: chấm bằng mô hình đang phục vụ, ghi bằng ``COPY``, ``threshold_used`` lấy theo
ngưỡng hiện hành. Ba phần:

1. Tải lên ``--rows`` dòng đầu của tập kiểm thử (mặc định 10.000, kèm cột ``Class``), bỏ các dòng
   đã có trong thư viện mẫu để hàng đợi không có hai bản của cùng một giao dịch.
2. Nạp thư viện mẫu, **trừ** nhóm ``fraud_hard`` — để dành cho đoạn trình diễn trực tiếp: nạp nhóm
   đó, mở một dòng, bấm ``A`` và giao diện báo kết luận sai (lenh-chay §8.4).
3. Thẩm định 5 gian lận và 3 cảnh báo giả điểm cao nhất của hàng đợi theo đúng nhãn thật, có ghi
   chú — để bộ lọc "đã thẩm định" có nội dung và kiểm được AC-A7.

``dump`` và ``restore`` chạy ``pg_dump``/``pg_restore`` **bên trong** container ``db`` rồi chép
tệp bằng ``docker compose cp``: chuyển hướng ``>`` của PowerShell 5.1 làm hỏng dữ liệu nhị phân.
Máy không có Python: chạy tay đúng các lệnh in ra khi thêm ``--dry-run``.
"""

from __future__ import annotations

import argparse
import io
import subprocess
import sys
import time
from pathlib import Path

import httpx
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import PROJECT_ROOT, SAMPLE_POOL_PATH, TEST_SET_PATH  # noqa: E402
from src.features import RAW_REQUIRED_COLUMNS  # noqa: E402

DEFAULT_API = "http://127.0.0.1:8000/api/v1"
DEFAULT_DUMP = PROJECT_ROOT / "backup" / "fraud-demo.dump"
CONTAINER_DUMP = "/tmp/fraud-demo.dump"
PSQL = ["docker", "compose", "exec", "-T", "db", "psql", "-U", "fraud", "-d", "fraud", "-v", "ON_ERROR_STOP=1"]

#: Nhóm mẫu nạp sẵn. fraud_hard để dành cho trình diễn trực tiếp.
SEED_CATEGORIES = ("fraud_easy", "legit_hard", "legit_easy")

#: Ghi chú thẩm định theo nhãn thật: mỗi ghi chú dùng cho một giao dịch, lấy từ đầu hàng đợi xuống
#: tới khi hết ghi chú — tức 5 gian lận và 3 cảnh báo giả điểm cao nhất.
REVIEW_NOTES = {
    1: ["Đã gọi chủ thẻ: không thực hiện giao dịch này — khóa thẻ",
        "Chuỗi giao dịch nhỏ liên tiếp trước đó, chủ thẻ xác nhận bị lộ thẻ",
        "Thiết bị lạ, chủ thẻ không nhận ra giao dịch",
        "Khớp mẫu gian lận đã biết",
        "Chủ thẻ báo mất thẻ hôm trước"],
    0: ["Chủ thẻ xác nhận đã mua — mở lại thẻ",
        "Giao dịch định kỳ của khách quen, số tiền lớn bất thường nhưng hợp lệ",
        "Khách đang đi du lịch, đã thông báo ngân hàng"],
}


def log(message: str) -> None:
    print(message, flush=True)


def run(cmd: list[str], *, dry_run: bool = False, capture: bool = False) -> str:
    log("$ " + " ".join(cmd))
    if dry_run:
        return ""
    done = subprocess.run(cmd, cwd=PROJECT_ROOT, check=True, text=True, encoding="utf-8",
                          stdout=subprocess.PIPE if capture else None)
    return done.stdout or ""


def counts() -> dict[str, int]:
    out = run(PSQL + ["-tAc", "SELECT (SELECT count(*) FROM transactions) || ' ' || (SELECT count(*) FROM reviews)"
                                 " || ' ' || (SELECT count(*) FROM settings)"], capture=True)
    tx, rv, st = (int(x) for x in out.split())
    return {"transactions": tx, "reviews": rv, "settings": st}


# --------------------------------------------------------------------------
# seed
# --------------------------------------------------------------------------

def seed(api: str, rows: int, force: bool) -> None:
    client = httpx.Client(base_url=api.rstrip("/"), timeout=120)
    try:
        health = client.get("/health")
    except httpx.ConnectError:
        sys.exit(f"Không kết nối được API tại {api}. Chạy `docker compose up` trước.")
    if health.status_code != 200:
        sys.exit(f"/health trả {health.status_code}: {health.text}")
    log(f"API sẵn sàng: {health.json()['model_version']}, ngưỡng {health.json()['threshold']:.6f}")

    existing = client.get("/transactions", params={"min_score": 0, "page_size": 1}).json()["total"]
    if existing and not force:
        sys.exit(f"Cơ sở dữ liệu đã có {existing} giao dịch. Chạy `python scripts/demo_db.py reset` trước, "
                 "hoặc thêm --force để nạp chồng lên.")

    pool = pd.read_json(SAMPLE_POOL_PATH, typ="series")
    sample_rows = {item["test_row"] for item in pool["items"]}
    test = pd.read_parquet(TEST_SET_PATH)
    upload = test.loc[~test.index.isin(sample_rows), RAW_REQUIRED_COLUMNS + ["Class"]].head(rows)
    buffer = io.BytesIO(upload.to_csv(index=False).encode("utf-8"))
    started = time.perf_counter()
    r = client.post("/score/upload", files={"file": ("giao-dich-demo.csv", buffer, "text/csv")})
    r.raise_for_status()
    body = r.json()
    log(f"1. Tải lên {body['count']:,} dòng ({int(upload['Class'].sum())} gian lận) trong "
        f"{time.perf_counter() - started:.1f} giây — {body['alerts']} vào hàng đợi, batch {body['batch_id']}")

    loaded = flagged = 0
    for item in pool["items"]:
        if item["category"] not in SEED_CATEGORIES:
            continue
        r = client.post("/score", json={"transaction": item["features"], "sample_id": item["id"], "persist": True})
        r.raise_for_status()
        loaded += 1
        flagged += r.json()["decision"] != "allow"
    log(f"2. Nạp {loaded} mẫu ({', '.join(SEED_CATEGORIES)}) — {flagged} vào hàng đợi; fraud_hard để dành trình diễn")

    queue = client.get("/transactions", params={"page_size": 50}).json()["items"]
    notes = {label: list(texts) for label, texts in REVIEW_NOTES.items()}
    reviewed = []
    for tx in queue:
        texts = notes.get(tx["true_label"])
        if not texts:
            continue
        decision = "confirmed_fraud" if tx["true_label"] == 1 else "false_alarm"
        r = client.post("/reviews", json={"transaction_id": tx["id"], "decision": decision, "note": texts.pop(0)})
        r.raise_for_status()
        reviewed.append(f"{tx['id']} ({decision}, điểm {tx['risk_score']:.4f})")
    log(f"3. Thẩm định {len(reviewed)} giao dịch điểm cao nhất theo nhãn thật:")
    for line in reviewed:
        log(f"   {line}")

    total = client.get("/transactions", params={"page_size": 1}).json()
    log(f"Xong: hàng đợi {total['total']} giao dịch ở ngưỡng {total['threshold']:.6f}. "
        "Sao lưu: python scripts/demo_db.py dump")


# --------------------------------------------------------------------------
# dump / restore / reset
# --------------------------------------------------------------------------

def dump(path: Path, dry_run: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    run(["docker", "compose", "exec", "-T", "db", "pg_dump", "-U", "fraud", "-d", "fraud", "-Fc", "-f", CONTAINER_DUMP],
        dry_run=dry_run)
    run(["docker", "compose", "cp", f"db:{CONTAINER_DUMP}", str(path)], dry_run=dry_run)
    run(["docker", "compose", "exec", "-T", "db", "rm", "-f", CONTAINER_DUMP], dry_run=dry_run)
    if not dry_run:
        c = counts()
        log(f"Đã sao lưu {path} ({path.stat().st_size / 1024:,.0f} KB): {c['transactions']:,} giao dịch, "
            f"{c['reviews']} thẩm định, {c['settings']} khóa settings")


def restore(path: Path, dry_run: bool) -> None:
    if not dry_run and not path.exists():
        sys.exit(f"Không có {path}. Tạo bằng: python scripts/demo_db.py seed && python scripts/demo_db.py dump")
    started = time.perf_counter()
    run(["docker", "compose", "cp", str(path), f"db:{CONTAINER_DUMP}"], dry_run=dry_run)
    # --clean --if-exists: xóa bảng hiện có rồi dựng lại; --single-transaction: lỗi giữa chừng thì
    # cơ sở dữ liệu giữ nguyên như trước, không nửa nọ nửa kia
    run(["docker", "compose", "exec", "-T", "db", "pg_restore", "-U", "fraud", "-d", "fraud",
         "--clean", "--if-exists", "--single-transaction", "--no-owner", CONTAINER_DUMP], dry_run=dry_run)
    run(["docker", "compose", "exec", "-T", "db", "rm", "-f", CONTAINER_DUMP], dry_run=dry_run)
    if not dry_run:
        elapsed = time.perf_counter() - started
        c = counts()
        log(f"Đã khôi phục trong {elapsed:.1f} giây: {c['transactions']:,} giao dịch, {c['reviews']} thẩm định. "
            "Tải lại trang giao diện là thấy.")


def reset(dry_run: bool) -> None:
    run(PSQL + ["-c", "TRUNCATE transactions, reviews RESTART IDENTITY CASCADE; "
                      "DELETE FROM settings WHERE key = 'threshold';"], dry_run=dry_run)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("seed", help="nạp dữ liệu demo qua API")
    p.add_argument("--api", default=DEFAULT_API)
    p.add_argument("--rows", type=int, default=10_000, help="số dòng tải lên (mặc định 10.000)")
    p.add_argument("--force", action="store_true", help="nạp cả khi cơ sở dữ liệu đã có giao dịch")
    for name in ("dump", "restore"):
        p = sub.add_parser(name)
        p.add_argument("--file", type=Path, default=DEFAULT_DUMP)
        p.add_argument("--dry-run", action="store_true", help="chỉ in lệnh docker")
    p = sub.add_parser("reset")
    p.add_argument("--dry-run", action="store_true")
    sub.add_parser("status")
    args = parser.parse_args()

    if args.command == "seed":
        seed(args.api, args.rows, args.force)
    elif args.command == "dump":
        dump(args.file, args.dry_run)
    elif args.command == "restore":
        restore(args.file, args.dry_run)
    elif args.command == "reset":
        reset(args.dry_run)
    else:
        log(str(counts()))


if __name__ == "__main__":
    main()
