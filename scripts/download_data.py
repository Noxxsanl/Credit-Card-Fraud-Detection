"""Tải bộ dữ liệu Credit Card Fraud Detection từ Kaggle về ``data/creditcard.csv``.

    python scripts/download_data.py                # tải rồi kiểm tra toàn vẹn
    python scripts/download_data.py --check        # chỉ kiểm tra tệp đang có
    python scripts/download_data.py --force        # tải lại dù đã có tệp
    python scripts/download_data.py --keep-zip     # giữ lại tệp nén
    python scripts/download_data.py --mirror       # tải bản sao công khai, khỏi cần token
    python scripts/download_data.py --url <URL>    # tải từ nguồn khác

Kaggle bắt buộc xác thực để tải dataset, kể cả với dữ liệu công khai. Script gọi
thẳng REST API bằng HTTP Basic Auth nên không cần cài thêm gói ``kaggle``.

Lấy thông tin xác thực
----------------------
1. Đăng nhập kaggle.com → ảnh đại diện → Settings → mục API → *Create New Token*.
   Trình duyệt tải về ``kaggle.json``.
2. Đặt tệp đó vào ``C:\\Users\\<tên>\\.kaggle\\kaggle.json`` (Windows) hoặc
   ``~/.kaggle/kaggle.json`` (macOS/Linux).
   Hoặc đặt biến môi trường ``KAGGLE_USERNAME`` và ``KAGGLE_KEY``.
3. Mở trang dataset một lần và bấm đồng ý điều khoản, nếu chưa từng bấm.

Giấy phép: Database Contents License (DbCL) v1.0 — phải ghi nguồn trong báo cáo.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import zipfile
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import DATA_DIR, EXPECTED_FRAUD, EXPECTED_ROWS, RAW_DATA_PATH  # noqa: E402

DATASET_SLUG = "mlg-ulb/creditcardfraud"
TARGET_FILE = "creditcard.csv"
KAGGLE_API = f"https://www.kaggle.com/api/v1/datasets/download/{DATASET_SLUG}"

#: Bản sao công khai trên Hugging Face, dùng khi không có tài khoản Kaggle. Đây
#: là nguồn thứ cấp do bên thứ ba đăng lại, nên bước kiểm tra toàn vẹn ở cuối
#: script là bắt buộc — không được bỏ qua.
MIRROR_URL = (
    "https://huggingface.co/datasets/David-Egea/Creditcard-fraud-detection/"
    "resolve/main/creditcard.csv?download=true"
)

#: SHA-256 của tệp tải ngày 2026-09-12 từ MIRROR_URL, đã đối chiếu đủ 284.807
#: dòng và 492 mẫu gian lận. Khác giá trị này thì cảnh báo chứ không chặn: nguồn
#: có thể đăng lại tệp mà nội dung vẫn đúng.
KNOWN_SHA256 = "76274b691b16a6c49d3f159c883398e03ccd6d1ee12d9d8ee38f4b4b98551a89"

CHUNK = 1 << 20  # 1 MB
TIMEOUT = 60


class DownloadError(RuntimeError):
    """Không tải được dữ liệu — kèm hướng dẫn xử lý cụ thể."""


# --------------------------------------------------------------------------
# Xác thực
# --------------------------------------------------------------------------

def resolve_credentials() -> tuple[str, str]:
    """Tìm thông tin xác thực Kaggle theo thứ tự: biến môi trường → kaggle.json."""
    username = os.environ.get("KAGGLE_USERNAME")
    key = os.environ.get("KAGGLE_KEY")
    if username and key:
        return username, key

    candidates = [
        Path(os.environ.get("KAGGLE_CONFIG_DIR", "")) / "kaggle.json",
        Path.home() / ".kaggle" / "kaggle.json",
    ]
    for path in candidates:
        if path.is_file():
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
                return payload["username"], payload["key"]
            except (json.JSONDecodeError, KeyError) as exc:
                raise DownloadError(
                    f"{path} không đọc được hoặc thiếu trường username/key: {exc}"
                ) from exc

    raise DownloadError(
        "Không tìm thấy thông tin xác thực Kaggle.\n"
        "  Cách 1 — tệp token:\n"
        "    kaggle.com → Settings → API → Create New Token, rồi đặt kaggle.json vào\n"
        f"    {Path.home() / '.kaggle' / 'kaggle.json'}\n"
        "  Cách 2 — biến môi trường:\n"
        "    $env:KAGGLE_USERNAME = 'ten_dang_nhap'   # PowerShell\n"
        "    $env:KAGGLE_KEY      = 'khoa_api'\n"
        "  Cách 3 — tải tay:\n"
        f"    https://www.kaggle.com/datasets/{DATASET_SLUG}\n"
        f"    rồi đặt {TARGET_FILE} vào {DATA_DIR}"
    )


# --------------------------------------------------------------------------
# Tải
# --------------------------------------------------------------------------

def _stream_to_file(response: requests.Response, destination: Path) -> Path:
    """Ghi phản hồi xuống đĩa theo từng mẻ, in tiến độ."""
    total = int(response.headers.get("Content-Length", 0))
    downloaded = 0
    destination.parent.mkdir(parents=True, exist_ok=True)

    with destination.open("wb") as handle:
        for chunk in response.iter_content(chunk_size=CHUNK):
            if not chunk:
                continue
            handle.write(chunk)
            downloaded += len(chunk)
            if total:
                print(
                    f"\r  {downloaded / 1e6:7.1f} / {total / 1e6:.1f} MB "
                    f"({downloaded / total:5.1%})",
                    end="",
                    flush=True,
                )
            else:
                print(f"\r  {downloaded / 1e6:7.1f} MB", end="", flush=True)
    print()
    return destination


def download_from_kaggle(destination: Path) -> Path:
    """Tải tệp nén dataset qua REST API của Kaggle."""
    username, key = resolve_credentials()
    print(f"Xác thực với tài khoản Kaggle: {username}")
    print(f"Tải {DATASET_SLUG} …")

    response = requests.get(
        KAGGLE_API, auth=(username, key), stream=True, timeout=TIMEOUT
    )

    if response.status_code in (401, 403):
        raise DownloadError(
            f"Kaggle từ chối yêu cầu (HTTP {response.status_code}).\n"
            "  - Khoá API sai hoặc đã bị thu hồi → tạo token mới.\n"
            "  - Hoặc chưa đồng ý điều khoản dataset: mở\n"
            f"    https://www.kaggle.com/datasets/{DATASET_SLUG}\n"
            "    đăng nhập và bấm nút Download một lần để chấp nhận điều khoản."
        )
    if response.status_code == 404:
        raise DownloadError(f"Không tìm thấy dataset {DATASET_SLUG} (HTTP 404).")
    response.raise_for_status()

    return _stream_to_file(response, destination)


def download_from_url(url: str, destination: Path) -> Path:
    """Tải từ một địa chỉ trực tiếp bất kỳ (nguồn dự phòng do người dùng cung cấp).

    Dùng khi không có tài khoản Kaggle. Tính đúng đắn của nguồn này không được
    bảo đảm — vì vậy bước kiểm tra toàn vẹn ở cuối là bắt buộc, không phải tuỳ chọn.
    """
    print(f"Tải từ nguồn ngoài: {url}")
    response = requests.get(url, stream=True, timeout=TIMEOUT)
    response.raise_for_status()
    return _stream_to_file(response, destination)


def extract_csv(archive: Path, destination: Path) -> Path:
    """Rút ``creditcard.csv`` ra khỏi tệp nén. Nếu tệp tải về vốn đã là CSV thì giữ nguyên."""
    if not zipfile.is_zipfile(archive):
        if archive != destination:
            archive.replace(destination)
        return destination

    with zipfile.ZipFile(archive) as zf:
        names = zf.namelist()
        if TARGET_FILE not in names:
            raise DownloadError(
                f"Tệp nén không chứa {TARGET_FILE}. Nội dung: {', '.join(names)}"
            )
        print(f"Giải nén {TARGET_FILE} …")
        destination.parent.mkdir(parents=True, exist_ok=True)
        with zf.open(TARGET_FILE) as source, destination.open("wb") as target:
            while chunk := source.read(CHUNK):
                target.write(chunk)
    return destination


# --------------------------------------------------------------------------
# Kiểm tra
# --------------------------------------------------------------------------

def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(CHUNK):
            digest.update(chunk)
    return digest.hexdigest()


def verify(path: Path) -> bool:
    """Đối chiếu tệp với kỳ vọng trong đặc tả (02 §6).

    Sai lệch về số dòng hay số mẫu gian lận nghĩa là tệp không phải bản chuẩn —
    mọi kết quả về sau sẽ không so sánh được với báo cáo, nên phải dừng ngay ở đây.
    """
    import pandas as pd

    from src.data import check_integrity

    print(f"\nKiểm tra {path} …")
    size_mb = path.stat().st_size / 1e6
    digest = sha256(path)
    print(f"  Kích thước : {size_mb:.1f} MB")
    print(f"  SHA-256    : {digest}")
    print(
        "               "
        + ("khớp bản đã đối chiếu" if digest == KNOWN_SHA256 else "KHÁC bản đã đối chiếu")
    )

    report = check_integrity(pd.read_csv(path))
    print("  " + report.summary().replace("\n", "\n  "))

    if report.ok:
        print(
            f"\nĐạt. Đúng {EXPECTED_ROWS:,} giao dịch và {EXPECTED_FRAUD} mẫu gian lận."
        )
        print("Bước tiếp theo: chạy notebooks/01_eda.ipynb")
    else:
        print("\nTệp KHÔNG khớp đặc tả — không dùng để huấn luyện.")
    return report.ok


# --------------------------------------------------------------------------

def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Tải bộ dữ liệu Credit Card Fraud Detection (mlg-ulb) từ Kaggle.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--force", action="store_true", help="tải lại dù tệp đã tồn tại")
    parser.add_argument("--check", action="store_true", help="chỉ kiểm tra tệp hiện có")
    parser.add_argument("--keep-zip", action="store_true", help="giữ lại tệp nén sau khi giải")
    parser.add_argument("--url", help="tải từ địa chỉ trực tiếp thay vì Kaggle API")
    parser.add_argument(
        "--mirror",
        action="store_true",
        help="dùng bản sao công khai trên Hugging Face (không cần token Kaggle)",
    )
    parser.add_argument(
        "--output", type=Path, default=RAW_DATA_PATH, help="đường dẫn tệp CSV đích"
    )
    args = parser.parse_args(argv)

    destination: Path = args.output

    if args.check:
        if not destination.exists():
            print(f"Chưa có {destination}. Chạy lại không kèm --check để tải.")
            return 1
        return 0 if verify(destination) else 1

    if destination.exists() and not args.force:
        print(f"{destination} đã tồn tại ({destination.stat().st_size / 1e6:.1f} MB).")
        print("Dùng --force để tải lại.")
        return 0 if verify(destination) else 1

    archive = destination.with_suffix(".zip")
    url = args.url or (MIRROR_URL if args.mirror else None)
    try:
        if url:
            download_from_url(url, archive)
        else:
            download_from_kaggle(archive)
        extract_csv(archive, destination)
    except DownloadError as exc:
        print(f"\n{exc}", file=sys.stderr)
        return 2
    except requests.RequestException as exc:
        print(f"\nLỗi mạng khi tải: {exc}", file=sys.stderr)
        return 2
    finally:
        if archive.exists() and not args.keep_zip and archive != destination:
            archive.unlink()

    return 0 if verify(destination) else 1


if __name__ == "__main__":
    raise SystemExit(main())
