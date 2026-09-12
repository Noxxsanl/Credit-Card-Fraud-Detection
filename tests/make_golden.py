"""Sinh lại dữ liệu vàng cho TC-01. Chạy tay, không phải một phần của pytest.

    python tests/make_golden.py

Chỉ chạy lại khi CỐ Ý đổi cách sinh đặc trưng — và khi đó phải kiểm tra bằng mắt
tệp sinh ra trước khi commit. Nếu chạy lại một cách máy móc mỗi lần kiểm thử đỏ
thì kiểm thử vàng mất hết tác dụng.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.features import V_COLUMNS, build_features  # noqa: E402

OUT_DIR = Path(__file__).parent / "data"

# Các mốc Time chọn có chủ đích: đầu giờ, cuối giờ, ranh giới ngày, giá trị lớn
# nhất trong bộ dữ liệu gốc.
TIMES = [
    0.0, 1.0, 3_599.0, 3_600.0, 7_200.0, 43_200.0,
    86_399.0, 86_400.0, 86_401.0, 90_000.0,
    100_000.0, 120_000.0, 140_000.0, 160_000.0, 172_792.0,
    45_000.0, 61_200.0, 75_600.0, 12_345.0, 54_321.0,
]

# Amount chọn để phủ cả biên: 0 (giao dịch thử thẻ) và giá trị rất lớn.
AMOUNTS = [
    0.0, 1.0, 2.69, 9.99, 22.00, 49.50, 99.99, 149.62, 250.00, 378.66,
    529.00, 812.35, 1_809.68, 2_500.00, 4_200.75, 7_777.77, 12_910.93,
    19_656.53, 25_691.16, 0.76,
]


def main() -> None:
    rng = np.random.default_rng(20260912)
    n_rows = len(TIMES)

    raw = pd.DataFrame({"Time": TIMES})
    for column in V_COLUMNS:
        raw[column] = np.round(rng.normal(size=n_rows), 6)
    raw["Amount"] = AMOUNTS

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    raw.to_csv(OUT_DIR / "golden_input.csv", index=False)
    build_features(raw).to_csv(OUT_DIR / "golden_features.csv", index=False)

    print(f"Đã ghi {n_rows} dòng vào {OUT_DIR}")


if __name__ == "__main__":
    main()
