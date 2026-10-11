"""Chạy lưới theo mức cân bằng (docs/11 M5) ngoài notebook.

    python scripts/run_ratio_grid.py               # 4 mô hình × 3 chiến lược × 4 mức = 48 tổ hợp (~1 giờ)
    python scripts/run_ratio_grid.py --smoke       # chạy thử 12.000 dòng (~1 phút)
    python scripts/run_ratio_grid.py --fresh       # bỏ điểm lưu, chạy lại từ đầu
    python scripts/run_ratio_grid.py --status      # xem đã chạy được bao nhiêu tổ hợp

Lưới chuẩn (scripts/run_grid.py) so ``class_weight`` cân bằng hoàn toàn (1:1) với SMOTE và
undersample đưa về 1:10 — hai chiến lược khác nhau ở cả *loại* lẫn *mức độ* cân bằng. Lưới này cho
mọi chiến lược chạy ở CÙNG các tỷ lệ dương:âm mục tiêu r ∈ {0,02; 0,1; 0,3; 1}.

Kết quả ghi vào ``reports/ratio_grid.csv`` và ``reports/ratio_grid.npz`` sau **mỗi** tổ hợp. Chạy
lại thì tổ hợp nào có điểm lưu khớp ``config_hash`` được dùng lại. Notebook 04 §5.7 đọc hai tệp này.
Tham số ở đây PHẢI trùng với ô đọc kết quả ở notebook 04.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# Console Windows mặc định là cp1252 — xem ghi chú trong run_grid.py
for luong in (sys.stdout, sys.stderr):
    if hasattr(luong, "reconfigure"):
        luong.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)

from sklearn.model_selection import train_test_split  # noqa: E402

from src.config import DEFAULT_COST_FN, DEFAULT_COST_FP, RANDOM_STATE, REPORTS_DIR  # noqa: E402
from src.data import load_prepared, split_data  # noqa: E402
from src.features import build_features  # noqa: E402
from src.modeling import (  # noqa: E402
    DEFAULT_MODELS,
    RATIO_GRID_RATIOS,
    RATIO_GRID_STRATEGIES,
    make_cv,
    run_ratio_grid,
)

DUONG_DAN = REPORTS_DIR / "ratio_grid.csv"


def in_trang_thai() -> int:
    import pandas as pd

    tong = len(DEFAULT_MODELS) * len(RATIO_GRID_STRATEGIES) * len(RATIO_GRID_RATIOS)
    if not DUONG_DAN.exists():
        print(f"Chưa có {DUONG_DAN.name}. Đã chạy 0/{tong} tổ hợp.")
        return 0
    bang = pd.read_csv(DUONG_DAN)
    print(f"{DUONG_DAN}: {len(bang)}/{tong} tổ hợp, {bang['total_seconds'].sum() / 60:.1f} phút huấn luyện")
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--smoke", action="store_true", help="chạy thử trên 12.000 dòng (~1 phút)")
    parser.add_argument("--fresh", action="store_true", help="bỏ qua điểm lưu, chạy lại từ đầu")
    parser.add_argument("--status", action="store_true", help="chỉ in tiến độ của điểm lưu rồi thoát")
    args = parser.parse_args(argv)
    if args.status:
        return in_trang_thai()

    print("Nạp dữ liệu…")
    df, _ = load_prepared(verbose=False)
    X_train_tho, _, y_train, _ = split_data(df, as_features=False)
    duong_dan = DUONG_DAN
    if args.smoke:
        X_train_tho, _, y_train, _ = train_test_split(X_train_tho, y_train, train_size=12_000,
                                                      stratify=y_train, random_state=RANDOM_STATE)
        duong_dan = REPORTS_DIR / "ratio_grid_smoke.csv"
        print("CHẾ ĐỘ CHẠY THỬ: 12.000 dòng — KHÔNG dùng cho báo cáo")

    X_train = build_features(X_train_tho)
    print(f"Tập huấn luyện: {len(y_train):,} dòng, {int(y_train.sum())} gian lận")
    print(f"Mức cân bằng r: {', '.join(f'{r:g}' for r in RATIO_GRID_RATIOS)}; "
          f"chiến lược: {', '.join(RATIO_GRID_STRATEGIES)}")
    print(f"Điểm lưu: {duong_dan}")
    print("-" * 78)

    bat_dau = time.perf_counter()
    bang = run_ratio_grid(
        X_train, y_train,
        models=DEFAULT_MODELS,
        strategies=RATIO_GRID_STRATEGIES,
        ratios=RATIO_GRID_RATIOS,
        cv=make_cv(random_state=RANDOM_STATE),
        criterion="min_expected_cost",
        cost_fn=DEFAULT_COST_FN,
        cost_fp=DEFAULT_COST_FP,
        sample_fraction=len(y_train) / len(df),
        random_state=RANDOM_STATE,
        checkpoint_path=duong_dan,
        resume=not args.fresh,
        verbose=True,
    )
    print("-" * 78)
    print(f"Xong {len(bang)} tổ hợp trong {(time.perf_counter() - bat_dau) / 60:.1f} phút.")
    print(bang[["model", "strategy", "ratio", "pr_auc_mean", "pr_auc_std", "brier", "alerts_per_day"]]
          .to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
