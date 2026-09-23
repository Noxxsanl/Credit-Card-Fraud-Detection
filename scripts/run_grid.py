"""Chạy lưới 20 tổ hợp của T-20 ngoài notebook.

    python scripts/run_grid.py                 # chạy đủ 20 tổ hợp (khoảng 20-40 phút)
    python scripts/run_grid.py --smoke         # chạy thử 12.000 dòng (khoảng 1 phút)
    python scripts/run_grid.py --fast          # hạ n_estimators xuống 100 (phương án T-20)
    python scripts/run_grid.py --fresh         # bỏ điểm lưu, chạy lại từ đầu
    python scripts/run_grid.py --status        # xem đã chạy được bao nhiêu tổ hợp

Vì sao tách ra khỏi notebook: lưới chạy 20-40 phút, và chạy trong notebook thì
kernel chết là mất trắng, lại khó theo dõi tiến độ. Script này ghi điểm lưu vào
``reports/grid_results.csv`` sau **mỗi** tổ hợp. Chạy xong rồi mở
``notebooks/04_imbalance_strategies.ipynb`` và Run All: ``run_grid`` thấy điểm
lưu nên bỏ qua toàn bộ phần huấn luyện, cả notebook xong trong khoảng một phút.

Tham số truyền cho ``run_grid`` ở đây PHẢI trùng với ô ở mục 2 của notebook.
Lệch một tham số thì các dòng nối tiếp nhau sẽ không so sánh được với nhau.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# Console Windows mặc định là cp1252: in tiếng Việt sẽ ném UnicodeEncodeError và
# giết cả lần chạy 30 phút ở giữa chừng. line_buffering để tiến độ hiện ra ngay cả
# khi đầu ra bị chuyển hướng vào tệp (mặc định Python đệm theo khối, người chạy sẽ
# không thấy gì suốt nhiều phút).
for luong in (sys.stdout, sys.stderr):
    if hasattr(luong, "reconfigure"):
        luong.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)

from sklearn.model_selection import train_test_split  # noqa: E402

from src.config import (  # noqa: E402
    DEFAULT_COST_FN,
    DEFAULT_COST_FP,
    RANDOM_STATE,
    REPORTS_DIR,
)
from src.data import load_prepared, split_data  # noqa: E402
from src.features import build_features  # noqa: E402
from src.modeling import (  # noqa: E402
    DEFAULT_MODELS,
    DEFAULT_STRATEGIES,
    make_cv,
    run_grid,
)

DUONG_DAN_LUOI = REPORTS_DIR / "grid_results.csv"

#: Phương án dự phòng đã ghi sẵn trong T-20, bật bằng --fast
THAM_SO_RUT_GON = {"random_forest": {"n_estimators": 100},
                   "xgboost": {"n_estimators": 100}}


def in_trang_thai() -> int:
    """Xem điểm lưu hiện có mà không chạy gì."""
    import pandas as pd

    tong = len(DEFAULT_MODELS) * len(DEFAULT_STRATEGIES)
    if not DUONG_DAN_LUOI.exists():
        print(f"Chưa có {DUONG_DAN_LUOI.name}. Đã chạy 0/{tong} tổ hợp.")
        return 0

    bang = pd.read_csv(DUONG_DAN_LUOI)
    print(f"{DUONG_DAN_LUOI}: {len(bang)}/{tong} tổ hợp")
    print(f"  thời gian đã bỏ ra: {bang['total_seconds'].sum() / 60:.1f} phút")
    if len(bang) < tong:
        con_lai = {(m, s) for m in DEFAULT_MODELS for s in DEFAULT_STRATEGIES}
        con_lai -= set(map(tuple, bang[["model", "strategy"]].to_numpy()))
        print(f"  còn thiếu {len(con_lai)}: "
              + ", ".join(f"{m}+{s}" for m, s in sorted(con_lai)))
        uoc = bang["total_seconds"].mean() * len(con_lai) / 60
        print(f"  ước tính còn khoảng {uoc:.0f} phút")
    else:
        tot = bang.sort_values("pr_auc_mean", ascending=False).iloc[0]
        print(f"  ĐỦ 20 DÒNG. Tốt nhất: {tot['model']} + {tot['strategy']} "
              f"→ PR-AUC {tot['pr_auc_mean']:.4f}")
        print("  Bước tiếp theo: mở notebooks/04_imbalance_strategies.ipynb và Run All.")
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--smoke", action="store_true",
                        help="chạy thử trên 12.000 dòng để kiểm mọi thứ hoạt động (~1 phút)")
    parser.add_argument("--fast", action="store_true",
                        help="hạ n_estimators xuống 100 cho mô hình ensemble (phương án T-20)")
    parser.add_argument("--fresh", action="store_true",
                        help="bỏ qua điểm lưu, chạy lại toàn bộ từ đầu")
    parser.add_argument("--status", action="store_true",
                        help="chỉ in tiến độ của điểm lưu rồi thoát")
    args = parser.parse_args(argv)

    if args.status:
        return in_trang_thai()

    print("Nạp dữ liệu…")
    df, _ = load_prepared(verbose=False)
    X_train_tho, _, y_train, y_test = split_data(df, as_features=False)

    duong_dan = DUONG_DAN_LUOI
    if args.smoke:
        X_train_tho, _, y_train, _ = train_test_split(
            X_train_tho, y_train, train_size=12_000, stratify=y_train,
            random_state=RANDOM_STATE)
        duong_dan = REPORTS_DIR / "grid_results_smoke.csv"
        print("CHẾ ĐỘ CHẠY THỬ: 12.000 dòng, ghi ra grid_results_smoke.csv")
        print("   (kết quả KHÔNG dùng cho báo cáo — chỉ để xác nhận mọi thứ chạy được)")

    X_train = build_features(X_train_tho)
    phan_du_lieu = len(y_train) / len(df)

    print(f"Tập huấn luyện: {len(y_train):,} dòng, {int(y_train.sum())} gian lận "
          f"({y_train.mean():.4%})")
    print(f"Chế độ: {'RÚT GỌN (n_estimators=100)' if args.fast else 'đầy đủ theo 04 §3.2'}")
    print(f"Điểm lưu: {duong_dan}")
    print(f"Nối tiếp điểm lưu: {'không (--fresh)' if args.fresh else 'có'}")
    print("-" * 78)

    bat_dau = time.perf_counter()
    bang = run_grid(
        X_train,
        y_train,
        models=DEFAULT_MODELS,
        strategies=DEFAULT_STRATEGIES,
        model_params=THAM_SO_RUT_GON if args.fast else None,
        cv=make_cv(random_state=RANDOM_STATE),
        criterion="min_expected_cost",
        cost_fn=DEFAULT_COST_FN,
        cost_fp=DEFAULT_COST_FP,
        sample_fraction=phan_du_lieu,
        random_state=RANDOM_STATE,
        checkpoint_path=duong_dan,
        resume=not args.fresh,
        verbose=True,
    )
    troi = time.perf_counter() - bat_dau

    print("-" * 78)
    print(f"Xong {len(bang)} tổ hợp trong {troi / 60:.1f} phút.")
    print()
    print(bang[["model", "strategy", "pr_auc_mean", "pr_auc_std",
                "recall", "precision", "n_distinct_scores", "total_seconds"]]
          .to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    cao_nhat = bang["pr_auc_mean"].max()
    print()
    if cao_nhat > 0.95:
        print(f"!! BÁO ĐỘNG (T-23): PR-AUC cao nhất {cao_nhat:.4f} > 0,95 — dừng lại và rà")
        print("   danh sách rò rỉ ở docs/08 §3.1 trước khi đi tiếp.")
    else:
        print(f"T-23: PR-AUC cao nhất {cao_nhat:.4f}, dưới ngưỡng báo động 0,95. "
              "Dải kỳ vọng 04 §3.4 là 0,80–0,87.")

    if not args.smoke:
        print()
        print(f"Đã ghi {duong_dan.name} ({len(bang)} dòng) và "
              f"{duong_dan.with_suffix('.npz').name} (điểm out-of-fold).")
        print("Bước tiếp theo: mở notebooks/04_imbalance_strategies.ipynb và Run All —")
        print("notebook sẽ thấy điểm lưu và bỏ qua phần huấn luyện.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
