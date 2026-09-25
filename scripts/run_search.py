"""Chạy RandomizedSearchCV của T-24 ngoài notebook.

    python scripts/run_search.py               # 30 lần thử × 5 fold (khoảng 10-25 phút)
    python scripts/run_search.py --smoke       # chạy thử 12.000 dòng, 4 lần thử (~1 phút)
    python scripts/run_search.py --status      # xem kết quả đã lưu

Kết quả ghi vào ``reports/search_results.csv`` (đủ 30 dòng, sắp theo PR-AUC) và
``reports/best_params.json``. Chạy xong rồi mở
``notebooks/05_advanced_models.ipynb`` và Run All: notebook thấy tệp kết quả nên
bỏ qua bước tìm kiếm.

Tham số truyền cho ``run_search`` ở đây PHẢI trùng với ô ở mục 1 của notebook.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# Console Windows mặc định là cp1252 — xem ghi chú trong run_grid.py
for luong in (sys.stdout, sys.stderr):
    if hasattr(luong, "reconfigure"):
        luong.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)

from sklearn.model_selection import train_test_split  # noqa: E402

from src.config import RANDOM_STATE, REPORTS_DIR  # noqa: E402
from src.data import load_prepared, split_data  # noqa: E402
from src.features import build_features  # noqa: E402
from src.modeling import (  # noqa: E402
    SEARCH_N_ITER,
    best_params_from,
    imbalance_ratio,
    make_cv,
    run_search,
    search_table,
)

DUONG_DAN_KET_QUA = REPORTS_DIR / "search_results.csv"
DUONG_DAN_THAM_SO = REPORTS_DIR / "best_params.json"


def in_trang_thai() -> int:
    import pandas as pd

    if not DUONG_DAN_KET_QUA.exists():
        print(f"Chưa có {DUONG_DAN_KET_QUA.name}. Chạy: python scripts/run_search.py")
        return 0
    bang = pd.read_csv(DUONG_DAN_KET_QUA)
    print(f"{DUONG_DAN_KET_QUA}: {len(bang)} lần thử")
    print(bang.head(5).to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--smoke", action="store_true",
                        help="chạy thử trên 12.000 dòng, 4 lần thử (~1 phút)")
    parser.add_argument("--status", action="store_true",
                        help="chỉ in kết quả đã lưu rồi thoát")
    args = parser.parse_args(argv)

    if args.status:
        return in_trang_thai()

    print("Nạp dữ liệu…")
    df, _ = load_prepared(verbose=False)
    X_train_tho, _, y_train, _ = split_data(df, as_features=False)

    n_iter = SEARCH_N_ITER
    duong_dan, duong_dan_tham_so = DUONG_DAN_KET_QUA, DUONG_DAN_THAM_SO
    if args.smoke:
        X_train_tho, _, y_train, _ = train_test_split(
            X_train_tho, y_train, train_size=12_000, stratify=y_train,
            random_state=RANDOM_STATE)
        n_iter = 4
        duong_dan = REPORTS_DIR / "search_results_smoke.csv"
        duong_dan_tham_so = REPORTS_DIR / "best_params_smoke.json"
        print("CHẾ ĐỘ CHẠY THỬ: 12.000 dòng, 4 lần thử — KHÔNG dùng cho báo cáo")

    X_train = build_features(X_train_tho)
    print(f"Tập huấn luyện: {len(y_train):,} dòng, {int(y_train.sum())} gian lận; "
          f"tỷ lệ âm/dương {imbalance_ratio(y_train):.1f}")
    print(f"RandomizedSearchCV: {n_iter} lần thử × 5 fold = {n_iter * 5} lần huấn luyện")
    print("-" * 78)

    bat_dau = time.perf_counter()
    search = run_search(
        X_train, y_train,
        n_iter=n_iter,
        cv=make_cv(random_state=RANDOM_STATE),
        refit=False,          # notebook tự huấn luyện lại cấu hình thắng (T-25)
        random_state=RANDOM_STATE,
        verbose=1,
    )
    troi = time.perf_counter() - bat_dau

    bang = search_table(search.cv_results_)
    bang.to_csv(duong_dan, index=False, encoding="utf-8")

    tham_so = best_params_from(bang)
    duong_dan_tham_so.write_text(json.dumps({
        "model": "xgboost",
        "strategy": "class_weight",
        "params": tham_so,
        "cv_pr_auc_mean": float(bang.loc[0, "pr_auc_mean"]),
        "cv_pr_auc_std": float(bang.loc[0, "pr_auc_std"]),
        "n_iter": n_iter,
        "search_minutes": round(troi / 60, 2),
        "random_state": RANDOM_STATE,
    }, indent=2, ensure_ascii=False), encoding="utf-8")

    print("-" * 78)
    print(f"Xong {len(bang)} lần thử trong {troi / 60:.1f} phút.")
    print(bang.head(5).to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print()
    print(f"Đã ghi {duong_dan.name} và {duong_dan_tham_so.name}.")
    if bang["pr_auc_mean"].max() > 0.95:
        print("!! BÁO ĐỘNG: PR-AUC > 0,95 — rà danh sách rò rỉ ở docs/08 §3.1.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
