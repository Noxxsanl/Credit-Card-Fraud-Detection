"""CV lặp cho các cặp cấu hình cần kết luận (docs/11 m2, M4) — chạy SAU run_grid và run_ratio_grid.

    python scripts/run_repeated_cv.py              # khoảng 1 giờ
    python scripts/run_repeated_cv.py --smoke      # chạy thử 12.000 dòng, 2 lần lặp (~2 phút)
    python scripts/run_repeated_cv.py --status     # xem kết quả đã lưu

Vì sao: bảng 20 tổ hợp dựa trên MỘT lần CV 5 fold, và bootstrap trên điểm out-of-fold gộp chỉ lấy
mẫu lại các dòng — bỏ qua phương sai do huấn luyện. Ở đây mỗi cấu hình chạy
``RepeatedStratifiedKFold`` cùng ``random_state``, nên mọi cấu hình gặp đúng cùng các fold và so
được theo cặp bằng kiểm định t hiệu chỉnh Nadeau–Bengio (``src/evaluate.py``).

Ứng viên:

* **Họ XGBoost** (để chọn mô hình cuối, notebook 04 §5.8): mọi cấu hình XGBoost của lưới chuẩn và lưới
  theo mức có PR-AUC CV trong vòng ``MARGIN`` của cấu hình XGBoost cao nhất. Bỏ ``smote_tomek`` vì
  điểm của nó trùng ``smote`` từng bit. Lặp 5 × 5 fold.
* **Hai khẳng định của báo cáo về Random Forest**: RF + SMOTE (1:10) so với RF + class_weight, và
  XGBoost + class_weight so với RF + class_weight. RF chậm nên lặp 3 × 5 fold; cặp với XGBoost so
  trên 3 lần lặp chung (``n_repeats`` lần đầu luôn trùng nhau).

Kết quả: ``reports/repeated_cv.csv`` (mỗi dòng một fold của một cấu hình, kèm ``config_hash`` để
chạy lại thì dùng lại) và ``reports/repeated_cv_tests.csv`` (các cặp so sánh).
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

for luong in (sys.stdout, sys.stderr):
    if hasattr(luong, "reconfigure"):
        luong.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.model_selection import train_test_split  # noqa: E402

from src.config import RANDOM_STATE, REPORTS_DIR  # noqa: E402
from src.data import load_prepared, split_data  # noqa: E402
from src.evaluate import compare_repeated_cv  # noqa: E402
from src.features import build_features  # noqa: E402
from src.modeling import (  # noqa: E402
    build_pipeline,
    config_fingerprint,
    data_fingerprint,
    imbalance_ratio,
    repeated_cv_scores,
    score_key,
)

DUONG_DAN = REPORTS_DIR / "repeated_cv.csv"
DUONG_DAN_SO_SANH = REPORTS_DIR / "repeated_cv_tests.csv"

#: Lọc thô ứng viên XGBoost bằng CV một lần: kém cấu hình cao nhất quá một phần ba độ lệch chuẩn
#: giữa các fold (≈ 0,03) thì không đáng tốn CV lặp
MARGIN = 0.01
XGB_REPEATS = 5
RF_REPEATS = 3


def _ratio(value) -> float | None:
    return None if value is None or pd.isna(value) else float(value)


def ung_vien_xgboost(grid: pd.DataFrame, ratio_grid: pd.DataFrame) -> pd.DataFrame:
    """Các cấu hình XGBoost đáng so bằng CV lặp — cùng quy tắc lọc mà notebook 04 §5.8 trình bày."""
    xgb = pd.concat([
        grid[grid["model"] == "xgboost"].assign(ratio=np.nan),
        ratio_grid[ratio_grid["model"] == "xgboost"] if not ratio_grid.empty else None,
    ], ignore_index=True)
    xgb = xgb[xgb["strategy"] != "smote_tomek"]
    best = xgb["pr_auc_mean"].max()
    return xgb[xgb["pr_auc_mean"] >= best - MARGIN].reset_index(drop=True)


def in_trang_thai() -> int:
    if not DUONG_DAN_SO_SANH.exists():
        print(f"Chưa có {DUONG_DAN_SO_SANH.name}. Chạy: python scripts/run_repeated_cv.py")
        return 0
    print(pd.read_csv(DUONG_DAN_SO_SANH).to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--smoke", action="store_true", help="12.000 dòng, 2 lần lặp — KHÔNG dùng cho báo cáo")
    parser.add_argument("--status", action="store_true", help="chỉ in kết quả đã lưu")
    args = parser.parse_args(argv)
    if args.status:
        return in_trang_thai()

    hau_to = "_smoke" if args.smoke else ""
    grid = pd.read_csv(REPORTS_DIR / f"grid_results{hau_to}.csv")
    ratio_path = REPORTS_DIR / f"ratio_grid{hau_to}.csv"
    ratio_grid = pd.read_csv(ratio_path) if ratio_path.exists() else pd.DataFrame()

    print("Nạp dữ liệu…")
    df, _ = load_prepared(verbose=False)
    X_train_tho, _, y_train, _ = split_data(df, as_features=False)
    xgb_repeats, rf_repeats = XGB_REPEATS, RF_REPEATS
    duong_dan, duong_dan_so_sanh = DUONG_DAN, DUONG_DAN_SO_SANH
    if args.smoke:
        X_train_tho, _, y_train, _ = train_test_split(X_train_tho, y_train, train_size=12_000,
                                                      stratify=y_train, random_state=RANDOM_STATE)
        xgb_repeats = rf_repeats = 2
        duong_dan = REPORTS_DIR / "repeated_cv_smoke.csv"
        duong_dan_so_sanh = REPORTS_DIR / "repeated_cv_tests_smoke.csv"
        print("CHẾ ĐỘ CHẠY THỬ — KHÔNG dùng cho báo cáo")
    X_train = build_features(X_train_tho)
    y_train = y_train.reset_index(drop=True)
    X_train = X_train.reset_index(drop=True)
    pos_weight = imbalance_ratio(y_train)
    data_fp = data_fingerprint(X_train, y_train)

    xgb = ung_vien_xgboost(grid, ratio_grid)
    configs = [("xgboost", row["strategy"], _ratio(row["ratio"]), xgb_repeats) for _, row in xgb.iterrows()]
    for extra in (("random_forest", "smote", None), ("random_forest", "class_weight", None),
                  ("xgboost", "class_weight", None)):
        if not any(c[:3] == extra for c in configs):
            configs.append((*extra, rf_repeats))
    print(f"{len(configs)} cấu hình: " + ", ".join(score_key(m, s, r) + f" ×{n}" for m, s, r, n in configs))

    cached = pd.read_csv(duong_dan) if duong_dan.exists() else pd.DataFrame()
    ket_qua = []
    da_gap = {}          # dấu vân tay → khoá đầu tiên mang nó: cấu hình trùng chỉ chạy một lần
    bi_dan = {}          # khoá trùng → khoá gốc
    bat_dau = time.perf_counter()
    for model, strategy, ratio, n_repeats in configs:
        key = score_key(model, strategy, ratio)
        pipeline = build_pipeline(model, strategy, pos_weight=pos_weight, ratio=ratio, random_state=RANDOM_STATE)
        fingerprint = config_fingerprint(pipeline, cv=f"RepeatedStratifiedKFold(5, random_state={RANDOM_STATE})",
                                         data=data_fp)
        if fingerprint in da_gap:
            # Ví dụ xgboost__smote của lưới chuẩn chính là xgboost__smote__r0.1 của lưới theo mức
            bi_dan[key] = da_gap[fingerprint]
            print(f"  {key:40s} trùng cấu hình {da_gap[fingerprint]} — không chạy lại")
            continue
        da_gap[fingerprint] = key
        old = cached[(cached["config_key"] == key) & (cached["config_hash"] == fingerprint)] if not cached.empty else cached
        if len(old) >= n_repeats * 5:
            ket_qua.append(old[old["repeat"] < n_repeats])
            print(f"  {key:40s} dùng lại {len(old)} fold đã lưu")
            continue
        started = time.perf_counter()
        scores = repeated_cv_scores(pipeline, X_train, y_train, n_repeats=n_repeats, random_state=RANDOM_STATE)
        scores.insert(0, "config_key", key)
        scores.insert(1, "model", model)
        scores.insert(2, "strategy", strategy)
        scores.insert(3, "ratio", ratio)
        scores["config_hash"] = fingerprint
        ket_qua.append(scores)
        # Ghi ngay sau mỗi cấu hình, giữ cả các cấu hình cũ chưa tới lượt: chết giữa chừng không mất gì
        da_xong = {frame["config_key"].iloc[0] for frame in ket_qua}
        con_lai = cached[~cached["config_key"].isin(da_xong)] if not cached.empty else cached
        pd.concat([*ket_qua, con_lai], ignore_index=True).to_csv(duong_dan, index=False, encoding="utf-8")
        print(f"  {key:40s} PR-AUC {scores['pr_auc'].mean():.4f} ± {scores['pr_auc'].std(ddof=1):.4f} "
              f"({len(scores)} fold, {(time.perf_counter() - started) / 60:.1f} phút; "
              f"đã {(time.perf_counter() - bat_dau) / 60:.0f} phút)")

    tat_ca = pd.concat(ket_qua, ignore_index=True)
    tat_ca.to_csv(duong_dan, index=False, encoding="utf-8")
    theo_cau_hinh = {key: frame for key, frame in tat_ca.groupby("config_key")}
    for trung, goc in bi_dan.items():
        theo_cau_hinh[trung] = theo_cau_hinh[goc]

    # Cấu hình XGBoost tốt nhất theo trung bình CV lặp, rồi so mọi ứng viên XGBoost với nó
    xgb_keys = [score_key("xgboost", s, _ratio(r)) for s, r in zip(xgb["strategy"], xgb["ratio"])]
    xgb_keys = [k for k in xgb_keys if k not in bi_dan]          # mỗi cấu hình thật một dòng
    best_key = max(xgb_keys, key=lambda k: theo_cau_hinh[k]["pr_auc"].mean())
    so_sanh = []
    for key in xgb_keys:
        so_sanh.append({"purpose": "chọn mô hình cuối (04 §5.8)", "a": key, "b": best_key,
                        **compare_repeated_cv(theo_cau_hinh[key], theo_cau_hinh[best_key])})
    for a, b in ((score_key("random_forest", "smote"), score_key("random_forest", "class_weight")),
                 (score_key("xgboost", "class_weight"), score_key("random_forest", "class_weight"))):
        so_sanh.append({"purpose": "khẳng định cũ của báo cáo §6", "a": a, "b": b,
                        **compare_repeated_cv(theo_cau_hinh[a], theo_cau_hinh[b])})
    bang = pd.DataFrame(so_sanh)
    bang.to_csv(duong_dan_so_sanh, index=False, encoding="utf-8")

    print("-" * 78)
    print(f"Xong trong {(time.perf_counter() - bat_dau) / 60:.1f} phút. Cấu hình XGBoost tốt nhất: {best_key}")
    print(bang[["a", "b", "mean_a", "mean_b", "mean_diff", "ci_low", "ci_high", "p_value", "contains_zero"]]
          .to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
