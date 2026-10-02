"""Kiểm tra tái lập (T-38, AC-M5, NFR-07): chạy lại notebook trong kernel sạch rồi so số.

Quy trình ở docs/10 §6, viết thành một lệnh:

  1. Chụp số hiện tại: chỉ số chính và dấu vân tay điểm trong models/metrics.json,
     ngưỡng trong models/threshold.json, cùng mọi bảng reports/*.csv mà notebook ghi ra.
  2. Chạy lại lần lượt từng notebook bằng nbconvert — mỗi notebook một kernel mới.
  3. Chụp lại và so. Đạt khi PR-AUC lệch dưới 0,001 (AC-M5); các dòng còn lại để biết
     lệch nằm ở đâu nếu có.

Kết quả ghi vào reports/reproducibility.csv.

Lưu ý: notebook 04 và 05 đọc điểm lưu của scripts/run_grid.py và scripts/run_search.py
nếu có. Script này KHÔNG chạy lại hai bước đó (45 + 20 phút); notebook 08 tự tính lại
điểm out-of-fold của mô hình được chọn và so với điểm lưu của lưới.

Chạy:
    python scripts/check_reproducibility.py              # 01 → 08, khoảng 45 phút
    python scripts/check_reproducibility.py --only 08    # chỉ notebook 08, khoảng 5 phút
    python scripts/check_reproducibility.py --no-run     # chỉ in ảnh chụp hiện tại
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Console Windows mặc định là cp1252 — xem ghi chú trong run_grid.py
for luong in (sys.stdout, sys.stderr):
    if hasattr(luong, "reconfigure"):
        luong.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.artifacts import read_json  # noqa: E402
from src.config import METRICS_PATH, REPORTS_DIR, THRESHOLD_PATH  # noqa: E402

NOTEBOOKS = (
    "01_eda",
    "02_statistics",
    "03_baseline",
    "04_imbalance_strategies",
    "05_advanced_models",
    "06_threshold_and_cost",
    "06b_autoencoder",
    "07_explainability",
    "08_export_artifacts",
)

#: AC-M5, NFR-07
PR_AUC_TOLERANCE = 1e-3

OUTPUT_PATH = REPORTS_DIR / "reproducibility.csv"

#: Cột đo thời gian chạy — đổi mỗi lần chạy, không phải kết quả
_TIMING = re.compile(r"(giây|second|minute|phút|fit_time)", re.IGNORECASE)


def snapshot() -> dict:
    """Ảnh chụp mọi con số cần so."""
    if not METRICS_PATH.exists():
        raise SystemExit(f"Chưa có {METRICS_PATH.relative_to(ROOT)} — chạy notebook 08 trước.")
    metrics = read_json(METRICS_PATH)
    headline = metrics["headline"]
    numbers = {f"headline.{m}.{p}": headline[m][p]
               for m in ("pr_auc", "roc_auc", "recall", "precision") for p in ("value", "ci_low", "ci_high")}
    numbers["threshold.default_threshold"] = read_json(THRESHOLD_PATH)["default_threshold"]
    tables = {
        path.name: pd.read_csv(path)
        for path in sorted(REPORTS_DIR.glob("*.csv"))
        if not path.stem.endswith("_smoke") and path != OUTPUT_PATH
    }
    return {"numbers": numbers, "fingerprint": metrics.get("fingerprint", {}), "tables": tables,
            "model_version": metrics["model_version"], "trained_at": metrics["trained_at"]}


def compare_tables(before: pd.DataFrame, after: pd.DataFrame) -> tuple[float, int, str]:
    """(lệch số lớn nhất, số ô chữ khác nhau, ghi chú) giữa hai phiên bản một bảng."""
    if before.shape != after.shape or list(before.columns) != list(after.columns):
        return float("nan"), -1, f"khác kích thước: {before.shape} → {after.shape}"
    columns = [c for c in before.columns if not _TIMING.search(str(c))]
    numeric = [c for c in columns if pd.api.types.is_numeric_dtype(before[c]) and pd.api.types.is_numeric_dtype(after[c])]
    text = [c for c in columns if c not in numeric]
    diff = 0.0
    if numeric:
        a, b = before[numeric].to_numpy(dtype="float64"), after[numeric].to_numpy(dtype="float64")
        both_nan = np.isnan(a) & np.isnan(b)
        delta = np.where(both_nan, 0.0, np.abs(a - b))
        diff = float(np.nan_to_num(delta, nan=np.inf).max()) if delta.size else 0.0
    changed = int((before[text].astype(str) != after[text].astype(str)).to_numpy().sum()) if text else 0
    skipped = len(before.columns) - len(columns)
    return diff, changed, f"bỏ qua {skipped} cột thời gian" if skipped else ""


def run_notebook(name: str, timeout: int) -> float:
    path = ROOT / "notebooks" / f"{name}.ipynb"
    command = [sys.executable, "-m", "jupyter", "nbconvert", "--to", "notebook", "--execute",
               "--inplace", f"--ExecutePreprocessor.timeout={timeout}", str(path)]
    started = time.perf_counter()
    print(f"\n▶ {name} …", flush=True)
    subprocess.run(command, check=True, cwd=ROOT)
    elapsed = time.perf_counter() - started
    print(f"  xong sau {elapsed / 60:.1f} phút", flush=True)
    return elapsed


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--only", default="",
                        help="chỉ chạy các notebook có tiền tố này, cách nhau dấu phẩy (ví dụ 03,08)")
    parser.add_argument("--no-run", action="store_true", help="không chạy notebook, chỉ in ảnh chụp")
    parser.add_argument("--timeout", type=int, default=7200, help="giới hạn mỗi ô, giây")
    args = parser.parse_args(argv)

    before = snapshot()
    print(f"Ảnh chụp TRƯỚC: {before['model_version']} lúc {before['trained_at']}, "
          f"PR-AUC {before['numbers']['headline.pr_auc.value']:.10f}, "
          f"{len(before['tables'])} bảng trong reports/")
    if args.no_run:
        for key, value in before["numbers"].items():
            print(f"  {key:32s} {value:.10f}")
        for key, value in before["fingerprint"].items():
            print(f"  fingerprint.{key:20s} {value}")
        return 0

    prefixes = [p.strip() for p in args.only.split(",") if p.strip()]
    selected = [n for n in NOTEBOOKS if not prefixes or any(n.startswith(p) for p in prefixes)]
    if not selected:
        raise SystemExit(f"Không notebook nào khớp --only {args.only!r}")
    if selected[-1] != "08_export_artifacts":
        print("Cảnh báo: không chạy notebook 08 thì models/metrics.json không đổi — PR-AUC sẽ so với chính nó.")

    rows = []
    for name in selected:
        rows.append({"kind": "notebook", "name": name, "seconds": run_notebook(name, args.timeout)})

    after = snapshot()
    for key, old in before["numbers"].items():
        new = after["numbers"][key]
        tolerance = PR_AUC_TOLERANCE if key == "headline.pr_auc.value" else np.nan
        rows.append({"kind": "number", "name": key, "before": old, "after": new, "abs_diff": abs(new - old),
                     "tolerance": tolerance})
    for key, old in before["fingerprint"].items():
        new = after["fingerprint"].get(key)
        rows.append({"kind": "fingerprint", "name": key, "before": old, "after": new,
                     "identical": old == new})
    for name, table in before["tables"].items():
        if name not in after["tables"]:
            rows.append({"kind": "table", "name": name, "note": "không còn sau khi chạy lại"})
            continue
        diff, changed, note = compare_tables(table, after["tables"][name])
        rows.append({"kind": "table", "name": name, "abs_diff": diff, "text_cells_changed": changed,
                     "identical": diff == 0 and changed == 0, "note": note})

    result = pd.DataFrame(rows)
    result.insert(0, "checked_at", after["trained_at"])
    result.to_csv(OUTPUT_PATH, index=False, encoding="utf-8")

    pr_diff = abs(after["numbers"]["headline.pr_auc.value"] - before["numbers"]["headline.pr_auc.value"])
    passed = pr_diff < PR_AUC_TOLERANCE
    print("\n" + "=" * 72)
    print(f"Notebook đã chạy lại: {len(selected)}, tổng {result['seconds'].sum() / 60:.1f} phút")
    print(f"PR-AUC: {before['numbers']['headline.pr_auc.value']:.10f} → "
          f"{after['numbers']['headline.pr_auc.value']:.10f}  (lệch {pr_diff:.1e}, dung sai {PR_AUC_TOLERANCE:g})")
    for row in rows:
        if row["kind"] == "fingerprint":
            print(f"Dấu vân tay {row['name']}: {'TRÙNG' if row['identical'] else 'KHÁC'}")
    tables = result[result["kind"] == "table"]
    print(f"Bảng reports/*.csv trùng hoàn toàn: {int(tables['identical'].fillna(False).sum())}/{len(tables)}")
    for _, row in tables[~tables["identical"].fillna(False).astype(bool)].iterrows():
        print(f"  {row['name']:32s} lệch số {row['abs_diff']:.1e}, ô chữ khác {row.get('text_cells_changed')}"
              f"  {row.get('note') or ''}")
    print(f"T-38 / AC-M5: {'ĐẠT' if passed else 'KHÔNG ĐẠT'} — chi tiết ở {OUTPUT_PATH.relative_to(ROOT)}")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
