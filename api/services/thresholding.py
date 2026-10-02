"""Chỉ số theo ngưỡng và chọn ngưỡng theo chi phí — API-11, API-12 (docs/03 §5.2).

Hai nguồn điểm, hai vai trò, giữ đúng quy tắc ML-08 của notebook 06:

- ``preview`` **đo** trên điểm tập kiểm thử (``metrics.json → test_scores``), giống hệt
  phép tính UI-D1 trong trình duyệt (TC-12). Không chạy mô hình, không đụng ổ đĩa.
- ``optimize`` **chọn** ngưỡng trên điểm out-of-fold của tập huấn luyện
  (``oof_scores.npz``), dò trên mọi điểm khác nhau như notebook 06, rồi đo kết quả trên
  tập kiểm thử. Với chi phí mặc định nó ra đúng τ* của ``threshold.json``.
"""

from __future__ import annotations

from src.threshold import cost_curve, metrics_at_threshold

from ..loader import Artifacts

CURVE_POINTS = 200


def preview(loaded: Artifacts, value: float, cost_fn: float, cost_fp: float) -> dict:
    return metrics_at_threshold(loaded.y_test, loaded.s_test, value, cost_fn=cost_fn, cost_fp=cost_fp,
                                sample_fraction=loaded.test_fraction, days=loaded.days).as_dict()


def _oof_kwargs(loaded: Artifacts, cost_fn: float, cost_fp: float) -> dict:
    return dict(cost_fn=cost_fn, cost_fp=cost_fp, sample_fraction=loaded.oof["train_fraction"],
                days=loaded.oof["days"])


def optimize(loaded: Artifacts, cost_fn: float, cost_fp: float, constraint_type: str = "none",
             constraint_value: float | None = None) -> dict:
    """Ngưỡng cực tiểu chi phí kỳ vọng, có thể kèm một ràng buộc vận hành (FR-32, FR-33).

    ``constraint_binding`` trả lời câu hỏi "tôi đang bị giới hạn bởi ngân sách thẩm định
    hay bởi chính chi phí": ``true`` khi nghiệm tự do vi phạm ràng buộc và phải dời đi.
    """
    y, s = loaded.oof["y_true"], loaded.oof["y_score"]
    kwargs = _oof_kwargs(loaded, cost_fn, cost_fp)
    table = cost_curve(y, s, thresholds=loaded.oof_candidates, **kwargs)
    free = float(table.loc[table["expected_cost"].idxmin(), "threshold"])

    chosen, satisfied = free, True
    if constraint_type != "none":
        column = "recall" if constraint_type == "min_recall" else "alerts_per_day"
        ok = table[column] >= constraint_value if column == "recall" else table[column] <= constraint_value
        feasible = table[ok]
        if feasible.empty:
            # Không ngưỡng nào đạt: lấy ngưỡng gần đạt nhất, giống pick_threshold
            satisfied = False
            chosen = float(table["threshold"].min() if column == "recall" else table["threshold"].max())
        else:
            chosen = float(feasible.loc[feasible["expected_cost"].idxmin(), "threshold"])

    curve = cost_curve(y, s, n_steps=CURVE_POINTS, **kwargs)
    return {
        "optimal_threshold": chosen,
        "unconstrained_threshold": free,
        "constraint_binding": bool(chosen != free),
        "constraint_satisfied": satisfied,
        "cost_fn": cost_fn,
        "cost_fp": cost_fp,
        "metrics_at_optimal": preview(loaded, chosen, cost_fn, cost_fp),
        "metrics_at_optimal_oof": metrics_at_threshold(y, s, chosen, **kwargs).as_dict(),
        "curve": [
            {"threshold": float(r.threshold), "cost": float(r.expected_cost), "alerts": int(r.alerts),
             "alerts_per_day": float(r.alerts_per_day), "recall": float(r.recall), "precision": float(r.precision)}
            for r in curve.itertuples()
        ],
    }

