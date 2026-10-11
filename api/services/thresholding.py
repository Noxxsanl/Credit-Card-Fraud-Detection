"""Chỉ số theo ngưỡng và chọn ngưỡng theo chi phí — API-11, API-12 (docs/03 §5.2).

Cả hai phép tính chạy trên **điểm chọn ngưỡng** (``oof_scores.npz``): điểm out-of-fold của tập
huấn luyện, từ review 2026-10-10 sinh bằng forward-chaining theo thời gian (docs/11 M2).

- ``preview`` **ước lượng** chỉ số ở một ngưỡng giả định, giống hệt phép tính UI-D1 trong trình
  duyệt (TC-12). Trước review nó đo trên tập kiểm thử, nên người dùng kéo thanh trượt là đang chọn
  ngưỡng bằng chính tập kiểm thử — trái ML-08 (docs/11 M3). Tập kiểm thử giờ chỉ còn ở các màn
  hình báo cáo, cho ngưỡng đã áp dụng.
- ``optimize`` **chọn** ngưỡng, dò trên mọi điểm khác nhau như notebook 06 và 08, bằng cùng hàm
  ``src.threshold.constrained_min_cost`` mà notebook 08 dùng để chọn ngưỡng mặc định. Với chi phí
  và ngân sách mặc định nó ra đúng ngưỡng mặc định của ``threshold.json``.
"""

from __future__ import annotations

from src.threshold import constrained_min_cost, cost_curve, metrics_at_threshold

from ..loader import Artifacts

CURVE_POINTS = 200


def _selection_kwargs(loaded: Artifacts, cost_fn: float, cost_fp: float) -> dict:
    return dict(cost_fn=cost_fn, cost_fp=cost_fp, sample_fraction=loaded.oof["train_fraction"],
                days=loaded.oof["days"])


def preview(loaded: Artifacts, value: float, cost_fn: float, cost_fp: float) -> dict:
    return metrics_at_threshold(loaded.oof["y_true"], loaded.oof["y_score"], value,
                                **_selection_kwargs(loaded, cost_fn, cost_fp)).as_dict()


def optimize(loaded: Artifacts, cost_fn: float, cost_fp: float, constraint_type: str = "none",
             constraint_value: float | None = None) -> dict:
    """Ngưỡng cực tiểu chi phí kỳ vọng, có thể kèm một ràng buộc vận hành (FR-32, FR-33).

    ``constraint_binding`` trả lời câu hỏi "tôi đang bị giới hạn bởi ngân sách thẩm định
    hay bởi chính chi phí": ``true`` khi nghiệm tự do vi phạm ràng buộc và phải dời đi.
    """
    y, s = loaded.oof["y_true"], loaded.oof["y_score"]
    kwargs = _selection_kwargs(loaded, cost_fn, cost_fp)
    table = cost_curve(y, s, thresholds=loaded.oof_candidates, **kwargs)
    free, _ = constrained_min_cost(table, "none", None)
    chosen, satisfied = constrained_min_cost(table, constraint_type, constraint_value)

    curve = cost_curve(y, s, n_steps=CURVE_POINTS, **kwargs)
    return {
        "optimal_threshold": chosen,
        "unconstrained_threshold": free,
        "constraint_binding": bool(chosen != free),
        "constraint_satisfied": satisfied,
        "cost_fn": cost_fn,
        "cost_fp": cost_fp,
        "selection_method": loaded.selection_method,
        "metrics_at_optimal": metrics_at_threshold(y, s, chosen, **kwargs).as_dict(),
        "curve": [
            {"threshold": float(r.threshold), "cost": float(r.expected_cost), "alerts": int(r.alerts),
             "alerts_per_day": float(r.alerts_per_day), "recall": float(r.recall), "precision": float(r.precision)}
            for r in curve.itertuples()
        ],
    }

