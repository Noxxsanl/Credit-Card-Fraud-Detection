"""Theo dõi vận hành: tỷ lệ cảnh báo và precision theo thời gian, so với kỳ vọng (docs/11 M2).

Ngưỡng chọn một lần rồi để yên sẽ trôi. Khi chia theo thời gian, cùng một quy trình chọn ngưỡng mà
sang ngày 2 sinh gần gấp đôi số cảnh báo dự kiến, precision giảm một nửa. Endpoint này so, theo từng
khoảng thời gian của dữ liệu (``transactions.time_offset``) và chỉ với mô hình đang phục vụ:

- **tỷ lệ cảnh báo** quan sát được (giao dịch có điểm ≥ ngưỡng hiện hành / giao dịch đã chấm) với tỷ
  lệ kỳ vọng trên điểm chọn ngưỡng, và quy ra cảnh báo mỗi ngày trên toàn luồng để so ngân sách;
- **precision theo kết luận thẩm định** (bảng ``reviews``) với precision kỳ vọng;
- **recall theo nhãn thật** khi giao dịch có nhãn (thư viện mẫu, phát lại).

Mọi tỷ lệ kèm khoảng tin cậy Wilson 95%. Một khoảng thời gian bị đánh dấu khi cả khoảng tin cậy nằm
về phía xấu của kỳ vọng — chênh lệch nhỏ hơn dao động mẫu thì không báo.
"""

from __future__ import annotations

from sqlalchemy import text
from sqlalchemy.orm import Session

from src.evaluate import wilson_interval
from src.threshold import metrics_at_threshold

from ..loader import Artifacts

SECONDS_PER_HOUR = 3600.0

_BUCKETS = text("""
    SELECT floor(t.time_offset / :width) AS bucket,
           count(*) AS n_scored,
           count(*) FILTER (WHERE t.risk_score >= :tau) AS n_alerts,
           count(r.id) FILTER (WHERE t.risk_score >= :tau) AS n_reviewed,
           count(r.id) FILTER (WHERE t.risk_score >= :tau AND r.decision = 'confirmed_fraud') AS n_confirmed,
           count(*) FILTER (WHERE t.true_label = 1) AS n_labelled_fraud,
           count(*) FILTER (WHERE t.true_label = 1 AND t.risk_score >= :tau) AS n_caught
    FROM transactions t
    LEFT JOIN reviews r ON r.transaction_id = t.id
    WHERE t.model_version = :version
    GROUP BY bucket
    ORDER BY bucket
""")


def expected_rates(loaded: Artifacts, tau: float) -> dict:
    """Tỷ lệ kỳ vọng ở ngưỡng ``tau``, ước lượng trên điểm chọn ngưỡng."""
    oof = loaded.oof
    m = metrics_at_threshold(oof["y_true"], oof["y_score"], tau, sample_fraction=oof["train_fraction"],
                             days=oof["days"])
    n = int(oof["y_true"].size)
    return {"alert_rate": m.alerts / n, "alerts_per_day": m.alerts_per_day, "precision": m.precision,
            "recall": m.recall, "source": loaded.selection_method}


def transactions_per_day(loaded: Artifacts) -> float:
    """Khối lượng giao dịch mỗi ngày của luồng mà dữ liệu đại diện — để quy tỷ lệ ra số cảnh báo/ngày."""
    dataset = loaded.metrics["dataset"]
    return float(dataset["n_total"]) / float(dataset["days"])


def _bucket(row: dict, *, width_hours: float, expected: dict, volume: float, budget: float | None) -> dict:
    n, alerts = int(row["n_scored"]), int(row["n_alerts"])
    reviewed, confirmed = int(row["n_reviewed"]), int(row["n_confirmed"])
    labelled, caught = int(row["n_labelled_fraud"]), int(row["n_caught"])
    rate_ci = wilson_interval(alerts, n)
    precision_ci = wilson_interval(confirmed, reviewed)
    recall_ci = wilson_interval(caught, labelled)
    flags = []
    if n and rate_ci[0] > expected["alert_rate"]:
        flags.append("alerts_above_expected")
    if reviewed and precision_ci[1] < expected["precision"]:
        flags.append("precision_below_expected")
    if labelled and recall_ci[1] < expected["recall"]:
        flags.append("recall_below_expected")
    if budget is not None and n and rate_ci[0] * volume > budget:
        flags.append("over_budget")
    start = None if row["bucket"] is None else float(row["bucket"]) * width_hours
    return {
        "start_hour": start,
        "end_hour": None if start is None else start + width_hours,
        "n_scored": n,
        "n_alerts": alerts,
        "alert_rate": alerts / n if n else 0.0,
        "alert_rate_ci": list(rate_ci),
        "alerts_per_day": (alerts / n) * volume if n else 0.0,
        "n_reviewed": reviewed,
        "n_confirmed": confirmed,
        "precision_reviewed": confirmed / reviewed if reviewed else None,
        "precision_ci": list(precision_ci) if reviewed else None,
        "n_labelled_fraud": labelled,
        "n_caught": caught,
        "recall_labelled": caught / labelled if labelled else None,
        "recall_ci": list(recall_ci) if labelled else None,
        "flags": flags,
    }


def report(session: Session, loaded: Artifacts, tau: float, *, bucket_hours: float = 24.0) -> dict:
    expected = expected_rates(loaded, tau)
    volume = transactions_per_day(loaded)
    budget = loaded.threshold.get("constraints", {}).get("alert_budget_per_day")
    rows = [dict(r._mapping) for r in session.execute(
        _BUCKETS, {"width": bucket_hours * SECONDS_PER_HOUR, "tau": tau, "version": loaded.model_version})]
    common = dict(width_hours=bucket_hours, expected=expected, volume=volume, budget=budget)
    total = {key: sum(int(r[key]) for r in rows) for key in
             ("n_scored", "n_alerts", "n_reviewed", "n_confirmed", "n_labelled_fraud", "n_caught")}
    overall = _bucket({**total, "bucket": None}, **common)
    return {
        "model_version": loaded.model_version,
        "threshold": tau,
        "bucket_hours": bucket_hours,
        "transactions_per_day": volume,
        "alert_budget_per_day": budget,
        "expected": expected,
        "overall": overall,
        "buckets": [_bucket(r, **common) for r in rows],
    }
