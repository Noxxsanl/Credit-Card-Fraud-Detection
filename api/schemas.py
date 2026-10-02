"""Mô hình Pydantic vào/ra — bản thi hành của hợp đồng ở docs/05.

Khi có tranh cãi về hành vi giữa backend và frontend, tệp này là nguồn chân lý.

``TransactionInput`` được dựng từ ``RAW_REQUIRED_COLUMNS`` của ``src/features.py``
chứ không gõ tay 30 trường, để hợp đồng API và hợp đồng dữ liệu (02 §5) không thể
lệch nhau.
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, create_model, model_validator

from src.config import DEFAULT_COST_FN, DEFAULT_COST_FP
from src.features import AMOUNT_COLUMN, RAW_REQUIRED_COLUMNS, TIME_COLUMN, V_COLUMNS

log = logging.getLogger("api")

RiskBand = Literal["low", "medium", "high", "critical"]
Decision = Literal["allow", "review", "block"]
ReviewDecision = Literal["confirmed_fraud", "false_alarm"]
Source = Literal["upload", "sample", "replay", "manual"]
ThresholdSource = Literal["artifact", "user"]

#: NUMERIC(12, 2) của cột amount (ST-07) chứa được tới đây
MAX_AMOUNT = 9_999_999_999.99

FiniteFloat = Annotated[float, Field(allow_inf_nan=False)]

_EXAMPLE = {
    "Time": 7834.0,
    **dict(zip(V_COLUMNS, [
        -1.3598, -0.0728, 2.5363, 1.3782, -0.3383, 0.4624, 0.2396, 0.0987, 0.3638, 0.0908,
        -0.5516, -0.6178, -0.9914, -0.3112, 1.4682, -0.4704, 0.2080, 0.0258, 0.4040, 0.2514,
        -0.0183, 0.2778, -0.1105, 0.0669, 0.1285, -0.1891, 0.1336, -0.0211,
    ])),
    "Amount": 149.62,
}


class _TransactionBase(BaseModel):
    """30 cột thô của một giao dịch (DS-10…DS-14). Cột thừa bị bỏ qua, có ghi log."""

    model_config = ConfigDict(extra="ignore", json_schema_extra={"example": _EXAMPLE})

    @model_validator(mode="before")
    @classmethod
    def _log_extra_columns(cls, data):
        if isinstance(data, dict):
            extra = [k for k in data if k not in RAW_REQUIRED_COLUMNS]
            if extra:
                log.warning("Bỏ qua cột thừa ngoài lược đồ (DS-14): %s", ", ".join(map(str, extra[:10])))
        return data


_FIELDS = {TIME_COLUMN: (FiniteFloat, Field(ge=0, description="Số giây kể từ giao dịch đầu tiên"))}
_FIELDS.update({name: (FiniteFloat, Field(description="Thành phần chính sau PCA")) for name in V_COLUMNS})
_FIELDS[AMOUNT_COLUMN] = (FiniteFloat, Field(ge=0, le=MAX_AMOUNT, description="Số tiền, EUR"))

TransactionInput = create_model("TransactionInput", __base__=_TransactionBase, **_FIELDS)


# --------------------------------------------------------------------------
# Chấm điểm — API-02, API-03, API-04
# --------------------------------------------------------------------------

class ScoreRequest(BaseModel):
    transaction: TransactionInput  # type: ignore[valid-type]
    persist: bool = Field(True, description="false: chỉ chấm thử, không ghi vào cơ sở dữ liệu")
    sample_id: str | None = Field(
        None, description="Mã mẫu trong /samples. Có thì giao dịch lưu với source='sample' và nhãn thật của mẫu"
    )


class ScoreResponse(BaseModel):
    transaction_id: str | None
    risk_score: float
    threshold: float
    decision: Decision
    risk_band: RiskBand
    model_version: str
    scored_at: datetime
    latency_ms: float


class BatchRequest(BaseModel):
    transactions: list[TransactionInput]  # type: ignore[valid-type]
    batch_id: str | None = Field(None, max_length=64, pattern=r"^[A-Za-z0-9._\-]+$")
    persist: bool = True


class BatchResult(BaseModel):
    transaction_id: str | None
    risk_score: float
    risk_band: RiskBand
    decision: Decision


class ScoreDistribution(BaseModel):
    low: int = 0
    medium: int = 0
    high: int = 0
    critical: int = 0


class BatchResponse(BaseModel):
    batch_id: str
    count: int
    alerts: int
    threshold: float
    score_distribution: ScoreDistribution
    elapsed_ms: float
    model_version: str
    results: list[BatchResult]


class RejectedRow(BaseModel):
    row: int = Field(description="Số thứ tự dòng dữ liệu trong CSV, bắt đầu từ 1, không tính dòng tiêu đề")
    reason: str


class ActualMetrics(BaseModel):
    precision: float | None
    recall: float | None
    pr_auc: float | None


class UploadResponse(BatchResponse):
    rows_read: int
    rows_rejected: int
    rejection_reasons: list[RejectedRow]
    rejection_reasons_truncated: bool
    has_labels: bool
    actual_metrics: ActualMetrics | None
    results_truncated: bool = Field(description="results chỉ giữ các giao dịch điểm cao nhất")


# --------------------------------------------------------------------------
# Giải thích — API-05
# --------------------------------------------------------------------------

class ExplainRequest(BaseModel):
    transaction_id: str | None = None
    transaction: TransactionInput | None = None  # type: ignore[valid-type]

    @model_validator(mode="after")
    def _exactly_one(self):
        if (self.transaction_id is None) == (self.transaction is None):
            raise ValueError("Gửi đúng một trong hai: transaction_id hoặc transaction")
        return self


class Contribution(BaseModel):
    feature: str
    value: float = Field(description="Giá trị đặc trưng chưa chuẩn hoá (Amount theo đơn vị tiền)")
    shap: float = Field(description="Đóng góp vào logit của điểm rủi ro")


class ExplainResponse(BaseModel):
    transaction_id: str | None
    risk_score: float
    base_value: float = Field(description="Giá trị kỳ vọng của mô hình, thang log-odds")
    margin: float = Field(description="base_value + Σ SHAP = logit của điểm rủi ro")
    shap_output: Literal["log-odds"] = "log-odds"
    top_positive: list[Contribution]
    top_negative: list[Contribution]
    remaining_shap: float = Field(description="Tổng SHAP của các đặc trưng không nằm trong hai danh sách trên")
    contributions: list[Contribution] = Field(description="Toàn bộ 31 đặc trưng, sắp theo |SHAP| giảm dần")
    model_version: str
    elapsed_ms: float


# --------------------------------------------------------------------------
# Giao dịch và thẩm định — API-06, API-07, API-08
# --------------------------------------------------------------------------

class ReviewInfo(BaseModel):
    decision: ReviewDecision
    threshold_used: float
    reviewed_at: datetime
    note: str | None


class TransactionItem(BaseModel):
    id: str
    risk_score: float
    risk_band: RiskBand
    decision: Decision
    amount: float
    hour: int
    true_label: int | None
    reviewed: bool
    review_decision: ReviewDecision | None
    source: Source
    batch_id: str | None
    model_version: str
    created_at: datetime


class TransactionPage(BaseModel):
    items: list[TransactionItem]
    page: int
    page_size: int
    total: int
    threshold: float


class TransactionDetail(BaseModel):
    id: str
    features: dict[str, float] = Field(description="30 cột thô: Time, V1…V28, Amount")
    risk_score: float
    risk_band: RiskBand
    decision: Decision
    threshold: float
    amount: float
    amount_percentile: float = Field(description="Phần giao dịch của tập kiểm thử có số tiền nhỏ hơn (US-04)")
    hour: int
    true_label: int | None
    review: ReviewInfo | None
    source: Source
    batch_id: str | None
    model_version: str
    model_version_current: bool = Field(description="false: điểm do một mô hình khác chấm (06 §8)")
    created_at: datetime


class ReviewRequest(BaseModel):
    transaction_id: str = Field(min_length=1)
    decision: ReviewDecision
    note: str | None = Field(None, max_length=2000)


class ReviewResponse(BaseModel):
    transaction_id: str
    decision: ReviewDecision
    threshold_used: float
    reviewed_at: datetime
    note: str | None
    true_label: int | None = Field(description="Chỉ trả sau khi đã có quyết định (UI-02, FR-44)")
    matches_label: bool | None


# --------------------------------------------------------------------------
# Ngưỡng — API-09 … API-12
# --------------------------------------------------------------------------

class ThresholdState(BaseModel):
    current: float
    source: ThresholdSource
    default: float = Field(description="τ* trong threshold.json, chọn trên out-of-fold")
    cost_fn: float
    cost_fp: float
    alternatives: dict[str, float]
    model_version: str


class ThresholdUpdate(BaseModel):
    value: FiniteFloat
    cost_fn: FiniteFloat | None = Field(None, gt=0)
    cost_fp: FiniteFloat | None = Field(None, ge=0)


class ThresholdMetricsOut(BaseModel):
    threshold: float
    alerts: int
    tp: int
    fp: int
    fn: int
    tn: int
    precision: float
    recall: float
    f1: float
    expected_cost: float
    alerts_per_day: float


class Constraint(BaseModel):
    type: Literal["none", "min_recall", "max_alerts_per_day"] = "none"
    value: FiniteFloat | None = None

    @model_validator(mode="after")
    def _check_value(self):
        if self.type == "none":
            return self
        if self.value is None:
            raise ValueError(f"Ràng buộc {self.type} cần value")
        if self.type == "min_recall" and not 0 < self.value <= 1:
            raise ValueError("min_recall phải nằm trong (0, 1]")
        if self.type == "max_alerts_per_day" and self.value <= 0:
            raise ValueError("max_alerts_per_day phải dương")
        return self


class OptimizeRequest(BaseModel):
    cost_fn: FiniteFloat = Field(DEFAULT_COST_FN, gt=0)
    cost_fp: FiniteFloat = Field(DEFAULT_COST_FP, ge=0)
    constraint: Constraint = Constraint()


class CurvePoint(BaseModel):
    threshold: float
    cost: float
    alerts: int
    alerts_per_day: float
    recall: float
    precision: float


class OptimizeResponse(BaseModel):
    optimal_threshold: float
    unconstrained_threshold: float
    constraint_binding: bool = Field(description="true: ràng buộc đẩy nghiệm khỏi cực tiểu chi phí")
    constraint_satisfied: bool
    selected_on: Literal["out_of_fold"] = "out_of_fold"
    cost_fn: float
    cost_fp: float
    metrics_at_optimal: ThresholdMetricsOut = Field(description="Đo trên tập kiểm thử tại ngưỡng đã chọn")
    metrics_at_optimal_oof: ThresholdMetricsOut = Field(description="Trên out-of-fold, nơi ngưỡng được chọn")
    curve: list[CurvePoint]
    curve_source: Literal["out_of_fold"] = "out_of_fold"


# --------------------------------------------------------------------------
# Còn lại — API-01, API-14
# --------------------------------------------------------------------------

class SampleItem(BaseModel):
    id: str
    category: Literal["fraud_easy", "fraud_hard", "legit_easy", "legit_hard"]
    label: int
    risk_score: float
    amount: float
    description: str
    features: dict[str, float]


class SampleList(BaseModel):
    items: list[SampleItem]
    categories: dict[str, dict]
    model_version: str
    reference_threshold: float


class HealthResponse(BaseModel):
    status: Literal["ok"]
    model_version: str
    model_loaded_at: datetime
    threshold: float
    db: Literal["ok", "down"]
    uptime_seconds: float


class ErrorBody(BaseModel):
    code: str
    message: str
    details: dict | None = None


class ErrorResponse(BaseModel):
    error: ErrorBody
