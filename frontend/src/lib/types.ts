/*
 * Kiểu dữ liệu của API — bản TypeScript của api/schemas.py (hợp đồng ở docs/05).
 * Khi hai bên lệch nhau, api/schemas.py là nguồn chân lý.
 */

export type RiskBand = "low" | "medium" | "high" | "critical";
export type Decision = "allow" | "review" | "block";
export type ReviewDecision = "confirmed_fraud" | "false_alarm";
export type ReviewStatus = "pending" | ReviewDecision;
export type Source = "upload" | "sample" | "replay" | "manual";
export type SampleCategory = "fraud_easy" | "fraud_hard" | "legit_easy" | "legit_hard";

// ---------------------------------------------------------------- API-01, API-09 … API-12

export interface Health {
  status: "ok";
  model_version: string;
  model_loaded_at: string;
  threshold: number;
  db: "ok" | "down";
  uptime_seconds: number;
}

export interface ThresholdState {
  current: number;
  source: "artifact" | "user";
  default: number;
  cost_fn: number;
  cost_fp: number;
  alternatives: Record<string, number>;
  model_version: string;
}

export interface ThresholdMetricsOut {
  threshold: number;
  alerts: number;
  tp: number;
  fp: number;
  fn: number;
  tn: number;
  precision: number;
  recall: number;
  f1: number;
  expected_cost: number;
  alerts_per_day: number;
}

export interface CurvePoint {
  threshold: number;
  cost: number;
  alerts: number;
  alerts_per_day: number;
  recall: number;
  precision: number;
}

export type ConstraintType = "none" | "min_recall" | "max_alerts_per_day";

export interface OptimizeResponse {
  optimal_threshold: number;
  unconstrained_threshold: number;
  constraint_binding: boolean;
  constraint_satisfied: boolean;
  selected_on: "out_of_fold";
  cost_fn: number;
  cost_fp: number;
  metrics_at_optimal: ThresholdMetricsOut;
  metrics_at_optimal_oof: ThresholdMetricsOut;
  curve: CurvePoint[];
  curve_source: "out_of_fold";
}

// ---------------------------------------------------------------- API-02 … API-05

export interface ScoreResponse {
  transaction_id: string | null;
  risk_score: number;
  threshold: number;
  decision: Decision;
  risk_band: RiskBand;
  model_version: string;
  scored_at: string;
  latency_ms: number;
}

export interface UploadResponse {
  batch_id: string;
  count: number;
  alerts: number;
  threshold: number;
  elapsed_ms: number;
  model_version: string;
  rows_read: number;
  rows_rejected: number;
  rejection_reasons: { row: number; reason: string }[];
  rejection_reasons_truncated: boolean;
  has_labels: boolean;
  actual_metrics: { precision: number | null; recall: number | null; pr_auc: number | null } | null;
  results_truncated: boolean;
}

export interface Contribution {
  feature: string;
  value: number;
  shap: number;
}

export interface Explanation {
  transaction_id: string | null;
  risk_score: number;
  base_value: number;
  margin: number;
  shap_output: "log-odds";
  top_positive: Contribution[];
  top_negative: Contribution[];
  remaining_shap: number;
  contributions: Contribution[];
  model_version: string;
  elapsed_ms: number;
}

// ---------------------------------------------------------------- API-06 … API-08

export interface TransactionItem {
  id: string;
  risk_score: number;
  risk_band: RiskBand;
  decision: Decision;
  amount: number;
  hour: number;
  true_label: number | null;
  reviewed: boolean;
  review_decision: ReviewDecision | null;
  source: Source;
  batch_id: string | null;
  model_version: string;
  created_at: string;
}

export interface TransactionPage {
  items: TransactionItem[];
  page: number;
  page_size: number;
  total: number;
  threshold: number;
}

export interface ReviewInfo {
  decision: ReviewDecision;
  threshold_used: number;
  reviewed_at: string;
  note: string | null;
}

export interface TransactionDetail {
  id: string;
  features: Record<string, number>;
  risk_score: number;
  risk_band: RiskBand;
  decision: Decision;
  threshold: number;
  amount: number;
  amount_percentile: number;
  hour: number;
  true_label: number | null;
  review: ReviewInfo | null;
  source: Source;
  batch_id: string | null;
  model_version: string;
  model_version_current: boolean;
  created_at: string;
}

export interface ReviewResponse {
  transaction_id: string;
  decision: ReviewDecision;
  threshold_used: number;
  reviewed_at: string;
  note: string | null;
  true_label: number | null;
  matches_label: boolean | null;
}

// ---------------------------------------------------------------- API-13, API-14

export interface SampleItem {
  id: string;
  category: SampleCategory;
  label: number;
  risk_score: number;
  amount: number;
  description: string;
  features: Record<string, number>;
}

export interface SampleList {
  items: SampleItem[];
  categories: Record<SampleCategory, { count: number; rule: string }>;
  model_version: string;
  reference_threshold: number;
}

export interface Dataset {
  n_test: number;
  n_fraud_test: number;
  test_fraction: number;
  days: number;
}

export interface Interval {
  value: number;
  ci_low: number;
  ci_high: number;
}

export interface Headline {
  pr_auc: Interval;
  roc_auc: Interval;
  recall: Interval;
  precision: Interval;
  f1: { value: number };
  baseline_pr_auc: number;
  n_boot: number;
}

export interface GridRow {
  model: string;
  strategy: string;
  pr_auc_mean: number;
  pr_auc_std: number;
  roc_auc: number;
  recall: number;
  precision: number;
}

export interface StrategyCurve {
  strategy: string;
  pr_auc_mean: number;
  pr_auc_std: number;
  recall: number[];
  precision: number[];
}

export interface StrategyCurves {
  model: string;
  evaluated_on: string;
  baseline: number;
  curves: StrategyCurve[];
}

export interface BaselineRow {
  model: string;
  accuracy: number;
  pr_auc: number;
  pr_auc_ci_low: number;
  pr_auc_ci_high: number;
  threshold: number;
  tp: number;
  fp: number;
  fn: number;
  tn: number;
}

export interface ShapRow {
  feature: string;
  mean_abs_shap: number;
  rank_shap: number;
  rank_cohens_d: number | null;
  rank_cliffs_delta: number | null;
  rank_pearson: number | null;
  rank_lr: number | null;
}

export interface SplitRow {
  n_train: number;
  n_test: number;
  n_fraud_test: number;
  pr_auc_oof: number;
  pr_auc: number;
  pr_auc_ci_low: number;
  pr_auc_ci_high: number;
  recall: number;
  precision: number;
}

export interface Training {
  model: string;
  strategy: string;
}

export interface PerformanceData {
  headline: Headline;
  grid_results: GridRow[];
  strategy_pr_curves: StrategyCurves | null;
  baseline_comparison: BaselineRow[];
  shap_global: ShapRow[];
  split_comparison: Record<string, SplitRow>;
  pr_curve: { recall: number[]; precision: number[] };
  roc_curve: { fpr: number[]; tpr: number[] };
  training: Training;
  dataset: Dataset;
}

// ---------------------------------------------------------------- API-15

export interface ReplayEvent {
  id: string;
  risk_score: number;
  risk_band: RiskBand;
  amount: number;
  sim_time: string;
  sim_seconds: number;
}
