/** Kiểu cho threshold.mjs — xem chú thích ở tệp đó. */

export interface Prepared {
  all: Float64Array;
  positives: Float64Array;
  n: number;
  nPos: number;
}

export interface Counts {
  tp: number;
  fp: number;
  fn: number;
  tn: number;
  alerts: number;
}

export interface ThresholdMetrics extends Counts {
  threshold: number;
  precision: number;
  recall: number;
  f1: number;
  expected_cost: number;
  alerts_per_day: number;
}

export interface CostOptions {
  costFn: number;
  costFp: number;
  sampleFraction: number;
  days: number;
}

export declare function lowerBound(sorted: ArrayLike<number>, x: number): number;
export declare function prepare(yTrue: ArrayLike<number>, yScore: ArrayLike<number>): Prepared;
export declare function counts(prepared: Prepared, tau: number): Counts;
export declare function metricsAt(prepared: Prepared, tau: number, options: CostOptions): ThresholdMetrics;
