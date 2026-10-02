/*
 * Điểm của tập kiểm thử — UI-D1 (docs/07 §2).
 *
 * Tải `GET /metrics?section=test_scores` MỘT lần, sắp sẵn (threshold.mjs → prepare), giữ trong
 * state của AppContext. Mọi lần kéo thanh trượt sau đó chỉ là một lần tìm nhị phân — không gọi
 * mạng, không chạy mô hình. Các hàm ở đây thuần: cùng đầu vào, cùng kết quả.
 */

import { metricsAt, prepare, type Prepared, type ThresholdMetrics } from "./threshold.mjs";
import type { Dataset } from "./types";

export interface TestScores {
  prepared: Prepared;
  dataset: Dataset;
}

export function loadTestScores(yTrue: number[], yScore: number[], dataset: Dataset): TestScores {
  return { prepared: prepare(yTrue, yScore), dataset };
}

export function metrics(scores: TestScores, tau: number, costFn: number, costFp: number): ThresholdMetrics {
  return metricsAt(scores.prepared, tau, {
    costFn,
    costFp,
    sampleFraction: scores.dataset.test_fraction,
    days: scores.dataset.days,
  });
}

/** Quy đổi một đại lượng đếm trên tập kiểm thử (20% của hai ngày) ra mỗi ngày trên toàn luồng. */
export function perDay(scores: TestScores, value: number): number {
  return value / scores.dataset.test_fraction / scores.dataset.days;
}

export type { ThresholdMetrics };
