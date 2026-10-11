/*
 * Hai bộ điểm cho UI-D1 (docs/07 §2), mỗi bộ một vai trò (docs/11 M3):
 *
 * - **Điểm chọn ngưỡng** — `GET /threshold/oof-scores`: điểm out-of-fold của tập huấn luyện (forward-
 *   chaining theo thời gian). Màn hình Ngưỡng tính mọi con số trên bộ này: kéo thanh trượt là đang
 *   CHỌN ngưỡng, nên không được nhìn tập kiểm thử (ML-08).
 * - **Điểm tập kiểm thử** — `GET /metrics?section=test_scores`: chỉ để BÁO CÁO ngưỡng đang áp dụng
 *   (dòng tóm tắt của hàng đợi, ma trận ở màn hình Hiệu năng).
 *
 * Mỗi bộ tải MỘT lần, sắp sẵn (threshold.mjs), giữ trong state của AppContext. Mọi lần kéo thanh
 * trượt sau đó chỉ là một lần tìm nhị phân — không gọi mạng, không chạy mô hình.
 */

import { decodeSortedScores, metricsAt, prepare, type Prepared, type ThresholdMetrics } from "./threshold.mjs";
import type { Dataset } from "./types";

export interface ScoreSet {
  prepared: Prepared;
  /** Phần luồng giao dịch mà bộ điểm đại diện, và số ngày nó trải qua — để quy ra mỗi ngày */
  sampleFraction: number;
  days: number;
  source: "test" | "selection";
}

/** @deprecated tên cũ — dùng ScoreSet */
export type TestScores = ScoreSet;

export function loadTestScores(yTrue: number[], yScore: number[], dataset: Dataset): ScoreSet {
  return { prepared: prepare(yTrue, yScore), sampleFraction: dataset.test_fraction, days: dataset.days, source: "test" };
}

export function loadSelectionScores(buffer: ArrayBuffer): ScoreSet {
  const decoded = decodeSortedScores(buffer);
  return { prepared: decoded.prepared, sampleFraction: decoded.sampleFraction, days: decoded.days, source: "selection" };
}

export function metrics(scores: ScoreSet, tau: number, costFn: number, costFp: number): ThresholdMetrics {
  return metricsAt(scores.prepared, tau, { costFn, costFp, sampleFraction: scores.sampleFraction, days: scores.days });
}

/** Quy một đại lượng đếm trên bộ điểm ra mỗi ngày trên toàn luồng giao dịch. */
export function perDay(scores: ScoreSet, value: number): number {
  return value / scores.sampleFraction / scores.days;
}

export type { ThresholdMetrics };
