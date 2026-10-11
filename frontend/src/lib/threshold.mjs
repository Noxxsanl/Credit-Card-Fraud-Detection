/*
 * UI-D1 — chỉ số theo ngưỡng, tính ngay trong trình duyệt (docs/07 §2).
 *
 * Bản JavaScript của src/threshold.py: confusion_counts() và metrics_at_threshold(). Hai bên
 * phải cho cùng TP/FP/FN/TN với sai lệch 0 — tests/test_threshold_parity.py (TC-12) chạy chính
 * tệp này bằng Node rồi so với Python. Sửa một bên thì phải sửa bên kia.
 *
 * Viết bằng JavaScript thuần (ES module, đuôi .mjs) thay vì TypeScript để Node chạy được trực
 * tiếp trong TC-12, không cần bước biên dịch. Kiểu cho TypeScript nằm ở threshold.d.mts.
 *
 * Quy ước giống hệt phía máy chủ: dự đoán dương khi score >= τ, so bằng số thực IEEE 754
 * nguyên độ chính xác. Sắp điểm một lần lúc nạp, rồi mỗi ngưỡng chỉ là một lần tìm nhị phân
 * O(log n): với 56.746 điểm, một lần kéo thanh trượt tốn vài micro giây (NFR-03).
 */

/** Chỉ số đầu tiên i mà sorted[i] >= x (numpy.searchsorted side="left"). */
export function lowerBound(sorted, x) {
  let lo = 0;
  let hi = sorted.length;
  while (lo < hi) {
    const mid = (lo + hi) >>> 1;
    if (sorted[mid] < x) lo = mid + 1;
    else hi = mid;
  }
  return lo;
}

/** Sắp sẵn điểm của toàn tập và của riêng lớp dương. Gọi một lần sau khi tải test_scores. */
export function prepare(yTrue, yScore) {
  if (yTrue.length !== yScore.length) {
    throw new Error(`y_true và y_score khác độ dài: ${yTrue.length} vs ${yScore.length}`);
  }
  const all = Float64Array.from(yScore);
  let nPos = 0;
  for (let i = 0; i < yTrue.length; i++) if (yTrue[i] === 1) nPos++;
  const positives = new Float64Array(nPos);
  for (let i = 0, j = 0; i < yTrue.length; i++) if (yTrue[i] === 1) positives[j++] = yScore[i];
  all.sort(); // mảng có kiểu sắp theo giá trị số, không theo chuỗi
  positives.sort();
  return { all, positives, n: all.length, nPos };
}

/** Dựng `prepared` từ hai mảng ĐÃ SẮP tăng dần — mọi điểm và điểm của lớp dương. */
export function fromSorted(all, positives) {
  return { all: Float64Array.from(all), positives: Float64Array.from(positives), n: all.length, nPos: positives.length };
}

/**
 * Đọc payload nhị phân của GET /threshold/oof-scores (src/artifacts.py → sorted_scores_payload):
 * 32 byte đầu "FDSC", phiên bản, kiểu số (1 = float32, 2 = float64), số điểm, số điểm dương,
 * tỷ lệ mẫu và số ngày; rồi hai mảng đã sắp. Mỗi float32 nâng lên float64 đúng từng bit, nên
 * phép so với ngưỡng trùng phía Python (TC-12 kiểm cả đường này).
 */
export function decodeSortedScores(buffer) {
  const view = new DataView(buffer);
  const magic = String.fromCharCode(view.getUint8(0), view.getUint8(1), view.getUint8(2), view.getUint8(3));
  if (magic !== "FDSC" || view.getUint8(4) !== 1) throw new Error("Không phải payload điểm đã sắp phiên bản 1");
  const code = view.getUint8(5);
  const Kind = code === 1 ? Float32Array : code === 2 ? Float64Array : null;
  if (!Kind) throw new Error(`Kiểu số không hỗ trợ: ${code}`);
  const nAll = view.getUint32(8, true);
  const nPos = view.getUint32(12, true);
  const sampleFraction = view.getFloat64(16, true);
  const days = view.getFloat64(24, true);
  const all = new Kind(buffer, 32, nAll);
  const positives = new Kind(buffer, 32 + Kind.BYTES_PER_ELEMENT * nAll, nPos);
  return { prepared: fromSorted(all, positives), sampleFraction, days };
}

/** (tp, fp, fn, tn) tại ngưỡng τ — tương đương src.threshold.confusion_counts. */
export function counts(prepared, tau) {
  const alerts = prepared.n - lowerBound(prepared.all, tau);
  const tp = prepared.nPos - lowerBound(prepared.positives, tau);
  const fp = alerts - tp;
  const fn = prepared.nPos - tp;
  const tn = prepared.n - alerts - fn;
  return { tp, fp, fn, tn, alerts };
}

/**
 * Toàn bộ chỉ số tại τ — cùng tên trường và cùng thứ tự phép tính với
 * src.threshold.metrics_at_threshold, nên số thực cũng trùng từng bit.
 */
export function metricsAt(prepared, tau, { costFn, costFp, sampleFraction, days }) {
  const { tp, fp, fn, tn, alerts } = counts(prepared, tau);
  const precision = alerts ? tp / alerts : 0.0;
  const recall = tp + fn ? tp / (tp + fn) : 0.0;
  const f1 = precision + recall ? (2 * precision * recall) / (precision + recall) : 0.0;
  return {
    threshold: tau,
    tp,
    fp,
    fn,
    tn,
    alerts,
    precision,
    recall,
    f1,
    expected_cost: fn * costFn + fp * costFp,
    alerts_per_day: alerts / sampleFraction / days,
  };
}
