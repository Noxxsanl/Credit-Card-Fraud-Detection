"use client";

/*
 * Theo dõi vận hành (API-16, docs/11 M2): tỷ lệ cảnh báo, precision theo kết luận thẩm định và recall
 * theo nhãn thật, theo từng khoảng thời gian của dữ liệu, so với kỳ vọng ước lượng trên điểm chọn
 * ngưỡng. Ngưỡng chọn một lần rồi để yên sẽ trôi — khung này là chỗ thấy điều đó.
 */

import { useCallback, useEffect, useState } from "react";

import { useApp } from "@/context/AppContext";
import { api } from "@/lib/api";
import { fmt } from "@/lib/format";
import type { MonitoringBucket, MonitoringReport } from "@/lib/types";

import { Card } from "./ui";

const FLAG_TEXT: Record<string, string> = {
  alerts_above_expected: "cảnh báo nhiều hơn kỳ vọng",
  precision_below_expected: "precision thẩm định thấp hơn kỳ vọng",
  recall_below_expected: "recall thấp hơn kỳ vọng",
  over_budget: "vượt ngân sách thẩm định",
};

const ratio = (v: number | null) => (v == null ? "—" : fmt.pct(v, 1));

export function MonitoringCard() {
  const { threshold, queueVersion } = useApp();
  const [report, setReport] = useState<MonitoringReport | null>(null);
  const [error, setError] = useState<string | null>(null);

  const load = useCallback(() => {
    api<MonitoringReport>("/monitoring?bucket_hours=6").then(
      (r) => {
        setReport(r);
        setError(null);
      },
      (e: Error) => setError(e.message),
    );
  }, []);

  // Tải lại khi ngưỡng áp dụng đổi hoặc hàng đợi có dữ liệu mới (tải CSV, nạp mẫu, phát lại)
  useEffect(load, [load, threshold?.current, queueVersion]);

  return (
    <Card
      title="Theo dõi vận hành"
      sub="Giao dịch đã chấm trong hệ thống, theo từng khoảng 6 giờ của dữ liệu, so với kỳ vọng ước lượng trên điểm chọn ngưỡng. Khoảng nào lệch quá dao động mẫu (KTC Wilson 95%) thì được đánh dấu — dấu hiệu ngưỡng cần chỉnh lại."
    >
      {error && <p className="text-[12.5px] font-semibold text-critical-text">Không tải được: {error}</p>}
      {report && report.overall.n_scored === 0 && (
        <p className="text-muted">Chưa có giao dịch nào của mô hình đang chạy — tải CSV, nạp mẫu hoặc phát lại ngày 2.</p>
      )}
      {report && report.overall.n_scored > 0 && (
        <>
          <p className="mb-2 text-[12.5px] text-ink-2 tabular-nums">
            Kỳ vọng ở τ = {fmt.tau(report.threshold)}: {ratio(report.expected.alert_rate)} giao dịch bị cảnh báo
            (≈ {fmt.int(report.expected.alerts_per_day)}/ngày), precision {ratio(report.expected.precision)}, recall{" "}
            {ratio(report.expected.recall)}
            {report.alert_budget_per_day != null && <> · ngân sách thẩm định {fmt.int(report.alert_budget_per_day)}/ngày</>}.
          </p>
          <div className="overflow-x-auto">
            <table className="data-table compact">
              <thead>
                <tr>
                  <th scope="col">Khoảng (giờ của dữ liệu)</th>
                  <th scope="col" className="num">Đã chấm</th>
                  <th scope="col" className="num">Cảnh báo</th>
                  <th scope="col" className="num">Quy ra/ngày</th>
                  <th scope="col" className="num">Precision thẩm định</th>
                  <th scope="col" className="num">Recall theo nhãn</th>
                  <th scope="col">Đánh dấu</th>
                </tr>
              </thead>
              <tbody>
                {[...report.buckets, report.overall].map((b, i) => (
                  <MonitoringRow key={b.start_hour ?? `total-${i}`} bucket={b} />
                ))}
              </tbody>
            </table>
          </div>
        </>
      )}
    </Card>
  );
}

function MonitoringRow({ bucket: b }: { bucket: MonitoringBucket }) {
  const label = b.start_hour == null ? "Tổng" : `${fmt.int(b.start_hour)}–${fmt.int(b.end_hour ?? 0)} h`;
  return (
    <tr className={b.start_hour == null ? "font-semibold" : ""}>
      <td>{label}</td>
      <td className="num">{fmt.int(b.n_scored)}</td>
      <td className="num">
        {fmt.int(b.n_alerts)} ({ratio(b.alert_rate)})
      </td>
      <td className="num">{fmt.int(b.alerts_per_day)}</td>
      <td className="num">
        {b.n_reviewed ? `${ratio(b.precision_reviewed)} (${b.n_confirmed}/${b.n_reviewed})` : "—"}
      </td>
      <td className="num">{b.n_labelled_fraud ? `${ratio(b.recall_labelled)} (${b.n_caught}/${b.n_labelled_fraud})` : "—"}</td>
      <td className={b.flags.length ? "text-critical-text" : "text-muted"}>
        {b.flags.length ? b.flags.map((f) => FLAG_TEXT[f] ?? f).join("; ") : "trong kỳ vọng"}
      </td>
    </tr>
  );
}
