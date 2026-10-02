"use client";

/*
 * UI-04 — Hiệu năng mô hình (docs/07 §6). Đủ 7 mục bắt buộc, thêm đường PR/ROC của mô hình
 * xuất (FR-42). Các mục của metrics.json tải một lần rồi giữ trong bộ nhớ — quay lại màn hình
 * không tải lại. Ma trận nhầm lẫn tính trong trình duyệt tại ngưỡng hiện hành (UI-D1).
 */

import { useEffect, useMemo, useState } from "react";

import { useApp } from "@/context/AppContext";
import { api } from "@/lib/api";
import { COLORS, STRATEGY_COLORS } from "@/lib/colors";
import { fmt } from "@/lib/format";
import { MODEL, modelLabel, SPLIT, STRATEGY } from "@/lib/labels";
import type { PerformanceData } from "@/lib/types";

import { UnitCurveChart } from "../charts/UnitCurveChart";
import { Card, Tile } from "../ui";

const SECTIONS = ["headline", "grid_results", "strategy_pr_curves", "baseline_comparison", "shap_global",
  "split_comparison", "pr_curve", "roc_curve", "training", "dataset"] as const;
const STRATEGY_ORDER = ["class_weight", "smote", "smote_tomek", "none", "undersample"];

let cache: PerformanceData | null = null;

async function loadPerformance(): Promise<PerformanceData> {
  if (cache) return cache;
  const parts = await Promise.all(SECTIONS.map((name) => api<Record<string, unknown>>(`/metrics?section=${name}`)));
  const data = Object.fromEntries(SECTIONS.map((name, i) => [name, parts[i][name]])) as unknown as PerformanceData;
  cache = data;
  return data;
}

const pct = (v: number) => `${Math.min(Math.max(v, 0), 1) * 100}%`;
const rank = (v: number | null | undefined) => (v == null || Number.isNaN(v) ? "—" : fmt.int(v));

export function PerformanceScreen() {
  const { summary, threshold } = useApp();
  const [data, setData] = useState<PerformanceData | null>(cache);
  const [error, setError] = useState<string | null>(null);
  const [attempt, setAttempt] = useState(0);

  useEffect(() => {
    if (data) return;
    let cancelled = false;
    loadPerformance()
      .then((d) => !cancelled && setData(d))
      .catch((e) => !cancelled && setError((e as Error).message));
    return () => {
      cancelled = true;
    };
  }, [data, attempt]);

  const grid = useMemo(() => {
    if (!data) return [];
    return [...data.grid_results]
      .sort((a, b) => b.pr_auc_mean - a.pr_auc_mean)
      .map((r, i) => ({ ...r, rank: i + 1, exported: r.model === data.training.model && r.strategy === data.training.strategy }));
  }, [data]);

  const strategies = useMemo(() => {
    const curves = data?.strategy_pr_curves?.curves ?? [];
    // Màu đi theo chiến lược, không theo thứ hạng — lọc hay sắp lại không đổi màu
    return [...curves]
      .sort((a, b) => STRATEGY_ORDER.indexOf(a.strategy) - STRATEGY_ORDER.indexOf(b.strategy))
      .map((c) => ({ ...c, label: STRATEGY[c.strategy] ?? c.strategy, color: STRATEGY_COLORS[c.strategy] ?? COLORS.ink3 }));
  }, [data]);

  const shapRows = useMemo(() => (data ? [...data.shap_global].sort((a, b) => a.rank_shap - b.rank_shap) : []), [data]);

  if (!data) {
    return (
      <div className="rounded-[10px] border border-dashed border-line-strong bg-surface p-7 text-center">
        {error ? (
          <div role="alert">
            <p>Không tải được chỉ số: {error}</p>
            <button type="button" className="btn mt-2" onClick={() => { setError(null); setAttempt((n) => n + 1); }}>Thử lại</button>
          </div>
        ) : (
          <p>Đang tải metrics.json…</p>
        )}
      </div>
    );
  }

  const h = data.headline;
  const ciRows = [
    { key: "pr_auc", label: "PR-AUC", ...h.pr_auc },
    { key: "roc_auc", label: "ROC-AUC", ...h.roc_auc },
    { key: "recall", label: "Recall tại τ*", ...h.recall },
    { key: "precision", label: "Precision tại τ*", ...h.precision },
  ];
  const modelCiRows = data.baseline_comparison.map((b) => ({
    key: b.model, label: modelLabel(b.model), value: b.pr_auc, ci_low: b.pr_auc_ci_low, ci_high: b.pr_auc_ci_high,
  }));

  return (
    <div className="grid gap-4">
      <div className="grid grid-cols-2 gap-3 xl:grid-cols-4">
        {ciRows.map((r) => (
          <Tile key={r.key} label={r.label} value={fmt.num(r.value, 3)} sub={`KTC 95%: ${fmt.num(r.ci_low, 3)} – ${fmt.num(r.ci_high, 3)}`} />
        ))}
      </div>
      <p className="-mt-1 text-[12.5px] text-muted">
        Tập kiểm thử: {fmt.int(data.dataset.n_test)} giao dịch, {fmt.int(data.dataset.n_fraud_test)} gian lận. Khoảng tin cậy từ {fmt.int(h.n_boot)} lần
        bootstrap phân tầng — với chưa tới 100 mẫu dương, khoảng rất rộng.
      </p>

      {/* 7 — Khoảng tin cậy dạng thanh sai số */}
      <div className="grid gap-4 lg:grid-cols-2">
        <Card title="Khoảng tin cậy bootstrap 95%" sub="Mô hình xuất trên tập kiểm thử. Chấm là giá trị đo được, thanh là khoảng tin cậy.">
          <CiChart rows={ciRows} label="Chỉ số kèm khoảng tin cậy" showRange />
        </Card>
        <Card title="PR-AUC theo mô hình, kèm khoảng tin cậy" sub="Khoảng của hồi quy logistic và cây quyết định chồng lên nhau: 95 mẫu dương không đủ để tách chúng.">
          <CiChart rows={modelCiRows} label="PR-AUC theo mô hình kèm khoảng tin cậy" />
        </Card>
      </div>

      {/* 1 — Bảng 20 tổ hợp */}
      <Card title="20 tổ hợp mô hình × chiến lược xử lý mất cân bằng" sub="Kiểm định chéo 5 fold phân tầng trên tập huấn luyện, sắp theo PR-AUC. Dòng tô đậm là mô hình đang phục vụ.">
        <div className="overflow-x-auto">
          <table className="data-table">
            <thead>
              <tr>
                <th scope="col" className="num">#</th>
                <th scope="col">Mô hình</th>
                <th scope="col">Chiến lược</th>
                <th scope="col" className="num">PR-AUC</th>
                <th scope="col" className="num">± độ lệch chuẩn</th>
                <th scope="col" className="w-[120px]"><span className="sr-only">PR-AUC dạng thanh</span></th>
                <th scope="col" className="num">ROC-AUC</th>
                <th scope="col" className="num">Recall</th>
                <th scope="col" className="num">Precision</th>
              </tr>
            </thead>
            <tbody>
              {grid.map((r) => (
                <tr key={r.model + r.strategy} className={r.exported ? "[&>td]:bg-accent-soft [&>td]:font-semibold" : ""}>
                  <td className="num">{r.rank}</td>
                  <td>
                    {MODEL[r.model] ?? r.model}
                    {r.exported && <span className="ml-2 rounded-full bg-accent-soft px-2 text-[11px] font-bold text-accent-strong">đang phục vụ</span>}
                  </td>
                  <td>{STRATEGY[r.strategy] ?? r.strategy}</td>
                  <td className="num">{fmt.num(r.pr_auc_mean, 3)}</td>
                  <td className="num text-muted">± {fmt.num(r.pr_auc_std, 3)}</td>
                  <td aria-hidden="true"><InlineBar value={r.pr_auc_mean} /></td>
                  <td className="num">{fmt.num(r.roc_auc, 3)}</td>
                  <td className="num">{fmt.num(r.recall, 2)}</td>
                  <td className="num">{fmt.num(r.precision, 2)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Card>

      <div className="grid gap-4 lg:grid-cols-2">
        {/* 2 — Đường PR của 5 chiến lược */}
        <Card title="Đường Precision–Recall của 5 chiến lược (XGBoost)" sub="Out-of-fold trên tập huấn luyện. Đường xám nằm ngang là đường cơ sở — tỷ lệ gian lận.">
          {data.strategy_pr_curves ? (
            <>
              <UnitCurveChart
                series={strategies.map((c) => ({ key: c.strategy, label: c.label, color: c.color, x: c.recall, y: c.precision }))}
                baseline={data.strategy_pr_curves.baseline}
                xLabel="Recall"
                yLabel="Precision"
                height={330}
                legend
                label="Đường precision–recall của XGBoost với năm chiến lược xử lý mất cân bằng; giá trị PR-AUC ở bảng ngay dưới."
              />
              <table className="data-table compact mt-3">
                <caption className="sr-only">PR-AUC của từng chiến lược</caption>
                <thead>
                  <tr><th scope="col">Chiến lược</th><th scope="col" className="num">PR-AUC</th><th scope="col" className="num">± độ lệch chuẩn</th></tr>
                </thead>
                <tbody>
                  {strategies.map((c) => (
                    <tr key={c.strategy}>
                      <td>
                        <span aria-hidden="true" className="mr-2 inline-block h-[3px] w-4 rounded-sm align-middle" style={{ background: c.color }} />
                        {c.label}
                      </td>
                      <td className="num">{fmt.num(c.pr_auc_mean, 3)}</td>
                      <td className="num text-muted">± {fmt.num(c.pr_auc_std, 3)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
              <p className="mt-2 text-[12.5px] text-muted">Đường cơ sở {fmt.num(data.strategy_pr_curves.baseline, 5)}.</p>
            </>
          ) : (
            <p className="text-muted">metrics.json không có đường PR của các chiến lược (thiếu reports/grid_results.npz khi xuất).</p>
          )}
        </Card>

        {/* 3 — Ma trận nhầm lẫn tại ngưỡng hiện hành */}
        <Card
          title="Ma trận nhầm lẫn tại ngưỡng hiện hành"
          sub={<>Tập kiểm thử, τ = <span className="tabular-nums">{fmt.tau(threshold?.current)}</span>. Tính trong trình duyệt, cập nhật ngay khi áp dụng ngưỡng mới.</>}
        >
          {summary && (
            <>
              <table className="-mx-1.5 mt-1 w-full border-separate border-spacing-1.5 tabular-nums">
                <thead>
                  <tr>
                    <td />
                    <th scope="col" className="text-[12.5px] font-bold text-ink-2">Cảnh báo</th>
                    <th scope="col" className="text-[12.5px] font-bold text-ink-2">Cho qua</th>
                  </tr>
                </thead>
                <tbody>
                  <tr>
                    <th scope="row" className="pr-1.5 text-right text-[12.5px] font-bold text-ink-2">Thực tế gian lận</th>
                    <ConfusionCell good value={summary.tp} label="bắt được (TP)" />
                    <ConfusionCell value={summary.fn} label="bỏ lọt (FN)" />
                  </tr>
                  <tr>
                    <th scope="row" className="pr-1.5 text-right text-[12.5px] font-bold text-ink-2">Thực tế hợp lệ</th>
                    <ConfusionCell value={summary.fp} label="cảnh báo giả (FP)" />
                    <ConfusionCell good value={summary.tn} label="cho qua đúng (TN)" />
                  </tr>
                </tbody>
              </table>
              <p className="mt-2 text-[12.5px] text-muted">
                Recall {fmt.pct(summary.recall)} · precision {fmt.pct(summary.precision)} · chi phí kỳ vọng {fmt.eur(summary.expected_cost)}.
              </p>
            </>
          )}
          <h3 className="mb-1 mt-4 text-sm font-bold">Mô hình xuất trên tập kiểm thử</h3>
          <div className="grid gap-3 sm:grid-cols-2">
            <figure>
              <figcaption className="mb-1 text-[12.5px] font-semibold text-ink-2">Precision–Recall</figcaption>
              <UnitCurveChart
                series={[{ key: "pr", label: "Mô hình xuất", color: COLORS.series1, x: data.pr_curve.recall, y: data.pr_curve.precision }]}
                baseline={h.baseline_pr_auc}
                xLabel="Recall"
                yLabel="Precision"
                height={220}
                label={`Đường precision–recall của mô hình xuất trên tập kiểm thử, PR-AUC ${fmt.num(h.pr_auc.value, 3)}`}
              />
            </figure>
            <figure>
              <figcaption className="mb-1 text-[12.5px] font-semibold text-ink-2">ROC</figcaption>
              <UnitCurveChart
                series={[{ key: "roc", label: "Mô hình xuất", color: COLORS.series1, x: data.roc_curve.fpr, y: data.roc_curve.tpr }]}
                diagonal
                xLabel="FPR"
                yLabel="TPR"
                height={220}
                label={`Đường ROC của mô hình xuất trên tập kiểm thử, ROC-AUC ${fmt.num(h.roc_auc.value, 3)}`}
              />
            </figure>
          </div>
        </Card>
      </div>

      {/* 4 — Accuracy không phân biệt được mô hình, PR-AUC thì có (G-2) */}
      <Card
        title="Accuracy gần như không phân biệt được các mô hình — PR-AUC thì có"
        sub={'Tập kiểm thử. Mô hình rỗng luôn đoán "hợp lệ" vẫn đạt accuracy 99,83% mà không bắt được gian lận nào.'}
      >
        <div className="overflow-x-auto">
          <table className="data-table">
            <thead>
              <tr>
                <th scope="col">Mô hình</th>
                <th scope="col" className="num">Accuracy</th>
                <th scope="col" className="w-[120px]"><span className="sr-only">Accuracy dạng thanh</span></th>
                <th scope="col" className="num">PR-AUC</th>
                <th scope="col" className="w-[120px]"><span className="sr-only">PR-AUC dạng thanh</span></th>
                <th scope="col" className="num">Ngưỡng</th>
                <th scope="col" className="num">Bắt được</th>
                <th scope="col" className="num">Cảnh báo giả</th>
              </tr>
            </thead>
            <tbody>
              {data.baseline_comparison.map((b) => (
                <tr key={b.model}>
                  <td>{modelLabel(b.model)}</td>
                  <td className="num">{fmt.pct(b.accuracy, 2)}</td>
                  <td aria-hidden="true"><InlineBar value={b.accuracy} /></td>
                  <td className="num">{fmt.num(b.pr_auc, 3)}</td>
                  <td aria-hidden="true"><InlineBar value={b.pr_auc} /></td>
                  <td className="num">{fmt.tau(b.threshold)}</td>
                  <td className="num">{b.tp}/{b.tp + b.fn}</td>
                  <td className="num">{fmt.int(b.fp)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Card>

      <div className="grid gap-4 lg:grid-cols-2">
        {/* 5 — SHAP cạnh kiểm định thống kê */}
        <Card title="Xếp hạng đặc trưng: SHAP đặt cạnh kiểm định thống kê" sub={'Hạng 1 là quan trọng nhất. Cột thống kê lấy từ notebook 02; "—" là đặc trưng không có trong phép kiểm định đó.'}>
          <div className="max-h-[420px] overflow-auto">
            <table className="data-table compact">
              <thead>
                <tr>
                  <th scope="col">Đặc trưng</th>
                  <th scope="col" className="num">Hạng SHAP</th>
                  <th scope="col" className="num">Mean |SHAP|</th>
                  <th scope="col" className="num">Cohen&apos;s d</th>
                  <th scope="col" className="num">Cliff&apos;s δ</th>
                  <th scope="col" className="num">Pearson</th>
                  <th scope="col" className="num">Logistic</th>
                </tr>
              </thead>
              <tbody>
                {shapRows.map((r) => (
                  <tr key={r.feature}>
                    <td className="font-mono">{r.feature}</td>
                    <td className="num">{rank(r.rank_shap)}</td>
                    <td className="num">{fmt.num(r.mean_abs_shap, 3)}</td>
                    <td className="num">{rank(r.rank_cohens_d)}</td>
                    <td className="num">{rank(r.rank_cliffs_delta)}</td>
                    <td className="num">{rank(r.rank_pearson)}</td>
                    <td className="num">{rank(r.rank_lr)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <p className="mt-2.5 text-[12.5px] text-muted">
            V1–V28 là thành phần chính sau PCA: bảng cho biết đặc trưng nào ảnh hưởng tới dự đoán, không cho biết nguyên nhân nghiệp vụ.
          </p>
        </Card>

        {/* 6 — Hai cách chia tập */}
        <Card title="Đối chiếu cách chia tập" sub="Chia theo thời gian gần với triển khai thật hơn; PR-AUC giảm là bằng chứng mô hình chịu ảnh hưởng của trôi dạt giữa hai ngày.">
          <div className="grid gap-3">
            {Object.entries(data.split_comparison).map(([key, r]) => (
              <div key={key} className="rounded-md border border-line px-3.5 py-3">
                <h3 className="text-[13.5px] font-bold">{SPLIT[key] ?? key}</h3>
                <dl className="mt-1.5 grid gap-1 tabular-nums">
                  <SplitFact label="Huấn luyện / kiểm thử" value={`${fmt.int(r.n_train)} / ${fmt.int(r.n_test)}`} />
                  <SplitFact label="Gian lận trong tập kiểm thử" value={fmt.int(r.n_fraud_test)} />
                  <SplitFact label="PR-AUC out-of-fold" value={fmt.num(r.pr_auc_oof, 3)} />
                  <SplitFact label="PR-AUC kiểm thử (KTC 95%)" value={`${fmt.num(r.pr_auc, 3)} (${fmt.num(r.pr_auc_ci_low, 3)}–${fmt.num(r.pr_auc_ci_high, 3)})`} />
                  <SplitFact label="Recall / precision tại ngưỡng chọn trên OOF" value={`${fmt.num(r.recall, 2)} / ${fmt.num(r.precision, 2)}`} />
                </dl>
              </div>
            ))}
          </div>
        </Card>
      </div>
    </div>
  );
}

function InlineBar({ value }: { value: number }) {
  return <span className="block h-2 min-w-0.5 rounded-r bg-series-1" style={{ width: pct(value) }} />;
}

function ConfusionCell({ value, label, good = false }: { value: number; label: string; good?: boolean }) {
  return (
    <td className={`rounded-md p-3 text-center ${good ? "bg-good-soft" : "bg-critical-soft"}`}>
      <strong className="block text-2xl">{fmt.int(value)}</strong>
      <small className="text-ink-2">{label}</small>
    </td>
  );
}

function SplitFact({ label, value }: { label: string; value: string }) {
  return (
    <div className="flex items-baseline justify-between gap-3">
      <dt className="text-xs text-muted">{label}</dt>
      <dd className="text-right font-semibold">{value}</dd>
    </div>
  );
}

/** Thanh sai số HTML: đọc được bằng trình đọc màn hình mà không cần bảng thay thế. */
function CiChart({ rows, label, showRange = false }: {
  rows: { key: string; label: string; value: number; ci_low: number; ci_high: number }[];
  label: string;
  showRange?: boolean;
}) {
  const grid = "grid grid-cols-[190px_minmax(0,1fr)_90px] items-center gap-3";
  return (
    <div role="table" aria-label={label} className="grid gap-1">
      {rows.map((r) => (
        <div key={r.key} role="row" className={`${grid} min-h-[30px]`}>
          <span role="rowheader" className="text-[13px] font-semibold">{r.label}</span>
          <span
            role="cell"
            aria-label={`${fmt.num(r.value, 3)}, từ ${fmt.num(r.ci_low, 3)} tới ${fmt.num(r.ci_high, 3)}`}
            className="relative h-[22px] border-r border-grid bg-[linear-gradient(to_right,var(--color-grid)_1px,transparent_1px)] bg-[length:50%_100%] bg-repeat-x"
          >
            <span className="absolute top-[9px] h-1 rounded-sm bg-series-1/45" style={{ left: pct(r.ci_low), width: `calc(${pct(r.ci_high)} - ${pct(r.ci_low)})` }} />
            <span className="absolute top-[5px] -ml-1.5 size-3 rounded-full bg-series-1 shadow-[0_0_0_2px_#fff]" style={{ left: pct(r.value) }} />
          </span>
          <span role="cell" className="text-right text-[12.5px] tabular-nums text-ink-2">
            {showRange ? `${fmt.num(r.ci_low, 2)}–${fmt.num(r.ci_high, 2)}` : fmt.num(r.value, 3)}
          </span>
        </div>
      ))}
      <div aria-hidden="true" className={`${grid} min-h-[18px] text-[11.5px] text-muted`}>
        <span />
        <span className="relative h-4 tabular-nums">
          <span className="absolute -translate-x-1/2" style={{ left: "0%" }}>0</span>
          <span className="absolute -translate-x-1/2" style={{ left: "50%" }}>0,5</span>
          <span className="absolute -translate-x-1/2" style={{ left: "100%" }}>1</span>
        </span>
        <span />
      </div>
    </div>
  );
}
