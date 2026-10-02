"use client";

/*
 * UI-03 — Cấu hình ngưỡng (docs/07 §5). Màn hình có giá trị học thuật cao nhất.
 *
 * Mọi con số ở các ô tính NGAY trong trình duyệt bằng lib/threshold.mjs (UI-D1), không gọi
 * mạng — đó là thứ bảo đảm AC-A3 (< 200 ms). Đường cong chi phí lấy từ POST /threshold/optimize,
 * đo trên out-of-fold (nơi ngưỡng được chọn), còn các ô đo trên tập kiểm thử; cả hai quy ra
 * EUR/ngày nên cùng thang. Biểu đồ vẽ lại ở mức ưu tiên thấp (useDeferredValue) để thanh trượt
 * và các ô không phải chờ nó.
 */

import { useDeferredValue, useMemo, useRef, useState, type ChangeEvent, type KeyboardEvent, type ReactNode } from "react";

import { useApp } from "@/context/AppContext";
import { useThresholdDraft } from "@/context/ThresholdDraftContext";
import { api } from "@/lib/api";
import { clamp, fmt, niceCeil } from "@/lib/format";
import { ALTERNATIVES } from "@/lib/labels";
import { metrics, perDay } from "@/lib/scores";
import type { ConstraintType, ThresholdState } from "@/lib/types";

import { CostCurveChart, type CostPoint } from "../charts/CostCurveChart";
import { Card, Tile } from "../ui";

const LOG_MIN = -4; // τ = 0,0001
const LOG_MAX = Math.log10(0.999);
const SCALE_TICKS = [0.0001, 0.001, 0.01, 0.1, 1];
const scalePos = (v: number) => ((Math.log10(v) - LOG_MIN) / (LOG_MAX - LOG_MIN)) * 100;

export function ThresholdScreen() {
  const app = useApp();
  const draft = useThresholdDraft();
  const { threshold, scores } = app;
  const { tau, costs, opt, constraintType } = draft;
  const [applying, setApplying] = useState(false);
  const [latencyMs, setLatencyMs] = useState<number | null>(null);
  const inputStarted = useRef(0);

  const costFn = costs.valid ? costs.fn : threshold?.cost_fn ?? 0;
  const costFp = costs.valid ? costs.fp : threshold?.cost_fp ?? 0;
  const current = threshold?.current ?? null;

  // Chỉ số tại τ đang xem và tại ngưỡng đang áp dụng — tìm nhị phân, vài micro giây
  const m = useMemo(() => (scores ? metrics(scores, tau, costFn, costFp) : null), [scores, tau, costFn, costFp]);
  const applied = useMemo(
    () => (scores && current !== null ? metrics(scores, current, costFn, costFp) : null),
    [scores, current, costFn, costFp],
  );

  // Đường cong: chi phí out-of-fold quy ra EUR/ngày, cùng thang với ô "Chi phí mỗi ngày"
  const curve = useMemo(() => {
    if (!opt) return null;
    const ref = opt.curve.find((p) => p.alerts > 0);
    const perDay = ref ? ref.alerts_per_day / ref.alerts : 0; // = 1 / (tỷ lệ mẫu × số ngày)
    const points: CostPoint[] = opt.curve
      .filter((p) => p.threshold >= 1e-4)
      .map((p) => ({ x: p.threshold, y: p.cost * perDay, apd: p.alerts_per_day, recall: p.recall }));
    // Hai đầu vọt lên hàng chục nghìn EUR/ngày và đè bẹp vùng đáy chữ U: cắt ở ~2,5 lần cực tiểu
    const ys = points.map((p) => p.y);
    const lowest = Math.min(...ys);
    return { points, yMax: niceCeil(lowest > 0 ? lowest * 2.5 : Math.max(...ys)) };
  }, [opt]);

  const chartTau = useDeferredValue(tau);

  const presets = useMemo(() => {
    if (!scores || !threshold) return [];
    const list: { key: string; label: string; hint: string; value: number }[] = ALTERNATIVES
      .filter((a) => a.key in threshold.alternatives)
      .map((a) => ({ key: a.key, label: a.label, hint: a.hint, value: threshold.alternatives[a.key] }));
    if (opt && opt.optimal_threshold !== threshold.alternatives.min_expected_cost) {
      list.push({
        key: "optimized",
        label: constraintType === "none" ? "Tối ưu với tham số chi phí đang nhập" : "Tối ưu với tham số và ràng buộc đang nhập",
        hint: `Chi phí bỏ lọt ${fmt.num(opt.cost_fn)} EUR, thẩm định ${fmt.num(opt.cost_fp)} EUR — chọn trên out-of-fold`,
        value: opt.optimal_threshold,
      });
    }
    if (threshold.source === "user" && current !== null && !list.some((p) => p.value === current)) {
      list.push({ key: "applied", label: "Ngưỡng đang áp dụng", hint: "Người dùng đã đặt", value: current });
    }
    return list.map((p) => {
      const r = metrics(scores, p.value, costFn, costFp);
      return { ...p, recall: r.recall, alertsPerDay: r.alerts_per_day, costPerDay: perDay(scores, r.expected_cost), tp: r.tp, nPos: r.tp + r.fn };
    });
  }, [scores, threshold, current, opt, constraintType, costFn, costFp]);

  if (!draft.initialized || !scores || !threshold || !m || !applied) {
    return (
      <div className="rounded-[10px] border border-dashed border-line-strong bg-surface p-7 text-center">
        {app.thresholdError ? `Không đọc được ngưỡng hiện hành: ${app.thresholdError}` : "Đang nạp điểm của tập kiểm thử…"}
      </div>
    );
  }

  const sliderValue = clamp(Math.log10(tau), LOG_MIN, LOG_MAX);
  const dirty = tau !== threshold.current || (costs.valid && (costs.fn !== threshold.cost_fn || costs.fp !== threshold.cost_fp));

  function measure(started: number) {
    requestAnimationFrame(() => requestAnimationFrame(() => setLatencyMs(performance.now() - started)));
  }

  /** Thanh trượt theo thang log10: vùng quyết định 0,0005…0,97 trải ba bậc độ lớn. */
  function onSlider(event: ChangeEvent<HTMLInputElement>) {
    inputStarted.current = event.timeStamp || performance.now();
    draft.setTau(10 ** Number(event.target.value));
    measure(inputStarted.current);
  }

  /** ←/→ một bước nhỏ, PgUp/PgDn một bước lớn, Home/End hai đầu (07 §5). */
  function onSliderKey(event: KeyboardEvent<HTMLInputElement>) {
    const steps: Record<string, number> = { ArrowLeft: -0.01, ArrowDown: -0.01, ArrowRight: 0.01, ArrowUp: 0.01, PageDown: -0.1, PageUp: 0.1 };
    let next: number | null = null;
    if (event.key in steps) next = sliderValue + steps[event.key];
    else if (event.key === "Home") next = LOG_MIN;
    else if (event.key === "End") next = LOG_MAX;
    if (next === null) return;
    event.preventDefault();
    const started = performance.now();
    draft.setTau(10 ** clamp(next, LOG_MIN, LOG_MAX));
    measure(started);
  }

  function onTauInput(event: ChangeEvent<HTMLInputElement>) {
    const value = Number(event.target.value.replace(",", "."));
    if (Number.isFinite(value) && value > 0 && value < 1) draft.setTau(value);
    else event.target.value = String(Number(tau.toPrecision(6)));
  }

  /** Chênh lệch so với ngưỡng đang áp dụng — thứ khiến kịch bản kéo 0,5 → τ* dễ thấy (07 §5). */
  const delta = (field: "alerts_per_day" | "tp" | "fp" | "expected_cost", goodIf: "up" | "down" | null, toPerDay = false, unit = "") => {
    if (tau === applied.threshold) return { text: null, cls: "" };
    const raw = m[field] - applied[field];
    const v = toPerDay ? perDay(scores, raw) : raw;
    if (Math.abs(v) < 0.5) return { text: "như ngưỡng đang áp dụng", cls: "" };
    const text = `${v > 0 ? "▲ +" : "▼ −"}${fmt.int(Math.abs(v))}${unit} so với ngưỡng đang áp dụng`;
    const cls = goodIf === null ? "" : (v > 0) === (goodIf === "up") ? "text-good-text" : "text-critical-text";
    return { text, cls };
  }

  async function apply() {
    if (!costs.valid || applying) return;
    setApplying(true);
    try {
      const state = await api<ThresholdState>("/threshold", { method: "PUT", json: { value: tau, cost_fn: costs.fn, cost_fp: costs.fp } });
      app.setThreshold(state);
      app.toast(`Đã áp dụng ngưỡng ${fmt.tau(state.current)} cho toàn ứng dụng`, "success");
    } catch (e) {
      app.toast(`Không áp dụng được: ${(e as Error).message}`, "error");
    } finally {
      setApplying(false);
    }
  }

  const dAlerts = delta("alerts_per_day", null);
  const dTp = delta("tp", "up");
  const dFp = delta("fp", "down");
  const dCost = delta("expected_cost", "down", true, " EUR");
  const curveLabel = opt
    ? `Đường cong chi phí mỗi ngày theo ngưỡng, trục hoành thang log. Cực tiểu tại τ = ${fmt.tau(opt.optimal_threshold)}; tại đó trên tập kiểm thử bắt ${opt.metrics_at_optimal.tp}/${opt.metrics_at_optimal.tp + opt.metrics_at_optimal.fn} gian lận với ${fmt.int(opt.metrics_at_optimal.alerts_per_day)} cảnh báo mỗi ngày.`
    : "Đường cong chi phí đang tính";

  return (
    <div className="grid gap-4">
      <Card className="pb-3">
        <div className="flex flex-wrap items-end justify-between gap-4">
          <div>
            <label htmlFor="tau-slider" className="block text-xs font-semibold uppercase tracking-wide text-muted">Ngưỡng đang xem</label>
            <div className="flex items-center gap-3.5">
              <span className="text-[34px] font-bold leading-tight tabular-nums">{fmt.tau(tau)}</span>
              <input
                key={tau}
                type="text"
                inputMode="decimal"
                autoComplete="off"
                defaultValue={String(Number(tau.toPrecision(6)))}
                onBlur={onTauInput}
                onKeyDown={(e) => e.key === "Enter" && (e.target as HTMLInputElement).blur()}
                aria-label="Nhập ngưỡng chính xác, trong khoảng 0 đến 1"
                className="field-input min-h-[30px]! w-[130px] px-2 py-1 text-[13px]"
              />
            </div>
          </div>
          <div className="flex gap-2">
            <button type="button" className="btn" disabled={tau === threshold.default} onClick={() => draft.setTau(threshold.default)}>
              Về τ*
            </button>
            <button type="button" className="btn btn-primary" disabled={!dirty || applying || !costs.valid} onClick={apply}>
              {applying ? "Đang áp dụng…" : "Áp dụng ngưỡng này"}
            </button>
          </div>
        </div>
        <input
          id="tau-slider"
          className="tau-slider"
          type="range"
          min={LOG_MIN}
          max={LOG_MAX}
          step={0.001}
          value={sliderValue}
          onChange={onSlider}
          onKeyDown={onSliderKey}
          aria-valuetext={`Ngưỡng ${fmt.tau(tau)}`}
        />
        <div aria-hidden="true" className="relative mx-[11px] h-9 text-xs tabular-nums text-muted">
          {SCALE_TICKS.map((v) => (
            <span key={v} className="absolute top-1 -translate-x-1/2 before:absolute before:-top-1.5 before:left-1/2 before:h-[5px] before:border-l before:border-axis" style={{ left: `${scalePos(v)}%` }}>
              {fmt.tauTick(v)}
            </span>
          ))}
          <span className="absolute top-[19px] -translate-x-1/2 whitespace-nowrap font-bold text-ink" style={{ left: `${scalePos(threshold.default)}%` }}>
            ▲ τ*
          </span>
        </div>
        <p className="mt-2 text-[12.5px] text-muted">
          Thanh trượt theo thang log — vùng quyết định trải từ 0,0005 tới 0,97. Bàn phím: ← → bước nhỏ, PgUp/PgDn bước lớn, Home/End hai đầu;
          hoặc gõ số vào ô bên cạnh. Mọi con số dưới đây tính ngay trong trình duyệt, không gọi mạng: từ lúc kéo tới lúc vẽ xong{" "}
          {latencyMs === null ? "—" : `${fmt.num(latencyMs, 0)} ms`} (yêu cầu dưới 200 ms).
        </p>
      </Card>

      <div aria-live="polite" className="grid grid-cols-2 gap-3 xl:grid-cols-4">
        <Tile label="Cảnh báo mỗi ngày" value={fmt.int(m.alerts_per_day)} sub={`${fmt.int(m.alerts)} cảnh báo trên tập kiểm thử`} delta={dAlerts.text} />
        <Tile
          label="Gian lận bắt được"
          value={
            <>
              {m.tp}
              <span className="text-[22px] font-semibold text-muted"> / {m.tp + m.fn}</span>
            </>
          }
          sub={`recall ${fmt.pct(m.recall)}`}
          delta={dTp.text}
          deltaClass={dTp.cls}
        />
        <Tile label="Precision" value={fmt.num(m.precision, 2)} sub={`${fmt.int(m.fp)} cảnh báo giả · F1 ${fmt.num(m.f1, 2)}`} delta={dFp.text} deltaClass={dFp.cls} />
        <Tile
          label="Chi phí mỗi ngày"
          value={
            <>
              {fmt.int(perDay(scores, m.expected_cost))} <span className="text-base font-semibold text-muted">EUR</span>
            </>
          }
          sub={`tập kiểm thử: ${fmt.eur(m.expected_cost)}`}
          delta={dCost.text}
          deltaClass={dCost.cls}
        />
      </div>

      <div className="grid gap-4 2xl:grid-cols-[minmax(300px,0.8fr)_minmax(0,1.4fr)] xl:grid-cols-[minmax(300px,0.8fr)_minmax(0,1.4fr)]">
        <CostForm />
        <Card
          title="Tổng chi phí theo ngưỡng"
          sub="Đường cong: out-of-fold của tập huấn luyện, nơi ngưỡng được chọn. Các ô số phía trên: tập kiểm thử. Cả hai quy ra EUR mỗi ngày trên toàn luồng giao dịch."
        >
          <div className={`transition-opacity ${draft.optLoading ? "opacity-55" : ""}`}>
            {curve ? (
              <CostCurveChart points={curve.points} yMax={curve.yMax} tau={chartTau} optimal={opt?.optimal_threshold ?? null} label={curveLabel} />
            ) : (
              <div className="grid h-[320px] place-items-center text-muted">Đang tính đường cong…</div>
            )}
          </div>
          <ul className="mt-2.5 flex flex-wrap gap-x-4.5 gap-y-1.5 text-[12.5px] text-ink-2">
            <li className="inline-flex items-center"><span aria-hidden="true" className="mr-2 inline-block h-0.5 w-4 bg-ink" />Ngưỡng đang xem</li>
            <li className="inline-flex items-center"><span aria-hidden="true" className="mr-2 inline-block h-[3px] w-4 rounded-sm bg-series-2" />Ngưỡng tối ưu theo tham số đang nhập</li>
            <li className="inline-flex items-center"><span aria-hidden="true" className="mr-2 inline-block h-[3px] w-4 rounded-sm bg-series-1" />Chi phí ước tính</li>
          </ul>
        </Card>
      </div>

      <fieldset className="card min-w-0">
        <legend className="float-left card-title w-full p-0">Ngưỡng gợi ý sẵn</legend>
        <p className="card-sub clear-both">
          Mỗi ngưỡng được chọn trên out-of-fold; số bên phải đo trên tập kiểm thử với tham số chi phí đang nhập (FR-34).
        </p>
        <div className="grid gap-1.5">
          {presets.map((p) => {
            const selected = tau === p.value;
            return (
              <label
                key={p.key}
                className={`grid cursor-pointer grid-cols-[20px_80px_minmax(0,1fr)] items-center gap-3 rounded-md border px-3 py-2.5 hover:bg-surface-2 lg:grid-cols-[20px_92px_minmax(0,1fr)_auto] ${selected ? "border-accent bg-accent-soft" : "border-line"}`}
              >
                <input type="radio" name="preset" value={p.key} checked={selected} onChange={() => draft.setTau(p.value)} className="size-4 accent-accent" />
                <span className="text-[15px] font-bold tabular-nums">{fmt.tau(p.value)}</span>
                <span className="grid">
                  <strong>{p.label}</strong>
                  <small className="text-xs text-muted">{p.hint}</small>
                </span>
                <span className="col-start-3 flex flex-wrap gap-x-4 gap-y-1 text-[12.5px] tabular-nums text-ink-2 lg:col-start-auto">
                  <span>bắt <strong>{p.tp}/{p.nPos}</strong></span>
                  <span><strong>{fmt.int(p.alertsPerDay)}</strong> cảnh báo/ngày</span>
                  <span><strong>{fmt.int(p.costPerDay)}</strong> EUR/ngày</span>
                </span>
              </label>
            );
          })}
        </div>
      </fieldset>
    </div>
  );
}

function CostForm() {
  const draft = useThresholdDraft();
  const { opt, costs, constraint, constraintType } = draft;
  return (
    <form className="card" onSubmit={(e) => e.preventDefault()} noValidate>
      <h2 className="card-title">Tham số chi phí</h2>
      <p className="card-sub">Đổi tham số làm dịch chuyển ngưỡng tối ưu (FR-32). Máy chủ tính lại sau 300 ms kể từ lần gõ cuối.</p>
      <Field id="cost-fn" label="Chi phí bỏ lọt 1 gian lận" unit="EUR">
        <input id="cost-fn" type="number" min={0.01} step={0.01} value={draft.costFnInput} onChange={(e) => draft.setCostFnInput(e.target.value)} className="field-input max-w-40" />
      </Field>
      <Field id="cost-fp" label="Chi phí 1 lần thẩm định" unit="EUR">
        <input id="cost-fp" type="number" min={0} step={0.01} value={draft.costFpInput} onChange={(e) => draft.setCostFpInput(e.target.value)} className="field-input max-w-40" />
      </Field>
      {!costs.valid && <p role="alert" className="mb-3 text-[12.5px] font-semibold text-critical-text">Chi phí bỏ lọt phải lớn hơn 0; chi phí thẩm định không âm.</p>}

      <div className="mb-3 grid gap-1">
        <label htmlFor="constraint-type" className="text-[13px] font-semibold text-ink-2">Ràng buộc vận hành (FR-33)</label>
        <select id="constraint-type" value={constraintType} onChange={(e) => draft.setConstraintType(e.target.value as ConstraintType)} className="field-input">
          <option value="none">Không ràng buộc</option>
          <option value="min_recall">Recall tối thiểu</option>
          <option value="max_alerts_per_day">Số cảnh báo tối đa mỗi ngày</option>
        </select>
      </div>
      {constraintType !== "none" && (
        <Field id="constraint-value" label={constraintType === "min_recall" ? "Recall tối thiểu" : "Ngân sách thẩm định"} unit={constraintType === "min_recall" ? "%" : "cảnh báo/ngày"}>
          <input
            id="constraint-value"
            type="number"
            min={0}
            step="any"
            placeholder={constraintType === "min_recall" ? "90" : "200"}
            value={draft.constraintValueInput}
            onChange={(e) => draft.setConstraintValueInput(e.target.value)}
            className="field-input max-w-40"
          />
        </Field>
      )}
      {!constraint.valid && <p role="alert" className="mb-3 text-[12.5px] font-semibold text-critical-text">Nhập giá trị ràng buộc: recall trong (0, 100]%, ngân sách lớn hơn 0.</p>}

      <div aria-live="polite" className="mt-4 min-h-[90px] border-t border-grid pt-3.5">
        {draft.optLoading && <p className="text-muted">Đang tính ngưỡng tối ưu…</p>}
        {draft.optError && <p className="text-[12.5px] font-semibold text-critical-text">{draft.optError}</p>}
        {opt && (
          <div>
            <p className="mb-2 flex flex-wrap items-center gap-2.5 text-[15px]">
              Ngưỡng tối ưu: <strong className="tabular-nums">{fmt.tau(opt.optimal_threshold)}</strong>
              <button type="button" className="btn btn-sm" disabled={draft.tau === opt.optimal_threshold} onClick={() => draft.setTau(opt.optimal_threshold)}>
                Xem ngưỡng này
              </button>
            </p>
            {constraintType !== "none" && opt.constraint_binding && (
              <p className="mb-2">
                <strong>Ràng buộc đang chặn nghiệm.</strong> Không có nó, ngưỡng tối ưu là <span className="tabular-nums">{fmt.tau(opt.unconstrained_threshold)}</span> — bạn
                đang bị giới hạn bởi ràng buộc vận hành, không phải bởi chi phí.
              </p>
            )}
            {constraintType !== "none" && !opt.constraint_binding && <p className="mb-2">Ràng buộc không chặn: nghiệm cực tiểu chi phí đã thỏa ràng buộc.</p>}
            {!opt.constraint_satisfied && <p className="mb-2 text-[12.5px] font-semibold text-critical-text">Không ngưỡng nào đạt ràng buộc — đã lấy ngưỡng gần đạt nhất.</p>}
            <p className="text-[12.5px] text-muted">
              Chọn trên out-of-fold của tập huấn luyện (ML-08). Đo lại trên tập kiểm thử: bắt {opt.metrics_at_optimal.tp}/
              {opt.metrics_at_optimal.tp + opt.metrics_at_optimal.fn} gian lận, {fmt.int(opt.metrics_at_optimal.alerts_per_day)} cảnh báo/ngày.
            </p>
          </div>
        )}
      </div>
    </form>
  );
}

function Field({ id, label, unit, children }: { id: string; label: string; unit: string; children: ReactNode }) {
  return (
    <div className="mb-3 grid gap-1">
      <label htmlFor={id} className="text-[13px] font-semibold text-ink-2">{label}</label>
      <div className="flex items-center gap-2">
        {children}
        <span className="text-[13px] text-muted">{unit}</span>
      </div>
    </div>
  );
}
