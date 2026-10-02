"use client";

/*
 * UI-03 — tổng chi phí ước tính mỗi ngày theo ngưỡng. Trục hoành thang log; hai vạch dọc khác
 * màu: ngưỡng đang xem và ngưỡng tối ưu (07 §5). Trục tung do màn hình truyền vào (cắt quanh
 * cực tiểu), phần vượt khung bị cắt.
 */

import { memo } from "react";
import { CartesianGrid, Line, LineChart, ReferenceLine, Tooltip, XAxis, YAxis, type TooltipContentProps, type TooltipValueType } from "recharts";

import { COLORS } from "@/lib/colors";
import { fmt } from "@/lib/format";

export interface CostPoint {
  x: number; // ngưỡng
  y: number; // EUR/ngày
  apd: number; // cảnh báo/ngày
  recall: number;
}

const TICKS = [1e-4, 1e-3, 1e-2, 1e-1, 1];

function CostTooltip({ active, payload }: TooltipContentProps<TooltipValueType, string | number>) {
  if (!active || !payload?.length) return null;
  const p = payload[0].payload as CostPoint;
  return (
    <div className="rounded-md border border-line-strong bg-surface px-2.5 py-2 text-[12.5px] shadow-float">
      <div className="font-semibold">τ = {fmt.tau(p.x)}</div>
      <div className="font-semibold tabular-nums">≈ {fmt.int(p.y)} EUR/ngày</div>
      <div className="text-ink-2">Cảnh báo ≈ {fmt.int(p.apd)}/ngày · recall {fmt.pct(p.recall)}</div>
    </div>
  );
}

export const CostCurveChart = memo(function CostCurveChart({ points, yMax, tau, optimal, label }: {
  points: CostPoint[];
  yMax: number;
  tau: number;
  optimal: number | null;
  label: string;
}) {
  return (
    <div role="img" aria-label={label}>
      <LineChart responsive data={points} style={{ width: "100%", height: 320 }} margin={{ top: 36, right: 18, bottom: 18, left: 8 }}>
        <CartesianGrid stroke={COLORS.grid} />
        <XAxis
          dataKey="x"
          type="number"
          scale="log"
          domain={[1e-4, 1]}
          ticks={TICKS}
          tickFormatter={(v: number) => fmt.tauTick(v)}
          allowDataOverflow
          stroke={COLORS.axis}
          tick={{ fill: COLORS.ink2, fontSize: 12 }}
          label={{ value: "Ngưỡng τ (thang log)", position: "insideBottom", offset: -12, fill: COLORS.ink2, fontSize: 12, fontWeight: 600 }}
        />
        <YAxis
          type="number"
          domain={[0, yMax]}
          allowDataOverflow
          tickFormatter={(v: number) => fmt.int(v)}
          width={72}
          stroke={COLORS.axis}
          tick={{ fill: COLORS.ink2, fontSize: 12 }}
          label={{ value: "EUR mỗi ngày", angle: -90, position: "insideLeft", offset: 4, fill: COLORS.ink2, fontSize: 12, fontWeight: 600, style: { textAnchor: "middle" } }}
        />
        <Tooltip content={CostTooltip} cursor={{ stroke: COLORS.axis, strokeWidth: 1 }} isAnimationActive={false} />
        <ReferenceLine
          x={tau}
          ifOverflow="hidden"
          stroke={COLORS.ink}
          strokeWidth={1.5}
          label={{ value: `đang xem ${fmt.tau(tau)}`, position: "top", fill: COLORS.ink, fontSize: 11, fontWeight: 600, offset: 18 }}
        />
        {optimal !== null && (
          <ReferenceLine
            x={optimal}
            ifOverflow="hidden"
            stroke={COLORS.series2}
            strokeWidth={2}
            label={{ value: `tối ưu ${fmt.tau(optimal)}`, position: "top", fill: COLORS.ink, fontSize: 11, fontWeight: 600, offset: 2 }}
          />
        )}
        <Line
          dataKey="y"
          type="linear"
          stroke={COLORS.series1}
          strokeWidth={2}
          dot={false}
          activeDot={{ r: 5, stroke: COLORS.surface, strokeWidth: 2 }}
          isAnimationActive={false}
        />
      </LineChart>
    </div>
  );
});
