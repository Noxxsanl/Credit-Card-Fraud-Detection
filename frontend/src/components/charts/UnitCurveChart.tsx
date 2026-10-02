"use client";

/*
 * Đường cong trên mặt phẳng [0, 1] × [0, 1]: PR hoặc ROC, một hoặc nhiều đường (UI-04).
 * Mỗi đường mang dữ liệu riêng vì các đường có lưới recall khác nhau.
 * Đường tham chiếu: đường cơ sở nằm ngang (PR) hoặc đường chéo đoán ngẫu nhiên (ROC).
 */

import { memo } from "react";
import { CartesianGrid, Legend, Line, LineChart, ReferenceLine, XAxis, YAxis } from "recharts";

import { COLORS } from "@/lib/colors";
import { fmt } from "@/lib/format";

export interface UnitSeries {
  key: string;
  label: string;
  color: string;
  x: number[];
  y: number[];
}

const TICKS = [0, 0.2, 0.4, 0.6, 0.8, 1];

export const UnitCurveChart = memo(function UnitCurveChart({ series, baseline, diagonal = false, xLabel, yLabel, height, legend = false, label }: {
  series: UnitSeries[];
  baseline?: number;
  diagonal?: boolean;
  xLabel: string;
  yLabel: string;
  height: number;
  legend?: boolean;
  label: string;
}) {
  return (
    <div role="img" aria-label={label}>
      <LineChart responsive style={{ width: "100%", height }} margin={{ top: 8, right: 14, bottom: 18, left: 0 }}>
        <CartesianGrid stroke={COLORS.grid} />
        <XAxis
          dataKey="x"
          type="number"
          domain={[0, 1]}
          ticks={TICKS}
          tickFormatter={(v: number) => fmt.num(v, 1)}
          stroke={COLORS.axis}
          tick={{ fill: COLORS.ink2, fontSize: 11 }}
          label={{ value: xLabel, position: "insideBottom", offset: -12, fill: COLORS.ink2, fontSize: 12, fontWeight: 600 }}
        />
        <YAxis
          type="number"
          domain={[0, 1]}
          ticks={TICKS}
          tickFormatter={(v: number) => fmt.num(v, 1)}
          width={44}
          stroke={COLORS.axis}
          tick={{ fill: COLORS.ink2, fontSize: 11 }}
          label={{ value: yLabel, angle: -90, position: "insideLeft", offset: 10, fill: COLORS.ink2, fontSize: 12, fontWeight: 600, style: { textAnchor: "middle" } }}
        />
        {baseline !== undefined && <ReferenceLine y={baseline} stroke={COLORS.ink3} strokeWidth={1} />}
        {diagonal && <ReferenceLine segment={[{ x: 0, y: 0 }, { x: 1, y: 1 }]} stroke={COLORS.ink3} strokeWidth={1} />}
        {series.map((s) => (
          <Line
            key={s.key}
            data={s.x.map((x, i) => ({ x, y: s.y[i] }))}
            dataKey="y"
            name={s.label}
            type="linear"
            stroke={s.color}
            strokeWidth={2}
            dot={false}
            activeDot={false}
            isAnimationActive={false}
          />
        ))}
        {legend && (
          // Giữ thứ tự của chuỗi (mặc định Recharts sắp theo chữ cái); chữ màu mực, màu chỉ ở nét kẻ
          <Legend
            verticalAlign="bottom"
            align="left"
            iconType="plainline"
            itemSorter={null}
            formatter={(value: string) => <span style={{ color: COLORS.ink2 }}>{value}</span>}
            wrapperStyle={{ fontSize: 12, paddingTop: 18 }}
          />
        )}
      </LineChart>
    </div>
  );
});
