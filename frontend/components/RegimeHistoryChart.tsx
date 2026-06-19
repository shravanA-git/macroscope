// frontend/components/RegimeHistoryChart.tsx
"use client";

import {
  AreaChart,
  Area,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
  ReferenceArea,
} from "recharts";
import { REGIME_COLORS, type RegimeHistoryPoint } from "@/lib/types";

interface Props {
  history: RegimeHistoryPoint[];
}

function buildData(history: RegimeHistoryPoint[]) {
  let cum = 1;
  return history.map((pt) => {
    if (pt.sp500_return != null) {
      cum *= 1 + pt.sp500_return;
    }
    return { date: pt.date, regime: pt.regime, value: parseFloat(cum.toFixed(4)) };
  });
}

function buildBands(history: RegimeHistoryPoint[]) {
  const bands: { regime: string; start: string; end: string }[] = [];
  if (!history.length) return bands;
  let current = history[0].regime;
  let start = history[0].date;
  for (let i = 1; i < history.length; i++) {
    if (history[i].regime !== current) {
      bands.push({ regime: current, start, end: history[i].date });
      current = history[i].regime;
      start = history[i].date;
    }
  }
  bands.push({ regime: current, start, end: history[history.length - 1].date });
  return bands;
}

const TOOLTIP_STYLE = {
  background: "#111111",
  border: "1px solid #1f1f1f",
  borderRadius: 6,
  fontFamily: "JetBrains Mono, monospace",
  fontSize: 11,
  color: "#a3a3a3",
};

export function RegimeHistoryChart({ history }: Props) {
  const data = buildData(history);
  const bands = buildBands(history);

  return (
    <section>
      <h2 className="mono text-xs tracking-widest uppercase mb-3" style={{ color: "#525252" }}>
        Regime History — S&amp;P 500 Cumulative Return
      </h2>
      <div className="card p-3" style={{ height: 340 }}>
        <ResponsiveContainer width="100%" height="100%">
          <AreaChart data={data} margin={{ top: 8, right: 8, left: 0, bottom: 0 }}>
            <defs>
              <linearGradient id="spGrad" x1="0" y1="0" x2="0" y2="1">
                <stop offset="5%" stopColor="#3b82f6" stopOpacity={0.25} />
                <stop offset="95%" stopColor="#3b82f6" stopOpacity={0} />
              </linearGradient>
            </defs>
            {bands.map((b, i) => (
              <ReferenceArea
                key={i}
                x1={b.start}
                x2={b.end}
                fill={REGIME_COLORS[b.regime] ?? "#525252"}
                fillOpacity={0.07}
              />
            ))}
            <XAxis
              dataKey="date"
              tickLine={false}
              axisLine={false}
              tick={{ fill: "#404040", fontSize: 10, fontFamily: "JetBrains Mono" }}
              tickFormatter={(v: string) => v.slice(0, 4)}
              interval={35}
            />
            <YAxis
              tickLine={false}
              axisLine={false}
              tick={{ fill: "#404040", fontSize: 10, fontFamily: "JetBrains Mono" }}
              tickFormatter={(v: number) => `${v.toFixed(0)}x`}
              width={36}
            />
            <Tooltip
              contentStyle={TOOLTIP_STYLE}
              // eslint-disable-next-line @typescript-eslint/no-explicit-any
              formatter={(v: any) => [`${Number(v).toFixed(2)}x`, "Cumulative"]}
            />
            <Area
              type="monotone"
              dataKey="value"
              stroke="#3b82f6"
              strokeWidth={1.5}
              fill="url(#spGrad)"
              dot={false}
            />
          </AreaChart>
        </ResponsiveContainer>
      </div>
      <div className="flex flex-wrap gap-4 mt-2">
        {Object.entries(REGIME_COLORS).map(([name, color]) => (
          <div key={name} className="flex items-center gap-1.5">
            <div
              className="w-2.5 h-2.5 rounded-sm"
              style={{ background: color, opacity: 0.5 }}
            />
            <span className="mono text-xs" style={{ color: "#525252" }}>
              {name}
            </span>
          </div>
        ))}
      </div>
    </section>
  );
}
