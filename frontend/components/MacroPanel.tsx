// frontend/components/MacroPanel.tsx
import type { MacroIndicator } from "@/lib/types";

const META: Record<string, { label: string; unit: string; note: string }> = {
  Vix: { label: "VIX", unit: "", note: "Market fear — spikes in crises" },
  "Yield Spread": {
    label: "10Y–2Y Spread",
    unit: "%",
    note: "Inverts before recessions",
  },
  Pmi: {
    label: "Mfg. Employment",
    unit: "K",
    note: "Manufacturing activity proxy",
  },
  "Credit Spread": {
    label: "Baa/10Y Spread",
    unit: "%",
    note: "Corporate default risk",
  },
  Unemployment: {
    label: "Unemployment",
    unit: "%",
    note: "Labor market health",
  },
};

function Arrow({ trend }: { trend: string }) {
  if (trend === "up") return <span style={{ color: "#22c55e" }}>↑</span>;
  if (trend === "down") return <span style={{ color: "#ef4444" }}>↓</span>;
  return <span style={{ color: "#525252" }}>→</span>;
}

function ZBadge({ z }: { z: number }) {
  const abs = Math.abs(z);
  const color = abs > 2 ? "#ef4444" : abs > 1 ? "#f59e0b" : "#525252";
  return (
    <span
      className="mono text-xs px-1.5 py-0.5 rounded"
      style={{ color, background: `${color}1a` }}
    >
      z={z >= 0 ? "+" : ""}
      {z.toFixed(2)}
    </span>
  );
}

export function MacroPanel({ indicators }: { indicators: MacroIndicator[] }) {
  return (
    <section>
      <h2 className="mono text-xs tracking-widest uppercase mb-3" style={{ color: "#525252" }}>
        Macro Indicators
      </h2>
      <div className="grid grid-cols-2 md:grid-cols-5 gap-3">
        {indicators.map((ind) => {
          const meta = META[ind.name] ?? {
            label: ind.name,
            unit: "",
            note: "",
          };
          return (
            <div key={ind.name} className="card space-y-2">
              <div className="flex items-center justify-between">
                <span className="mono text-xs" style={{ color: "#525252" }}>
                  {meta.label}
                </span>
                <Arrow trend={ind.trend} />
              </div>
              <div className="mono text-lg font-semibold" style={{ color: "#e5e5e5" }}>
                {ind.value != null ? ind.value.toFixed(2) : "—"}
                <span className="text-xs ml-1" style={{ color: "#525252" }}>
                  {meta.unit}
                </span>
              </div>
              <ZBadge z={ind.zscore} />
              <p className="text-xs leading-tight" style={{ color: "#404040" }}>
                {meta.note}
              </p>
            </div>
          );
        })}
      </div>
    </section>
  );
}
