// frontend/components/Hero.tsx
import { REGIME_COLORS, type CurrentRegimeData } from "@/lib/types";

interface Props {
  data: CurrentRegimeData;
}

export function Hero({ data }: Props) {
  const { regime, description, last_updated } = data;
  const color = REGIME_COLORS[regime.regime] ?? "#737373";
  const pct = Math.round(regime.probability * 100);

  const circumference = 2 * Math.PI * 48;
  const dash = (regime.probability * circumference).toFixed(1);

  const sorted = Object.entries(regime.probabilities).sort(
    ([, a], [, b]) => b - a
  );

  return (
    <section className="card flex flex-col md:flex-row items-center gap-8 py-8 px-6">
      {/* Gauge */}
      <div className="relative flex-shrink-0">
        <svg width="120" height="120" viewBox="0 0 120 120">
          <circle
            cx="60"
            cy="60"
            r="48"
            fill="none"
            stroke="#1f1f1f"
            strokeWidth="7"
          />
          <circle
            cx="60"
            cy="60"
            r="48"
            fill="none"
            stroke={color}
            strokeWidth="7"
            strokeLinecap="round"
            strokeDasharray={`${dash} ${circumference}`}
            strokeDashoffset={circumference * 0.25}
            style={{ transition: "stroke-dasharray 0.8s ease" }}
          />
        </svg>
        <div className="absolute inset-0 flex flex-col items-center justify-center">
          <span className="mono text-xl font-bold" style={{ color }}>
            {pct}%
          </span>
          <span className="mono text-xs" style={{ color: "#525252" }}>
            conf.
          </span>
        </div>
      </div>

      {/* Regime label + description */}
      <div className="flex-1 min-w-0">
        <p
          className="mono text-xs tracking-widest uppercase mb-1"
          style={{ color: "#525252" }}
        >
          Current Macro Regime
        </p>
        <h1 className="text-4xl font-bold tracking-tight mb-3" style={{ color }}>
          {regime.regime}
        </h1>
        <p className="text-sm leading-relaxed max-w-lg" style={{ color: "#a3a3a3" }}>
          {description}
        </p>
        <p className="mono text-xs mt-3" style={{ color: "#404040" }}>
          Updated{" "}
          {new Date(last_updated).toLocaleDateString("en-US", {
            month: "short",
            day: "numeric",
            year: "numeric",
          })}
        </p>
      </div>

      {/* Probability breakdown */}
      <div className="flex-shrink-0 space-y-2 w-48">
        {sorted.map(([name, prob]) => (
          <div key={name} className="flex items-center gap-2">
            <div
              className="w-1.5 h-1.5 rounded-full flex-shrink-0"
              style={{ background: REGIME_COLORS[name] ?? "#525252" }}
            />
            <span className="mono text-xs w-20 truncate" style={{ color: "#737373" }}>
              {name}
            </span>
            <div
              className="flex-1 h-1 rounded-full overflow-hidden"
              style={{ background: "#1f1f1f" }}
            >
              <div
                className="h-full rounded-full"
                style={{
                  width: `${(prob * 100).toFixed(0)}%`,
                  background: REGIME_COLORS[name] ?? "#525252",
                }}
              />
            </div>
            <span className="mono text-xs w-7 text-right" style={{ color: "#525252" }}>
              {(prob * 100).toFixed(0)}%
            </span>
          </div>
        ))}
      </div>
    </section>
  );
}
