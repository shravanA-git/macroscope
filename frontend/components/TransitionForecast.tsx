// frontend/components/TransitionForecast.tsx
import { REGIME_COLORS, type TransitionForecast } from "@/lib/types";

export function TransitionForecastPanel({ data }: { data: TransitionForecast }) {
  const sorted = Object.entries(data.transition_probabilities).sort(
    ([, a], [, b]) => b - a
  );
  const nextColor = REGIME_COLORS[data.predicted_next_regime] ?? "#737373";

  return (
    <section>
      <h2 className="mono text-xs tracking-widest uppercase mb-3" style={{ color: "#525252" }}>
        Transition Forecast — Next Month
      </h2>
      <div className="card space-y-4 h-full">
        <div className="flex items-baseline gap-2 flex-wrap">
          <span className="text-sm" style={{ color: "#737373" }}>
            Predicted:
          </span>
          <span className="mono font-bold text-base" style={{ color: nextColor }}>
            {data.predicted_next_regime}
          </span>
          <span className="mono text-xs" style={{ color: "#525252" }}>
            ({(sorted[0][1] * 100).toFixed(0)}% · AutoGluon)
          </span>
        </div>
        <div className="space-y-2.5">
          {sorted.map(([name, prob]) => (
            <div key={name} className="flex items-center gap-2">
              <span className="mono text-xs w-20 truncate" style={{ color: "#737373" }}>
                {name}
              </span>
              <div
                className="flex-1 h-1.5 rounded-full overflow-hidden"
                style={{ background: "#1f1f1f" }}
              >
                <div
                  className="h-full rounded-full transition-all duration-500"
                  style={{
                    width: `${(prob * 100).toFixed(1)}%`,
                    background: REGIME_COLORS[name] ?? "#525252",
                  }}
                />
              </div>
              <span className="mono text-xs w-8 text-right" style={{ color: "#525252" }}>
                {(prob * 100).toFixed(0)}%
              </span>
            </div>
          ))}
        </div>
        <p className="mono text-xs" style={{ color: "#333333" }}>
          ExtraTreesGini ensemble · walk-forward validated
        </p>
      </div>
    </section>
  );
}
