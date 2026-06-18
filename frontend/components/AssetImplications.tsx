// frontend/components/AssetImplications.tsx
import type { AssetImplication } from "@/lib/types";
import { REGIME_COLORS } from "@/lib/types";

function Pct({ v }: { v: number }) {
  const s = `${v >= 0 ? "+" : ""}${(v * 100).toFixed(2)}%`;
  const color = v > 0.005 ? "#22c55e" : v < -0.005 ? "#ef4444" : "#737373";
  return (
    <span className="mono text-sm font-medium" style={{ color }}>
      {s}
    </span>
  );
}

export function AssetImplications({
  implications,
}: {
  implications: AssetImplication[];
}) {
  return (
    <section>
      <h2 className="mono text-xs tracking-widest uppercase mb-3" style={{ color: "#525252" }}>
        Historical Returns by Regime — S&amp;P 500
      </h2>
      <div className="card overflow-x-auto">
        <table className="w-full">
          <thead>
            <tr style={{ borderBottom: "1px solid #1f1f1f" }}>
              {["Regime", "Median/Month", "Mean/Month", "Observations"].map(
                (h) => (
                  <th
                    key={h}
                    className="mono text-xs text-left py-2 pr-4"
                    style={{ color: "#525252" }}
                  >
                    {h}
                  </th>
                )
              )}
            </tr>
          </thead>
          <tbody>
            {implications.map((imp) => (
              <tr
                key={imp.regime}
                style={{ borderBottom: "1px solid #141414" }}
                className="hover:bg-white/[0.02]"
              >
                <td className="py-2.5 pr-4">
                  <div className="flex items-center gap-2">
                    <div
                      className="w-2 h-2 rounded-full"
                      style={{
                        background: REGIME_COLORS[imp.regime] ?? "#525252",
                      }}
                    />
                    <span className="mono text-sm" style={{ color: "#d4d4d4" }}>
                      {imp.regime}
                    </span>
                  </div>
                </td>
                <td className="py-2.5 pr-4">
                  <Pct v={imp.median_monthly_return} />
                </td>
                <td className="py-2.5 pr-4">
                  <Pct v={imp.mean_monthly_return} />
                </td>
                <td className="py-2.5 pr-4">
                  <span className="mono text-xs" style={{ color: "#525252" }}>
                    {imp.observation_count} mo
                  </span>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
        <p className="text-xs mt-3" style={{ color: "#333333" }}>
          Historical performance does not guarantee future results.
        </p>
      </div>
    </section>
  );
}
