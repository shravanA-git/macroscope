// frontend/components/Methodology.tsx
"use client";

import { useState } from "react";

const SECTIONS = [
  {
    title: "Hidden Markov Model (Layer 1 — Generative)",
    body: `The HMM assumes the economy moves through unobservable hidden states — Expansion, Late-Cycle, Contraction, Recovery — that generate the macro indicator readings we observe. The Baum-Welch algorithm (a special case of Expectation-Maximization) learns the transition probabilities and emission distributions from 30+ years of FRED data without being told which periods were recessions. We use 5 hidden states: 3 growth archetypes and 2 contraction archetypes, because the 2022 rate-shock bear market (tight labor + inverted yield curve) is economically distinct from 2008 (high unemployment + credit freeze).`,
  },
  {
    title: "AutoGluon Ensemble (Layer 2 — Discriminative)",
    body: `While the HMM classifies the current regime, AutoGluon predicts the next one — a harder, more valuable problem. It trains an ensemble of gradient-boosted trees, random forests, and linear models on the same macro features, optimizing directly for next-period regime prediction. Walk-forward validation ensures no future data leaks into training: the model always predicts strictly future months using strictly past data. The benchmark is the naive "stay in current regime" rule; the paper measures how much the ensemble improves on it.`,
  },
  {
    title: "Feature Engineering",
    body: `Raw FRED series are non-stationary — VIX has different baselines pre/post-2008, unemployment trends over decades. Rolling 12-month z-scores transform each series into "how unusual is this reading vs. the past year?" making features comparable across time and across series with different units. Month-over-month rate-of-change captures momentum. 1- and 3-month lags let the model see trajectory, not just the current snapshot. All 20 features are constructed without look-ahead.`,
  },
  {
    title: "Data Sources",
    body: `All macro data from FRED (Federal Reserve Economic Data): VIX (VIXCLS), 10Y-2Y Treasury yield spread (T10Y2Y), Manufacturing Employment as PMI proxy (MANEMP), Moody's Baa corporate spread over 10Y Treasury as credit risk proxy (BAA10Y), and Unemployment Rate (UNRATE). Monthly frequency, data since 1991. S&P 500 returns from Yahoo Finance via yfinance for backtesting.`,
  },
  {
    title: "Backtesting",
    body: `A regime-conditioned strategy goes long S&P 500 in Expansion and Recovery, short in Contraction, and flat in Late-Cycle. Signals are set at month-end and applied to next month's returns — no look-ahead bias. Performance is measured by annualized Sharpe ratio and maximum drawdown, compared against buy-and-hold. Results are honest: the naive long/short strategy does not always beat the market, which is expected — regime detection is most valuable as a risk management overlay, not a standalone alpha strategy.`,
  },
];

export function Methodology() {
  const [open, setOpen] = useState<number | null>(null);

  return (
    <section>
      <h2
        className="mono text-xs tracking-widest uppercase mb-3"
        style={{ color: "#525252" }}
      >
        Methodology
      </h2>
      <div className="space-y-2">
        {SECTIONS.map((s, i) => (
          <div key={i} className="card">
            <button
              className="w-full flex items-center justify-between text-left gap-4"
              onClick={() => setOpen(open === i ? null : i)}
            >
              <span className="mono text-sm font-medium" style={{ color: "#d4d4d4" }}>
                {s.title}
              </span>
              <span className="mono text-lg flex-shrink-0" style={{ color: "#525252" }}>
                {open === i ? "−" : "+"}
              </span>
            </button>
            {open === i && (
              <p
                className="text-sm leading-relaxed mt-3 pt-3"
                style={{
                  color: "#a3a3a3",
                  borderTop: "1px solid #1f1f1f",
                }}
              >
                {s.body}
              </p>
            )}
          </div>
        ))}
      </div>
    </section>
  );
}
