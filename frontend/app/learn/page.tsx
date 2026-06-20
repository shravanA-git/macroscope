// frontend/app/learn/page.tsx

export const metadata = {
  title: "MacroScope — Learn",
  description: "A plain-English guide to everything on the MacroScope dashboard.",
};

const REGIME_CARDS = [
  {
    name: "Expansion",
    color: "#22c55e",
    emoji: "↑",
    what: "The economy is growing. Companies are hiring, GDP is rising, and investors feel confident.",
    signals: "Low unemployment, tight credit spreads, moderate VIX, positive yield curve.",
    stocks: "Historically the best period for equities. Growth stocks outperform.",
  },
  {
    name: "Late-Cycle",
    color: "#f59e0b",
    emoji: "→",
    what: "Growth is still positive but slowing. The expansion is aging — cracks are starting to appear.",
    signals: "Yield curve flattening, early credit stress, VIX starting to rise.",
    stocks: "Reduce risk. Defensive sectors (utilities, healthcare) start to outperform.",
  },
  {
    name: "Recovery",
    color: "#3b82f6",
    emoji: "↗",
    what: "The economy is healing after a downturn. Growth is resuming from a low base.",
    signals: "Credit spreads compressing, unemployment falling from peak, VIX declining.",
    stocks: "Early opportunity — beaten-down cyclicals and financials tend to lead.",
  },
  {
    name: "Contraction",
    color: "#ef4444",
    emoji: "↓",
    what: "The economy is shrinking or stagnant. Fear is elevated and credit is stressed.",
    signals: "High VIX, wide credit spreads, rising unemployment, inverted or flat yield curve.",
    stocks: "Defensive positioning. Cash, bonds, and defensive equities. Avoid cyclicals.",
  },
];

const INDICATORS = [
  {
    name: "VIX",
    aka: "The Fear Index",
    what: "The VIX measures how much volatility the options market expects in the S&P 500 over the next 30 days. Think of it as a fear gauge — when investors are calm it's low (under 15), when they're panicking it spikes (above 40 in 2008 and 2020).",
    normal: "10–20",
    elevated: "20–30",
    extreme: "30+",
    direction: "High VIX = bad for stocks. Low VIX = calm markets.",
  },
  {
    name: "10Y–2Y Yield Spread",
    aka: "The Yield Curve",
    what: "The difference between what the US government pays to borrow money for 10 years vs. 2 years. Normally 10-year rates are higher (you want more compensation for locking up money longer). When 2-year rates exceed 10-year rates (the spread goes negative), it's called an \"inverted yield curve\" — and it has predicted every US recession since the 1970s.",
    normal: "> 0%",
    elevated: "0% to −0.5%",
    extreme: "< −0.5% (inverted)",
    direction: "Negative spread = recession warning. Positive spread = healthy.",
  },
  {
    name: "Manufacturing Employment",
    aka: "PMI Proxy (MANEMP)",
    what: "The number of people employed in manufacturing in the US (in thousands). When factories are hiring, the economy is producing goods and expanding. When they're laying off, it signals contraction. This is used as a proxy for the ISM Manufacturing PMI (a classic business activity survey) because the original PMI isn't freely available from FRED.",
    normal: "Rising trend",
    elevated: "Flat or declining",
    extreme: "Sharp decline",
    direction: "Rising = expansion signal. Falling = contraction signal.",
  },
  {
    name: "Baa/10Y Credit Spread",
    aka: "Corporate Bond Spread",
    what: "The extra interest rate that medium-quality (\"Baa\" rated) companies must pay to borrow money, compared to the US government. When this spread is wide, it means investors distrust corporate borrowers and want compensation for the risk of default. Wide spreads are a classic sign of financial stress — they spiked sharply in 2008, 2016, and 2020.",
    normal: "< 1.5%",
    elevated: "1.5%–2.5%",
    extreme: "> 2.5%",
    direction: "Wide spread = market stress. Tight spread = confidence.",
  },
  {
    name: "Unemployment Rate",
    aka: "Labor Market Health",
    what: "The percentage of people in the US labor force who are actively looking for work but can't find it. It's the most widely-watched economic indicator in the world. The Federal Reserve's primary job is keeping unemployment low while controlling inflation. High unemployment = recession. Low unemployment = healthy economy — but also potential inflation pressure.",
    normal: "3%–5%",
    elevated: "5%–7%",
    extreme: "> 7% (recession territory)",
    direction: "Rising = economic stress. Falling = recovery/expansion.",
  },
];

function Section({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <section className="space-y-4">
      <h2 className="mono text-xs tracking-widest uppercase" style={{ color: "#525252" }}>
        {title}
      </h2>
      {children}
    </section>
  );
}

function Callout({ children }: { children: React.ReactNode }) {
  return (
    <div
      className="card text-sm leading-relaxed"
      style={{ color: "#a3a3a3", borderColor: "#262626" }}
    >
      {children}
    </div>
  );
}

export default function LearnPage() {
  return (
    <div className="space-y-12 max-w-3xl">

      {/* Header */}
      <div className="space-y-2">
        <h1 className="text-2xl font-semibold" style={{ color: "#e5e5e5" }}>
          How MacroScope Works
        </h1>
        <p style={{ color: "#737373" }} className="text-sm leading-relaxed">
          A plain-English guide to everything on the dashboard — no finance or machine learning background required.
        </p>
      </div>

      {/* What is a macro regime */}
      <Section title="What is a macro regime?">
        <Callout>
          <p>
            The economy doesn&apos;t move in a straight line. It cycles through distinct phases — sometimes
            growing fast, sometimes slowing, sometimes contracting, sometimes recovering. Each phase is
            called a <strong style={{ color: "#e5e5e5" }}>macro regime</strong>.
          </p>
          <p className="mt-3">
            The regime matters because different investments behave very differently depending on which
            phase the economy is in. Stocks that soar in Expansion can collapse in Contraction. Bonds
            that look boring in Recovery can be the safest asset in a downturn.
          </p>
          <p className="mt-3">
            MacroScope classifies the current economy into one of four regimes every month, using five
            publicly available economic signals.
          </p>
        </Callout>
      </Section>

      {/* The four regimes */}
      <Section title="The four regimes">
        <div className="space-y-3">
          {REGIME_CARDS.map((r) => (
            <div key={r.name} className="card space-y-2" style={{ borderColor: "#1f1f1f" }}>
              <div className="flex items-center gap-2">
                <span
                  className="mono text-xs font-bold px-2 py-0.5 rounded"
                  style={{ background: r.color + "22", color: r.color }}
                >
                  {r.emoji} {r.name.toUpperCase()}
                </span>
              </div>
              <p className="text-sm" style={{ color: "#d4d4d4" }}>{r.what}</p>
              <div className="text-xs space-y-1" style={{ color: "#737373" }}>
                <p><span style={{ color: "#525252" }}>Signals:</span> {r.signals}</p>
                <p><span style={{ color: "#525252" }}>For stocks:</span> {r.stocks}</p>
              </div>
            </div>
          ))}
        </div>
      </Section>

      {/* The five indicators */}
      <Section title="The five macro indicators">
        <Callout>
          MacroScope watches five signals from FRED — the Federal Reserve&apos;s free public data
          repository. Each signal is transformed into a <strong style={{ color: "#e5e5e5" }}>z-score</strong> (explained below)
          so the model can compare all five on the same scale.
        </Callout>
        <div className="space-y-4">
          {INDICATORS.map((ind) => (
            <div key={ind.name} className="card space-y-2" style={{ borderColor: "#1f1f1f" }}>
              <div className="flex items-baseline gap-2">
                <span className="mono text-sm font-semibold" style={{ color: "#e5e5e5" }}>
                  {ind.name}
                </span>
                <span className="mono text-xs" style={{ color: "#525252" }}>
                  — {ind.aka}
                </span>
              </div>
              <p className="text-sm leading-relaxed" style={{ color: "#a3a3a3" }}>{ind.what}</p>
              <div className="flex flex-wrap gap-4 pt-1 text-xs mono">
                <span style={{ color: "#22c55e" }}>Normal: {ind.normal}</span>
                <span style={{ color: "#f59e0b" }}>Elevated: {ind.elevated}</span>
                <span style={{ color: "#ef4444" }}>Extreme: {ind.extreme}</span>
              </div>
              <p className="text-xs" style={{ color: "#525252" }}>{ind.direction}</p>
            </div>
          ))}
        </div>
      </Section>

      {/* Z-score */}
      <Section title="What is a z-score?">
        <Callout>
          <p>
            A <strong style={{ color: "#e5e5e5" }}>z-score</strong> measures how unusual a reading is
            compared to its historical average.
          </p>
          <ul className="mt-3 space-y-1.5 text-xs mono" style={{ color: "#737373" }}>
            <li><span style={{ color: "#22c55e" }}>z = 0</span> — exactly average</li>
            <li><span style={{ color: "#f59e0b" }}>z = +1.0</span> — one standard deviation above average (unusual but not extreme)</li>
            <li><span style={{ color: "#ef4444" }}>z = +2.0</span> — two standard deviations above average (very unusual — happens ~5% of the time)</li>
            <li><span style={{ color: "#3b82f6" }}>z = −1.5</span> — below average</li>
          </ul>
          <p className="mt-3">
            For example: an Unemployment z-score of +1.77 means unemployment is significantly higher
            than its 35-year average — a contraction signal. A VIX z-score of −0.62 means VIX is
            slightly below average — markets are calm.
          </p>
          <p className="mt-3">
            Z-scores let the model compare apples to oranges: a VIX reading of 35 and an unemployment
            rate of 7% are on completely different scales, but z = +2.5 and z = +1.8 are directly comparable.
          </p>
        </Callout>
      </Section>

      {/* Trend arrows */}
      <Section title="What do the trend arrows mean?">
        <Callout>
          <p>
            Each indicator card shows an arrow indicating recent momentum — whether the indicator is
            moving up, down, or sideways over the last month.
          </p>
          <div className="mt-3 space-y-1.5 text-xs mono" style={{ color: "#737373" }}>
            <div><span style={{ color: "#22c55e" }}>↑ Up</span> — indicator rose month-over-month (can be good or bad depending on the indicator)</div>
            <div><span style={{ color: "#ef4444" }}>↓ Down</span> — indicator fell month-over-month</div>
            <div><span style={{ color: "#737373" }}>→ Flat</span> — minimal change</div>
          </div>
          <p className="mt-3">
            Context matters: rising unemployment (↑) is bad; rising manufacturing employment (↑) is good.
            The z-score and description tell you the full picture.
          </p>
        </Callout>
      </Section>

      {/* Layer 1: HMM */}
      <Section title="Layer 1 — Hidden Markov Model (the regime classifier)">
        <Callout>
          <p>
            The <strong style={{ color: "#e5e5e5" }}>Hidden Markov Model (HMM)</strong> is the engine
            that classifies the current regime. Here&apos;s the intuition:
          </p>
          <p className="mt-3">
            Imagine you&apos;re blindfolded and trying to figure out what season it is just by feeling the
            temperature, humidity, and wind each day. You can&apos;t see the season directly — it&apos;s
            &quot;hidden&quot; — but you can observe clues. Over time you learn that certain combinations
            of clues (cold + dry + low wind) reliably signal winter.
          </p>
          <p className="mt-3">
            The HMM does the same thing with the economy. The &quot;season&quot; is the regime
            (Expansion / Contraction / etc.). The &quot;clues&quot; are the five z-scored indicators.
            The model was trained on 35 years of monthly FRED data and learned — entirely without being
            told in advance — which combinations of signals cluster together into distinct economic environments.
          </p>
          <p className="mt-3">
            It uses 5 hidden states internally (because the 2022 rate-shock bear market looks different
            from the 2008 unemployment-driven recession), but maps them to 4 named regimes for display.
          </p>
          <p className="mt-3">
            The <strong style={{ color: "#e5e5e5" }}>confidence percentage</strong> on the dashboard
            (e.g., &quot;100% conf.&quot;) is the HMM&apos;s posterior probability — how certain it is
            about the current classification given the observed data. 100% means all five signals are
            unambiguously pointing to Contraction.
          </p>
        </Callout>
      </Section>

      {/* Layer 2: AutoGluon */}
      <Section title="Layer 2 — AutoGluon (the transition forecaster)">
        <Callout>
          <p>
            <strong style={{ color: "#e5e5e5" }}>AutoGluon</strong> is a machine learning toolkit
            from Amazon that automatically trains and combines many different models (decision trees,
            gradient boosting, etc.) and picks the best combination.
          </p>
          <p className="mt-3">
            While the HMM tells you &quot;what regime are we in right now,&quot; AutoGluon answers
            &quot;what regime are we most likely to be in next month?&quot;
          </p>
          <p className="mt-3">
            It was trained using <strong style={{ color: "#e5e5e5" }}>walk-forward validation</strong>:
            at each step, it was only allowed to train on data from the past and was tested on the
            future — exactly like a real investor who can only use historical information. This prevents
            the model from &quot;cheating&quot; by learning from future data it wouldn&apos;t have had access to.
          </p>
          <p className="mt-3">
            The <strong style={{ color: "#e5e5e5" }}>Transition Forecast</strong> panel shows the
            probability distribution over next month&apos;s regime. A reading of &quot;Contraction 62%&quot;
            means the model assigns a 62% probability to staying in Contraction — it&apos;s not certain,
            but it&apos;s the most likely outcome.
          </p>
        </Callout>
      </Section>

      {/* Dashboard walkthrough */}
      <Section title="What each section of the dashboard shows">
        <div className="space-y-3">
          {[
            {
              title: "Current Macro Regime (top panel)",
              desc: "The HMM's current classification with a confidence dial. The probability bars on the right show how the confidence is distributed across all four regimes. 100% Contraction means the model sees no ambiguity.",
            },
            {
              title: "Macro Indicators (5 cards)",
              desc: "The actual values of the five FRED signals as of the most recent fully-published month (data releases lag by 2–4 weeks). Each card shows the raw value, z-score, trend arrow, and a plain-English description of what the indicator measures.",
            },
            {
              title: "Regime History — S&P 500 Cumulative Return (chart)",
              desc: "The blue line shows what $1 invested in the S&P 500 in 1990 would be worth today (about 26x). The colored background bands show what regime MacroScope classified each historical month as. Notice how the red Contraction bands (2001, 2008, 2020, 2022) align with the biggest dips in the blue line.",
            },
            {
              title: "Transition Forecast (right panel)",
              desc: "AutoGluon's probability distribution for next month's regime. These are model probabilities — not certainties. The model was validated on out-of-sample data with 90% accuracy at regime identification.",
            },
            {
              title: "Historical Returns by Regime (table)",
              desc: "The actual median and mean monthly S&P 500 returns across all months in each regime, from 1990–2026. This is empirical — it shows what historically happened in each regime, not what a model predicts. 'Observations' is how many months of data underlie each figure.",
            },
            {
              title: "Methodology (accordion)",
              desc: "Detailed technical explanations of every modeling decision: why 5 HMM states instead of 4, how the z-score weighting was calibrated, what walk-forward validation means, and an honest account of the backtesting results (including why the strategy underperforms buy-and-hold).",
            },
          ].map((item) => (
            <div key={item.title} className="card" style={{ borderColor: "#1f1f1f" }}>
              <p className="text-sm font-medium" style={{ color: "#e5e5e5" }}>{item.title}</p>
              <p className="text-sm mt-1.5 leading-relaxed" style={{ color: "#737373" }}>{item.desc}</p>
            </div>
          ))}
        </div>
      </Section>

      {/* How often does it update */}
      <Section title="How often does the data update?">
        <Callout>
          <p>
            MacroScope updates <strong style={{ color: "#e5e5e5" }}>monthly</strong>, because the
            underlying FRED data is monthly. Unemployment, credit spreads, and manufacturing employment
            are all released with a 2–4 week lag, so the &quot;current&quot; reading always reflects
            the most recently completed month.
          </p>
          <p className="mt-3">
            The &quot;Updated&quot; date on the dashboard shows when the data pipeline was last run
            and the predictions were last regenerated. This is not a real-time feed — it is a
            monthly snapshot that gets refreshed when new FRED data is available.
          </p>
        </Callout>
      </Section>

      {/* Disclaimer */}
      <Section title="Is this investment advice?">
        <Callout>
          <p>
            No. MacroScope is a research and education project. It demonstrates machine learning
            techniques applied to public macroeconomic data.
          </p>
          <p className="mt-3">
            The backtesting results show that a naive long/short strategy based on regime
            classification <em>underperforms</em> simple buy-and-hold over 35 years. Regime
            detection is most useful as a risk awareness tool — not as a trading signal.
          </p>
          <p className="mt-3" style={{ color: "#525252" }}>
            Past regime classifications do not guarantee future accuracy. Do not make investment
            decisions based on this site.
          </p>
        </Callout>
      </Section>

    </div>
  );
}
