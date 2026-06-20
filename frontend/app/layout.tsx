// frontend/app/layout.tsx
import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "MacroScope — Macro Regime Detection",
  description:
    "Live macroeconomic regime classification via Hidden Markov Model and AutoGluon ensemble.",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <body className="min-h-screen" style={{ background: "#0a0a0a" }}>
        <nav
          style={{ borderBottom: "1px solid #1f1f1f" }}
          className="px-6 py-3 flex items-center justify-between"
        >
          <div className="flex items-center gap-3">
            <span
              className="mono font-bold text-sm"
              style={{ color: "#22c55e" }}
            >
              ◆ MACROSCOPE
            </span>
            <span className="mono text-xs" style={{ color: "#404040" }}>
              v0.1 · FRED + HMM + AutoGluon
            </span>
          </div>
          <div className="flex items-center gap-4">
            <a
              href="/learn"
              className="mono text-xs transition-colors"
              style={{ color: "#525252" }}
            >
              Learn
            </a>
            <a
              href="https://github.com/shravan-anand/macroscope"
              className="mono text-xs transition-colors"
              style={{ color: "#525252" }}
              target="_blank"
              rel="noopener noreferrer"
            >
              GitHub →
            </a>
          </div>
        </nav>
        <main className="max-w-7xl mx-auto px-4 sm:px-6 py-8 space-y-6">
          {children}
        </main>
        <footer
          style={{ borderTop: "1px solid #1f1f1f" }}
          className="px-6 py-4 mt-12 text-center"
        >
          <p className="mono text-xs" style={{ color: "#404040" }}>
            MacroScope · Shravan Anand · Duke University 2029 · Data: FRED,
            yfinance · Models: hmmlearn, AutoGluon · Not investment advice.
          </p>
        </footer>
      </body>
    </html>
  );
}
