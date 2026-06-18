# MacroScope

A quantitative macroeconomic regime detection platform combining Hidden Markov Models and AutoGluon ensemble learning to classify current market regimes and forecast transitions — published as a live web application and Zenodo research paper.

## Architecture

**Layer 1 — HMM (Generative):** Classifies the current macro regime (Expansion / Late-Cycle / Contraction / Recovery) from FRED macro indicators using the Baum-Welch algorithm.

**Layer 2 — AutoGluon (Discriminative):** Predicts regime transitions 1–3 months ahead using the same feature set. The publishable comparison: generative vs. discriminative for regime forecasting.

## Features
- Live macro regime dashboard (Bloomberg Terminal aesthetic)
- Historical regime bands overlaid on S&P 500 returns
- Transition probability forecast with confidence intervals
- Asset class return implications per regime
- Full methodology explanation for non-quant audiences

## Stack
- **Backend:** Python (hmmlearn, AutoGluon, FastAPI) → Vercel serverless functions
- **Frontend:** Next.js 16 App Router + Tailwind + Recharts
- **Data:** FRED API, yfinance
- **Deployment:** Vercel

## Structure
```
macroscope/
├── backend/
│   ├── data/         # FRED pipeline, feature engineering
│   ├── models/       # HMM, AutoGluon, backtesting
│   ├── api/          # FastAPI endpoints
│   └── tests/        # Unit + integration tests
├── frontend/         # Next.js 16 App Router
├── research/
│   ├── notebooks/    # EDA, validation, figures
│   ├── figures/      # Generated charts for paper
│   └── paper/        # Zenodo draft
└── data/             # Cached FRED pulls (.gitignored)
```

## Research Paper
Target: Zenodo preprint before September 2026.
Topic: *Generative vs. Discriminative Approaches to Macroeconomic Regime Transition Forecasting*

## Author
Shravan Anand | Duke University, Class of 2029 | CS + Economics
