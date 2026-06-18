#!/usr/bin/env python
# backend/scripts/refresh_predictions.py
"""
Regenerates all static JSON files consumed by the frontend.
Run from the project root: python -m backend.scripts.refresh_predictions
Or directly:              python backend/scripts/refresh_predictions.py
"""
import json
import warnings
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from datetime import datetime, timezone

warnings.filterwarnings("ignore")

STATIC_DIR = Path(__file__).parent.parent.parent / "static"
LABEL_MAP_PATH = Path(__file__).parent.parent / "models" / "artifacts" / "label_map.pkl"


def main():
    from backend.data.pipeline import run
    from backend.models.hmm_model import MacroHMM
    from backend.models.regime_labeler import label_series, REGIME_COLORS, REGIME_DESCRIPTIONS
    from backend.models.autogluon_model import RegimeTransitionPredictor
    from backend.models.backtester import run_regime_backtest

    STATIC_DIR.mkdir(exist_ok=True)

    print("Loading data...")
    raw, features, sp500 = run()

    zscore_cols = [c for c in features.columns if c.endswith("_zscore")]
    hmm_features = features[zscore_cols]

    print("Loading HMM from artifacts...")
    model = MacroHMM.load()
    label_map = joblib.load(LABEL_MAP_PATH)

    states = model.decode(hmm_features)
    regime_series = label_series(states, label_map)
    regime_series.index = hmm_features.index

    print(f"Current regime: {regime_series.iloc[-1]}")
    print(f"Regime counts: {dict(regime_series.value_counts())}")

    # --- Train AutoGluon ---
    print("\nTraining AutoGluon (time_limit=180s)...")
    ag = RegimeTransitionPredictor(lookahead=1)
    train_df = ag.prepare_training_data(features, regime_series)
    print(f"Training data: {train_df.shape}, target classes: {train_df['next_regime'].unique().tolist()}")
    ag.fit(train_df, time_limit=180)
    print("AutoGluon training complete.")
    print(ag.leaderboard()[["model", "score_val"]].head(5).to_string())

    # --- current_regime.json ---
    last_features = features[zscore_cols].iloc[[-1]]
    proba = model.predict_proba(last_features)[0]

    proba_dict: dict[str, float] = {}
    for state_i, prob in enumerate(proba):
        label = label_map[state_i]
        proba_dict[label] = proba_dict.get(label, 0) + float(prob)

    current_label = regime_series.iloc[-1]
    current_regime = {
        "regime": {
            "regime": current_label,
            "probability": round(proba_dict.get(current_label, 0), 4),
            "probabilities": {k: round(v, 4) for k, v in proba_dict.items()},
        },
        "indicators": [],
        "last_updated": datetime.now(timezone.utc).isoformat(),
        "description": REGIME_DESCRIPTIONS.get(current_label, ""),
    }

    raw_last = raw.iloc[-1]
    for col in ["vix", "yield_spread", "pmi", "credit_spread", "unemployment"]:
        z_col = f"{col}_zscore"
        mom_col = f"{col}_mom"
        z = float(features[z_col].iloc[-1]) if z_col in features.columns else 0.0
        mom = float(features[mom_col].iloc[-1]) if mom_col in features.columns else 0.0
        trend = "up" if mom > 0.002 else ("down" if mom < -0.002 else "flat")
        current_regime["indicators"].append({
            "name": col.replace("_", " ").title(),
            "value": float(raw_last[col]) if col in raw_last.index else 0.0,
            "zscore": round(z, 3),
            "trend": trend,
        })

    (STATIC_DIR / "current_regime.json").write_text(json.dumps(current_regime, indent=2))
    print(f"\nWrote current_regime.json: {current_label}")

    # --- regime_history.json ---
    sp500_aligned = sp500.reindex(regime_series.index)
    history = []
    for date, regime in regime_series.items():
        ret = sp500_aligned.get(date)
        history.append({
            "date": str(date.date()),
            "regime": regime,
            "sp500_return": (
                None
                if (ret is None or (isinstance(ret, float) and np.isnan(ret)))
                else round(float(ret), 5)
            ),
        })
    (STATIC_DIR / "regime_history.json").write_text(json.dumps({"history": history}, indent=2))
    print(f"Wrote regime_history.json: {len(history)} points")

    # --- transition_probs.json ---
    last_feature_row = features.iloc[[-1]]
    feature_cols = [c for c in train_df.columns if c != "next_regime"]
    trans_proba = ag.predict_proba(last_feature_row[feature_cols]).iloc[0].to_dict()
    next_regime = max(trans_proba, key=lambda k: trans_proba[k])
    transition = {
        "current_regime": current_label,
        "predicted_next_regime": next_regime,
        "transition_probabilities": {k: round(float(v), 4) for k, v in trans_proba.items()},
        "forecast_horizon_months": 1,
    }
    (STATIC_DIR / "transition_probs.json").write_text(json.dumps(transition, indent=2))
    print(f"Wrote transition_probs.json: → {next_regime}")

    # --- asset_implications.json ---
    implications = []
    sp500_aligned2 = sp500.reindex(regime_series.index).dropna()
    for regime_name in ["Expansion", "Late-Cycle", "Recovery", "Contraction"]:
        mask = regime_series == regime_name
        aligned_returns = sp500_aligned2[mask]
        implications.append({
            "regime": regime_name,
            "color": REGIME_COLORS.get(regime_name, "#888888"),
            "asset_class": "US Equities (S&P 500)",
            "median_monthly_return": round(float(aligned_returns.median()), 5) if len(aligned_returns) > 0 else 0.0,
            "mean_monthly_return": round(float(aligned_returns.mean()), 5) if len(aligned_returns) > 0 else 0.0,
            "observation_count": int(mask.sum()),
        })
    (STATIC_DIR / "asset_implications.json").write_text(json.dumps({"implications": implications}, indent=2))
    print("Wrote asset_implications.json")

    # --- backtest.json ---
    backtest = run_regime_backtest(regime_series, sp500)
    (STATIC_DIR / "backtest.json").write_text(json.dumps(backtest, indent=2))
    print(
        f"Wrote backtest.json: strategy Sharpe={backtest['strategy_sharpe']:.2f} "
        f"vs benchmark={backtest['benchmark_sharpe']:.2f}"
    )

    print("\n=== ALL STATIC JSON FILES WRITTEN ===")
    for f in sorted(STATIC_DIR.glob("*.json")):
        print(f"  {f.name}: {f.stat().st_size} bytes")


if __name__ == "__main__":
    main()
