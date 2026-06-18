# backend/models/regime_labeler.py
import numpy as np
import pandas as pd

# Labels assigned in order from most expansion-like to most contraction-like
ORDERED_LABELS = ["Expansion", "Late-Cycle", "Recovery", "Contraction"]

REGIME_COLORS = {
    "Expansion": "#22c55e",
    "Late-Cycle": "#f59e0b",
    "Recovery": "#3b82f6",
    "Contraction": "#ef4444",
}

REGIME_DESCRIPTIONS = {
    "Expansion": "GDP growing, unemployment falling, credit spreads tight. Historically the best period for equities.",
    "Late-Cycle": "Growth slowing, yield curve flattening, early credit stress. Reduce risk exposure.",
    "Recovery": "Economy healing post-contraction. Credit spreads compressing. Early equity opportunity.",
    "Contraction": "GDP declining or stagnant, VIX elevated, credit spreads wide. Defensive positioning.",
}


def _expansion_score(state_means: dict[str, float]) -> float:
    """Higher score = more expansion-like. Linear combination of key z-scores."""
    return (
        state_means.get("yield_spread_zscore", 0)
        + state_means.get("pmi_zscore", 0)
        - state_means.get("vix_zscore", 0)
        - state_means.get("credit_spread_zscore", 0)
    )


def assign_labels(
    n_states: int,
    features: pd.DataFrame,
    states: np.ndarray,
) -> dict[int, str]:
    """
    Assign economic regime names to HMM integer state indices.
    States sorted by expansion_score: highest -> 'Expansion', lowest -> 'Contraction'.
    """
    zscore_cols = [c for c in features.columns if "_zscore" in c]

    state_means: dict[int, dict[str, float]] = {}
    for s in range(n_states):
        mask = states == s
        if mask.sum() == 0:
            state_means[s] = {}
            continue
        state_means[s] = features.loc[mask, zscore_cols].mean().to_dict()

    sorted_states = sorted(
        range(n_states),
        key=lambda s: _expansion_score(state_means[s]),
        reverse=True,
    )

    return {state: ORDERED_LABELS[i] for i, state in enumerate(sorted_states)}


def label_series(states: np.ndarray, label_map: dict[int, str]) -> pd.Series:
    """Convert integer state array to string regime series."""
    return pd.Series([label_map[s] for s in states])
