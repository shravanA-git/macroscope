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
    """Higher score = more expansion-like. Linear combination of key z-scores.

    Weight rationale:
    - unemployment_zscore: strongest single contraction signal (2008, COVID).
      High unemployment = bad. Weight: -2.0
    - credit_spread_zscore: captures both classic crises (2008) and rate-shock
      bear markets (2022, credit spreads widened despite low unemployment).
      Weight: -2.0
    - yield_spread_zscore: an *inverted* curve (negative z-score) is a leading
      recession indicator and dominated 2022. BUT the Fed steepens the curve
      during crises via rate cuts, so this must not dominate. Weight: +0.8
      (positive: a steep healthy curve is expansion-like; inversion is bearish)
    - vix_zscore: elevated fear = stress. Weight: -1.0
    - pmi_zscore: activity indicator; 2022 had high PMI despite being a bear
      market, so intentionally lower weight than stress signals. Weight: +0.5
    """
    return (
        -2.0 * state_means.get("unemployment_zscore", 0)
        - 2.0 * state_means.get("credit_spread_zscore", 0)
        - 1.0 * state_means.get("vix_zscore", 0)
        + 0.5 * state_means.get("pmi_zscore", 0)
        + 0.8 * state_means.get("yield_spread_zscore", 0)
    )


def assign_labels(
    n_states: int,
    features: pd.DataFrame,
    states: np.ndarray,
) -> dict[int, str]:
    """
    Assign economic regime names to HMM integer state indices.

    States are sorted by expansion_score (highest = most expansion-like).
    The label sequence is always: Expansion, Late-Cycle, Recovery, Contraction.
    When n_states > 4, all states beyond the third are mapped to 'Contraction'
    — this correctly handles the 5-state model where the two lowest-scoring
    states correspond to distinct contraction archetypes (e.g. the classic
    unemployment-driven 2008 recession and the 2022 rate-shock bear market).
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

    # Pad with 'Contraction' for any states beyond the 4 named labels.
    # WARNING: label ordering is data-driven — retraining the HMM can produce a
    # different sorted_states order, invalidating any AutoGluon model trained on
    # the old label_map. Always retrain HMM + AutoGluon together (refresh_predictions.py).
    padded_labels = ORDERED_LABELS + ["Contraction"] * max(0, n_states - len(ORDERED_LABELS))
    return {state: padded_labels[i] for i, state in enumerate(sorted_states)}


def label_series(states: np.ndarray, label_map: dict[int, str]) -> pd.Series:
    """Convert integer state array to string regime series."""
    return pd.Series([label_map[s] for s in states])
