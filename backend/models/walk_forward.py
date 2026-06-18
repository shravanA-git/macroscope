# backend/models/walk_forward.py
from __future__ import annotations
import pandas as pd
from typing import Iterator
from sklearn.metrics import accuracy_score, classification_report


def walk_forward_splits(
    df: pd.DataFrame,
    initial_train_years: int = 5,
    step_months: int = 1,
) -> Iterator[tuple[pd.DataFrame, pd.DataFrame]]:
    """
    Expanding walk-forward cross-validation.
    Training window grows by step_months; test is always strictly future.
    """
    start_idx = initial_train_years * 12
    i = start_idx
    while i < len(df):
        train = df.iloc[:i]
        test = df.iloc[i:i + step_months]
        if len(test) == 0:
            break
        yield train, test
        i += step_months


def naive_baseline_accuracy(df: pd.DataFrame, initial_train_years: int = 5) -> float:
    """Baseline: predict 'stay in current regime'. The bar AutoGluon must beat."""
    start_idx = initial_train_years * 12
    test_df = df.iloc[start_idx:]
    current = test_df["next_regime"].shift(1).dropna()
    target = test_df["next_regime"].iloc[1:]
    return float(accuracy_score(target, current))
