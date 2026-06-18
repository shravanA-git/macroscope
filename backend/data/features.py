# backend/data/features.py
import pandas as pd
import numpy as np

ROLLING_WINDOW = 12  # 12-month rolling window for z-score normalization


def add_rolling_zscores(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize each series to 12-month rolling z-scores for stationarity."""
    result = {}
    for col in df.columns:
        mu = df[col].rolling(ROLLING_WINDOW).mean()
        sigma = df[col].rolling(ROLLING_WINDOW).std()
        result[f"{col}_zscore"] = (df[col] - mu) / sigma.replace(0, np.nan)
    return pd.DataFrame(result, index=df.index)


def add_rate_of_change(df: pd.DataFrame, periods: int = 1) -> pd.DataFrame:
    """Month-over-month percentage change captures momentum."""
    result = {f"{col}_mom": df[col].pct_change(periods) for col in df.columns}
    return pd.DataFrame(result, index=df.index)


def add_lag_features(df: pd.DataFrame, lags: list[int] = [1, 3]) -> pd.DataFrame:
    """Lagged values let the model see recent trajectory."""
    result = {}
    for col in df.columns:
        for lag in lags:
            result[f"{col}_lag{lag}"] = df[col].shift(lag)
    return pd.DataFrame(result, index=df.index)


def build_feature_matrix(raw_df: pd.DataFrame) -> pd.DataFrame:
    """Full pipeline: z-scores + MoM changes + lags, then drop NaN rows.

    Columns with >50% missing values are dropped before row-level dropna so
    that a sparse series (e.g. one that only starts recently) does not
    collapse the entire history to just its coverage window.
    """
    # Remove sparse columns before feature engineering to preserve row history
    threshold = 0.5
    coverage = raw_df.notna().mean()
    raw_df = raw_df.loc[:, coverage >= threshold]

    zscores = add_rolling_zscores(raw_df)
    moms = add_rate_of_change(raw_df)
    lags = add_lag_features(zscores)
    combined = pd.concat([zscores, moms, lags], axis=1)
    # Drop any remaining all-NaN columns, then drop incomplete rows
    combined = combined.dropna(how="all", axis=1)
    return combined.dropna()
