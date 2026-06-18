# backend/data/pipeline.py
import pandas as pd
from .fred_client import fetch_all
from .yfinance_client import fetch_sp500_returns
from .features import build_feature_matrix
from . import cache


def run(force_refresh: bool = False) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    """
    Returns (raw_df, features_df, sp500_returns).
    Uses parquet cache unless force_refresh=True.
    """
    raw = None if force_refresh else cache.load("raw_fred")
    if raw is None:
        raw = fetch_all()
        cache.save(raw, "raw_fred")

    features = build_feature_matrix(raw)
    cache.save(features, "features")

    sp500 = fetch_sp500_returns(start=str(features.index[0].date()))
    return raw, features, sp500
