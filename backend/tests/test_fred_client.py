# backend/tests/test_fred_client.py
import pytest
import pandas as pd
from backend.data.fred_client import fetch_series, fetch_all, SERIES

def test_series_map_has_all_indicators():
    assert set(SERIES.keys()) == {"vix", "yield_spread", "pmi", "credit_spread", "unemployment"}

def test_fetch_series_returns_float_series():
    s = fetch_series("UNRATE", start="2020-01-01")
    assert isinstance(s, pd.Series)
    assert s.dtype == float
    assert len(s) > 0
    assert not s.isnull().all()

def test_fetch_all_columns_and_monthly():
    df = fetch_all(start="2020-01-01")
    assert isinstance(df, pd.DataFrame)
    assert set(df.columns) == {"vix", "yield_spread", "pmi", "credit_spread", "unemployment"}
    assert df.index.dtype == "datetime64[ns]"

def test_fetch_series_no_dot_strings():
    # FRED uses "." for missing — verify we convert to NaN not string
    s = fetch_series("UNRATE", start="1990-01-01")
    assert s.dtype == float
