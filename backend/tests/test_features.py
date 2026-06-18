# backend/tests/test_features.py
import pytest
import pandas as pd
import numpy as np
from backend.data.features import add_rolling_zscores, add_rate_of_change, build_feature_matrix

def mock_raw(n: int = 60) -> pd.DataFrame:
    idx = pd.date_range("2015-01-31", periods=n, freq="ME")
    rng = np.random.default_rng(42)
    return pd.DataFrame({
        "vix": rng.uniform(10, 80, n),
        "yield_spread": rng.uniform(-1, 3, n),
        "pmi": rng.uniform(45, 65, n),
        "credit_spread": rng.uniform(2, 10, n),
        "unemployment": rng.uniform(3, 10, n),
    }, index=idx)

def test_rolling_zscores_produces_zscore_columns():
    raw = mock_raw()
    z = add_rolling_zscores(raw)
    assert all(c.endswith("_zscore") for c in z.columns)
    assert z.shape[1] == raw.shape[1]

def test_rolling_zscores_roughly_unit_variance():
    raw = mock_raw(120)
    z = add_rolling_zscores(raw).dropna()
    for col in z.columns:
        assert z[col].std() < 3.0

def test_rate_of_change_produces_mom_columns():
    raw = mock_raw()
    mom = add_rate_of_change(raw)
    assert all(c.endswith("_mom") for c in mom.columns)

def test_build_feature_matrix_no_nan():
    raw = mock_raw(60)
    features = build_feature_matrix(raw)
    assert not features.isnull().any().any()

def test_build_feature_matrix_has_all_feature_types():
    raw = mock_raw(60)
    features = build_feature_matrix(raw)
    cols = features.columns.tolist()
    assert any("_zscore" in c for c in cols)
    assert any("_mom" in c for c in cols)
    assert any("_lag" in c for c in cols)
