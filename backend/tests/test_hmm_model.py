# backend/tests/test_hmm_model.py
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import pytest
from backend.models.hmm_model import MacroHMM

def make_features(n: int = 120) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    idx = pd.date_range("2010-01-31", periods=n, freq="ME")
    cols = ["vix_zscore", "yield_spread_zscore", "pmi_zscore",
            "credit_spread_zscore", "unemployment_zscore"]
    data = rng.standard_normal((n, len(cols)))
    return pd.DataFrame(data, index=idx, columns=cols)

def test_hmm_fits_without_error():
    features = make_features()
    model = MacroHMM(n_states=4)
    model.fit(features)

def test_decode_returns_integer_states():
    features = make_features()
    model = MacroHMM(n_states=4)
    model.fit(features)
    states = model.decode(features)
    assert states.dtype in [np.int32, np.int64]
    assert set(np.unique(states)).issubset({0, 1, 2, 3})
    assert len(states) == len(features)

def test_predict_proba_sums_to_one():
    features = make_features()
    model = MacroHMM(n_states=4)
    model.fit(features)
    proba = model.predict_proba(features)
    assert proba.shape == (len(features), 4)
    np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-6)

def test_transition_matrix_rows_sum_to_one():
    features = make_features()
    model = MacroHMM(n_states=4)
    model.fit(features)
    tm = model.get_transition_matrix()
    np.testing.assert_allclose(tm.values.sum(axis=1), 1.0, atol=1e-6)

def test_save_and_load(tmp_path):
    features = make_features()
    model = MacroHMM(n_states=4)
    model.fit(features)
    path = tmp_path / "hmm.pkl"
    model.save(path)
    loaded = MacroHMM.load(path)
    states_orig = model.decode(features)
    states_loaded = loaded.decode(features)
    np.testing.assert_array_equal(states_orig, states_loaded)
