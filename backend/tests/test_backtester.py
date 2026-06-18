# backend/tests/test_backtester.py
import numpy as np
import pandas as pd
import pytest
from backend.models.backtester import compute_sharpe, compute_max_drawdown, run_regime_backtest


def make_returns(n: int = 120) -> pd.Series:
    rng = np.random.default_rng(42)
    idx = pd.date_range("2010-01-31", periods=n, freq="ME")
    return pd.Series(rng.normal(0.008, 0.04, n), index=idx)


def make_regimes(n: int = 120) -> pd.Series:
    rng = np.random.default_rng(42)
    idx = pd.date_range("2010-01-31", periods=n, freq="ME")
    return pd.Series(
        rng.choice(["Expansion", "Late-Cycle", "Recovery", "Contraction"], n),
        index=idx,
    )


def test_sharpe_is_finite():
    assert np.isfinite(compute_sharpe(make_returns()))


def test_max_drawdown_is_negative_or_zero():
    assert compute_max_drawdown(make_returns()) <= 0.0


def test_backtest_returns_required_keys():
    result = run_regime_backtest(make_regimes(), make_returns())
    required = {
        "strategy_sharpe", "benchmark_sharpe",
        "strategy_max_drawdown", "benchmark_max_drawdown",
        "strategy_total_return", "benchmark_total_return",
    }
    assert required.issubset(result.keys())


def test_backtest_all_finite():
    result = run_regime_backtest(make_regimes(), make_returns())
    for v in result.values():
        assert np.isfinite(v), f"Non-finite value in backtest result: {v}"
