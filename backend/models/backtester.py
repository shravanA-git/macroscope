# backend/models/backtester.py
import pandas as pd
import numpy as np

LONG_REGIMES = {"Expansion", "Recovery"}
SHORT_REGIMES = {"Contraction"}


def compute_sharpe(returns: pd.Series, risk_free_annual: float = 0.04) -> float:
    """Annualized Sharpe ratio using monthly returns."""
    excess = returns - risk_free_annual / 12
    std = excess.std()
    if std == 0 or np.isnan(std):
        return 0.0
    return float((excess.mean() / std) * np.sqrt(12))


def compute_max_drawdown(returns: pd.Series) -> float:
    """Maximum peak-to-trough percentage drawdown."""
    cumulative = (1 + returns).cumprod()
    rolling_max = cumulative.cummax()
    drawdown = (cumulative - rolling_max) / rolling_max
    return float(drawdown.min())


def run_regime_backtest(
    regime_series: pd.Series,
    sp500_returns: pd.Series,
) -> dict[str, float]:
    """
    Long/short equity strategy:
    +1 (long) in Expansion/Recovery, -1 (short) in Contraction, 0 (flat) in Late-Cycle.
    Signal set at month-end, applied to next month (no look-ahead bias).
    """
    aligned = pd.concat([regime_series, sp500_returns], axis=1).dropna()
    aligned.columns = ["regime", "returns"]

    def position(regime: str) -> float:
        if regime in LONG_REGIMES:
            return 1.0
        if regime in SHORT_REGIMES:
            return -1.0
        return 0.0

    aligned["position"] = aligned["regime"].map(position).shift(1)
    aligned["strategy_returns"] = aligned["position"] * aligned["returns"]
    aligned = aligned.dropna()

    strategy = aligned["strategy_returns"]
    benchmark = aligned["returns"]

    return {
        "strategy_sharpe": compute_sharpe(strategy),
        "benchmark_sharpe": compute_sharpe(benchmark),
        "strategy_max_drawdown": compute_max_drawdown(strategy),
        "benchmark_max_drawdown": compute_max_drawdown(benchmark),
        "strategy_total_return": float((1 + strategy).prod() - 1),
        "benchmark_total_return": float((1 + benchmark).prod() - 1),
    }
