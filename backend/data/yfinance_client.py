# backend/data/yfinance_client.py
import pandas as pd
import yfinance as yf

def fetch_sp500_returns(start: str = "1990-01-01") -> pd.Series:
    """Fetch S&P 500 monthly total returns for backtest benchmarking."""
    ticker = yf.Ticker("^GSPC")
    hist = ticker.history(start=start, interval="1mo")
    closes = hist["Close"]
    # Strip timezone if present, then align to month-end timestamps to match FRED data
    if closes.index.tz is not None:
        closes.index = closes.index.tz_localize(None)
    closes.index = closes.index.to_period("M").to_timestamp("M")
    returns = closes.pct_change().dropna()
    returns.name = "sp500_return"
    return returns
