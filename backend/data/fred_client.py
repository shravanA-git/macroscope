# backend/data/fred_client.py
import pandas as pd

FRED_CSV_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv"

SERIES = {
    "vix": "VIXCLS",
    "yield_spread": "T10Y2Y",
    "pmi": "MANEMP",          # Manufacturing employment — proxy for ISM PMI (NAPM deprecated on FRED)
    "credit_spread": "BAMLH0A0HYM2",
    "unemployment": "UNRATE",
}

def fetch_series(series_id: str, start: str = "1990-01-01") -> pd.Series:
    """Fetch FRED series via public CSV endpoint — no API key needed."""
    url = f"{FRED_CSV_URL}?id={series_id}"
    df = pd.read_csv(url, index_col=0, parse_dates=True, na_values=".")
    series = df.iloc[:, 0].astype(float)
    return series[series.index >= start]

def fetch_all(start: str = "1990-01-01") -> pd.DataFrame:
    """Fetch all macro indicators and resample to month-end frequency."""
    frames = {}
    for name, sid in SERIES.items():
        try:
            frames[name] = fetch_series(sid, start).resample("ME").last()
        except Exception:
            frames[name] = pd.Series(dtype=float)
    df = pd.DataFrame(frames)
    df.index = pd.to_datetime(df.index).astype("datetime64[ns]")
    df.dropna(how="all", inplace=True)
    return df
