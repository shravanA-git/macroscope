# backend/data/cache.py
import pandas as pd
from pathlib import Path

CACHE_DIR = Path(__file__).parent.parent.parent / "data"


def save(df: pd.DataFrame, name: str) -> None:
    CACHE_DIR.mkdir(exist_ok=True)
    df.to_parquet(CACHE_DIR / f"{name}.parquet")


def load(name: str) -> pd.DataFrame | None:
    path = CACHE_DIR / f"{name}.parquet"
    return pd.read_parquet(path) if path.exists() else None
