# backend/api/schemas.py
from pydantic import BaseModel
from typing import Optional


class RegimeState(BaseModel):
    regime: str
    probability: float
    probabilities: dict[str, float]


class MacroIndicator(BaseModel):
    name: str
    value: Optional[float] = None
    zscore: float
    trend: str  # "up" | "down" | "flat"


class CurrentRegimeResponse(BaseModel):
    regime: RegimeState
    indicators: list[MacroIndicator]
    last_updated: str
    description: str


class RegimeHistoryPoint(BaseModel):
    date: str
    regime: str
    sp500_return: Optional[float] = None


class RegimeHistoryResponse(BaseModel):
    history: list[RegimeHistoryPoint]


class TransitionForecastResponse(BaseModel):
    current_regime: str
    predicted_next_regime: str
    transition_probabilities: dict[str, float]
    forecast_horizon_months: int


class AssetImplication(BaseModel):
    regime: str
    color: str
    asset_class: str
    median_monthly_return: float
    mean_monthly_return: float
    observation_count: int


class AssetImplicationsResponse(BaseModel):
    implications: list[AssetImplication]
