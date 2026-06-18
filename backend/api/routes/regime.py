# backend/api/routes/regime.py
import json
from pathlib import Path
from fastapi import APIRouter, HTTPException
from ..schemas import (
    CurrentRegimeResponse,
    RegimeHistoryResponse,
    TransitionForecastResponse,
    AssetImplicationsResponse,
)

router = APIRouter(prefix="/regime", tags=["regime"])
STATIC_DIR = Path(__file__).parent.parent.parent.parent / "static"


def _load(name: str) -> dict:
    path = STATIC_DIR / f"{name}.json"
    if not path.exists():
        raise HTTPException(
            status_code=503,
            detail=f"Data not yet computed. Run: python -m backend.scripts.refresh_predictions",
        )
    return json.loads(path.read_text())


@router.get("/current", response_model=CurrentRegimeResponse)
def get_current():
    return _load("current_regime")


@router.get("/history", response_model=RegimeHistoryResponse)
def get_history():
    return _load("regime_history")


@router.get("/transition", response_model=TransitionForecastResponse)
def get_transition():
    return _load("transition_probs")


@router.get("/implications", response_model=AssetImplicationsResponse)
def get_implications():
    return _load("asset_implications")
