# backend/tests/test_api.py
import pytest
from fastapi.testclient import TestClient
from backend.api.main import app

client = TestClient(app)


def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_current_regime_responds():
    r = client.get("/regime/current")
    assert r.status_code in (200, 503)


def test_current_regime_schema_when_present():
    r = client.get("/regime/current")
    if r.status_code == 200:
        data = r.json()
        assert "regime" in data
        assert "indicators" in data
        assert "last_updated" in data
        assert "description" in data
        assert isinstance(data["regime"]["probability"], float)


def test_regime_history_responds():
    r = client.get("/regime/history")
    assert r.status_code in (200, 503)
    if r.status_code == 200:
        data = r.json()
        assert "history" in data
        assert len(data["history"]) > 0


def test_transition_forecast_responds():
    r = client.get("/regime/transition")
    assert r.status_code in (200, 503)
    if r.status_code == 200:
        data = r.json()
        assert "current_regime" in data
        assert "predicted_next_regime" in data
        assert "transition_probabilities" in data


def test_asset_implications_responds():
    r = client.get("/regime/implications")
    assert r.status_code in (200, 503)
    if r.status_code == 200:
        data = r.json()
        assert "implications" in data


def test_unknown_endpoint_404():
    r = client.get("/not/a/real/endpoint")
    assert r.status_code == 404
