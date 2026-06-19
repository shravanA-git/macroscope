# backend/api/main.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from .routes.regime import router as regime_router

app = FastAPI(
    title="MacroScope API",
    description="Macroeconomic regime detection — HMM + AutoGluon",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET"],
    allow_headers=["Content-Type", "Accept"],
)

app.include_router(regime_router)


@app.get("/health")
def health():
    return {"status": "ok"}
