from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="UK Electricity Demand Forecast API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/models")
def models():
    return {
        "models": [
            {"id": "linear", "name": "Linear Regression"},
            {"id": "rf", "name": "Random Forest"},
            {"id": "xgb", "name": "XGBoost"},
            {"id": "ebm", "name": "EBM (Explainable Boosting Machine)"},
        ]
    }

@app.get("/metrics")
def metrics(model: str = "xgb", horizon: int = 24):
    return {
        "model": model,
        "horizon_hours": horizon,
        "mae": 950.2,
        "rmse": 1310.7,
        "smape": 3.9,
    }

@app.get("/predict")
def predict(model: str = "xgb", horizon: int = 24):
    return {
        "model": model,
        "horizon_hours": horizon,
        "series": [
            {"t": "2026-01-01T00:00:00Z", "actual": 32000, "predicted": 31800},
            {"t": "2026-01-01T01:00:00Z", "actual": 31000, "predicted": 31200},
            {"t": "2026-01-01T02:00:00Z", "actual": 30500, "predicted": 30000},
        ],
    }
