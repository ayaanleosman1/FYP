import json
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

BASE_DIR = Path(__file__).resolve().parent.parent
OUTPUTS_DIR = BASE_DIR / "outputs"

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

def _read_json(filename: str):
    path = OUTPUTS_DIR / filename
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"File not found: {filename}")
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to read {filename}: {e}")

@app.get("/metrics")
def metrics(model: str = "xgb", horizon: int = 24):
    return _read_json(f"metrics_{model}_{horizon}.json")

@app.get("/predict")
def predict(model: str = "xgb", horizon: int = 24):
    return _read_json(f"preds_{model}_{horizon}.json")
