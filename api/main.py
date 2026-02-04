"""
UK Electricity Demand Forecast API

Provides endpoints for accessing demand forecasts at various granularities.
"""

import json
import sys
from pathlib import Path
from typing import Optional, Dict

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from anthropic import Anthropic
import joblib
import numpy as np

# Add ml directory to path for imports
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR / "ml"))

from utils.granularity import Granularity, get_all_granularities, GRANULARITY_CONFIG
from utils.io import (
    get_outputs_root,
    load_outputs,
    load_legacy_outputs,
    list_available_models,
)

OUTPUTS_DIR = BASE_DIR / "outputs"

# Initialize Anthropic client (uses ANTHROPIC_API_KEY env var)
anthropic_client = Anthropic()

app = FastAPI(
    title="UK Electricity Demand Forecast API",
    description="API for accessing demand forecasts at various granularities (hourly, daily, weekly, monthly, yearly)",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:5174", "http://localhost:5175", "http://localhost:5176"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health():
    """Health check endpoint."""
    return {"status": "ok"}


@app.get("/models")
def models():
    """List available model types."""
    return {
        "models": [
            {"id": "linear", "name": "Linear Regression"},
            {"id": "rf", "name": "Random Forest"},
            {"id": "xgb", "name": "XGBoost"},
            {"id": "ebm", "name": "EBM (Explainable Boosting Machine)"},
        ]
    }


@app.get("/granularities")
def granularities():
    """
    List available forecast granularities.

    Returns information about each granularity including code, name,
    and default forecast horizon.
    """
    return {
        "granularities": get_all_granularities()
    }


@app.get("/available")
def available():
    """
    List all available trained models organized by granularity.

    Returns a dictionary mapping granularity codes to lists of
    available model/horizon combinations.
    """
    return {
        "available": list_available_models()
    }


def _read_output(
    granularity_code: str,
    file_type: str,
    model: str,
    horizon: int,
) -> dict:
    """
    Read an output file, checking both new granularity folders and legacy location.

    Args:
        granularity_code: Granularity code (H, D, W, M, Y)
        file_type: Either "metrics" or "preds"
        model: Model name
        horizon: Forecast horizon

    Returns:
        Parsed JSON dict

    Raises:
        HTTPException if file not found or read error
    """
    # Try new granularity-organized location first
    try:
        granularity = Granularity.from_code(granularity_code)
        data = load_outputs(granularity, file_type, model, horizon)
        if data is not None:
            return data
    except ValueError:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid granularity: {granularity_code}. Valid codes: H, D, W, M, Y"
        )

    # Fall back to legacy location for hourly data
    if granularity_code == "H":
        data = load_legacy_outputs(file_type, model, horizon)
        if data is not None:
            # Add granularity fields for consistency
            data["granularity"] = "H"
            data["granularity_name"] = "hourly"
            return data

    raise HTTPException(
        status_code=404,
        detail=f"No {file_type} found for model={model}, granularity={granularity_code}, horizon={horizon}"
    )


@app.get("/metrics")
def metrics(
    model: str = Query(default="xgb", description="Model name (xgb, rf, linear)"),
    horizon: int = Query(default=24, description="Forecast horizon in periods"),
    granularity: str = Query(default="H", description="Granularity code (H, D, W, M, Y)"),
):
    """
    Get metrics for a trained model.

    Args:
        model: Model name (xgb, rf, linear)
        horizon: Forecast horizon in periods (default varies by granularity)
        granularity: Granularity code - H(ourly), D(aily), W(eekly), M(onthly), Y(early)

    Returns:
        Model metrics including MAE, RMSE, SMAPE, MAPE
    """
    return _read_output(granularity, "metrics", model, horizon)


@app.get("/interpret")
def interpret(
    granularity: str = Query(default="W", description="Granularity code (H, D, W, M, Y)"),
    horizon: int = Query(default=4, description="Forecast horizon in periods"),
):
    """
    Get EBM model interpretation data (feature importances).

    EBM (Explainable Boosting Machine) is a glassbox model that provides
    full interpretability - you can see exactly how each feature affects predictions.

    Args:
        granularity: Granularity code - H(ourly), D(aily), W(eekly), M(onthly)
        horizon: Forecast horizon in periods

    Returns:
        Feature importances showing how much each feature contributes to predictions
    """
    try:
        gran = Granularity.from_code(granularity)
    except ValueError:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid granularity: {granularity}. Valid codes: H, D, W, M"
        )

    interp_path = OUTPUTS_DIR / gran.config.folder_name / f"interpretation_ebm_{horizon}.json"
    if not interp_path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"No interpretation data found for EBM at granularity={granularity}, horizon={horizon}"
        )

    with open(interp_path) as f:
        return json.load(f)


class WhatIfRequest(BaseModel):
    """Request body for what-if prediction."""
    features: Dict[str, float]
    granularity: str = "H"
    horizon: int = 24


@app.post("/whatif")
def whatif_predict(request: WhatIfRequest):
    """
    What-If Analysis: Get a prediction based on custom feature values.

    Users can adjust input features (temperature, hour, day of week, etc.)
    and see how the predicted demand changes.

    This endpoint uses the EBM (Explainable Boosting Machine) model.
    """
    try:
        gran = Granularity.from_code(request.granularity)
    except ValueError:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid granularity: {request.granularity}"
        )

    # Load the saved model
    model_path = OUTPUTS_DIR / gran.config.folder_name / "models" / f"ebm_{request.horizon}.joblib"
    if not model_path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"No EBM model found. Train with: python train_ebm.py --granularity {request.granularity}"
        )

    # Load interpretation data for feature info
    interp_path = OUTPUTS_DIR / gran.config.folder_name / f"interpretation_ebm_{request.horizon}.json"
    with open(interp_path) as f:
        interp_data = json.load(f)

    expected_features = interp_data["features"]
    feature_ranges = interp_data.get("feature_ranges", {})

    # Build feature vector in correct order
    feature_values = []
    for feat in expected_features:
        if feat in request.features:
            feature_values.append(request.features[feat])
        elif feat in feature_ranges:
            # Use median as default
            feature_values.append(feature_ranges[feat]["median"])
        else:
            feature_values.append(0)

    # Load model and predict
    model_data = joblib.load(model_path)
    ebm = model_data["model"]

    X = np.array([feature_values])
    prediction = float(ebm.predict(X)[0])

    # Get feature contributions (local explanation)
    contributions = {}
    try:
        local_exp = ebm.explain_local(X)
        for i, feat in enumerate(expected_features):
            contributions[feat] = float(local_exp.data(0)["scores"][i])
    except:
        pass

    return {
        "prediction": round(prediction, 2),
        "unit": "MW",
        "features_used": {feat: feature_values[i] for i, feat in enumerate(expected_features)},
        "contributions": contributions,
        "feature_ranges": feature_ranges,
    }


@app.get("/whatif/features")
def whatif_features(
    granularity: str = Query(default="H", description="Granularity code"),
    horizon: int = Query(default=24, description="Forecast horizon"),
):
    """
    Get available features and their ranges for What-If analysis.
    """
    try:
        gran = Granularity.from_code(granularity)
    except ValueError:
        raise HTTPException(status_code=400, detail=f"Invalid granularity: {granularity}")

    interp_path = OUTPUTS_DIR / gran.config.folder_name / f"interpretation_ebm_{horizon}.json"
    if not interp_path.exists():
        raise HTTPException(status_code=404, detail="No EBM model found for this granularity")

    with open(interp_path) as f:
        data = json.load(f)

    return {
        "features": data["features"],
        "feature_ranges": data.get("feature_ranges", {}),
        "feature_importances": data.get("feature_importances", {}),
    }


@app.get("/predict")
def predict(
    model: str = Query(default="xgb", description="Model name (xgb, rf, linear, ebm)"),
    horizon: int = Query(default=24, description="Forecast horizon in periods"),
    granularity: str = Query(default="H", description="Granularity code (H, D, W, M, Y)"),
):
    """
    Get predictions from a trained model.

    Args:
        model: Model name (xgb, rf, linear)
        horizon: Forecast horizon in periods (default varies by granularity)
        granularity: Granularity code - H(ourly), D(aily), W(eekly), M(onthly), Y(early)

    Returns:
        Prediction series with actual and predicted values
    """
    return _read_output(granularity, "preds", model, horizon)


@app.get("/predict/aggregated")
def predict_aggregated(
    model: str = Query(default="xgb", description="Model name"),
    horizon: int = Query(default=24, description="Source horizon (hourly)"),
    target_granularity: str = Query(default="D", description="Target granularity (D, W, M, Y)"),
    aggregation: str = Query(default="sum", description="Aggregation method (sum, mean)"),
):
    """
    Aggregate hourly predictions on-the-fly to a coarser granularity.

    This provides a quick view without needing to train at the target granularity.
    For best results, use natively trained models at the target granularity.

    Args:
        model: Model name
        horizon: Source hourly horizon
        target_granularity: Target granularity code (D, W, M, Y - not H)
        aggregation: How to aggregate (sum or mean)

    Returns:
        Aggregated prediction series
    """
    import pandas as pd

    # Validate target granularity
    if target_granularity == "H":
        raise HTTPException(
            status_code=400,
            detail="Target granularity must be coarser than hourly (D, W, M, Y)"
        )

    try:
        target_gran = Granularity.from_code(target_granularity)
    except ValueError:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid target granularity: {target_granularity}"
        )

    # Get hourly predictions
    hourly_data = _read_output("H", "preds", model, horizon)

    # Convert to DataFrame
    series = hourly_data.get("series", [])
    if not series:
        raise HTTPException(status_code=404, detail="No prediction data found")

    df = pd.DataFrame(series)
    df["t"] = pd.to_datetime(df["t"])
    df = df.set_index("t")

    # Get target frequency
    target_config = target_gran.config
    freq = target_config.pandas_freq

    # Aggregate
    if aggregation == "sum":
        agg_df = df.resample(freq).sum()
    elif aggregation == "mean":
        agg_df = df.resample(freq).mean()
    else:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid aggregation: {aggregation}. Use 'sum' or 'mean'"
        )

    # Format output
    aggregated_series = [
        {
            "t": idx.isoformat() + "Z" if not str(idx).endswith("Z") else str(idx),
            "actual": round(row["actual"], 2),
            "predicted": round(row["predicted"], 2),
        }
        for idx, row in agg_df.iterrows()
    ]

    return {
        "model": model,
        "source_granularity": "H",
        "source_horizon": horizon,
        "target_granularity": target_granularity,
        "target_granularity_name": target_config.name,
        "aggregation": aggregation,
        "note": "On-the-fly aggregation of hourly predictions. For better accuracy, use natively trained models.",
        "series": aggregated_series,
    }


# ============================================
# Chat Endpoint
# ============================================

class ChatRequest(BaseModel):
    message: str
    context: dict = {}  # Optional: current view data


class ChatResponse(BaseModel):
    response: str
    model: str = "claude-sonnet-4-20250514"


def build_system_prompt(context: dict) -> str:
    """Build system prompt with optional dashboard context."""
    base_prompt = """You are an AI assistant for a UK electricity demand forecasting dashboard.

You help users understand:
- Electricity demand forecasts and predictions
- Model performance metrics (MAE, RMSE, SMAPE, MAPE)
- Comparisons between XGBoost, Random Forest, Linear Regression, and EBM (Explainable Boosting Machine) models
- UK National Grid demand patterns
- EBM feature importances - which factors most influence predictions

Data source: National Grid ESO/NESO historic demand data.
Demand values are in megawatts (MW) for hourly or megawatt-hours (MWh) for aggregated periods.

Be concise and helpful. Use the context provided about the current view when relevant."""

    if context:
        base_prompt += f"\n\nCurrent dashboard context:\n{json.dumps(context, indent=2)}"

    return base_prompt


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    """
    Chat with AI assistant about electricity demand forecasting.
    """
    # Build system prompt with data context
    system_prompt = build_system_prompt(request.context)

    # Call Claude API
    message = anthropic_client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=1024,
        system=system_prompt,
        messages=[{"role": "user", "content": request.message}]
    )

    return ChatResponse(response=message.content[0].text)
