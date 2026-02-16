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


@app.get("/dashboard")
def dashboard():
    """Aggregated dashboard data for the landing page. Single call, fast load."""
    models = ["xgb", "rf", "linear", "ebm"]
    granularity_horizons = {"H": 24, "D": 7, "W": 4, "M": 3}

    performance = {}
    best_smape = float("inf")
    best_model = None
    best_granularity = None

    for gran_code, horizon in granularity_horizons.items():
        performance[gran_code] = {}
        try:
            gran = Granularity.from_code(gran_code)
        except ValueError:
            continue
        for model_id in models:
            metrics_data = load_outputs(gran, "metrics", model_id, horizon)
            if metrics_data is None:
                metrics_data = load_legacy_outputs("metrics", model_id, horizon)
            if metrics_data:
                smape_val = metrics_data.get("smape")
                performance[gran_code][model_id] = {
                    "smape": smape_val,
                    "mae": metrics_data.get("mae"),
                    "rmse": metrics_data.get("rmse"),
                    "mape": metrics_data.get("mape"),
                }
                if smape_val is not None and smape_val < best_smape:
                    best_smape = smape_val
                    best_model = model_id
                    best_granularity = gran_code

    # Best daily model for the preview chart
    daily_models = performance.get("D", {})
    best_daily_model = min(daily_models, key=lambda m: daily_models[m]["smape"]) if daily_models else "xgb"
    best_daily_smape = daily_models.get(best_daily_model, {}).get("smape", 0)

    best_forecast = {"model": best_daily_model, "granularity": "D", "smape": best_daily_smape, "series": []}
    try:
        gran_d = Granularity.from_code("D")
        preds_data = load_outputs(gran_d, "preds", best_daily_model, 7)
        if preds_data and "series" in preds_data:
            best_forecast["series"] = preds_data["series"]
    except Exception:
        pass

    # Top features from best daily model's SHAP data
    FEATURE_CATEGORIES = {
        "hour": "calendar", "dow": "calendar", "month": "calendar",
        "day_of_year": "calendar", "is_weekend": "calendar", "is_holiday": "calendar",
        "has_holiday": "calendar", "week_of_year": "calendar", "quarter": "calendar",
        "lag_1": "lag", "lag_7": "lag", "lag_12": "lag", "lag_24": "lag",
        "lag_52": "lag", "lag_168": "lag",
        "roll_3_mean": "lag", "roll_4_mean": "lag", "roll_7_mean": "lag",
        "roll_12_mean": "lag", "roll_24_mean": "lag", "roll_30_mean": "lag",
        "temp": "weather", "humidity": "weather", "wind_speed": "weather",
        "temp_roll_7": "weather", "temp_lag_7": "weather", "temp_lag_24": "weather",
        "solar_rad": "weather", "solar_rad_lag_24": "weather", "direct_rad": "weather",
        "gen_solar": "energy", "gen_gas": "energy", "gen_wind": "energy",
        "gen_nuclear": "energy", "carbon_intensity": "energy",
    }

    top_features = []
    n_features = 19
    try:
        gran_d = Granularity.from_code("D")
        shap_path = OUTPUTS_DIR / gran_d.config.folder_name / f"shap_{best_daily_model}_7.json"
        if shap_path.exists():
            with open(shap_path) as f:
                shap_data = json.load(f)
            features_list = shap_data.get("features", [])
            importance_list = shap_data.get("importance", [])
            n_features = len(features_list)
            for feat, imp in zip(features_list[:8], importance_list[:8]):
                top_features.append({
                    "name": feat,
                    "importance": imp,
                    "category": FEATURE_CATEGORIES.get(feat, "other"),
                })
    except Exception:
        pass

    return {
        "stats": {
            "data_years": 16,
            "n_models": len(models),
            "n_features": n_features,
            "best_smape": round(best_smape, 2) if best_smape < float("inf") else None,
            "best_model": best_model,
            "best_granularity": best_granularity,
        },
        "performance": performance,
        "best_forecast": best_forecast,
        "top_features": top_features,
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


@app.get("/shap")
def shap_analysis(
    granularity: str = Query(default="D", description="Granularity code (H, D, W, M)"),
    horizon: int = Query(default=7, description="Forecast horizon in periods"),
    model: str = Query(default="xgb", description="Model name (xgb, rf, linear, ebm, hybrid)"),
):
    """
    Get SHAP analysis for a model.

    SHAP (SHapley Additive exPlanations) provides detailed feature importance
    and shows how each feature contributes to individual predictions.

    Args:
        granularity: Granularity code - H(ourly), D(aily), W(eekly), M(onthly)
        horizon: Forecast horizon in periods
        model: Model type - xgb, rf, linear, ebm, hybrid

    Returns:
        SHAP feature importances and distribution data for visualization
    """
    try:
        gran = Granularity.from_code(granularity)
    except ValueError:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid granularity: {granularity}. Valid codes: H, D, W, M"
        )

    shap_path = OUTPUTS_DIR / gran.config.folder_name / f"shap_{model}_{horizon}.json"
    if not shap_path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"No SHAP analysis found for model={model}, granularity={granularity}, horizon={horizon}. Run: python ml/generate_shap.py -g {granularity} -m {model}"
        )

    with open(shap_path) as f:
        return json.load(f)


@app.get("/shap/available")
def shap_available(
    granularity: str = Query(default="D", description="Granularity code (H, D, W, M)"),
    horizon: int = Query(default=7, description="Forecast horizon in periods"),
):
    """List which models have SHAP data for a given granularity and horizon."""
    try:
        gran = Granularity.from_code(granularity)
    except ValueError:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid granularity: {granularity}. Valid codes: H, D, W, M"
        )

    folder = OUTPUTS_DIR / gran.config.folder_name
    available = []
    for model_id in ["xgb", "rf", "linear", "ebm", "hybrid"]:
        path = folder / f"shap_{model_id}_{horizon}.json"
        if path.exists():
            available.append(model_id)

    return {"granularity": granularity, "horizon": horizon, "models": available}


class WhatIfRequest(BaseModel):
    """Request body for what-if prediction."""
    features: Dict[str, float]
    granularity: str = "H"
    horizon: int = 24


class SensitivityRequest(BaseModel):
    """Request body for sensitivity analysis."""
    feature: str
    granularity: str = "H"
    horizon: int = 24
    steps: int = 30
    base_features: Dict[str, float] = {}


@app.get("/ebm-shapes")
def ebm_shapes(
    granularity: str = Query(default="D", description="Granularity code (H, D, W, M)"),
    horizon: int = Query(default=7, description="Forecast horizon in periods"),
):
    """
    Get EBM shape functions for visualization.

    Shape functions show exactly how each feature value maps to its contribution
    to the prediction. This is a key advantage of EBM's glass-box interpretability.

    Args:
        granularity: Granularity code - H(ourly), D(aily), W(eekly), M(onthly)
        horizon: Forecast horizon in periods

    Returns:
        Shape function data for each feature (x values and corresponding y contributions)
    """
    try:
        gran = Granularity.from_code(granularity)
    except ValueError:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid granularity: {granularity}. Valid codes: H, D, W, M"
        )

    shapes_path = OUTPUTS_DIR / gran.config.folder_name / f"ebm_shapes_{horizon}.json"
    if not shapes_path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"No EBM shapes found for granularity={granularity}, horizon={horizon}. Run: python ml/generate_ebm_shapes.py -g {granularity}"
        )

    with open(shapes_path) as f:
        return json.load(f)


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


@app.post("/whatif/sensitivity")
def whatif_sensitivity(request: SensitivityRequest):
    """
    Sensitivity analysis: sweep one feature across its range and return predictions.
    """
    try:
        gran = Granularity.from_code(request.granularity)
    except ValueError:
        raise HTTPException(status_code=400, detail=f"Invalid granularity: {request.granularity}")

    model_path = OUTPUTS_DIR / gran.config.folder_name / "models" / f"ebm_{request.horizon}.joblib"
    if not model_path.exists():
        raise HTTPException(status_code=404, detail="No EBM model found for this configuration")

    interp_path = OUTPUTS_DIR / gran.config.folder_name / f"interpretation_ebm_{request.horizon}.json"
    if not interp_path.exists():
        raise HTTPException(status_code=404, detail="No interpretation data found")

    with open(interp_path) as f:
        interp_data = json.load(f)

    expected_features = interp_data["features"]
    feature_ranges = interp_data.get("feature_ranges", {})

    if request.feature not in expected_features:
        raise HTTPException(status_code=400, detail=f"Unknown feature: {request.feature}")

    feat_range = feature_ranges.get(request.feature)
    if not feat_range:
        raise HTTPException(status_code=400, detail=f"No range data for feature: {request.feature}")

    model_data = joblib.load(model_path)
    ebm = model_data["model"]

    # Build base feature vector
    base_values = {}
    for feat in expected_features:
        if feat in request.base_features:
            base_values[feat] = request.base_features[feat]
        elif feat in feature_ranges:
            base_values[feat] = feature_ranges[feat]["median"]
        else:
            base_values[feat] = 0

    sweep_values = np.linspace(feat_range["min"], feat_range["max"], request.steps)
    sweep = []

    for val in sweep_values:
        feature_vector = [base_values[f] if f != request.feature else float(val) for f in expected_features]
        X = np.array([feature_vector])
        pred = float(ebm.predict(X)[0])
        sweep.append({"value": round(float(val), 4), "prediction": round(pred, 2)})

    # Base prediction (at median of swept feature)
    base_vector = [base_values[f] for f in expected_features]
    base_prediction = float(ebm.predict(np.array([base_vector]))[0])

    return {
        "feature": request.feature,
        "sweep": sweep,
        "base_prediction": round(base_prediction, 2),
        "feature_range": {
            "min": feat_range["min"],
            "max": feat_range["max"],
            "median": feat_range["median"],
        },
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
