import json
import os
import sys
from pathlib import Path
from typing import Optional, Dict
from datetime import datetime

import httpx
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from openai import OpenAI
import joblib
import numpy as np

load_dotenv(Path(__file__).parent / ".env")

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

openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

app = FastAPI(
    title="UK Electricity Demand Forecast API",
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


@app.get("/granularities")
def granularities():
    return {"granularities": get_all_granularities()}


@app.get("/available")
def available():
    return {"available": list_available_models()}


@app.get("/dashboard")
def dashboard():
    model_ids = ["xgb", "rf", "linear", "ebm"]
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
        for model_id in model_ids:
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
            "n_models": len(model_ids),
            "n_features": n_features,
            "best_smape": round(best_smape, 2) if best_smape < float("inf") else None,
            "best_model": best_model,
            "best_granularity": best_granularity,
        },
        "performance": performance,
        "best_forecast": best_forecast,
        "top_features": top_features,
    }


def _read_output(granularity_code, file_type, model, horizon):
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

    if granularity_code == "H":
        data = load_legacy_outputs(file_type, model, horizon)
        if data is not None:
            data["granularity"] = "H"
            data["granularity_name"] = "hourly"
            return data

    raise HTTPException(
        status_code=404,
        detail=f"No {file_type} found for model={model}, granularity={granularity_code}, horizon={horizon}"
    )


@app.get("/metrics")
def metrics(
    model: str = Query(default="xgb"),
    horizon: int = Query(default=24),
    granularity: str = Query(default="H"),
):
    return _read_output(granularity, "metrics", model, horizon)


@app.get("/interpret")
def interpret(
    granularity: str = Query(default="W"),
    horizon: int = Query(default=4),
):
    try:
        gran = Granularity.from_code(granularity)
    except ValueError:
        raise HTTPException(status_code=400, detail=f"Invalid granularity: {granularity}")

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
    granularity: str = Query(default="D"),
    horizon: int = Query(default=7),
    model: str = Query(default="xgb"),
):
    try:
        gran = Granularity.from_code(granularity)
    except ValueError:
        raise HTTPException(status_code=400, detail=f"Invalid granularity: {granularity}")

    shap_path = OUTPUTS_DIR / gran.config.folder_name / f"shap_{model}_{horizon}.json"
    if not shap_path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"No SHAP analysis found for model={model}, granularity={granularity}, horizon={horizon}. "
                   f"Run: python ml/generate_shap.py -g {granularity} -m {model}"
        )

    with open(shap_path) as f:
        return json.load(f)


@app.get("/shap/available")
def shap_available(
    granularity: str = Query(default="D"),
    horizon: int = Query(default=7),
):
    try:
        gran = Granularity.from_code(granularity)
    except ValueError:
        raise HTTPException(status_code=400, detail=f"Invalid granularity: {granularity}")

    folder = OUTPUTS_DIR / gran.config.folder_name
    available = []
    for model_id in ["xgb", "rf", "linear", "ebm", "hybrid"]:
        path = folder / f"shap_{model_id}_{horizon}.json"
        if path.exists():
            available.append(model_id)

    return {"granularity": granularity, "horizon": horizon, "models": available}


class WhatIfRequest(BaseModel):
    features: Dict[str, float]
    granularity: str = "H"
    horizon: int = 24


class SensitivityRequest(BaseModel):
    feature: str
    granularity: str = "H"
    horizon: int = 24
    steps: int = 30
    base_features: Dict[str, float] = {}


@app.get("/ebm-shapes")
def ebm_shapes(
    granularity: str = Query(default="D"),
    horizon: int = Query(default=7),
):
    try:
        gran = Granularity.from_code(granularity)
    except ValueError:
        raise HTTPException(status_code=400, detail=f"Invalid granularity: {granularity}")

    shapes_path = OUTPUTS_DIR / gran.config.folder_name / f"ebm_shapes_{horizon}.json"
    if not shapes_path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"No EBM shapes found for granularity={granularity}, horizon={horizon}"
        )

    with open(shapes_path) as f:
        return json.load(f)


@app.post("/whatif")
def whatif_predict(request: WhatIfRequest):
    try:
        gran = Granularity.from_code(request.granularity)
    except ValueError:
        raise HTTPException(status_code=400, detail=f"Invalid granularity: {request.granularity}")

    model_path = OUTPUTS_DIR / gran.config.folder_name / "models" / f"ebm_{request.horizon}.joblib"
    if not model_path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"No EBM model found. Train with: python train_ebm.py --granularity {request.granularity}"
        )

    interp_path = OUTPUTS_DIR / gran.config.folder_name / f"interpretation_ebm_{request.horizon}.json"
    with open(interp_path) as f:
        interp_data = json.load(f)

    expected_features = interp_data["features"]
    feature_ranges = interp_data.get("feature_ranges", {})

    feature_values = []
    for feat in expected_features:
        if feat in request.features:
            feature_values.append(request.features[feat])
        elif feat in feature_ranges:
            feature_values.append(feature_ranges[feat]["median"])
        else:
            feature_values.append(0)

    model_data = joblib.load(model_path)
    ebm = model_data["model"]

    X = np.array([feature_values])
    prediction = float(ebm.predict(X)[0])

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
    granularity: str = Query(default="H"),
    horizon: int = Query(default=24),
):
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
    model: str = Query(default="xgb"),
    horizon: int = Query(default=24),
    granularity: str = Query(default="H"),
):
    return _read_output(granularity, "preds", model, horizon)


@app.get("/predict/aggregated")
def predict_aggregated(
    model: str = Query(default="xgb"),
    horizon: int = Query(default=24),
    target_granularity: str = Query(default="D"),
    aggregation: str = Query(default="sum"),
):
    import pandas as pd

    if target_granularity == "H":
        raise HTTPException(status_code=400, detail="Target granularity must be coarser than hourly (D, W, M, Y)")

    try:
        target_gran = Granularity.from_code(target_granularity)
    except ValueError:
        raise HTTPException(status_code=400, detail=f"Invalid target granularity: {target_granularity}")

    hourly_data = _read_output("H", "preds", model, horizon)

    series = hourly_data.get("series", [])
    if not series:
        raise HTTPException(status_code=404, detail="No prediction data found")

    df = pd.DataFrame(series)
    df["t"] = pd.to_datetime(df["t"])
    df = df.set_index("t")

    target_config = target_gran.config
    freq = target_config.pandas_freq

    if aggregation == "sum":
        agg_df = df.resample(freq).sum()
    elif aggregation == "mean":
        agg_df = df.resample(freq).mean()
    else:
        raise HTTPException(status_code=400, detail=f"Invalid aggregation: {aggregation}. Use 'sum' or 'mean'")

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



LONDON_LAT = 51.51
LONDON_LON = -0.13

@app.get("/live-forecast")
async def live_forecast(
    granularity: str = Query(default="H"),
    horizon: int = Query(default=24),
    temp_offset: float = Query(default=0.0),
    hour_override: Optional[int] = Query(default=None),
    dow_override: Optional[int] = Query(default=None),
):
    try:
        gran = Granularity.from_code(granularity)
    except ValueError:
        raise HTTPException(status_code=400, detail=f"Invalid granularity: {granularity}")

    weather = {}
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.get(
                "https://api.open-meteo.com/v1/forecast",
                params={
                    "latitude": LONDON_LAT,
                    "longitude": LONDON_LON,
                    "current": "temperature_2m,relative_humidity_2m,wind_speed_10m,cloud_cover",
                    "timezone": "Europe/London",
                }
            )
            if resp.status_code == 200:
                data = resp.json()
                current = data.get("current", {})
                weather = {
                    "temperature": current.get("temperature_2m", 10),
                    "humidity": current.get("relative_humidity_2m", 70),
                    "wind_speed": current.get("wind_speed_10m", 15),
                    "cloud_cover": current.get("cloud_cover", 50),
                    "source": "live",
                }
    except Exception:
        weather = {"temperature": 10, "humidity": 70, "wind_speed": 15, "cloud_cover": 50, "source": "fallback"}

    weather["temperature"] += temp_offset

    now = datetime.now()
    hour = hour_override if hour_override is not None else now.hour
    dow = dow_override if dow_override is not None else now.weekday()
    month = now.month

    model_path = OUTPUTS_DIR / gran.config.folder_name / "models" / f"ebm_{horizon}.joblib"
    if not model_path.exists():
        raise HTTPException(status_code=404, detail="EBM model not found for this granularity")

    interp_path = OUTPUTS_DIR / gran.config.folder_name / f"interpretation_ebm_{horizon}.json"
    if not interp_path.exists():
        raise HTTPException(status_code=404, detail="No interpretation data")

    model_data = joblib.load(model_path)
    ebm = model_data["model"]

    with open(interp_path) as f:
        interp = json.load(f)

    expected_features = interp["features"]
    feature_ranges = interp.get("feature_ranges", {})

    feature_map = {
        "hour": hour,
        "dow": dow,
        "month": month,
        "is_holiday": 0,
        "is_weekend": 1 if dow >= 5 else 0,
        "day_of_year": now.timetuple().tm_yday,
        "week_of_year": now.isocalendar()[1],
        "quarter": (month - 1) // 3 + 1,
        "temp": weather["temperature"],
        "humidity": weather["humidity"],
        "wind_speed": weather["wind_speed"],
    }

    feature_values = []
    for feat in expected_features:
        if feat in feature_map:
            feature_values.append(feature_map[feat])
        elif feat in feature_ranges:
            feature_values.append(feature_ranges[feat]["median"])
        else:
            feature_values.append(0)

    X = np.array([feature_values])
    prediction = float(ebm.predict(X)[0])

    contributions = {}
    try:
        import pandas as pd
        X_df = pd.DataFrame(X, columns=expected_features)
        local_exp = ebm.explain_local(X_df)
        exp_data = local_exp.data(0)
        for i, feat in enumerate(expected_features):
            if i < len(exp_data["scores"]):
                contributions[feat] = round(float(exp_data["scores"][i]), 1)
    except Exception:
        pass

    sorted_contribs = sorted(contributions.items(), key=lambda x: abs(x[1]), reverse=True)

    avg_uk_home_kw = 1.0  # ~1 kW average household consumption
    homes_equivalent = int(prediction / avg_uk_home_kw) if prediction > 0 else 0
    kettles = int(prediction / 2.0)  # a kettle is ~2 kW

    if prediction > 40000:
        level = "critical"
        level_label = "Very High"
    elif prediction > 35000:
        level = "high"
        level_label = "High"
    elif prediction > 28000:
        level = "moderate"
        level_label = "Moderate"
    elif prediction > 20000:
        level = "low"
        level_label = "Low"
    else:
        level = "very_low"
        level_label = "Very Low"

    hourly_predictions = []
    for h in range(24):
        h_features = dict(zip(expected_features, feature_values))
        if "hour" in h_features:
            h_features["hour"] = h
        h_vec = [h_features[f] for f in expected_features]
        h_pred = float(ebm.predict(np.array([h_vec]))[0])
        hourly_predictions.append({"hour": h, "demand": round(h_pred, 0)})

    peak_hour = max(hourly_predictions, key=lambda x: x["demand"])
    trough_hour = min(hourly_predictions, key=lambda x: x["demand"])

    return {
        "prediction": round(prediction, 0),
        "unit": "MW",
        "weather": weather,
        "time": {"hour": hour, "dow": dow, "month": month, "day_name": ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"][dow]},
        "level": level,
        "level_label": level_label,
        "contributions": sorted_contribs[:8],
        "context": {
            "homes_equivalent": homes_equivalent,
            "kettles_equivalent": kettles,
            "peak": {"hour": peak_hour["hour"], "demand": peak_hour["demand"]},
            "trough": {"hour": trough_hour["hour"], "demand": trough_hour["demand"]},
        },
        "hourly_curve": hourly_predictions,
        "features_used": {feat: round(feature_values[i], 2) for i, feat in enumerate(expected_features)},
    }



class ChatRequest(BaseModel):
    message: str
    context: dict = {}


class ChatResponse(BaseModel):
    response: str
    model: str = "gpt-4o-mini"


def build_system_prompt(context):
    base_prompt = """You are an AI assistant for a UK electricity demand forecasting dashboard.

You help users understand:
- Electricity demand forecasts and predictions
- Model performance metrics (MAE, RMSE, SMAPE, MAPE)
- Comparisons between XGBoost, Random Forest, Linear Regression, and EBM models
- UK National Grid demand patterns
- Feature importances

Data source: National Grid ESO/NESO historic demand data.
Demand values are in megawatts (MW) for hourly or megawatt-hours (MWh) for aggregated periods.

Be concise and helpful. Use the context provided about the current view when relevant."""

    if context:
        base_prompt += f"\n\nCurrent dashboard context:\n{json.dumps(context, indent=2)}"

    return base_prompt


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    system_prompt = build_system_prompt(request.context)

    response = openai_client.chat.completions.create(
        model="gpt-4o-mini",
        max_tokens=1024,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": request.message},
        ]
    )

    return ChatResponse(response=response.choices[0].message.content)


class ShapExplainRequest(BaseModel):
    model_name: str
    granularity: str
    top_features: list
    n_samples: int


class ShapExplainResponse(BaseModel):
    explanation: str


@app.post("/shap/explain", response_model=ShapExplainResponse)
def shap_explain(request: ShapExplainRequest):
    features_str = "\n".join(
        f"- {f['name']}: importance {f['importance']:.1f}" for f in request.top_features[:10]
    )

    prompt = f"""You are an expert data scientist explaining SHAP analysis results for a UK electricity demand forecasting model to a non-technical audience.

Model: {request.model_name}
Granularity: {request.granularity}
Test samples: {request.n_samples}

Top features by SHAP importance:
{features_str}

Write a clear 3-4 paragraph explanation of what these SHAP results tell us. Cover:
1. What the top features mean and why they matter for electricity demand
2. Any interesting patterns (e.g. lag features vs weather vs calendar)
3. What this means practically for forecasting UK electricity demand

Use plain English. Avoid jargon. Keep it concise but insightful. Don't use bullet points - write flowing paragraphs."""

    response = openai_client.chat.completions.create(
        model="gpt-4o-mini",
        max_tokens=512,
        temperature=0.7,
        messages=[
            {"role": "system", "content": "You are a data science expert who explains ML concepts clearly to non-technical audiences."},
            {"role": "user", "content": prompt},
        ]
    )

    return ShapExplainResponse(explanation=response.choices[0].message.content)


class AdvisorRequest(BaseModel):
    appliances: list
    total_kwh: float
    daily_cost: float
    monthly_cost: float
    yearly_cost: float
    cheapest_hour: int
    peak_hour: int
    current_price: float
    cheapest_price: float
    peak_price: float


class AdvisorResponse(BaseModel):
    tips: list


@app.post("/advisor", response_model=AdvisorResponse)
def advisor(request: AdvisorRequest):
    appliance_list = ", ".join(request.appliances)

    prompt = f"""You are a UK energy advisor AI. A household user has selected these appliances: {appliance_list}.

Their usage data:
- Total daily energy: {request.total_kwh:.2f} kWh
- Daily cost: \u00a3{request.daily_cost:.2f}, Monthly: \u00a3{request.monthly_cost:.2f}, Yearly: \u00a3{request.yearly_cost:.0f}
- Current electricity price: {request.current_price:.1f}p/kWh
- Cheapest hour today: {request.cheapest_hour}:00 ({request.cheapest_price:.1f}p/kWh)
- Peak hour today: {request.peak_hour}:00 ({request.peak_price:.1f}p/kWh)

Give exactly 4 short, specific, actionable energy-saving tips personalized to their appliances. Each tip should be 1-2 sentences max. Make them practical UK-focused advice. Be specific to THEIR appliances, not generic.

Return as a JSON array of 4 objects, each with "icon" (single emoji), "title" (max 5 words), and "body" (the tip). Return ONLY the JSON array, no markdown."""

    response = openai_client.chat.completions.create(
        model="gpt-4o-mini",
        max_tokens=512,
        temperature=0.7,
        messages=[
            {"role": "system", "content": "You are a helpful UK energy advisor. Always respond with valid JSON only."},
            {"role": "user", "content": prompt},
        ]
    )

    try:
        tips = json.loads(response.choices[0].message.content)
    except json.JSONDecodeError:
        tips = [
            {"icon": "\u26a1", "title": "Shift to off-peak", "body": f"Run flexible appliances at {request.cheapest_hour}:00 when prices are lowest."},
            {"icon": "\ud83c\udf21\ufe0f", "title": "Watch your heating", "body": "Heating is the biggest energy cost in UK homes. Drop 1\u00b0C to save ~\u00a380/year."},
            {"icon": "\ud83d\udca1", "title": "Standby wastes energy", "body": "UK households waste \u00a355/year on standby. Unplug devices when not in use."},
            {"icon": "\ud83c\udf0d", "title": "Go green at night", "body": "Grid carbon intensity drops overnight. Schedule appliances after 10pm for a greener footprint."},
        ]

    return AdvisorResponse(tips=tips)
