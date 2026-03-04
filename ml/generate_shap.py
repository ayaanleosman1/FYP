import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import shap
import joblib

sys.path.insert(0, str(Path(__file__).parent))

from utils.data import get_data_for_granularity, get_recommended_days_for_granularity, train_test_split_temporal
from utils.features import build_features, get_available_features
from utils.granularity import Granularity

BASE_DIR = Path(__file__).parent.parent
OUTPUTS_DIR = BASE_DIR / "outputs"

MODEL_NAMES = {
    "xgb": "XGBoost",
    "rf": "Random Forest",
    "linear": "Linear Regression",
    "ebm": "EBM (Explainable Boosting Machine)",
    "hybrid": "Hybrid Prophet+XGBoost (residual component)",
}

MODEL_JOBLIB_PATTERNS = {
    "xgb": "xgb_{horizon}.joblib",
    "rf": "rf_{horizon}.joblib",
    "linear": "linear_{horizon}.joblib",
    "ebm": "ebm_{horizon}.joblib",
    "hybrid": "hybrid_xgb_{horizon}.joblib",
}

ALL_MODELS = ["xgb", "rf", "linear", "ebm", "hybrid"]


def _get_ebm_shap_values(ebm_model, X_sample, feature_names):
    X_df = pd.DataFrame(X_sample, columns=feature_names)
    local_exp = ebm_model.explain_local(X_df)
    n_samples = len(X_sample)
    n_features = len(feature_names)
    shap_vals = np.zeros((n_samples, n_features))

    for i in range(n_samples):
        data = local_exp.data(i)
        names = data["names"]
        scores = data["scores"]
        # map scores back, skip interaction terms
        name_to_score = {}
        for name, score in zip(names, scores):
            if " x " not in str(name):
                name_to_score[str(name)] = score
        for j, feat in enumerate(feature_names):
            shap_vals[i, j] = name_to_score.get(feat, 0.0)

    return shap_vals


def _compute_shap_values(model_type, model, model_data, X_train, X_sample):
    if model_type in ("xgb", "rf", "hybrid"):
        explainer = shap.TreeExplainer(model)
        # RF can fail additivity checks due to floating point
        return explainer.shap_values(X_sample, check_additivity=(model_type != "rf"))
    elif model_type == "linear":
        background = model_data.get("background", X_train[:100])
        explainer = shap.LinearExplainer(model, background)
        return explainer.shap_values(X_sample)
    elif model_type == "ebm":
        feature_names = model_data["features"]
        return _get_ebm_shap_values(model, X_sample, feature_names)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def generate_shap_analysis(granularity_code="D", horizon=7, model_type="xgb", sample_size=500):
    print(f"Generating SHAP analysis for {MODEL_NAMES.get(model_type, model_type)} at {granularity_code} granularity...")

    gran = Granularity.from_code(granularity_code)

    joblib_name = MODEL_JOBLIB_PATTERNS[model_type].format(horizon=horizon)
    model_path = OUTPUTS_DIR / gran.config.folder_name / "models" / joblib_name

    if not model_path.exists():
        print(f"  Model not found: {model_path}")
        print(f"  Train the model first, then re-run SHAP generation.")
        return None

    print(f"  Loading model from {model_path}...")
    model_data = joblib.load(model_path)
    model = model_data["model"]
    saved_features = model_data["features"]

    print("  Loading data...")
    n_days = get_recommended_days_for_granularity(gran)
    df = get_data_for_granularity(n_days=n_days, granularity=gran)

    print("  Building features...")
    df_feat = build_features(df, granularity=gran, target_col="demand")

    available_features = [c for c in saved_features if c in df_feat.columns]
    if len(available_features) < len(saved_features):
        missing = set(saved_features) - set(available_features)
        print(f"  Warning: missing features {missing}, using {len(available_features)}/{len(saved_features)}")

    if not available_features:
        print(f"  Error: no matching features found. Model expects: {saved_features}")
        return None

    print(f"  Features ({len(available_features)}): {available_features}")

    test_periods = gran.config.default_test_periods
    train_df, test_df = train_test_split_temporal(df_feat, test_periods, gran)

    X_train = train_df[available_features].values
    X_test = test_df[available_features].values

    print(f"  Train: {len(X_train)}, Test: {len(X_test)}")

    if len(X_test) > sample_size:
        idx = np.random.choice(len(X_test), sample_size, replace=False)
        X_sample = X_test[idx]
    else:
        X_sample = X_test

    print(f"  Computing SHAP values for {len(X_sample)} samples...")

    shap_values = _compute_shap_values(model_type, model, model_data, X_train, X_sample)

    mean_abs_shap = np.abs(shap_values).mean(axis=0)

    feature_importance = {
        available_features[i]: float(mean_abs_shap[i])
        for i in range(len(available_features))
    }

    sorted_features = sorted(feature_importance.items(), key=lambda x: -x[1])

    print("  Top features by SHAP importance:")
    for feat, imp in sorted_features[:5]:
        print(f"    {feat}: {imp:.2f}")

    summary_data = {
        "model": model_type,
        "model_name": MODEL_NAMES.get(model_type, model_type),
        "granularity": granularity_code,
        "granularity_name": gran.config.name,
        "horizon": horizon,
        "n_samples": int(len(X_sample)),
        "n_train": int(len(X_train)),
        "features": [f[0] for f in sorted_features],
        "importance": [round(f[1], 2) for f in sorted_features],
        "feature_importance": {k: round(v, 2) for k, v in dict(sorted_features).items()},
    }

    if model_type == "hybrid":
        summary_data["note"] = (
            "This SHAP analysis covers the XGBoost residual component of the Hybrid model. "
            "Prophet captures trend and seasonality; XGBoost captures the remaining signal "
            "from features like lags and weather."
        )

    # distribution data for beeswarm plot (top 10 features)
    distribution_data = []
    for i, feat in enumerate(available_features):
        feat_values = X_sample[:, i].tolist()
        shap_vals = shap_values[:, i].tolist()

        # limit to 150 points for frontend performance
        if len(feat_values) > 150:
            sample_idx = np.random.choice(len(feat_values), 150, replace=False)
            feat_values = [round(feat_values[j], 2) for j in sample_idx]
            shap_vals = [round(shap_vals[j], 2) for j in sample_idx]
        else:
            feat_values = [round(v, 2) for v in feat_values]
            shap_vals = [round(v, 2) for v in shap_vals]

        distribution_data.append({
            "feature": feat,
            "importance": round(float(mean_abs_shap[i]), 2),
            "values": feat_values,
            "shap_values": shap_vals,
            "min_val": round(float(X_sample[:, i].min()), 2),
            "max_val": round(float(X_sample[:, i].max()), 2),
            "mean_val": round(float(X_sample[:, i].mean()), 2),
        })

    distribution_data.sort(key=lambda x: -x["importance"])
    summary_data["distribution"] = distribution_data[:10]

    output_path = OUTPUTS_DIR / gran.config.folder_name / f"shap_{model_type}_{horizon}.json"
    with open(output_path, "w") as f:
        json.dump(summary_data, f, indent=2)

    print(f"  Saved SHAP analysis to {output_path}")

    return summary_data


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Generate SHAP analysis for ML models")
    parser.add_argument("--granularity", "-g", default="D")
    parser.add_argument("--horizon", type=int, default=None)
    parser.add_argument("--model", "-m", default="xgb", choices=ALL_MODELS + ["all"])
    parser.add_argument("--all", action="store_true")
    args = parser.parse_args()

    default_horizons = {"H": 24, "D": 7, "W": 4, "M": 3}

    models = ALL_MODELS if args.model == "all" else [args.model]
    granularities = ["H", "D", "W", "M"] if args.all else [args.granularity]

    for g in granularities:
        horizon = args.horizon or default_horizons.get(g, 7)
        for m in models:
            try:
                generate_shap_analysis(g, horizon, model_type=m)
                print()
            except Exception as e:
                print(f"  Error for {m} at {g}: {e}")
                print()
