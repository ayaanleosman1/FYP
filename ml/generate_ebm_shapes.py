"""
Extract and save EBM shape functions for visualization.
Shows exactly how each feature value affects predictions.
"""

import json
import sys
from pathlib import Path
import numpy as np
import joblib

sys.path.insert(0, str(Path(__file__).parent))
from utils.granularity import Granularity

BASE_DIR = Path(__file__).parent.parent
OUTPUTS_DIR = BASE_DIR / "outputs"


def extract_ebm_shapes(granularity_code: str = "D", horizon: int = 7):
    """Extract shape functions from trained EBM model."""

    print(f"Extracting EBM shapes for {granularity_code}...")

    gran = Granularity.from_code(granularity_code)
    model_path = OUTPUTS_DIR / gran.config.folder_name / "models" / f"ebm_{horizon}.joblib"

    if not model_path.exists():
        print(f"No EBM model found at {model_path}")
        return None

    model_data = joblib.load(model_path)
    ebm = model_data["model"]

    # Get global explanation (shape functions)
    global_exp = ebm.explain_global()

    feature_names = global_exp.data()["names"]

    shape_data = {
        "granularity": granularity_code,
        "granularity_name": gran.config.name,
        "horizon": horizon,
        "features": [],
    }

    # Extract shape function for each feature
    for i, feat_name in enumerate(feature_names):
        if feat_name.startswith("interaction"):
            continue  # Skip interaction terms for now

        feat_data = global_exp.data(i)

        # Get x values (feature bins) and y values (contributions)
        if "names" in feat_data and "scores" in feat_data:
            x_vals = feat_data["names"]
            y_vals = feat_data["scores"]

            # Convert to lists and handle numpy types
            if hasattr(x_vals, 'tolist'):
                x_vals = x_vals.tolist()
            if hasattr(y_vals, 'tolist'):
                y_vals = y_vals.tolist()

            # Clean up values
            x_clean = []
            y_clean = []
            for x, y in zip(x_vals, y_vals):
                try:
                    if isinstance(x, (int, float)) and not np.isnan(x) and not np.isinf(x):
                        x_clean.append(round(float(x), 2))
                        y_clean.append(round(float(y), 2))
                    elif isinstance(x, str):
                        x_clean.append(x)
                        y_clean.append(round(float(y), 2))
                except:
                    pass

            if len(x_clean) > 0:
                shape_data["features"].append({
                    "name": feat_name,
                    "importance": round(float(np.mean(np.abs(y_clean))), 2),
                    "x": x_clean[:50],  # Limit points for frontend
                    "y": y_clean[:50],
                    "min_effect": round(min(y_clean), 2) if y_clean else 0,
                    "max_effect": round(max(y_clean), 2) if y_clean else 0,
                })

    # Sort by importance
    shape_data["features"].sort(key=lambda x: -x["importance"])

    # Save
    output_path = OUTPUTS_DIR / gran.config.folder_name / f"ebm_shapes_{horizon}.json"
    with open(output_path, "w") as f:
        json.dump(shape_data, f, indent=2)

    print(f"Saved {len(shape_data['features'])} feature shapes to {output_path}")
    print(f"Top features: {[f['name'] for f in shape_data['features'][:5]]}")

    return shape_data


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--granularity", "-g", default="D")
    parser.add_argument("--horizon", type=int, default=None)
    parser.add_argument("--all", action="store_true")
    args = parser.parse_args()

    default_horizons = {"H": 24, "D": 7, "W": 4, "M": 3}

    if args.all:
        for g in ["H", "D", "W", "M"]:
            try:
                extract_ebm_shapes(g, default_horizons[g])
            except Exception as e:
                print(f"Error for {g}: {e}")
    else:
        horizon = args.horizon or default_horizons.get(args.granularity, 7)
        extract_ebm_shapes(args.granularity, horizon)
