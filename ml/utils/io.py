"""
Input/Output utilities for saving and loading model outputs.

Outputs are organized by granularity:
    outputs/
    ├── hourly/
    │   ├── metrics_xgb_24.json
    │   └── preds_xgb_24.json
    ├── daily/
    │   ├── metrics_xgb_7.json
    │   └── preds_xgb_7.json
    └── ...
"""

import json
from pathlib import Path
from typing import Union, Optional
import pandas as pd

from .granularity import Granularity, GRANULARITY_CONFIG


def get_outputs_root() -> Path:
    """Get the root outputs directory."""
    # Navigate from ml/utils/ to project root
    return Path(__file__).parent.parent.parent / "outputs"


def get_output_path(
    granularity: Granularity,
    file_type: str,  # "metrics" or "preds"
    model: str,
    horizon: int,
) -> Path:
    """
    Get the path for an output file.

    Args:
        granularity: The forecast granularity
        file_type: Either "metrics" or "preds"
        model: Model name (xgb, rf, linear)
        horizon: Forecast horizon in periods

    Returns:
        Path to the output file
    """
    folder = get_outputs_root() / granularity.config.folder_name
    return folder / f"{file_type}_{model}_{horizon}.json"


def save_outputs(
    granularity: Granularity,
    model: str,
    horizon: int,
    metrics: dict,
    predictions: list[dict],
    extra_metrics: Optional[dict] = None,
) -> tuple[Path, Path]:
    """
    Save model metrics and predictions to JSON files.

    Args:
        granularity: The forecast granularity
        model: Model name (xgb, rf, linear)
        horizon: Forecast horizon in periods
        metrics: Dictionary of metric values
        predictions: List of {t, actual, predicted} dicts
        extra_metrics: Optional extra fields to include in metrics file

    Returns:
        Tuple of (metrics_path, preds_path)
    """
    config = granularity.config
    folder = get_outputs_root() / config.folder_name
    folder.mkdir(parents=True, exist_ok=True)

    # Build metrics output
    metrics_out = {
        "model": model,
        "granularity": config.code,
        "granularity_name": config.name,
        "horizon": horizon,
        **metrics,
    }
    if extra_metrics:
        metrics_out.update(extra_metrics)

    # Build predictions output
    preds_out = {
        "model": model,
        "granularity": config.code,
        "granularity_name": config.name,
        "horizon": horizon,
        "series": predictions,
    }

    # Write files
    metrics_path = folder / f"metrics_{model}_{horizon}.json"
    preds_path = folder / f"preds_{model}_{horizon}.json"

    with open(metrics_path, "w") as f:
        json.dump(metrics_out, f, indent=2)

    with open(preds_path, "w") as f:
        json.dump(preds_out, f, indent=2)

    return metrics_path, preds_path


def load_outputs(
    granularity: Granularity,
    file_type: str,
    model: str,
    horizon: int,
) -> Optional[dict]:
    """
    Load output file if it exists.

    Args:
        granularity: The forecast granularity
        file_type: Either "metrics" or "preds"
        model: Model name
        horizon: Forecast horizon

    Returns:
        Parsed JSON dict or None if file doesn't exist
    """
    path = get_output_path(granularity, file_type, model, horizon)
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def load_legacy_outputs(
    file_type: str,
    model: str,
    horizon: int,
) -> Optional[dict]:
    """
    Load legacy output files from the root outputs/ folder (pre-granularity structure).

    This provides backward compatibility with existing outputs.

    Args:
        file_type: Either "metrics" or "preds"
        model: Model name
        horizon: Forecast horizon

    Returns:
        Parsed JSON dict or None if file doesn't exist
    """
    path = get_outputs_root() / f"{file_type}_{model}_{horizon}.json"
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def list_available_models(granularity: Optional[Granularity] = None) -> dict:
    """
    List all available trained models by granularity.

    Args:
        granularity: Optional specific granularity to check

    Returns:
        Dict mapping granularity codes to lists of {model, horizon} dicts
    """
    outputs_root = get_outputs_root()
    available = {}

    granularities = [granularity] if granularity else list(Granularity)

    for g in granularities:
        folder = outputs_root / g.config.folder_name
        if not folder.exists():
            available[g.value] = []
            continue

        models = []
        for f in folder.glob("metrics_*.json"):
            # Parse filename: metrics_{model}_{horizon}.json
            parts = f.stem.split("_")
            if len(parts) >= 3:
                model = parts[1]
                try:
                    horizon = int(parts[2])
                    models.append({"model": model, "horizon": horizon})
                except ValueError:
                    continue
        available[g.value] = models

    # Also check legacy files in root outputs/
    if granularity is None or granularity == Granularity.HOURLY:
        legacy_models = []
        for f in outputs_root.glob("metrics_*.json"):
            # Skip files in subdirectories
            if f.parent != outputs_root:
                continue
            parts = f.stem.split("_")
            if len(parts) >= 3:
                model = parts[1]
                try:
                    horizon = int(parts[2])
                    legacy_models.append({"model": model, "horizon": horizon, "legacy": True})
                except ValueError:
                    continue
        if legacy_models:
            if "H" not in available:
                available["H"] = []
            # Add legacy models if not already present
            existing = {(m["model"], m["horizon"]) for m in available["H"]}
            for lm in legacy_models:
                if (lm["model"], lm["horizon"]) not in existing:
                    available["H"].append(lm)

    return available


def format_predictions_for_api(
    df: pd.DataFrame,
    actual_col: str = "demand",
    pred_col: str = "predicted",
) -> list[dict]:
    """
    Format a predictions DataFrame for JSON output.

    Args:
        df: DataFrame with datetime index, actual and predicted columns
        actual_col: Name of the actual values column
        pred_col: Name of the predicted values column

    Returns:
        List of {t, actual, predicted} dicts
    """
    records = []
    for idx, row in df.iterrows():
        records.append({
            "t": idx.isoformat() + "Z" if not str(idx).endswith("Z") else str(idx),
            "actual": round(row[actual_col], 2),
            "predicted": round(row[pred_col], 2),
        })
    return records
