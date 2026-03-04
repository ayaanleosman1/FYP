import json
from pathlib import Path
from typing import Union, Optional
import pandas as pd

from .granularity import Granularity, GRANULARITY_CONFIG


def get_outputs_root():
    return Path(__file__).parent.parent.parent / "outputs"


def get_output_path(granularity, file_type, model, horizon):
    folder = get_outputs_root() / granularity.config.folder_name
    return folder / f"{file_type}_{model}_{horizon}.json"


def save_outputs(granularity, model, horizon, metrics, predictions, extra_metrics=None):
    config = granularity.config
    folder = get_outputs_root() / config.folder_name
    folder.mkdir(parents=True, exist_ok=True)

    metrics_out = {
        "model": model,
        "granularity": config.code,
        "granularity_name": config.name,
        "horizon": horizon,
        **metrics,
    }
    if extra_metrics:
        metrics_out.update(extra_metrics)

    preds_out = {
        "model": model,
        "granularity": config.code,
        "granularity_name": config.name,
        "horizon": horizon,
        "series": predictions,
    }

    metrics_path = folder / f"metrics_{model}_{horizon}.json"
    preds_path = folder / f"preds_{model}_{horizon}.json"

    with open(metrics_path, "w") as f:
        json.dump(metrics_out, f, indent=2)

    with open(preds_path, "w") as f:
        json.dump(preds_out, f, indent=2)

    return metrics_path, preds_path


def load_outputs(granularity, file_type, model, horizon):
    path = get_output_path(granularity, file_type, model, horizon)
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def load_legacy_outputs(file_type, model, horizon):
    path = get_outputs_root() / f"{file_type}_{model}_{horizon}.json"
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def list_available_models(granularity=None):
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
            parts = f.stem.split("_")
            if len(parts) >= 3:
                model = parts[1]
                try:
                    horizon = int(parts[2])
                    models.append({"model": model, "horizon": horizon})
                except ValueError:
                    continue
        available[g.value] = models

    # also check legacy files in root outputs/
    if granularity is None or granularity == Granularity.HOURLY:
        legacy_models = []
        for f in outputs_root.glob("metrics_*.json"):
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
            existing = {(m["model"], m["horizon"]) for m in available["H"]}
            for lm in legacy_models:
                if (lm["model"], lm["horizon"]) not in existing:
                    available["H"].append(lm)

    return available


def format_predictions_for_api(df, actual_col="demand", pred_col="predicted"):
    records = []
    for idx, row in df.iterrows():
        records.append({
            "t": idx.isoformat() + "Z" if not str(idx).endswith("Z") else str(idx),
            "actual": round(row[actual_col], 2),
            "predicted": round(row[pred_col], 2),
        })
    return records
