import argparse
import numpy as np
import pandas as pd
import json
from pathlib import Path
from interpret.glassbox import ExplainableBoostingRegressor
from interpret import show, preserve

import sys
sys.path.insert(0, str(Path(__file__).parent))

from utils import (
    Granularity,
    get_data_for_granularity,
    build_features,
    get_available_features,
    compute_all_metrics,
    save_outputs,
    format_predictions_for_api,
)
from utils.data import train_test_split_temporal, get_recommended_days_for_granularity
import joblib


def parse_args():
    parser = argparse.ArgumentParser(description="Train EBM glassbox model")
    parser.add_argument("--granularity", "-g", type=str, default="H", choices=["H", "D", "W", "M", "Y"])
    parser.add_argument("--horizon", "-hz", type=int, default=None)
    parser.add_argument("--days", "-d", type=int, default=None)
    parser.add_argument("--source", "-s", type=str, default="real", choices=["real", "synthetic"])
    parser.add_argument("--save-plots", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()

    granularity = Granularity.from_code(args.granularity)
    config = granularity.config

    horizon = args.horizon or config.default_horizon
    n_days = args.days or get_recommended_days_for_granularity(granularity)
    test_periods = config.default_test_periods

    print("=" * 60)
    print("Training Explainable Boosting Machine (EBM)")
    print("=" * 60)
    print(f"  Granularity: {config.name}")
    print(f"  Horizon: {horizon} periods")
    print(f"  Data: {n_days} days")
    print()

    df = get_data_for_granularity(n_days=n_days, granularity=granularity, source=args.source)
    print(f"Data shape: {df.shape}")
    print(f"Date range: {df.index.min()} to {df.index.max()}")

    feat = build_features(df, granularity=granularity)
    feature_cols = get_available_features(feat, granularity)
    print(f"Features: {feature_cols}")

    train_df, test_df = train_test_split_temporal(feat, test_periods, granularity)
    X_train = train_df[feature_cols]
    y_train = train_df["demand"]
    X_test = test_df[feature_cols]
    y_test = test_df["demand"]

    print(f"Train: {len(X_train)}, Test: {len(X_test)}")
    print()

    val_size = max(1, int(len(X_train) * 0.15))
    X_train_fit = X_train.iloc[:-val_size]
    y_train_fit = y_train.iloc[:-val_size]
    print(f"Temporal validation: fit={len(X_train_fit)}, val={val_size}")

    print("Training EBM (this may take a moment)...")
    ebm = ExplainableBoostingRegressor(
        max_bins=256,
        max_interaction_bins=32,
        interactions=10,
        outer_bags=8,
        inner_bags=0,
        learning_rate=0.01,
        validation_size=0.15,
        early_stopping_rounds=50,
        n_jobs=1,
        random_state=args.seed,
    )
    ebm.fit(X_train_fit, y_train_fit)
    print("Training complete!")
    print()

    y_pred = ebm.predict(X_test)

    metrics = compute_all_metrics(y_test.values, y_pred)
    print("Metrics:")
    for k, v in metrics.items():
        print(f"  {k}: {v:.2f}")
    print()

    # feature importances
    print("Feature Importances:")
    print("-" * 50)
    importances = list(zip(feature_cols, ebm.term_importances()))
    importances.sort(key=lambda x: abs(x[1]), reverse=True)
    for name, imp in importances:
        bar = "█" * int(imp / max(i[1] for i in importances) * 20)
        print(f"  {name:20s} {imp:10.2f}  {bar}")
    print()

    importance_dict = {name: float(imp) for name, imp in importances}

    outputs_root = Path(__file__).parent.parent / "outputs" / config.folder_name
    outputs_root.mkdir(parents=True, exist_ok=True)

    # save feature ranges for what-if analysis
    feature_ranges = {}
    for col in feature_cols:
        feature_ranges[col] = {
            "min": float(X_train[col].min()),
            "max": float(X_train[col].max()),
            "mean": float(X_train[col].mean()),
            "median": float(X_train[col].median()),
        }

    interpretation_path = outputs_root / f"interpretation_ebm_{horizon}.json"
    with open(interpretation_path, 'w') as f:
        json.dump({
            "model": "ebm",
            "granularity": config.code,
            "feature_importances": importance_dict,
            "features": feature_cols,
            "feature_ranges": feature_ranges,
        }, f, indent=2)
    print(f"Saved interpretation: {interpretation_path}")

    models_dir = outputs_root / "models"
    models_dir.mkdir(exist_ok=True)
    model_path = models_dir / f"ebm_{horizon}.joblib"
    joblib.dump({"model": ebm, "features": feature_cols}, model_path)
    print(f"Saved model: {model_path}")

    if args.save_plots:
        plots_dir = outputs_root / "plots"
        plots_dir.mkdir(exist_ok=True)
        ebm_global = ebm.explain_global()
        html_path = plots_dir / f"ebm_global_{config.code}.html"
        preserve(ebm_global, file_name=str(html_path))
        print(f"Saved global explanation: {html_path}")

    test_df = test_df.copy()
    test_df["predicted"] = y_pred
    predictions = format_predictions_for_api(test_df, actual_col="demand", pred_col="predicted")

    metrics_path, preds_path = save_outputs(
        granularity=granularity,
        model="ebm",
        horizon=horizon,
        metrics=metrics,
        predictions=predictions,
    )

    print(f"Saved: {metrics_path}")
    print(f"Saved: {preds_path}")


if __name__ == "__main__":
    main()
