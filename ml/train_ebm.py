"""
Train Explainable Boosting Machine (EBM) - a glassbox model.

EBM provides:
- Competitive accuracy with XGBoost/Random Forest
- Full interpretability - see exactly how each feature affects predictions
- Feature importance with confidence intervals
- Visualisable feature effects (shape functions)

Usage:
    python train_ebm.py                        # Default: hourly
    python train_ebm.py --granularity D        # Daily
    python train_ebm.py --granularity D --save-plots  # Save interpretation plots
"""

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
    parser.add_argument(
        "--granularity", "-g",
        type=str,
        default="H",
        choices=["H", "D", "W", "M", "Y"],
        help="Forecast granularity"
    )
    parser.add_argument(
        "--horizon", "-hz",
        type=int,
        default=None,
        help="Forecast horizon in periods"
    )
    parser.add_argument(
        "--days", "-d",
        type=int,
        default=None,
        help="Days of training data"
    )
    parser.add_argument(
        "--source", "-s",
        type=str,
        default="real",
        choices=["real", "synthetic"],
        help="Data source"
    )
    parser.add_argument(
        "--save-plots",
        action="store_true",
        help="Save feature interpretation plots"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility"
    )
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

    # Get data
    df = get_data_for_granularity(
        n_days=n_days,
        granularity=granularity,
        source=args.source
    )
    print(f"Data shape: {df.shape}")
    print(f"Date range: {df.index.min()} to {df.index.max()}")

    # Build features
    feat = build_features(df, granularity=granularity)
    feature_cols = get_available_features(feat, granularity)
    print(f"Features: {feature_cols}")

    # Train/test split
    train_df, test_df = train_test_split_temporal(feat, test_periods, granularity)
    X_train = train_df[feature_cols]
    y_train = train_df["demand"]
    X_test = test_df[feature_cols]
    y_test = test_df["demand"]

    print(f"Train: {len(X_train)}, Test: {len(X_test)}")
    print()

    # Train EBM
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
        n_jobs=1,  # Single-threaded to avoid Python 3.14 pickle issues
        random_state=args.seed,
    )
    ebm.fit(X_train, y_train)
    print("Training complete!")
    print()

    # Predict
    y_pred = ebm.predict(X_test)

    # Compute metrics
    metrics = compute_all_metrics(y_test.values, y_pred)
    print("Metrics:")
    for k, v in metrics.items():
        print(f"  {k}: {v:.2f}")
    print()

    # Feature importances
    print("Feature Importances (how much each feature matters):")
    print("-" * 50)
    importances = list(zip(feature_cols, ebm.term_importances()))
    importances.sort(key=lambda x: abs(x[1]), reverse=True)
    for name, imp in importances:
        bar = "█" * int(imp / max(i[1] for i in importances) * 20)
        print(f"  {name:20s} {imp:10.2f}  {bar}")
    print()

    # Save feature importance to JSON
    importance_dict = {name: float(imp) for name, imp in importances}

    # Save interpretation data
    outputs_root = Path(__file__).parent.parent / "outputs" / config.folder_name
    outputs_root.mkdir(parents=True, exist_ok=True)

    # Get feature ranges from training data for what-if UI
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

    # Save the trained model for what-if predictions
    models_dir = outputs_root / "models"
    models_dir.mkdir(exist_ok=True)
    model_path = models_dir / f"ebm_{horizon}.joblib"
    joblib.dump({"model": ebm, "features": feature_cols}, model_path)
    print(f"Saved model: {model_path}")

    # Save plots if requested
    if args.save_plots:
        plots_dir = outputs_root / "plots"
        plots_dir.mkdir(exist_ok=True)

        # Global explanation
        ebm_global = ebm.explain_global()

        # Save as HTML
        html_path = plots_dir / f"ebm_global_{config.code}.html"
        preserve(ebm_global, file_name=str(html_path))
        print(f"Saved global explanation: {html_path}")

    # Prepare predictions for output
    test_df = test_df.copy()
    test_df["predicted"] = y_pred
    predictions = format_predictions_for_api(test_df, actual_col="demand", pred_col="predicted")

    # Save outputs
    metrics_path, preds_path = save_outputs(
        granularity=granularity,
        model="ebm",
        horizon=horizon,
        metrics=metrics,
        predictions=predictions,
    )

    print(f"Saved: {metrics_path}")
    print(f"Saved: {preds_path}")

    print()
    print("=" * 60)
    print("EBM Advantage: You can now explain WHY the model predicts")
    print("what it does - essential for regulatory compliance!")
    print("=" * 60)


if __name__ == "__main__":
    main()
