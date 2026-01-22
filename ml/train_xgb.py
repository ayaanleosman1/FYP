"""
Train XGBoost model for demand forecasting.

Usage:
    python train_xgb.py                           # Default: hourly, horizon=24
    python train_xgb.py --granularity D           # Daily with default horizon
    python train_xgb.py --granularity D --horizon 14  # Daily, 14-day horizon
    python train_xgb.py --granularity W --days 730    # Weekly with 2 years of data
"""

import argparse
import numpy as np
import pandas as pd
from xgboost import XGBRegressor

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


def parse_args():
    parser = argparse.ArgumentParser(description="Train XGBoost demand forecasting model")
    parser.add_argument(
        "--granularity", "-g",
        type=str,
        default="H",
        choices=["H", "D", "W", "M", "Y"],
        help="Forecast granularity: H=hourly, D=daily, W=weekly, M=monthly, Y=yearly"
    )
    parser.add_argument(
        "--horizon", "-hz",
        type=int,
        default=None,
        help="Forecast horizon in periods (default: granularity-specific)"
    )
    parser.add_argument(
        "--days", "-d",
        type=int,
        default=None,
        help="Days of training data (default: granularity-specific)"
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

    # Parse granularity
    granularity = Granularity.from_code(args.granularity)
    config = granularity.config

    # Set defaults based on granularity
    horizon = args.horizon or config.default_horizon
    n_days = args.days or get_recommended_days_for_granularity(granularity)
    test_periods = config.default_test_periods

    print(f"Training XGBoost model")
    print(f"  Granularity: {config.name} ({config.code})")
    print(f"  Horizon: {horizon} {config.name} periods")
    print(f"  Data: {n_days} days")
    print()

    # Get data at the specified granularity
    df = get_data_for_granularity(n_days=n_days, granularity=granularity, seed=args.seed)
    print(f"Data shape: {df.shape}")
    print(f"Date range: {df.index.min()} to {df.index.max()}")

    # Build features
    feat = build_features(df, granularity=granularity)
    print(f"Features shape after engineering: {feat.shape}")

    # Get available feature columns
    feature_cols = get_available_features(feat, granularity)
    print(f"Features: {feature_cols}")

    # Prepare X and y
    X = feat[feature_cols]
    y = feat["demand"]

    # Train/test split
    train_df, test_df = train_test_split_temporal(feat, test_periods, granularity)
    X_train = train_df[feature_cols]
    y_train = train_df["demand"]
    X_test = test_df[feature_cols]
    y_test = test_df["demand"]

    print(f"Train size: {len(X_train)}, Test size: {len(X_test)}")
    print()

    # Train model
    model = XGBRegressor(
        n_estimators=500,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=1.0,
        random_state=args.seed,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    # Predict
    y_pred = model.predict(X_test)

    # Compute metrics
    metrics = compute_all_metrics(y_test.values, y_pred)
    print("Metrics:")
    for k, v in metrics.items():
        print(f"  {k}: {v}")
    print()

    # Prepare predictions for output
    test_df = test_df.copy()
    test_df["predicted"] = y_pred
    predictions = format_predictions_for_api(test_df, actual_col="demand", pred_col="predicted")

    # Save outputs
    metrics_path, preds_path = save_outputs(
        granularity=granularity,
        model="xgb",
        horizon=horizon,
        metrics=metrics,
        predictions=predictions,
    )

    print("Saved:")
    print(f"  {metrics_path}")
    print(f"  {preds_path}")


if __name__ == "__main__":
    main()
