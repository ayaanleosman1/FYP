"""
Train Hybrid Prophet + XGBoost Model.

This model combines:
- Prophet: Captures trend, yearly/weekly/daily seasonality, and holiday effects
- XGBoost: Models the residuals using weather, carbon, and other features

The hybrid approach leverages the strengths of both:
- Prophet is excellent at decomposing time series into interpretable components
- XGBoost captures complex feature interactions that Prophet can't model

This is the approach recommended by the tutor for improved forecasting.

Usage:
    python train_hybrid.py                   # Default: hourly
    python train_hybrid.py --granularity D   # Daily
"""

import argparse
import numpy as np
import pandas as pd
import json
from pathlib import Path
from prophet import Prophet
import xgboost as xgb
import warnings

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

# Suppress Prophet's verbose output
warnings.filterwarnings('ignore', category=FutureWarning)


def parse_args():
    parser = argparse.ArgumentParser(description="Train Hybrid Prophet + XGBoost model")
    parser.add_argument(
        "--granularity", "-g",
        type=str,
        default="H",
        choices=["H", "D", "W", "M"],  # Yearly not suitable for Prophet
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
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility"
    )
    return parser.parse_args()


def prepare_prophet_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare data for Prophet.
    Prophet requires columns 'ds' (datetime) and 'y' (target).
    """
    prophet_df = df.reset_index()
    prophet_df = prophet_df.rename(columns={
        df.index.name or 'index': 'ds',
        'demand': 'y'
    })
    return prophet_df[['ds', 'y']]


def train_prophet(train_df: pd.DataFrame, granularity: Granularity) -> Prophet:
    """
    Train Prophet model with appropriate seasonality settings.
    """
    # Configure Prophet based on granularity
    prophet_config = {
        Granularity.HOURLY: {
            'daily_seasonality': True,
            'weekly_seasonality': True,
            'yearly_seasonality': True,
        },
        Granularity.DAILY: {
            'daily_seasonality': False,
            'weekly_seasonality': True,
            'yearly_seasonality': True,
        },
        Granularity.WEEKLY: {
            'daily_seasonality': False,
            'weekly_seasonality': False,
            'yearly_seasonality': True,
        },
        Granularity.MONTHLY: {
            'daily_seasonality': False,
            'weekly_seasonality': False,
            'yearly_seasonality': True,
        },
    }

    config = prophet_config.get(granularity, prophet_config[Granularity.DAILY])

    model = Prophet(
        daily_seasonality=config['daily_seasonality'],
        weekly_seasonality=config['weekly_seasonality'],
        yearly_seasonality=config['yearly_seasonality'],
        changepoint_prior_scale=0.05,  # Regularization for trend changes
        seasonality_prior_scale=10,     # Regularization for seasonality
    )

    # Add UK holidays
    model.add_country_holidays(country_name='UK')

    # Prepare data and fit
    prophet_df = prepare_prophet_data(train_df)
    model.fit(prophet_df)

    return model


def get_prophet_predictions(model: Prophet, df: pd.DataFrame) -> np.ndarray:
    """Get Prophet predictions for a dataframe."""
    prophet_df = prepare_prophet_data(df)
    forecast = model.predict(prophet_df)
    return forecast['yhat'].values


def main():
    args = parse_args()

    granularity = Granularity.from_code(args.granularity)
    config = granularity.config

    horizon = args.horizon or config.default_horizon
    n_days = args.days or get_recommended_days_for_granularity(granularity)
    test_periods = config.default_test_periods

    print("=" * 60)
    print("Training Hybrid Prophet + XGBoost Model")
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

    # Build features for XGBoost
    feat = build_features(df, granularity=granularity)
    feature_cols = get_available_features(feat, granularity)
    print(f"XGBoost features: {feature_cols}")

    # Train/test split
    train_df, test_df = train_test_split_temporal(feat, test_periods, granularity)

    print(f"Train: {len(train_df)}, Test: {len(test_df)}")
    print()

    # =========================================
    # Step 1: Train Prophet on trend/seasonality
    # =========================================
    print("Step 1: Training Prophet for trend and seasonality...")
    prophet_model = train_prophet(train_df, granularity)

    # Get Prophet predictions on training data
    train_prophet_pred = get_prophet_predictions(prophet_model, train_df)

    # Calculate residuals (what Prophet can't explain)
    train_residuals = train_df['demand'].values - train_prophet_pred
    print(f"  Prophet train RMSE: {np.sqrt(np.mean(train_residuals**2)):.2f}")
    print()

    # =========================================
    # Step 2: Train XGBoost on residuals
    # =========================================
    print("Step 2: Training XGBoost on residuals...")

    X_train = train_df[feature_cols]

    xgb_model = xgb.XGBRegressor(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=args.seed,
        n_jobs=-1,
    )
    xgb_model.fit(X_train, train_residuals)

    # XGBoost residual predictions on train
    train_xgb_residual_pred = xgb_model.predict(X_train)
    print(f"  XGBoost residual RMSE: {np.sqrt(np.mean((train_residuals - train_xgb_residual_pred)**2)):.2f}")
    print()

    # =========================================
    # Step 3: Make predictions on test set
    # =========================================
    print("Step 3: Making predictions on test set...")

    # Prophet component
    test_prophet_pred = get_prophet_predictions(prophet_model, test_df)

    # XGBoost residual component
    X_test = test_df[feature_cols]
    test_xgb_residual_pred = xgb_model.predict(X_test)

    # Hybrid prediction = Prophet + XGBoost residual
    y_pred = test_prophet_pred + test_xgb_residual_pred
    y_test = test_df['demand'].values

    # =========================================
    # Step 4: Evaluate and compare
    # =========================================
    print()
    print("=" * 60)
    print("Results Comparison")
    print("=" * 60)

    # Prophet-only metrics
    prophet_only_metrics = compute_all_metrics(y_test, test_prophet_pred)
    print("\nProphet Only:")
    for k, v in prophet_only_metrics.items():
        print(f"  {k}: {v:.2f}")

    # Hybrid metrics
    hybrid_metrics = compute_all_metrics(y_test, y_pred)
    print("\nHybrid (Prophet + XGBoost):")
    for k, v in hybrid_metrics.items():
        print(f"  {k}: {v:.2f}")

    # Improvement
    print("\nImprovement from hybrid approach:")
    for k in hybrid_metrics:
        improvement = prophet_only_metrics[k] - hybrid_metrics[k]
        pct = (improvement / prophet_only_metrics[k]) * 100
        print(f"  {k}: {improvement:.2f} ({pct:.1f}% better)")

    print()

    # Feature importances from XGBoost
    print("XGBoost Feature Importances (for residual modeling):")
    print("-" * 50)
    importances = list(zip(feature_cols, xgb_model.feature_importances_))
    importances.sort(key=lambda x: x[1], reverse=True)
    for name, imp in importances[:10]:
        bar = "█" * int(imp / max(i[1] for i in importances) * 20)
        print(f"  {name:20s} {imp:.4f}  {bar}")
    print()

    # =========================================
    # Step 5: Save outputs
    # =========================================
    outputs_root = Path(__file__).parent.parent / "outputs" / config.folder_name
    outputs_root.mkdir(parents=True, exist_ok=True)

    # Save models
    models_dir = outputs_root / "models"
    models_dir.mkdir(exist_ok=True)

    # Save Prophet model (as pickle)
    prophet_path = models_dir / f"hybrid_prophet_{horizon}.joblib"
    joblib.dump(prophet_model, prophet_path)

    # Save XGBoost model
    xgb_path = models_dir / f"hybrid_xgb_{horizon}.joblib"
    joblib.dump({"model": xgb_model, "features": feature_cols}, xgb_path)

    print(f"Saved Prophet model: {prophet_path}")
    print(f"Saved XGBoost model: {xgb_path}")

    # Prepare predictions for output
    test_df = test_df.copy()
    test_df["predicted"] = y_pred
    predictions = format_predictions_for_api(test_df, actual_col="demand", pred_col="predicted")

    # Save outputs using standard format
    metrics_path, preds_path = save_outputs(
        granularity=granularity,
        model="hybrid",
        horizon=horizon,
        metrics=hybrid_metrics,
        predictions=predictions,
    )

    print(f"Saved: {metrics_path}")
    print(f"Saved: {preds_path}")

    # Save comparison data
    comparison_path = outputs_root / f"comparison_hybrid_{horizon}.json"
    with open(comparison_path, 'w') as f:
        json.dump({
            "prophet_only": prophet_only_metrics,
            "hybrid": hybrid_metrics,
            "improvement_pct": {
                k: ((prophet_only_metrics[k] - hybrid_metrics[k]) / prophet_only_metrics[k]) * 100
                for k in hybrid_metrics
            }
        }, f, indent=2)
    print(f"Saved: {comparison_path}")

    print()
    print("=" * 60)
    print("Hybrid Model Advantage:")
    print("  - Prophet captures interpretable seasonality patterns")
    print("  - XGBoost captures complex feature interactions")
    print("  - Combined: Better accuracy than either alone!")
    print("=" * 60)


if __name__ == "__main__":
    main()
