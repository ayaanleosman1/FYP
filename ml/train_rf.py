import argparse
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor

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
    parser = argparse.ArgumentParser(description="Train Random Forest demand forecasting model")
    parser.add_argument("--granularity", "-g", type=str, default="H", choices=["H", "D", "W", "M", "Y"])
    parser.add_argument("--horizon", "-hz", type=int, default=None)
    parser.add_argument("--days", "-d", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--source", "-s", type=str, default="real", choices=["real", "synthetic"])
    return parser.parse_args()


def main():
    args = parse_args()

    granularity = Granularity.from_code(args.granularity)
    config = granularity.config

    horizon = args.horizon or config.default_horizon
    n_days = args.days or get_recommended_days_for_granularity(granularity)
    test_periods = config.default_test_periods

    print(f"Training Random Forest model")
    print(f"  Granularity: {config.name} ({config.code})")
    print(f"  Horizon: {horizon} {config.name} periods")
    print(f"  Data: {n_days} days")
    print(f"  Source: {args.source}")
    print()

    df = get_data_for_granularity(n_days=n_days, granularity=granularity, seed=args.seed, source=args.source)
    print(f"Data shape: {df.shape}")
    print(f"Date range: {df.index.min()} to {df.index.max()}")

    feat = build_features(df, granularity=granularity)
    print(f"Features shape after engineering: {feat.shape}")

    feature_cols = get_available_features(feat, granularity)
    print(f"Features: {feature_cols}")

    X = feat[feature_cols]
    y = feat["demand"]

    train_df, test_df = train_test_split_temporal(feat, test_periods, granularity)
    X_train = train_df[feature_cols]
    y_train = train_df["demand"]
    X_test = test_df[feature_cols]
    y_test = test_df["demand"]

    print(f"Train size: {len(X_train)}, Test size: {len(X_test)}")
    print()

    model = RandomForestRegressor(
        n_estimators=500,
        max_depth=15,
        min_samples_leaf=5,
        max_features="sqrt",
        random_state=args.seed,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    outputs_root = Path(__file__).parent.parent / "outputs" / config.folder_name / "models"
    outputs_root.mkdir(parents=True, exist_ok=True)
    model_path = outputs_root / f"rf_{horizon}.joblib"
    joblib.dump({"model": model, "features": feature_cols}, model_path)
    print(f"Saved model: {model_path}")

    y_pred = model.predict(X_test)

    metrics = compute_all_metrics(y_test.values, y_pred)
    print("Metrics:")
    for k, v in metrics.items():
        print(f"  {k}: {v}")
    print()

    test_df = test_df.copy()
    test_df["predicted"] = y_pred
    predictions = format_predictions_for_api(test_df, actual_col="demand", pred_col="predicted")

    metrics_path, preds_path = save_outputs(
        granularity=granularity,
        model="rf",
        horizon=horizon,
        metrics=metrics,
        predictions=predictions,
    )

    print("Saved:")
    print(f"  {metrics_path}")
    print(f"  {preds_path}")


if __name__ == "__main__":
    main()
