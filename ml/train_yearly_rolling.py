"""
Train models with expanding window for yearly predictions.

This gives more realistic predictions since each year uses all prior data:
- Predict 2020: Train on 2009-2019
- Predict 2021: Train on 2009-2020
- Predict 2022: Train on 2009-2021
- etc.
"""

import numpy as np
import pandas as pd
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from utils import (
    Granularity,
    build_features,
    get_available_features,
    compute_all_metrics,
    save_outputs,
)
from utils.data import load_demand_with_weather, resample_to_granularity


def train_with_expanding_window(granularity_code: str = "Y", test_years: int = 5):
    """
    Train models using expanding window cross-validation.

    Args:
        granularity_code: "Y" for yearly, "M" for monthly
        test_years: Number of years to use for testing
    """
    granularity = Granularity.from_code(granularity_code)
    config = granularity.config

    print(f"{'='*60}")
    print(f"Expanding Window Training - {config.name.title()}")
    print(f"{'='*60}")

    # Load all data
    print("Loading data...")
    hourly_df = load_demand_with_weather()
    df = resample_to_granularity(hourly_df, granularity, agg_func="sum")

    print(f"Total {config.name} periods: {len(df)}")
    print(f"Date range: {df.index.min()} to {df.index.max()}")

    # For yearly: get unique years
    if granularity == Granularity.YEARLY:
        all_years = sorted(df.index.year.unique())
        test_period_years = all_years[-test_years:]
        print(f"Test years: {test_period_years}")
    else:
        # For monthly: test on last N months
        test_start_idx = len(df) - (test_years * 12)
        test_period_years = None

    # Models to train
    models = {
        'xgb': lambda: XGBRegressor(
            n_estimators=100, max_depth=3, learning_rate=0.1,
            random_state=42, n_jobs=-1
        ),
        'rf': lambda: RandomForestRegressor(
            n_estimators=100, max_depth=5, random_state=42, n_jobs=-1
        ),
        'linear': lambda: LinearRegression()
    }

    results = {name: {'actuals': [], 'predictions': [], 'timestamps': []}
               for name in models}

    # Expanding window training
    if granularity == Granularity.YEARLY:
        for test_year in test_period_years:
            print(f"\n--- Predicting {test_year} ---")

            # Split: train on all years before test_year
            train_df = df[df.index.year < test_year].copy()
            test_df = df[df.index.year == test_year].copy()

            if len(train_df) < 3:
                print(f"  Skipping {test_year}: not enough training data")
                continue

            print(f"  Training on: {train_df.index.min().year}-{train_df.index.max().year} ({len(train_df)} years)")

            # Build features
            # For yearly, we need to be careful about features
            train_feat = _build_yearly_features_simple(train_df)
            test_feat = _build_yearly_features_simple(test_df, train_df)

            if len(test_feat) == 0:
                continue

            feature_cols = [c for c in train_feat.columns if c != 'demand']

            X_train = train_feat[feature_cols]
            y_train = train_feat['demand']
            X_test = test_feat[feature_cols]
            y_test = test_feat['demand']

            # Train and predict with each model
            for name, model_fn in models.items():
                model = model_fn()
                model.fit(X_train, y_train)
                pred = model.predict(X_test)[0]
                actual = y_test.values[0]

                results[name]['actuals'].append(actual)
                results[name]['predictions'].append(pred)
                results[name]['timestamps'].append(test_df.index[0])

                error_pct = ((pred - actual) / actual) * 100
                print(f"  {name:8s}: Actual={actual/1e6:.1f}M, Pred={pred/1e6:.1f}M, Error={error_pct:+.1f}%")

    else:
        # Monthly expanding window
        for i in range(test_years * 12):
            test_idx = test_start_idx + i
            if test_idx >= len(df):
                break

            train_df = df.iloc[:test_idx].copy()
            test_df = df.iloc[[test_idx]].copy()

            if len(train_df) < 12:
                continue

            # Build features
            full_df = pd.concat([train_df, test_df])
            feat = build_features(full_df, granularity=granularity)

            train_feat = feat.iloc[:-1]
            test_feat = feat.iloc[[-1]]

            if len(test_feat) == 0 or test_feat.isna().any().any():
                continue

            feature_cols = get_available_features(feat, granularity)

            X_train = train_feat[feature_cols].dropna()
            y_train = train_feat.loc[X_train.index, 'demand']
            X_test = test_feat[feature_cols]
            y_test = test_feat['demand']

            for name, model_fn in models.items():
                model = model_fn()
                model.fit(X_train, y_train)
                pred = model.predict(X_test)[0]
                actual = y_test.values[0]

                results[name]['actuals'].append(actual)
                results[name]['predictions'].append(pred)
                results[name]['timestamps'].append(test_df.index[0])

    # Compute metrics and save for each model
    print(f"\n{'='*60}")
    print("Results Summary")
    print(f"{'='*60}")

    for name in models:
        actuals = np.array(results[name]['actuals'])
        preds = np.array(results[name]['predictions'])
        timestamps = results[name]['timestamps']

        if len(actuals) == 0:
            continue

        metrics = compute_all_metrics(actuals, preds)

        print(f"\n{name.upper()}:")
        print(f"  MAE:   {metrics['mae']/1e6:.2f}M")
        print(f"  RMSE:  {metrics['rmse']/1e6:.2f}M")
        print(f"  MAPE:  {metrics['mape']:.2f}%")
        print(f"  SMAPE: {metrics['smape']:.2f}%")

        # Format predictions for API (just the series list, save_outputs wraps it)
        predictions = [
            {
                "t": ts.isoformat() + "Z" if not str(ts).endswith('Z') else str(ts),
                "actual": float(a),
                "predicted": float(p)
            }
            for ts, a, p in zip(timestamps, actuals, preds)
        ]

        # Save outputs
        save_outputs(
            granularity=granularity,
            model=name,
            horizon=1,
            metrics=metrics,
            predictions=predictions,
        )

    print(f"\n{'='*60}")
    print("Done! Outputs saved to outputs/{}/".format(config.folder_name))


def _build_yearly_features_simple(df: pd.DataFrame, train_df: pd.DataFrame = None) -> pd.DataFrame:
    """Build simple features for yearly predictions."""
    feat = df.copy()

    # Year as feature (trend)
    feat['year'] = feat.index.year

    # If we have training data, compute lag from last training year
    if train_df is not None and len(train_df) > 0:
        feat['lag_1'] = train_df['demand'].iloc[-1]
        if len(train_df) > 1:
            feat['lag_2'] = train_df['demand'].iloc[-2]
            feat['trend'] = train_df['demand'].iloc[-1] - train_df['demand'].iloc[-2]
        else:
            feat['lag_2'] = feat['lag_1']
            feat['trend'] = 0
        if len(train_df) > 2:
            feat['avg_3yr'] = train_df['demand'].iloc[-3:].mean()
        else:
            feat['avg_3yr'] = train_df['demand'].mean()
    else:
        # For training data, use shifted values
        feat['lag_1'] = feat['demand'].shift(1)
        feat['lag_2'] = feat['demand'].shift(2)
        feat['trend'] = feat['demand'].shift(1) - feat['demand'].shift(2)
        feat['avg_3yr'] = feat['demand'].shift(1).rolling(3).mean()
        feat = feat.dropna()

    return feat


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--granularity', '-g', default='Y', choices=['Y', 'M'])
    parser.add_argument('--test-years', '-t', type=int, default=5)
    args = parser.parse_args()

    train_with_expanding_window(args.granularity, args.test_years)
