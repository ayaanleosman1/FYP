import argparse
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from interpret.glassbox import ExplainableBoostingRegressor

sys.path.insert(0, str(Path(__file__).parent))

from utils import (
    Granularity,
    get_data_for_granularity,
    build_features,
    get_available_features,
    compute_all_metrics,
)
from utils.data import get_recommended_days_for_granularity
from utils.validation import walk_forward_validate, verify_no_leakage


def make_xgb_fit_predict(seed=42):
    def fit_predict(X_train, y_train, X_test):
        val_size = max(1, int(len(X_train) * 0.15))
        X_tr = X_train.iloc[:-val_size]
        y_tr = y_train.iloc[:-val_size]
        X_val = X_train.iloc[-val_size:]
        y_val = y_train.iloc[-val_size:]

        model = XGBRegressor(
            n_estimators=1000,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_lambda=1.0,
            early_stopping_rounds=50,
            random_state=seed,
            n_jobs=-1,
        )
        model.fit(
            X_tr, y_tr,
            eval_set=[(X_val, y_val)],
            verbose=False,
        )
        return model.predict(X_test)
    return fit_predict


def make_rf_fit_predict(seed=42):
    def fit_predict(X_train, y_train, X_test):
        model = RandomForestRegressor(
            n_estimators=500,
            max_depth=15,
            min_samples_leaf=5,
            max_features="sqrt",
            random_state=seed,
            n_jobs=-1,
        )
        model.fit(X_train, y_train)
        return model.predict(X_test)
    return fit_predict


def make_linear_fit_predict():
    def fit_predict(X_train, y_train, X_test):
        model = LinearRegression()
        model.fit(X_train, y_train)
        return model.predict(X_test)
    return fit_predict


def make_ebm_fit_predict(seed=42):
    def fit_predict(X_train, y_train, X_test):
        val_size = max(1, int(len(X_train) * 0.15))
        X_tr = X_train.iloc[:-val_size]
        y_tr = y_train.iloc[:-val_size]

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
            random_state=seed,
        )
        ebm.fit(X_tr, y_tr)
        return ebm.predict(X_test)
    return fit_predict


BASELINE_COLS = {
    "H": ("lag_168", "roll_24_mean"),
    "D": ("lag_7", "roll_7_mean"),
    "W": ("lag_52", "roll_4_mean"),
    "M": ("lag_12", "roll_3_mean"),
    "Y": ("lag_1", "roll_2_mean"),
}


def make_naive_last_predict():
    def fit_predict(X_train, y_train, X_test):
        return X_test["lag_1"].values
    return fit_predict


def make_seasonal_naive_predict(granularity_code):
    seasonal_col = BASELINE_COLS[granularity_code][0]
    def fit_predict(X_train, y_train, X_test):
        if seasonal_col in X_test.columns:
            return X_test[seasonal_col].values
        return X_test["lag_1"].values
    return fit_predict


def make_moving_avg_predict(granularity_code):
    avg_col = BASELINE_COLS[granularity_code][1]
    def fit_predict(X_train, y_train, X_test):
        if avg_col in X_test.columns:
            return X_test[avg_col].values
        return X_test["lag_1"].values
    return fit_predict


def parse_args():
    parser = argparse.ArgumentParser(description="Walk-forward validation for all models and baselines")
    parser.add_argument("--granularity", "-g", type=str, default="D", choices=["H", "D", "W", "M"])
    parser.add_argument("--folds", "-f", type=int, default=5)
    parser.add_argument("--test-fraction", "-t", type=float, default=0.3)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()
    granularity = Granularity.from_code(args.granularity)
    config = granularity.config
    n_days = get_recommended_days_for_granularity(granularity)

    print("=" * 70)
    print(f"Walk-Forward Validation: {config.name} granularity")
    print(f"Folds: {args.folds}, Test fraction: {args.test_fraction}")
    print(f"Data: {n_days} days")
    print("=" * 70)
    print()

    df = get_data_for_granularity(n_days=n_days, granularity=granularity, source="real")
    feat = build_features(df, granularity=granularity)
    feature_cols = get_available_features(feat, granularity)

    print(f"Total rows after feature engineering: {len(feat)}")
    print(f"Features ({len(feature_cols)}): {feature_cols}")
    print()

    warnings = verify_no_leakage(feat, feature_cols)
    if warnings:
        print("WARNINGS:")
        for w in warnings:
            print(f"  {w}")
        print()
    else:
        print("Leakage checks passed.")
        print()

    models = {
        "XGBoost": make_xgb_fit_predict(args.seed),
        "Random Forest": make_rf_fit_predict(args.seed),
        "Linear Reg": make_linear_fit_predict(),
        "EBM": make_ebm_fit_predict(args.seed),
        "Naive Last": make_naive_last_predict(),
        f"Seasonal Naive": make_seasonal_naive_predict(args.granularity),
        "Moving Avg": make_moving_avg_predict(args.granularity),
    }

    all_results = {}
    for name, fit_fn in models.items():
        print(f"Running {name}...", end=" ", flush=True)
        try:
            folds, avg = walk_forward_validate(
                feat, feature_cols, "demand", fit_fn,
                n_folds=args.folds, test_fraction=args.test_fraction,
            )
            all_results[name] = avg
            print(
                f"MAE={avg['mae']:.2f}  RMSE={avg['rmse']:.2f}  "
                f"SMAPE={avg['smape']:.2f}%"
            )
        except Exception as e:
            print(f"FAILED: {e}")
            all_results[name] = None

    # results table
    print()
    print("=" * 70)
    print(f"RESULTS: {config.name} granularity, {args.folds}-fold walk-forward")
    print("=" * 70)
    print(f"{'Model':<20} {'MAE':>12} {'RMSE':>12} {'SMAPE':>10} {'MAPE':>10}")
    print("-" * 70)

    sorted_results = sorted(
        [(k, v) for k, v in all_results.items() if v is not None],
        key=lambda x: x[1]["smape"],
    )
    for name, metrics in sorted_results:
        print(
            f"{name:<20} {metrics['mae']:>12.2f} {metrics['rmse']:>12.2f} "
            f"{metrics['smape']:>9.2f}% {metrics['mape']:>9.2f}%"
        )

    ml_models = ["XGBoost", "Random Forest", "Linear Reg", "EBM"]
    baselines = ["Naive Last", "Seasonal Naive", "Moving Avg"]

    best_ml = min(
        [(k, v) for k, v in all_results.items() if v and k in ml_models],
        key=lambda x: x[1]["smape"],
        default=None,
    )
    best_base = min(
        [(k, v) for k, v in all_results.items() if v and k in baselines],
        key=lambda x: x[1]["smape"],
        default=None,
    )

    print()
    if best_ml and best_base:
        ml_smape = best_ml[1]["smape"]
        base_smape = best_base[1]["smape"]
        improvement = ((base_smape - ml_smape) / base_smape) * 100
        print(f"Best ML model: {best_ml[0]} (SMAPE {ml_smape:.2f}%)")
        print(f"Best baseline: {best_base[0]} (SMAPE {base_smape:.2f}%)")
        if improvement > 0:
            print(f"ML improvement over baseline: {improvement:.1f}%")
        else:
            print(f"WARNING: Best baseline outperforms best ML model by {-improvement:.1f}%")


if __name__ == "__main__":
    main()
