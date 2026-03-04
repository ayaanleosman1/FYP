import numpy as np
import pandas as pd
from typing import Callable, Optional

from .metrics import compute_all_metrics


def walk_forward_validate(feat_df, feature_cols, target_col, fit_predict_fn,
                          n_folds=5, test_fraction=0.3, min_train_size=30):
    n = len(feat_df)
    test_total = int(n * test_fraction)
    fold_size = test_total // n_folds

    if fold_size < 1:
        raise ValueError(
            f"Not enough data for {n_folds} folds with test_fraction={test_fraction}. "
            f"Total rows={n}, test_total={test_total}, fold_size={fold_size}"
        )

    fold_results = []

    for i in range(n_folds):
        test_start = n - test_total + i * fold_size
        test_end = test_start + fold_size
        if i == n_folds - 1:
            test_end = n  # last fold takes remainder

        train = feat_df.iloc[:test_start]
        test = feat_df.iloc[test_start:test_end]

        if len(train) < min_train_size:
            continue
        if len(test) == 0:
            continue

        X_train = train[feature_cols]
        y_train = train[target_col]
        X_test = test[feature_cols]
        y_test = test[target_col]

        y_pred = fit_predict_fn(X_train, y_train, X_test)

        metrics = compute_all_metrics(y_test.values, y_pred)
        metrics["fold"] = i + 1
        metrics["train_size"] = len(train)
        metrics["test_size"] = len(test)
        metrics["test_start"] = str(test.index.min())
        metrics["test_end"] = str(test.index.max())
        fold_results.append(metrics)

    if not fold_results:
        raise ValueError("No folds completed — not enough data")

    metric_keys = ["mae", "rmse", "smape", "mape"]
    avg_metrics = {
        k: round(np.mean([r[k] for r in fold_results]), 2)
        for k in metric_keys
    }
    avg_metrics["n_folds"] = len(fold_results)

    return fold_results, avg_metrics


def verify_no_leakage(feat_df, feature_cols, target_col="demand"):
    warnings = []

    raw_suspicious = [
        "temperature", "solar_radiation", "direct_radiation",
        "cloud_cover", "precipitation",
    ]
    for col in raw_suspicious:
        if col in feature_cols:
            warnings.append(
                f"LEAKAGE: Raw column '{col}' in features — should be shifted/renamed"
            )

    for col in feature_cols:
        if col in feat_df.columns and target_col in feat_df.columns:
            corr = feat_df[col].corr(feat_df[target_col])
            if abs(corr) > 0.99:
                warnings.append(
                    f"SUSPICIOUS: Feature '{col}' has correlation {corr:.4f} "
                    f"with target — possible leakage"
                )

    if not feat_df.index.is_monotonic_increasing:
        warnings.append("DATA ORDER: Index is not sorted chronologically")

    return warnings
