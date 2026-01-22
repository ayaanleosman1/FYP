"""
Feature engineering for multi-timeframe forecasting.

Each granularity has its own set of features:

| Granularity | Calendar Features                     | Lag Features           | Rolling Features           |
|-------------|---------------------------------------|------------------------|----------------------------|
| Hourly (H)  | hour, dow, month                      | lag_1, lag_24, lag_168 | roll_24_mean               |
| Daily (D)   | dow, month, day_of_year, is_weekend   | lag_1, lag_7           | roll_7_mean, roll_30_mean  |
| Weekly (W)  | week_of_year, month, quarter          | lag_1, lag_4, lag_52   | roll_4_mean                |
| Monthly (M) | month, quarter                        | lag_1, lag_12          | roll_3_mean, roll_12_mean  |
| Yearly (Y)  | —                                     | lag_1                  | roll_2_mean                |
"""

import pandas as pd
import numpy as np
from typing import Optional

from .granularity import Granularity


def build_features(
    df: pd.DataFrame,
    granularity: Granularity = Granularity.HOURLY,
    target_col: str = "demand",
) -> pd.DataFrame:
    """
    Build features appropriate for the given granularity.

    Args:
        df: DataFrame with datetime index and target column
        granularity: The forecast granularity
        target_col: Name of the target column

    Returns:
        DataFrame with features and target, NaN rows dropped
    """
    feat = df.copy()

    if granularity == Granularity.HOURLY:
        feat = _build_hourly_features(feat, target_col)
    elif granularity == Granularity.DAILY:
        feat = _build_daily_features(feat, target_col)
    elif granularity == Granularity.WEEKLY:
        feat = _build_weekly_features(feat, target_col)
    elif granularity == Granularity.MONTHLY:
        feat = _build_monthly_features(feat, target_col)
    elif granularity == Granularity.YEARLY:
        feat = _build_yearly_features(feat, target_col)
    else:
        raise ValueError(f"Unknown granularity: {granularity}")

    # Drop rows with NaN (from lag/rolling features)
    feat = feat.dropna()

    return feat


def _build_hourly_features(df: pd.DataFrame, target_col: str) -> pd.DataFrame:
    """
    Build features for hourly forecasting.

    Calendar: hour, dow, month
    Lags: lag_1, lag_24, lag_168 (1 week)
    Rolling: roll_24_mean
    """
    feat = df.copy()

    # Calendar features
    feat["hour"] = feat.index.hour
    feat["dow"] = feat.index.dayofweek
    feat["month"] = feat.index.month

    # Lag features
    feat["lag_1"] = feat[target_col].shift(1)
    feat["lag_24"] = feat[target_col].shift(24)
    feat["lag_168"] = feat[target_col].shift(168)  # 1 week

    # Rolling features (shifted to avoid look-ahead)
    feat["roll_24_mean"] = feat[target_col].shift(1).rolling(24).mean()

    return feat


def _build_daily_features(df: pd.DataFrame, target_col: str) -> pd.DataFrame:
    """
    Build features for daily forecasting.

    Calendar: dow, month, day_of_year, is_weekend
    Lags: lag_1, lag_7
    Rolling: roll_7_mean, roll_30_mean
    """
    feat = df.copy()

    # Calendar features
    feat["dow"] = feat.index.dayofweek
    feat["month"] = feat.index.month
    feat["day_of_year"] = feat.index.dayofyear
    feat["is_weekend"] = (feat.index.dayofweek >= 5).astype(int)

    # Lag features
    feat["lag_1"] = feat[target_col].shift(1)
    feat["lag_7"] = feat[target_col].shift(7)

    # Rolling features (shifted to avoid look-ahead)
    feat["roll_7_mean"] = feat[target_col].shift(1).rolling(7).mean()
    feat["roll_30_mean"] = feat[target_col].shift(1).rolling(30).mean()

    return feat


def _build_weekly_features(df: pd.DataFrame, target_col: str) -> pd.DataFrame:
    """
    Build features for weekly forecasting.

    Calendar: week_of_year, month, quarter
    Lags: lag_1, lag_4, lag_52
    Rolling: roll_4_mean
    """
    feat = df.copy()

    # Calendar features
    feat["week_of_year"] = feat.index.isocalendar().week.astype(int)
    feat["month"] = feat.index.month
    feat["quarter"] = feat.index.quarter

    # Lag features
    feat["lag_1"] = feat[target_col].shift(1)
    feat["lag_4"] = feat[target_col].shift(4)

    # lag_52 only if we have enough data
    if len(feat) > 52:
        feat["lag_52"] = feat[target_col].shift(52)
    else:
        # Use available lag as fallback
        max_lag = min(len(feat) - 1, 52)
        if max_lag > 4:
            feat[f"lag_{max_lag}"] = feat[target_col].shift(max_lag)

    # Rolling features (shifted to avoid look-ahead)
    feat["roll_4_mean"] = feat[target_col].shift(1).rolling(4).mean()

    return feat


def _build_monthly_features(df: pd.DataFrame, target_col: str) -> pd.DataFrame:
    """
    Build features for monthly forecasting.

    Calendar: month, quarter
    Lags: lag_1, lag_12
    Rolling: roll_3_mean, roll_12_mean
    """
    feat = df.copy()

    # Calendar features
    feat["month"] = feat.index.month
    feat["quarter"] = feat.index.quarter

    # Lag features
    feat["lag_1"] = feat[target_col].shift(1)

    # lag_12 only if we have enough data
    if len(feat) > 12:
        feat["lag_12"] = feat[target_col].shift(12)
    else:
        max_lag = min(len(feat) - 1, 12)
        if max_lag > 1:
            feat[f"lag_{max_lag}"] = feat[target_col].shift(max_lag)

    # Rolling features (shifted to avoid look-ahead)
    feat["roll_3_mean"] = feat[target_col].shift(1).rolling(3).mean()

    if len(feat) > 12:
        feat["roll_12_mean"] = feat[target_col].shift(1).rolling(12).mean()

    return feat


def _build_yearly_features(df: pd.DataFrame, target_col: str) -> pd.DataFrame:
    """
    Build features for yearly forecasting.

    Calendar: (none - not much yearly periodicity within years)
    Lags: lag_1
    Rolling: roll_2_mean
    """
    feat = df.copy()

    # Minimal features for yearly (limited data typically)
    feat["lag_1"] = feat[target_col].shift(1)

    # Rolling mean if we have enough data
    if len(feat) > 2:
        feat["roll_2_mean"] = feat[target_col].shift(1).rolling(2).mean()

    return feat


def get_feature_columns(granularity: Granularity) -> list[str]:
    """
    Get the list of feature column names for a granularity.

    This is useful for knowing which columns to use as X in training.

    Args:
        granularity: The forecast granularity

    Returns:
        List of feature column names
    """
    feature_map = {
        Granularity.HOURLY: [
            "hour", "dow", "month",
            "lag_1", "lag_24", "lag_168",
            "roll_24_mean"
        ],
        Granularity.DAILY: [
            "dow", "month", "day_of_year", "is_weekend",
            "lag_1", "lag_7",
            "roll_7_mean", "roll_30_mean"
        ],
        Granularity.WEEKLY: [
            "week_of_year", "month", "quarter",
            "lag_1", "lag_4", "lag_52",
            "roll_4_mean"
        ],
        Granularity.MONTHLY: [
            "month", "quarter",
            "lag_1", "lag_12",
            "roll_3_mean", "roll_12_mean"
        ],
        Granularity.YEARLY: [
            "lag_1",
            "roll_2_mean"
        ],
    }
    return feature_map.get(granularity, [])


def get_available_features(df: pd.DataFrame, granularity: Granularity) -> list[str]:
    """
    Get the feature columns that are actually present in the DataFrame.

    This handles cases where some features couldn't be computed
    (e.g., lag_52 with less than a year of data).

    Args:
        df: Feature DataFrame
        granularity: The forecast granularity

    Returns:
        List of available feature column names
    """
    expected = get_feature_columns(granularity)
    available = [col for col in expected if col in df.columns]

    # Also include any additional lag columns that were created as fallbacks
    for col in df.columns:
        if col.startswith("lag_") and col not in available:
            available.append(col)

    return available
