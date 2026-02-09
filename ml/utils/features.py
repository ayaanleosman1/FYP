"""
Feature engineering for multi-timeframe forecasting.

Each granularity has its own set of features:

| Granularity | Calendar Features                              | Lag Features           | Rolling Features           | Weather Features                    |
|-------------|------------------------------------------------|------------------------|----------------------------|-------------------------------------|
| Hourly (H)  | hour, dow, month, is_holiday                   | lag_1, lag_24, lag_168 | roll_24_mean               | temperature, humidity, wind_speed   |
| Daily (D)   | dow, month, day_of_year, is_weekend, is_holiday| lag_1, lag_7           | roll_7_mean, roll_30_mean  | temp_mean, temp_range, precip_sum   |
| Weekly (W)  | week_of_year, month, quarter, has_holiday      | lag_1, lag_4, lag_52   | roll_4_mean                | temp_mean, precip_sum               |
| Monthly (M) | month, quarter                                 | lag_1, lag_12          | roll_3_mean, roll_12_mean  | temp_mean                           |
| Yearly (Y)  | —                                              | lag_1                  | roll_2_mean                | temp_mean                           |
"""

import pandas as pd
import numpy as np
from typing import Optional
import holidays

from .granularity import Granularity

# UK holidays instance (cached for performance)
_uk_holidays = None

def get_uk_holidays():
    """Get UK holidays instance (England + Wales + Scotland + Northern Ireland)."""
    global _uk_holidays
    if _uk_holidays is None:
        # UK includes England, Wales, Scotland, Northern Ireland
        _uk_holidays = holidays.UK(years=range(2000, 2030))
    return _uk_holidays


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

    Calendar: hour, dow, month, is_holiday
    Lags: lag_1, lag_24, lag_168 (1 week)
    Rolling: roll_24_mean
    Weather: temperature, humidity, wind_speed (if available)
    """
    feat = df.copy()

    # Calendar features
    feat["hour"] = feat.index.hour
    feat["dow"] = feat.index.dayofweek
    feat["month"] = feat.index.month

    # UK Bank Holiday feature
    uk_hols = get_uk_holidays()
    feat["is_holiday"] = feat.index.date
    feat["is_holiday"] = feat["is_holiday"].apply(lambda x: 1 if x in uk_hols else 0)

    # Lag features
    feat["lag_1"] = feat[target_col].shift(1)
    feat["lag_24"] = feat[target_col].shift(24)
    feat["lag_168"] = feat[target_col].shift(168)  # 1 week

    # Rolling features (shifted to avoid look-ahead)
    feat["roll_24_mean"] = feat[target_col].shift(1).rolling(24).mean()

    # Weather features (if available) - use lagged values to avoid look-ahead
    if "temperature" in df.columns:
        feat["temp"] = df["temperature"].shift(1)
        feat["temp_lag_24"] = df["temperature"].shift(24)
    if "humidity" in df.columns:
        feat["humidity"] = df["humidity"].shift(1)
    if "wind_speed" in df.columns:
        feat["wind_speed"] = df["wind_speed"].shift(1)
    # Solar radiation features (as suggested by tutor - affects demand via solar PV)
    if "solar_radiation" in df.columns:
        feat["solar_rad"] = df["solar_radiation"].shift(1)
        feat["solar_rad_lag_24"] = df["solar_radiation"].shift(24)
    if "direct_radiation" in df.columns:
        feat["direct_rad"] = df["direct_radiation"].shift(1)

    # Carbon intensity and generation mix features (from National Grid ESO)
    if "carbon_intensity" in df.columns:
        feat["carbon_intensity"] = df["carbon_intensity"].shift(1)
    # Generation mix - percentage of each fuel type (affects demand patterns)
    for fuel in ["gen_gas", "gen_wind", "gen_solar", "gen_nuclear"]:
        if fuel in df.columns:
            feat[fuel] = df[fuel].shift(1)

    return feat


def _build_daily_features(df: pd.DataFrame, target_col: str) -> pd.DataFrame:
    """
    Build features for daily forecasting.

    Calendar: dow, month, day_of_year, is_weekend, is_holiday
    Lags: lag_1, lag_7
    Rolling: roll_7_mean, roll_30_mean
    Weather: temp_mean, temp_lag_7 (if available)
    """
    feat = df.copy()

    # Calendar features
    feat["dow"] = feat.index.dayofweek
    feat["month"] = feat.index.month
    feat["day_of_year"] = feat.index.dayofyear
    feat["is_weekend"] = (feat.index.dayofweek >= 5).astype(int)

    # UK Bank Holiday feature
    uk_hols = get_uk_holidays()
    feat["is_holiday"] = feat.index.date
    feat["is_holiday"] = feat["is_holiday"].apply(lambda x: 1 if x in uk_hols else 0)

    # Lag features
    feat["lag_1"] = feat[target_col].shift(1)
    feat["lag_7"] = feat[target_col].shift(7)

    # Rolling features (shifted to avoid look-ahead)
    feat["roll_7_mean"] = feat[target_col].shift(1).rolling(7).mean()
    feat["roll_30_mean"] = feat[target_col].shift(1).rolling(30).mean()

    # Weather features (if available) - use lagged values
    if "temperature" in df.columns:
        feat["temp"] = df["temperature"].shift(1)
        feat["temp_lag_7"] = df["temperature"].shift(7)
        feat["temp_roll_7"] = df["temperature"].shift(1).rolling(7).mean()

    return feat


def _build_weekly_features(df: pd.DataFrame, target_col: str) -> pd.DataFrame:
    """
    Build features for weekly forecasting.

    Calendar: week_of_year, month, quarter, has_holiday
    Lags: lag_1, lag_4, lag_52
    Rolling: roll_4_mean
    """
    feat = df.copy()

    # Calendar features
    feat["week_of_year"] = feat.index.isocalendar().week.astype(int)
    feat["month"] = feat.index.month
    feat["quarter"] = feat.index.quarter

    # UK Bank Holiday feature - count holidays in each week
    uk_hols = get_uk_holidays()
    def count_holidays_in_week(start_date):
        """Count UK holidays in the week starting from start_date."""
        count = 0
        for i in range(7):
            day = (start_date + pd.Timedelta(days=i)).date()
            if day in uk_hols:
                count += 1
        return count

    feat["has_holiday"] = feat.index.to_series().apply(count_holidays_in_week)

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
            "hour", "dow", "month", "is_holiday",
            "lag_1", "lag_24", "lag_168",
            "roll_24_mean"
        ],
        Granularity.DAILY: [
            "dow", "month", "day_of_year", "is_weekend", "is_holiday",
            "lag_1", "lag_7",
            "roll_7_mean", "roll_30_mean"
        ],
        Granularity.WEEKLY: [
            "week_of_year", "month", "quarter", "has_holiday",
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

    # Include weather features if present
    weather_features = ["temp", "temp_lag_24", "temp_lag_7", "temp_roll_7",
                       "humidity", "wind_speed",
                       "solar_rad", "solar_rad_lag_24", "direct_rad"]
    for col in weather_features:
        if col in df.columns and col not in available:
            available.append(col)

    # Include holiday features if present
    holiday_features = ["is_holiday", "has_holiday"]
    for col in holiday_features:
        if col in df.columns and col not in available:
            available.append(col)

    # Include carbon intensity and generation mix features if present
    carbon_features = ["carbon_intensity", "gen_gas", "gen_wind", "gen_solar", "gen_nuclear"]
    for col in carbon_features:
        if col in df.columns and col not in available:
            available.append(col)

    return available
