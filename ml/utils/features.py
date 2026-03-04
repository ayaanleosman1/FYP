import pandas as pd
import numpy as np
from typing import Optional
import holidays

from .granularity import Granularity

_uk_holidays = None

def get_uk_holidays():
    global _uk_holidays
    if _uk_holidays is None:
        _uk_holidays = holidays.UK(years=range(2000, 2030))
    return _uk_holidays


def build_features(df, granularity=Granularity.HOURLY, target_col="demand"):
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

    feat = feat.dropna()

    _raw_cols = [
        "temperature", "solar_radiation", "direct_radiation",
        "cloud_cover", "precipitation",
        "gen_coal", "gen_hydro", "gen_biomass", "gen_imports",
    ]
    feat = feat.drop(columns=[c for c in _raw_cols if c in feat.columns])

    return feat


def _shift_exogenous(feat, df):
    if "temperature" in df.columns and "temp" not in feat.columns:
        feat["temp"] = df["temperature"].shift(1)
    if "humidity" in df.columns:
        feat["humidity"] = df["humidity"].shift(1)
    if "wind_speed" in df.columns:
        feat["wind_speed"] = df["wind_speed"].shift(1)

    if "carbon_intensity" in df.columns:
        feat["carbon_intensity"] = df["carbon_intensity"].shift(1)

    for fuel in ["gen_gas", "gen_wind", "gen_solar", "gen_nuclear"]:
        if fuel in df.columns:
            feat[fuel] = df[fuel].shift(1)

    return feat


def _build_hourly_features(df, target_col):
    feat = df.copy()

    # calendar
    feat["hour"] = feat.index.hour
    feat["dow"] = feat.index.dayofweek
    feat["month"] = feat.index.month

    uk_hols = get_uk_holidays()
    feat["is_holiday"] = feat.index.date
    feat["is_holiday"] = feat["is_holiday"].apply(lambda x: 1 if x in uk_hols else 0)

    # lags
    feat["lag_1"] = feat[target_col].shift(1)
    feat["lag_24"] = feat[target_col].shift(24)
    feat["lag_168"] = feat[target_col].shift(168)  # 1 week

    feat["roll_24_mean"] = feat[target_col].shift(1).rolling(24).mean()

    if "temperature" in df.columns:
        feat["temp"] = df["temperature"].shift(1)
        feat["temp_lag_24"] = df["temperature"].shift(24)
    if "humidity" in df.columns:
        feat["humidity"] = df["humidity"].shift(1)
    if "wind_speed" in df.columns:
        feat["wind_speed"] = df["wind_speed"].shift(1)
    if "solar_radiation" in df.columns:
        feat["solar_rad"] = df["solar_radiation"].shift(1)
        feat["solar_rad_lag_24"] = df["solar_radiation"].shift(24)
    if "direct_radiation" in df.columns:
        feat["direct_rad"] = df["direct_radiation"].shift(1)

    # carbon intensity and generation mix
    if "carbon_intensity" in df.columns:
        feat["carbon_intensity"] = df["carbon_intensity"].shift(1)
    for fuel in ["gen_gas", "gen_wind", "gen_solar", "gen_nuclear"]:
        if fuel in df.columns:
            feat[fuel] = df[fuel].shift(1)

    return feat


def _build_daily_features(df, target_col):
    feat = df.copy()

    feat["dow"] = feat.index.dayofweek
    feat["month"] = feat.index.month
    feat["day_of_year"] = feat.index.dayofyear
    feat["is_weekend"] = (feat.index.dayofweek >= 5).astype(int)

    uk_hols = get_uk_holidays()
    feat["is_holiday"] = feat.index.date
    feat["is_holiday"] = feat["is_holiday"].apply(lambda x: 1 if x in uk_hols else 0)

    feat["lag_1"] = feat[target_col].shift(1)
    feat["lag_7"] = feat[target_col].shift(7)

    feat["roll_7_mean"] = feat[target_col].shift(1).rolling(7).mean()
    feat["roll_30_mean"] = feat[target_col].shift(1).rolling(30).mean()

    if "temperature" in df.columns:
        feat["temp"] = df["temperature"].shift(1)
        feat["temp_lag_7"] = df["temperature"].shift(7)
        feat["temp_roll_7"] = df["temperature"].shift(1).rolling(7).mean()

    feat = _shift_exogenous(feat, df)

    return feat


def _build_weekly_features(df, target_col):
    feat = df.copy()

    feat["week_of_year"] = feat.index.isocalendar().week.astype(int)
    feat["month"] = feat.index.month
    feat["quarter"] = feat.index.quarter

    # count bank holidays in each week
    uk_hols = get_uk_holidays()
    def count_holidays_in_week(start_date):
        count = 0
        for i in range(7):
            day = (start_date + pd.Timedelta(days=i)).date()
            if day in uk_hols:
                count += 1
        return count

    feat["has_holiday"] = feat.index.to_series().apply(count_holidays_in_week)

    feat["lag_1"] = feat[target_col].shift(1)
    feat["lag_4"] = feat[target_col].shift(4)

    if len(feat) > 52:
        feat["lag_52"] = feat[target_col].shift(52)
    else:
        max_lag = min(len(feat) - 1, 52)
        if max_lag > 4:
            feat[f"lag_{max_lag}"] = feat[target_col].shift(max_lag)

    feat["roll_4_mean"] = feat[target_col].shift(1).rolling(4).mean()

    feat = _shift_exogenous(feat, df)

    return feat


def _build_monthly_features(df, target_col):
    feat = df.copy()

    feat["month"] = feat.index.month
    feat["quarter"] = feat.index.quarter

    feat["lag_1"] = feat[target_col].shift(1)

    if len(feat) > 12:
        feat["lag_12"] = feat[target_col].shift(12)
    else:
        max_lag = min(len(feat) - 1, 12)
        if max_lag > 1:
            feat[f"lag_{max_lag}"] = feat[target_col].shift(max_lag)

    feat["roll_3_mean"] = feat[target_col].shift(1).rolling(3).mean()

    if len(feat) > 12:
        feat["roll_12_mean"] = feat[target_col].shift(1).rolling(12).mean()

    feat = _shift_exogenous(feat, df)

    return feat


def _build_yearly_features(df, target_col):
    feat = df.copy()

    feat["lag_1"] = feat[target_col].shift(1)

    if len(feat) > 2:
        feat["roll_2_mean"] = feat[target_col].shift(1).rolling(2).mean()

    feat = _shift_exogenous(feat, df)

    return feat


def get_feature_columns(granularity):
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


def get_available_features(df, granularity):
    expected = get_feature_columns(granularity)
    available = [col for col in expected if col in df.columns]

    # include any fallback lag columns
    for col in df.columns:
        if col.startswith("lag_") and col not in available:
            available.append(col)

    # weather features
    weather_features = ["temp", "temp_lag_24", "temp_lag_7", "temp_roll_7",
                       "humidity", "wind_speed",
                       "solar_rad", "solar_rad_lag_24", "direct_rad"]
    for col in weather_features:
        if col in df.columns and col not in available:
            available.append(col)

    # holiday features
    holiday_features = ["is_holiday", "has_holiday"]
    for col in holiday_features:
        if col in df.columns and col not in available:
            available.append(col)

    # carbon/generation mix features
    carbon_features = ["carbon_intensity", "gen_gas", "gen_wind", "gen_solar", "gen_nuclear"]
    for col in carbon_features:
        if col in df.columns and col not in available:
            available.append(col)

    return available
