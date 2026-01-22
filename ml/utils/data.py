"""
Data generation and resampling utilities for multi-timeframe forecasting.
"""

import numpy as np
import pandas as pd
from typing import Optional

from .granularity import Granularity, GRANULARITY_CONFIG


def make_synthetic_hourly_demand(n_days: int = 90, seed: int = 42) -> pd.DataFrame:
    """
    Generate synthetic hourly electricity demand data.

    The synthetic data models realistic patterns:
    - Base demand around 30,000 units
    - Daily sinusoidal pattern with 2,500 amplitude (peak at midday)
    - Weekend reduction of 6,000 units
    - Random Gaussian noise with std=2,500

    Args:
        n_days: Number of days of data to generate
        seed: Random seed for reproducibility

    Returns:
        DataFrame with datetime index and 'demand' column
    """
    np.random.seed(seed)
    hours = n_days * 24
    idx = pd.date_range("2026-01-01", periods=hours, freq="h")

    # Base demand
    base = 30_000

    # Daily pattern: sinusoidal with peak around midday
    hour_of_day = idx.hour
    daily_pattern = 2_500 * np.sin(np.pi * (hour_of_day - 6) / 12)

    # Weekend effect
    is_weekend = idx.dayofweek >= 5
    weekend_effect = np.where(is_weekend, -6_000, 0)

    # Random noise
    noise = np.random.normal(0, 2_500, hours)

    # Combine components
    demand = base + daily_pattern + weekend_effect + noise

    return pd.DataFrame({"demand": demand}, index=idx)


def resample_to_granularity(
    df: pd.DataFrame,
    granularity: Granularity,
    agg_func: str = "sum",
) -> pd.DataFrame:
    """
    Resample hourly data to a different granularity.

    Args:
        df: DataFrame with datetime index and 'demand' column
        granularity: Target granularity (D, W, M, Y)
        agg_func: Aggregation function ('sum', 'mean', 'max', 'min')

    Returns:
        Resampled DataFrame
    """
    if granularity == Granularity.HOURLY:
        return df.copy()

    config = granularity.config
    resampled = df.resample(config.pandas_freq).agg(agg_func)
    return resampled


def get_data_for_granularity(
    n_days: int,
    granularity: Granularity,
    seed: int = 42,
    agg_func: str = "sum",
) -> pd.DataFrame:
    """
    Get synthetic data at the specified granularity.

    This generates hourly data and then resamples to the target granularity.

    Args:
        n_days: Number of days of hourly data to generate
        granularity: Target granularity
        seed: Random seed
        agg_func: Aggregation function for resampling

    Returns:
        DataFrame with data at the specified granularity
    """
    # Generate hourly data
    hourly = make_synthetic_hourly_demand(n_days=n_days, seed=seed)

    # Resample to target granularity
    return resample_to_granularity(hourly, granularity, agg_func=agg_func)


def get_minimum_days_for_granularity(granularity: Granularity) -> int:
    """
    Get the minimum number of days of data needed for a granularity.

    This ensures we have enough data for:
    - Test set (default_test_periods)
    - Lag features
    - Rolling features
    - Training data

    Args:
        granularity: Target granularity

    Returns:
        Minimum number of days
    """
    minimums = {
        Granularity.HOURLY: 30,    # 7 days test + lags + training
        Granularity.DAILY: 60,     # 7 days test + 30 day rolling + training
        Granularity.WEEKLY: 120,   # ~17 weeks, need lag_52 ideally but we'll use what we have
        Granularity.MONTHLY: 365,  # ~12 months, need lag_12
        Granularity.YEARLY: 1825,  # 5 years, need at least 2 for any lag
    }
    return minimums.get(granularity, 90)


def get_recommended_days_for_granularity(granularity: Granularity) -> int:
    """
    Get the recommended number of days of data for good model training.

    Args:
        granularity: Target granularity

    Returns:
        Recommended number of days
    """
    recommended = {
        Granularity.HOURLY: 90,     # ~3 months
        Granularity.DAILY: 365,     # 1 year
        Granularity.WEEKLY: 730,    # 2 years
        Granularity.MONTHLY: 1095,  # 3 years
        Granularity.YEARLY: 3650,   # 10 years
    }
    return recommended.get(granularity, 90)


def train_test_split_temporal(
    df: pd.DataFrame,
    test_periods: int,
    granularity: Granularity,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split data into train and test sets based on time.

    For time series, we always use the most recent data for testing
    to avoid look-ahead bias.

    Args:
        df: DataFrame with datetime index
        test_periods: Number of periods to use for testing
        granularity: Granularity to determine period length

    Returns:
        Tuple of (train_df, test_df)
    """
    config = granularity.config

    if granularity == Granularity.HOURLY:
        # test_periods is number of hours
        split_point = df.index.max() - pd.Timedelta(hours=test_periods)
    elif granularity == Granularity.DAILY:
        split_point = df.index.max() - pd.Timedelta(days=test_periods)
    elif granularity == Granularity.WEEKLY:
        split_point = df.index.max() - pd.Timedelta(weeks=test_periods)
    elif granularity == Granularity.MONTHLY:
        # Approximate: 30 days per month
        split_point = df.index.max() - pd.DateOffset(months=test_periods)
    elif granularity == Granularity.YEARLY:
        split_point = df.index.max() - pd.DateOffset(years=test_periods)
    else:
        raise ValueError(f"Unknown granularity: {granularity}")

    train = df[df.index <= split_point]
    test = df[df.index > split_point]

    return train, test
