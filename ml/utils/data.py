"""
Data generation and resampling utilities for multi-timeframe forecasting.

Supports both synthetic data generation and real UK electricity demand data
from National Grid ESO/NESO.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional, Literal

from .granularity import Granularity, GRANULARITY_CONFIG


def get_data_dir() -> Path:
    """Get the data directory path."""
    return Path(__file__).parent.parent.parent / "data"


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


def load_real_demand_data(
    years: Optional[list[int]] = None,
    demand_column: str = "ND",
) -> pd.DataFrame:
    """
    Load real UK electricity demand data from CSV files.

    The data is from National Grid ESO/NESO Historic Demand Data.
    - Half-hourly data (settlement periods 1-48)
    - Resampled to hourly by taking the mean of two half-hourly periods

    Args:
        years: List of years to load (default: all available)
        demand_column: Column to use for demand (default: "ND" = National Demand)

    Returns:
        DataFrame with datetime index and 'demand' column (hourly, in MW)
    """
    data_dir = get_data_dir()

    # Find available data files
    if years is None:
        csv_files = list(data_dir.glob("demanddata_*.csv"))
        years = [int(f.stem.split("_")[1]) for f in csv_files]
        years.sort()

    if not years:
        raise FileNotFoundError(
            f"No demand data files found in {data_dir}. "
            "Run 'python ml/utils/download_data.py' to download data."
        )

    all_data = []
    for year in years:
        file_path = data_dir / f"demanddata_{year}.csv"
        if not file_path.exists():
            print(f"Warning: {file_path} not found, skipping year {year}")
            continue

        df = pd.read_csv(file_path)

        # Create datetime from SETTLEMENT_DATE and SETTLEMENT_PERIOD
        # Settlement period 1 = 00:00-00:30, period 2 = 00:30-01:00, etc.
        df["datetime"] = pd.to_datetime(df["SETTLEMENT_DATE"]) + \
            pd.to_timedelta((df["SETTLEMENT_PERIOD"] - 1) * 30, unit="m")

        df = df[["datetime", demand_column]].rename(columns={demand_column: "demand"})
        all_data.append(df)

    if not all_data:
        raise FileNotFoundError(
            f"No valid demand data files found for years {years}"
        )

    # Combine all years
    combined = pd.concat(all_data, ignore_index=True)
    combined = combined.set_index("datetime").sort_index()

    # Remove any duplicates
    combined = combined[~combined.index.duplicated(keep='first')]

    # Resample half-hourly to hourly (mean of two periods)
    hourly = combined.resample("h").mean()

    # Drop any NaN rows
    hourly = hourly.dropna()

    # Drop rows with zero demand (likely missing/placeholder data)
    hourly = hourly[hourly["demand"] > 0]

    return hourly


def get_hourly_data(
    source: Literal["real", "synthetic", "auto"] = "auto",
    n_days: Optional[int] = None,
    years: Optional[list[int]] = None,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Get hourly demand data from specified source.

    Args:
        source: Data source ("real", "synthetic", or "auto")
            - "real": Load real UK data from CSV files
            - "synthetic": Generate synthetic data
            - "auto": Try real first, fall back to synthetic
        n_days: Number of days (used for synthetic data or to limit real data)
        years: Years to load (used for real data)
        seed: Random seed (used for synthetic data)

    Returns:
        DataFrame with datetime index and 'demand' column
    """
    if source == "synthetic":
        if n_days is None:
            n_days = 90
        return make_synthetic_hourly_demand(n_days=n_days, seed=seed)

    if source == "real":
        df = load_real_demand_data(years=years)
        if n_days is not None:
            # Take the last n_days of data
            df = df.tail(n_days * 24)
        return df

    # source == "auto"
    try:
        df = load_real_demand_data(years=years)
        if n_days is not None:
            df = df.tail(n_days * 24)
        print(f"Using real UK demand data ({len(df)} hours)")
        return df
    except FileNotFoundError:
        print("Real data not found, using synthetic data")
        if n_days is None:
            n_days = 90
        return make_synthetic_hourly_demand(n_days=n_days, seed=seed)


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
    source: Literal["real", "synthetic", "auto"] = "auto",
    years: Optional[list[int]] = None,
) -> pd.DataFrame:
    """
    Get data at the specified granularity.

    This gets hourly data (real or synthetic) and then resamples to the target granularity.

    Args:
        n_days: Number of days of hourly data to use
        granularity: Target granularity
        seed: Random seed (for synthetic data)
        agg_func: Aggregation function for resampling
        source: Data source ("real", "synthetic", or "auto")
        years: Years to load (for real data)

    Returns:
        DataFrame with data at the specified granularity
    """
    # Get hourly data from specified source
    hourly = get_hourly_data(source=source, n_days=n_days, years=years, seed=seed)

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
