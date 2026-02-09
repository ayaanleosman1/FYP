"""
Data loading and resampling utilities for multi-timeframe forecasting.

Uses real UK electricity demand data from National Grid ESO/NESO.
Data source: https://www.neso.energy/data-portal/historic-demand-data

Weather data from Open-Meteo Historical API.
Data source: https://open-meteo.com/en/docs/historical-weather-api
"""

import json
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


def load_carbon_data(years: Optional[list[int]] = None) -> pd.DataFrame:
    """
    Load UK Carbon Intensity and Generation Mix data from JSON files.

    Carbon data includes:
    - carbon_intensity: Actual carbon intensity (gCO2/kWh)
    - Generation mix percentages: gas, coal, wind, solar, nuclear, hydro, biomass, imports

    Note: Carbon Intensity API data is available from 2018 onwards.

    Args:
        years: List of years to load (default: all available)

    Returns:
        DataFrame with datetime index and carbon/generation columns
    """
    data_dir = get_data_dir()

    # Find available carbon files
    if years is None:
        json_files = list(data_dir.glob("carbon_*.json"))
        years = [int(f.stem.split("_")[1]) for f in json_files]
        years.sort()

    if not years:
        print("Warning: No carbon data found. Run 'python ml/utils/download_carbon.py'")
        return None

    all_data = []
    for year in years:
        file_path = data_dir / f"carbon_{year}.json"
        if not file_path.exists():
            continue

        with open(file_path, 'r') as f:
            data = json.load(f)

        intensity_records = data.get("intensity", [])
        generation_records = data.get("generation", [])

        # Build generation mix lookup by timestamp
        gen_lookup = {}
        for rec in generation_records:
            ts = rec.get("from")
            if ts and "generationmix" in rec:
                gen_lookup[ts] = {item["fuel"]: item["perc"] for item in rec["generationmix"]}

        # Process intensity records and join with generation
        for rec in intensity_records:
            ts = rec.get("from")
            if not ts:
                continue

            row = {
                "datetime": pd.to_datetime(ts),
                "carbon_intensity": rec.get("intensity", {}).get("actual"),
            }

            # Add generation mix if available
            if ts in gen_lookup:
                gen = gen_lookup[ts]
                row["gen_gas"] = gen.get("gas", 0)
                row["gen_coal"] = gen.get("coal", 0)
                row["gen_wind"] = gen.get("wind", 0)
                row["gen_solar"] = gen.get("solar", 0)
                row["gen_nuclear"] = gen.get("nuclear", 0)
                row["gen_hydro"] = gen.get("hydro", 0)
                row["gen_biomass"] = gen.get("biomass", 0)
                row["gen_imports"] = gen.get("imports", 0)

            all_data.append(row)

    if not all_data:
        return None

    df = pd.DataFrame(all_data)
    df = df.set_index("datetime").sort_index()
    df = df[~df.index.duplicated(keep='first')]

    # Remove timezone info to match demand data (which is timezone-naive)
    df.index = df.index.tz_localize(None)

    # Resample to hourly (carbon data is half-hourly) by taking mean
    df = df.resample("h").mean()

    return df


def load_weather_data(years: Optional[list[int]] = None) -> pd.DataFrame:
    """
    Load UK weather data from JSON files.

    Weather data includes:
    - temperature_2m: Temperature at 2m height (°C)
    - relative_humidity_2m: Relative humidity (%)
    - wind_speed_10m: Wind speed at 10m height (km/h)
    - cloud_cover: Cloud cover (%)
    - precipitation: Precipitation (mm)

    Args:
        years: List of years to load (default: all available)

    Returns:
        DataFrame with datetime index and weather columns
    """
    data_dir = get_data_dir()

    # Find available weather files
    if years is None:
        json_files = list(data_dir.glob("weather_*.json"))
        years = [int(f.stem.split("_")[1]) for f in json_files]
        years.sort()

    if not years:
        print("Warning: No weather data found. Run 'python ml/utils/download_weather.py'")
        return None

    all_data = []
    for year in years:
        file_path = data_dir / f"weather_{year}.json"
        if not file_path.exists():
            continue

        with open(file_path, 'r') as f:
            data = json.load(f)

        hourly = data.get("hourly", {})
        if not hourly:
            continue

        df = pd.DataFrame({
            "datetime": pd.to_datetime(hourly["time"]),
            "temperature": hourly.get("temperature_2m"),
            "humidity": hourly.get("relative_humidity_2m"),
            "wind_speed": hourly.get("wind_speed_10m"),
            "cloud_cover": hourly.get("cloud_cover"),
            "precipitation": hourly.get("precipitation"),
            # Solar radiation features
            "solar_radiation": hourly.get("shortwave_radiation"),
            "direct_radiation": hourly.get("direct_radiation"),
        })
        all_data.append(df)

    if not all_data:
        return None

    combined = pd.concat(all_data, ignore_index=True)
    combined = combined.set_index("datetime").sort_index()
    combined = combined[~combined.index.duplicated(keep='first')]

    return combined


def load_demand_with_weather(
    years: Optional[list[int]] = None,
    include_weather: bool = True,
    include_carbon: bool = True,
) -> pd.DataFrame:
    """
    Load demand data merged with weather and carbon intensity features.

    Args:
        years: List of years to load
        include_weather: Whether to include weather features
        include_carbon: Whether to include carbon intensity/generation mix features

    Returns:
        DataFrame with demand and optional weather/carbon columns
    """
    demand_df = load_real_demand_data(years=years)

    if not include_weather and not include_carbon:
        return demand_df

    merged = demand_df

    # Add weather features
    if include_weather:
        weather_df = load_weather_data(years=years)
        if weather_df is None:
            print("Warning: Weather data not available")
        else:
            merged = merged.join(weather_df, how='left')
            # Fill missing weather values
            weather_cols = ['temperature', 'humidity', 'wind_speed', 'cloud_cover', 'precipitation',
                            'solar_radiation', 'direct_radiation']
            for col in weather_cols:
                if col in merged.columns:
                    merged[col] = merged[col].ffill().bfill()

    # Add carbon intensity/generation mix features
    if include_carbon:
        carbon_df = load_carbon_data(years=years)
        if carbon_df is None:
            print("Warning: Carbon data not available (requires 2018+)")
        else:
            merged = merged.join(carbon_df, how='left')
            # Fill missing carbon values
            carbon_cols = ['carbon_intensity', 'gen_gas', 'gen_coal', 'gen_wind',
                          'gen_solar', 'gen_nuclear', 'gen_hydro', 'gen_biomass', 'gen_imports']
            for col in carbon_cols:
                if col in merged.columns:
                    merged[col] = merged[col].ffill().bfill()

    return merged


def get_hourly_data(
    source: Literal["real", "synthetic"] = "real",
    n_days: Optional[int] = None,
    years: Optional[list[int]] = None,
    seed: int = 42,
    include_weather: bool = True,
) -> pd.DataFrame:
    """
    Get hourly demand data from specified source.

    Args:
        source: Data source ("real" or "synthetic")
            - "real": Load real UK data from CSV files (default)
            - "synthetic": Generate synthetic data (for testing only)
        n_days: Number of days to limit data to
        years: Years to load (used for real data)
        seed: Random seed (used for synthetic data)
        include_weather: Whether to include weather features (real data only)

    Returns:
        DataFrame with datetime index and 'demand' column (plus weather if available)
    """
    if source == "synthetic":
        if n_days is None:
            n_days = 90
        return make_synthetic_hourly_demand(n_days=n_days, seed=seed)

    # source == "real" (default)
    df = load_demand_with_weather(years=years, include_weather=include_weather)
    if n_days is not None:
        # Take the last n_days of data
        df = df.tail(n_days * 24)

    weather_info = ""
    if include_weather and 'temperature' in df.columns:
        weather_info = " + weather features"
    print(f"Using real UK National Grid demand data ({len(df)} hours{weather_info})")
    return df


def resample_to_granularity(
    df: pd.DataFrame,
    granularity: Granularity,
    agg_func: str = "sum",
    drop_incomplete: bool = True,
) -> pd.DataFrame:
    """
    Resample hourly data to a different granularity.

    Args:
        df: DataFrame with datetime index and 'demand' column
        granularity: Target granularity (D, W, M, Y)
        agg_func: Aggregation function ('sum', 'mean', 'max', 'min')
        drop_incomplete: Drop incomplete periods at start/end

    Returns:
        Resampled DataFrame
    """
    if granularity == Granularity.HOURLY:
        return df.copy()

    config = granularity.config
    resampled = df.resample(config.pandas_freq).agg(agg_func)

    if drop_incomplete:
        # Count hours per period to detect incomplete ones
        hours_per_period = df['demand'].resample(config.pandas_freq).count()

        # Expected hours per period
        expected_hours = {
            'D': 24,
            'W-MON': 168,  # 7 * 24
            'MS': 28 * 24,  # minimum ~672 hours (28 days)
            'YS': 365 * 24,  # ~8760 hours
        }

        min_hours = expected_hours.get(config.pandas_freq, 24)

        # For monthly, be more lenient (28-31 days)
        if config.pandas_freq == 'MS':
            min_hours = 27 * 24  # Allow 27+ days

        # For weekly, require at least 6 days
        if config.pandas_freq == 'W-MON':
            min_hours = 6 * 24  # Allow 6+ days

        # Filter to only complete periods
        complete_mask = hours_per_period >= min_hours
        resampled = resampled[complete_mask]

    return resampled


def get_data_for_granularity(
    n_days: int,
    granularity: Granularity,
    seed: int = 42,
    agg_func: str = "sum",
    source: Literal["real", "synthetic"] = "real",
    years: Optional[list[int]] = None,
) -> pd.DataFrame:
    """
    Get data at the specified granularity.

    This gets hourly data (real UK National Grid data by default) and then
    resamples to the target granularity.

    Args:
        n_days: Number of days of hourly data to use
        granularity: Target granularity
        seed: Random seed (for synthetic data)
        agg_func: Aggregation function for resampling
        source: Data source ("real" or "synthetic")
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
