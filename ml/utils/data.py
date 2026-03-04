import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional, Literal

from .granularity import Granularity, GRANULARITY_CONFIG


def get_data_dir():
    return Path(__file__).parent.parent.parent / "data"


def make_synthetic_hourly_demand(n_days=90, seed=42):
    np.random.seed(seed)
    hours = n_days * 24
    idx = pd.date_range("2026-01-01", periods=hours, freq="h")

    base = 30_000
    hour_of_day = idx.hour
    daily_pattern = 2_500 * np.sin(np.pi * (hour_of_day - 6) / 12)
    is_weekend = idx.dayofweek >= 5
    weekend_effect = np.where(is_weekend, -6_000, 0)
    noise = np.random.normal(0, 2_500, hours)

    demand = base + daily_pattern + weekend_effect + noise
    return pd.DataFrame({"demand": demand}, index=idx)


def load_real_demand_data(years=None, demand_column="ND"):
    data_dir = get_data_dir()

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

        # settlement period 1 = 00:00-00:30, period 2 = 00:30-01:00, etc.
        df["datetime"] = pd.to_datetime(df["SETTLEMENT_DATE"]) + \
            pd.to_timedelta((df["SETTLEMENT_PERIOD"] - 1) * 30, unit="m")

        df = df[["datetime", demand_column]].rename(columns={demand_column: "demand"})
        all_data.append(df)

    if not all_data:
        raise FileNotFoundError(f"No valid demand data files found for years {years}")

    combined = pd.concat(all_data, ignore_index=True)
    combined = combined.set_index("datetime").sort_index()
    combined = combined[~combined.index.duplicated(keep='first')]

    # resample half-hourly to hourly
    hourly = combined.resample("h").mean()
    hourly = hourly.dropna()
    hourly = hourly[hourly["demand"] > 0]

    return hourly


def load_carbon_data(years=None):
    data_dir = get_data_dir()

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

        # build generation mix lookup by timestamp
        gen_lookup = {}
        for rec in generation_records:
            ts = rec.get("from")
            if ts and "generationmix" in rec:
                gen_lookup[ts] = {item["fuel"]: item["perc"] for item in rec["generationmix"]}

        for rec in intensity_records:
            ts = rec.get("from")
            if not ts:
                continue

            row = {
                "datetime": pd.to_datetime(ts),
                "carbon_intensity": rec.get("intensity", {}).get("actual"),
            }

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
    df.index = df.index.tz_localize(None)

    # resample to hourly (carbon data is half-hourly)
    df = df.resample("h").mean()

    return df


def load_weather_data(years=None):
    data_dir = get_data_dir()

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


def load_demand_with_weather(years=None, include_weather=True, include_carbon=True):
    demand_df = load_real_demand_data(years=years)

    if not include_weather and not include_carbon:
        return demand_df

    merged = demand_df

    if include_weather:
        weather_df = load_weather_data(years=years)
        if weather_df is None:
            print("Warning: Weather data not available")
        else:
            merged = merged.join(weather_df, how='left')
            weather_cols = ['temperature', 'humidity', 'wind_speed', 'cloud_cover', 'precipitation',
                            'solar_radiation', 'direct_radiation']
            for col in weather_cols:
                if col in merged.columns:
                    merged[col] = merged[col].ffill()

    if include_carbon:
        carbon_df = load_carbon_data(years=years)
        if carbon_df is None:
            print("Warning: Carbon data not available (requires 2018+)")
        else:
            merged = merged.join(carbon_df, how='left')
            carbon_cols = ['carbon_intensity', 'gen_gas', 'gen_coal', 'gen_wind',
                          'gen_solar', 'gen_nuclear', 'gen_hydro', 'gen_biomass', 'gen_imports']
            for col in carbon_cols:
                if col in merged.columns:
                    merged[col] = merged[col].ffill()

    return merged


def get_hourly_data(source="real", n_days=None, years=None, seed=42, include_weather=True):
    if source == "synthetic":
        if n_days is None:
            n_days = 90
        return make_synthetic_hourly_demand(n_days=n_days, seed=seed)

    df = load_demand_with_weather(years=years, include_weather=include_weather)
    if n_days is not None:
        df = df.tail(n_days * 24)

    weather_info = ""
    if include_weather and 'temperature' in df.columns:
        weather_info = " + weather features"
    print(f"Using real UK National Grid demand data ({len(df)} hours{weather_info})")
    return df


def resample_to_granularity(df, granularity, drop_incomplete=True):
    if granularity == Granularity.HOURLY:
        return df.copy()

    config = granularity.config

    # demand is summed (total energy), everything else is averaged
    agg_dict = {}
    for col in df.columns:
        if col == "demand":
            agg_dict[col] = "sum"
        else:
            agg_dict[col] = "mean"
    resampled = df.resample(config.pandas_freq).agg(agg_dict)

    if drop_incomplete:
        hours_per_period = df['demand'].resample(config.pandas_freq).count()

        expected_hours = {
            'D': 24,
            'W-MON': 168,
            'MS': 28 * 24,
            'YS': 365 * 24,
        }

        min_hours = expected_hours.get(config.pandas_freq, 24)

        if config.pandas_freq == 'MS':
            min_hours = 27 * 24
        if config.pandas_freq == 'W-MON':
            min_hours = 6 * 24

        complete_mask = hours_per_period >= min_hours
        resampled = resampled[complete_mask]

    return resampled


def get_data_for_granularity(n_days, granularity, seed=42, source="real", years=None):
    hourly = get_hourly_data(source=source, n_days=n_days, years=years, seed=seed)
    return resample_to_granularity(hourly, granularity)


def get_minimum_days_for_granularity(granularity):
    minimums = {
        Granularity.HOURLY: 30,
        Granularity.DAILY: 60,
        Granularity.WEEKLY: 120,
        Granularity.MONTHLY: 365,
        Granularity.YEARLY: 1825,
    }
    return minimums.get(granularity, 90)


def get_recommended_days_for_granularity(granularity):
    recommended = {
        Granularity.HOURLY: 365,
        Granularity.DAILY: 1095,
        Granularity.WEEKLY: 2190,
        Granularity.MONTHLY: 3650,
        Granularity.YEARLY: 5840,
    }
    return recommended.get(granularity, 90)


def train_test_split_temporal(df, test_periods, granularity):
    config = granularity.config

    if granularity == Granularity.HOURLY:
        split_point = df.index.max() - pd.Timedelta(hours=test_periods)
    elif granularity == Granularity.DAILY:
        split_point = df.index.max() - pd.Timedelta(days=test_periods)
    elif granularity == Granularity.WEEKLY:
        split_point = df.index.max() - pd.Timedelta(weeks=test_periods)
    elif granularity == Granularity.MONTHLY:
        split_point = df.index.max() - pd.DateOffset(months=test_periods)
    elif granularity == Granularity.YEARLY:
        split_point = df.index.max() - pd.DateOffset(years=test_periods)
    else:
        raise ValueError(f"Unknown granularity: {granularity}")

    train = df[df.index <= split_point]
    test = df[df.index > split_point]

    return train, test
