"""
Download historical UK weather data from Open-Meteo API.

Uses a central UK location (Birmingham) as representative for national demand.
Weather is a major driver of electricity demand (heating/cooling).

Data source: Open-Meteo Historical Weather API
https://open-meteo.com/en/docs/historical-weather-api
"""

import argparse
import json
from datetime import datetime, timedelta
from pathlib import Path
import urllib.request
import urllib.parse
import time


# Central UK location (Birmingham) - representative for national weather
UK_LAT = 52.4862
UK_LON = -1.8904

# Open-Meteo API base URL
OPEN_METEO_URL = "https://archive-api.open-meteo.com/v1/archive"


def get_data_dir() -> Path:
    """Get the data directory path."""
    return Path(__file__).parent.parent.parent / "data"


def download_weather_year(year: int, output_dir: Path) -> bool:
    """
    Download weather data for a specific year.

    Args:
        year: Year to download
        output_dir: Directory to save the data

    Returns:
        True if successful, False otherwise
    """
    start_date = f"{year}-01-01"
    end_date = f"{year}-12-31"

    # Don't download future dates
    today = datetime.now().date()
    if datetime.strptime(end_date, "%Y-%m-%d").date() > today:
        end_date = (today - timedelta(days=1)).strftime("%Y-%m-%d")

    params = {
        "latitude": UK_LAT,
        "longitude": UK_LON,
        "start_date": start_date,
        "end_date": end_date,
        "hourly": ",".join([
            "temperature_2m",
            "relative_humidity_2m",
            "wind_speed_10m",
            "cloud_cover",
            "precipitation",
            # Solar radiation features (as suggested by tutor)
            "shortwave_radiation",  # Total solar radiation (W/m²)
            "direct_radiation",     # Direct sunlight (W/m²)
        ]),
        "timezone": "Europe/London",
    }

    url = f"{OPEN_METEO_URL}?{urllib.parse.urlencode(params)}"
    output_file = output_dir / f"weather_{year}.json"

    print(f"Downloading {year} weather data...")
    print(f"  URL: {url[:80]}...")

    try:
        with urllib.request.urlopen(url, timeout=60) as response:
            data = json.loads(response.read().decode())

        # Save raw JSON
        with open(output_file, 'w') as f:
            json.dump(data, f)

        hours = len(data.get("hourly", {}).get("time", []))
        print(f"  Saved {hours} hours to {output_file}")
        return True

    except Exception as e:
        print(f"  Error: {e}")
        return False


def download_all_years(start_year: int, end_year: int) -> dict:
    """Download weather data for a range of years."""
    output_dir = get_data_dir()
    output_dir.mkdir(parents=True, exist_ok=True)

    results = {}
    for year in range(start_year, end_year + 1):
        results[year] = download_weather_year(year, output_dir)
        # Rate limiting - be nice to the free API
        time.sleep(1)

    return results


def parse_args():
    parser = argparse.ArgumentParser(description="Download UK weather data")
    parser.add_argument(
        "--start-year", "-s",
        type=int,
        default=2009,
        help="Start year (default: 2009)"
    )
    parser.add_argument(
        "--end-year", "-e",
        type=int,
        default=2024,
        help="End year (default: 2024)"
    )
    return parser.parse_args()


def main():
    args = parse_args()

    print("=" * 50)
    print("UK Weather Data Downloader")
    print("=" * 50)
    print(f"Location: Birmingham, UK ({UK_LAT}, {UK_LON})")
    print(f"Years: {args.start_year} to {args.end_year}")
    print()

    results = download_all_years(args.start_year, args.end_year)

    # Summary
    print("\n" + "=" * 50)
    print("Summary")
    print("=" * 50)

    successful = [y for y, s in results.items() if s]
    failed = [y for y, s in results.items() if not s]

    print(f"Successful: {len(successful)}/{len(results)}")
    if failed:
        print(f"Failed: {failed}")


if __name__ == "__main__":
    main()
