"""
Download historical UK Carbon Intensity and Generation Mix data.

Data source: National Grid ESO Carbon Intensity API
https://carbonintensity.org.uk/

This provides:
- Carbon intensity (gCO2/kWh)
- Generation mix percentages (gas, coal, wind, solar, nuclear, hydro, biomass, imports)

The generation mix affects electricity demand patterns as it reflects
the energy landscape and renewable penetration.
"""

import argparse
import json
from datetime import datetime, timedelta
from pathlib import Path
import urllib.request
import urllib.error
import time

# Carbon Intensity API base URL
CARBON_API_URL = "https://api.carbonintensity.org.uk"


def get_data_dir() -> Path:
    """Get the data directory path."""
    return Path(__file__).parent.parent.parent / "data"


def download_carbon_range(start_date: str, end_date: str) -> dict | None:
    """
    Download carbon intensity data for a date range.

    The API allows max 30 days per request for intensity data.

    Args:
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)

    Returns:
        JSON response data or None if failed
    """
    # Use the intensity endpoint with date range
    url = f"{CARBON_API_URL}/intensity/{start_date}T00:00Z/{end_date}T23:59Z"

    try:
        req = urllib.request.Request(
            url,
            headers={"Accept": "application/json"}
        )
        with urllib.request.urlopen(req, timeout=60) as response:
            return json.loads(response.read().decode())
    except urllib.error.HTTPError as e:
        print(f"    HTTP Error {e.code}: {e.reason}")
        return None
    except Exception as e:
        print(f"    Error: {e}")
        return None


def download_generation_range(start_date: str, end_date: str) -> dict | None:
    """
    Download generation mix data for a date range.

    Args:
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)

    Returns:
        JSON response data or None if failed
    """
    url = f"{CARBON_API_URL}/generation/{start_date}T00:00Z/{end_date}T23:59Z"

    try:
        req = urllib.request.Request(
            url,
            headers={"Accept": "application/json"}
        )
        with urllib.request.urlopen(req, timeout=60) as response:
            return json.loads(response.read().decode())
    except urllib.error.HTTPError as e:
        print(f"    HTTP Error {e.code}: {e.reason}")
        return None
    except Exception as e:
        print(f"    Error: {e}")
        return None


def download_carbon_year(year: int, output_dir: Path) -> bool:
    """
    Download carbon intensity and generation mix for a specific year.

    Downloads in monthly chunks due to API limitations.

    Args:
        year: Year to download
        output_dir: Directory to save the data

    Returns:
        True if successful, False otherwise
    """
    # Carbon Intensity API data starts from 2018
    if year < 2018:
        print(f"  Skipping {year} - Carbon Intensity API data starts from 2018")
        return False

    print(f"Downloading {year} carbon intensity data...")

    all_intensity = []
    all_generation = []

    # Download in monthly chunks (API has limits)
    for month in range(1, 13):
        start = f"{year}-{month:02d}-01"

        # Calculate end of month
        if month == 12:
            end = f"{year}-12-31"
        else:
            next_month = datetime(year, month + 1, 1) - timedelta(days=1)
            end = next_month.strftime("%Y-%m-%d")

        # Don't download future dates
        today = datetime.now().date()
        end_date = datetime.strptime(end, "%Y-%m-%d").date()
        if end_date > today:
            end = (today - timedelta(days=1)).strftime("%Y-%m-%d")
            if datetime.strptime(start, "%Y-%m-%d").date() > today:
                continue

        print(f"  Month {month}: {start} to {end}")

        # Download intensity data
        intensity_data = download_carbon_range(start, end)
        if intensity_data and "data" in intensity_data:
            all_intensity.extend(intensity_data["data"])

        # Rate limit - be nice to free API
        time.sleep(0.5)

        # Download generation mix data
        gen_data = download_generation_range(start, end)
        if gen_data and "data" in gen_data:
            all_generation.extend(gen_data["data"])

        time.sleep(0.5)

    if not all_intensity:
        print(f"  No data retrieved for {year}")
        return False

    # Save combined data
    output_file = output_dir / f"carbon_{year}.json"
    combined = {
        "year": year,
        "intensity": all_intensity,
        "generation": all_generation,
    }

    with open(output_file, 'w') as f:
        json.dump(combined, f)

    print(f"  Saved {len(all_intensity)} intensity records to {output_file}")
    return True


def download_all_years(start_year: int, end_year: int) -> dict:
    """Download carbon data for a range of years."""
    output_dir = get_data_dir()
    output_dir.mkdir(parents=True, exist_ok=True)

    results = {}
    for year in range(start_year, end_year + 1):
        results[year] = download_carbon_year(year, output_dir)
        time.sleep(1)  # Rate limit between years

    return results


def parse_args():
    parser = argparse.ArgumentParser(description="Download UK Carbon Intensity data")
    parser.add_argument(
        "--start-year", "-s",
        type=int,
        default=2018,  # API data starts from 2018
        help="Start year (default: 2018, API minimum)"
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

    # Ensure start year is not before 2018
    start_year = max(args.start_year, 2018)

    print("=" * 50)
    print("UK Carbon Intensity Data Downloader")
    print("=" * 50)
    print(f"Source: National Grid ESO Carbon Intensity API")
    print(f"Years: {start_year} to {args.end_year}")
    print()

    results = download_all_years(start_year, args.end_year)

    # Summary
    print("\n" + "=" * 50)
    print("Summary")
    print("=" * 50)

    successful = [y for y, s in results.items() if s]
    failed = [y for y, s in results.items() if not s]

    print(f"Successful: {len(successful)}/{len(results)}")
    if failed:
        print(f"Failed/Skipped: {failed}")


if __name__ == "__main__":
    main()
