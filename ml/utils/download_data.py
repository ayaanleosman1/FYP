"""
Download real UK electricity demand data from National Grid ESO/NESO.

Usage:
    python download_data.py                  # Download 2024 data
    python download_data.py --years 2023 2024  # Download specific years
    python download_data.py --list           # List available years

Data source: National Grid ESO/NESO Historic Demand Data
https://www.neso.energy/data-portal/historic-demand-data
"""

import argparse
import os
from pathlib import Path
import requests


# NESO API base URL for historic demand data
# The dataset ID for historic demand data
NESO_DATASET_ID = "7a12172a-939c-404c-b581-a6128b74f588"
NESO_API_BASE = "https://api.neso.energy/api/3/action"

# Direct download URLs for demand data (fallback)
DEMAND_DATA_URLS = {
    2024: "https://api.neso.energy/dataset/7a12172a-939c-404c-b581-a6128b74f588/resource/177f6fa4-ae49-4182-81ea-0c6b35f26ca6/download/demanddata_2024.csv",
    2023: "https://api.neso.energy/dataset/7a12172a-939c-404c-b581-a6128b74f588/resource/52eb8ec0-d903-4b34-a9b5-a12ab7cc5a47/download/demanddata_2023.csv",
    2022: "https://api.neso.energy/dataset/7a12172a-939c-404c-b581-a6128b74f588/resource/0ec03a49-e6ce-4e0b-941e-e06e44710921/download/demanddata_2022.csv",
    2021: "https://api.neso.energy/dataset/7a12172a-939c-404c-b581-a6128b74f588/resource/305c8efe-7e91-4428-b6f5-d7a4a7f18f35/download/demanddata_2021.csv",
    2020: "https://api.neso.energy/dataset/7a12172a-939c-404c-b581-a6128b74f588/resource/08be81b8-5004-4a6f-ac08-3cca4ae1f3a8/download/demanddata_2020.csv",
}


def get_data_dir() -> Path:
    """Get the data directory path."""
    # ml/utils/download_data.py -> ml/ -> project root -> data/
    return Path(__file__).parent.parent.parent / "data"


def download_file(url: str, dest_path: Path, chunk_size: int = 8192) -> bool:
    """
    Download a file from URL to destination path.

    Args:
        url: URL to download from
        dest_path: Path to save the file
        chunk_size: Size of chunks to download

    Returns:
        True if successful, False otherwise
    """
    try:
        print(f"  Downloading from: {url}")
        response = requests.get(url, stream=True, timeout=60)
        response.raise_for_status()

        total_size = int(response.headers.get('content-length', 0))
        downloaded = 0

        with open(dest_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=chunk_size):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total_size > 0:
                        pct = (downloaded / total_size) * 100
                        print(f"\r  Progress: {pct:.1f}%", end="", flush=True)

        print()  # Newline after progress
        return True

    except requests.exceptions.RequestException as e:
        print(f"  Error downloading: {e}")
        return False


def download_demand_data(year: int, data_dir: Path) -> bool:
    """
    Download demand data for a specific year.

    Args:
        year: Year to download (e.g., 2024)
        data_dir: Directory to save the data

    Returns:
        True if successful, False otherwise
    """
    if year not in DEMAND_DATA_URLS:
        print(f"  Error: No URL configured for year {year}")
        print(f"  Available years: {list(DEMAND_DATA_URLS.keys())}")
        return False

    url = DEMAND_DATA_URLS[year]
    dest_path = data_dir / f"demanddata_{year}.csv"

    print(f"\nDownloading {year} demand data...")

    if dest_path.exists():
        print(f"  File already exists: {dest_path}")
        print(f"  Skipping (delete file to re-download)")
        return True

    success = download_file(url, dest_path)

    if success:
        file_size = dest_path.stat().st_size / (1024 * 1024)  # MB
        print(f"  Saved to: {dest_path} ({file_size:.1f} MB)")

    return success


def list_available_years():
    """Print available years for download."""
    print("Available years for download:")
    for year in sorted(DEMAND_DATA_URLS.keys(), reverse=True):
        print(f"  - {year}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Download UK electricity demand data from NESO"
    )
    parser.add_argument(
        "--years", "-y",
        nargs="+",
        type=int,
        default=[2024],
        help="Years to download (default: 2024)"
    )
    parser.add_argument(
        "--list", "-l",
        action="store_true",
        help="List available years"
    )
    parser.add_argument(
        "--output-dir", "-o",
        type=str,
        default=None,
        help="Output directory (default: project data/ folder)"
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if args.list:
        list_available_years()
        return

    # Determine output directory
    if args.output_dir:
        data_dir = Path(args.output_dir)
    else:
        data_dir = get_data_dir()

    # Create data directory if it doesn't exist
    data_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 50)
    print("UK Electricity Demand Data Downloader")
    print("=" * 50)
    print(f"Output directory: {data_dir}")
    print(f"Years to download: {args.years}")

    # Download each year
    results = {}
    for year in args.years:
        results[year] = download_demand_data(year, data_dir)

    # Summary
    print("\n" + "=" * 50)
    print("Summary")
    print("=" * 50)

    successful = [y for y, s in results.items() if s]
    failed = [y for y, s in results.items() if not s]

    if successful:
        print(f"Successfully downloaded: {successful}")
    if failed:
        print(f"Failed: {failed}")

    return len(failed) == 0


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
