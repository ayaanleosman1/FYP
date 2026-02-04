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
NESO_DATASET_ID = "8f2fe0af-871c-488d-8bad-960426f24601"
NESO_API_BASE = "https://api.neso.energy/api/3/action"

# Direct download URLs for demand data (updated 29 January 2025)
DEMAND_DATA_URLS = {
    2025: "https://api.neso.energy/dataset/8f2fe0af-871c-488d-8bad-960426f24601/resource/b2bde559-3455-4021-b179-dfe60c0337b0/download/demanddata_2025.csv",
    2024: "https://api.neso.energy/dataset/8f2fe0af-871c-488d-8bad-960426f24601/resource/f6d02c0f-957b-48cb-82ee-09003f2ba759/download/demanddata_2024.csv",
    2023: "https://api.neso.energy/dataset/8f2fe0af-871c-488d-8bad-960426f24601/resource/bf5ab335-9b40-4ea4-b93a-ab4af7bce003/download/demanddata_2023.csv",
    2022: "https://api.neso.energy/dataset/8f2fe0af-871c-488d-8bad-960426f24601/resource/bb44a1b5-75b1-4db2-8491-257f23385006/download/demanddata_2022.csv",
    2021: "https://api.neso.energy/dataset/8f2fe0af-871c-488d-8bad-960426f24601/resource/18c69c42-f20d-46f0-84e9-e279045befc6/download/demanddata_2021.csv",
    2020: "https://api.neso.energy/dataset/8f2fe0af-871c-488d-8bad-960426f24601/resource/33ba6857-2a55-479f-9308-e5c4c53d4381/download/demanddata_2020.csv",
    2019: "https://api.neso.energy/dataset/8f2fe0af-871c-488d-8bad-960426f24601/resource/dd9de980-d724-415a-b344-d8ae11321432/download/demanddata_2019.csv",
    2018: "https://api.neso.energy/dataset/8f2fe0af-871c-488d-8bad-960426f24601/resource/fcb12133-0db0-4f27-a4a5-1669fd9f6d33/download/demanddata_2018.csv",
    2017: "https://api.neso.energy/dataset/8f2fe0af-871c-488d-8bad-960426f24601/resource/2f0f75b8-39c5-46ff-a914-ae38088ed022/download/demanddata_2017.csv",
    2016: "https://api.neso.energy/dataset/8f2fe0af-871c-488d-8bad-960426f24601/resource/3bb75a28-ab44-4a0b-9b1c-9be9715d3c44/download/demanddata_2016.csv",
    2015: "https://api.neso.energy/dataset/8f2fe0af-871c-488d-8bad-960426f24601/resource/cc505e45-65ae-4819-9b90-1fbb06880293/download/demanddata_2015.csv",
    2014: "https://api.neso.energy/dataset/8f2fe0af-871c-488d-8bad-960426f24601/resource/b9005225-49d3-40d1-921c-03ee2d83a2ff/download/demanddata_2014.csv",
    2013: "https://api.neso.energy/dataset/8f2fe0af-871c-488d-8bad-960426f24601/resource/2ff7aaff-8b42-4c1b-b234-9446573a1e27/download/demanddata_2013.csv",
    2012: "https://api.neso.energy/dataset/8f2fe0af-871c-488d-8bad-960426f24601/resource/4bf713a2-ea0c-44d3-a09a-63fc6a634b00/download/demanddata_2012.csv",
    2011: "https://api.neso.energy/dataset/8f2fe0af-871c-488d-8bad-960426f24601/resource/01522076-2691-4140-bfb8-c62284752efd/download/demanddata_2011.csv",
    2010: "https://api.neso.energy/dataset/8f2fe0af-871c-488d-8bad-960426f24601/resource/b3eae4a5-8c3c-4df1-b9de-7db243ac3a09/download/demanddata_2010.csv",
    2009: "https://api.neso.energy/dataset/8f2fe0af-871c-488d-8bad-960426f24601/resource/ed8a37cb-65ac-4581-8dbc-a3130780da3a/download/demanddata_2009.csv",
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
