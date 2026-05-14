"""
Step 1.1 & 1.2: Download CDC PLACES 2024 census tract data and data dictionary.

Downloads:
  - GIS-friendly format (wide, one row per tract) -> data/raw/places/places_tract_2024.csv
  - Data dictionary JSON -> data/raw/places/places_data_dictionary.json
"""
import json
import sys
from pathlib import Path

import requests
import pandas as pd

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from hrsn_analysis.paths import PATHS, ensure_dirs
from hrsn_analysis.io_utils import load_params

def download_file(url, dest_path, description, chunk_size=8192):
    """Download a file with progress reporting."""
    print(f"Downloading {description}...")
    print(f"  URL: {url}")
    print(f"  Dest: {dest_path}")

    resp = requests.get(url, stream=True, timeout=600)
    resp.raise_for_status()

    total = int(resp.headers.get("content-length", 0))
    downloaded = 0

    with open(dest_path, "wb") as f:
        for chunk in resp.iter_content(chunk_size=chunk_size):
            f.write(chunk)
            downloaded += len(chunk)
            if total > 0:
                pct = downloaded / total * 100
                print(f"\r  Progress: {downloaded / 1e6:.1f} MB / {total / 1e6:.1f} MB ({pct:.0f}%)", end="", flush=True)
            else:
                print(f"\r  Downloaded: {downloaded / 1e6:.1f} MB", end="", flush=True)

    print(f"\n  Done. File size: {dest_path.stat().st_size / 1e6:.1f} MB")


def download_places_data():
    """Download CDC PLACES GIS-friendly census tract data."""
    ensure_dirs()
    params = load_params()

    url = params["data_sources"]["places_gis_friendly"]["url"]
    dest = PATHS["places_raw"] / "places_tract_2024.csv"

    if dest.exists():
        size_mb = dest.stat().st_size / 1e6
        print(f"PLACES data already exists ({size_mb:.1f} MB). Skipping download.")
        print("  Delete the file and re-run to re-download.")
    else:
        download_file(url, dest, "CDC PLACES 2024 GIS-friendly census tract data")

    # Verify
    print("\nVerifying PLACES data...")
    df = pd.read_csv(dest, nrows=5)
    print(f"  Columns ({len(df.columns)}): {list(df.columns[:10])}...")

    # Count total rows (without loading full file)
    row_count = sum(1 for _ in open(dest)) - 1  # subtract header
    print(f"  Total rows: {row_count:,}")
    print(f"  Expected: ~72,000-73,000 census tracts")

    return dest


def download_data_dictionary():
    """Download PLACES data dictionary JSON."""
    ensure_dirs()
    params = load_params()

    url = params["data_sources"]["places_data_dictionary"]["url"]
    dest = PATHS["places_raw"] / "places_data_dictionary.json"

    if dest.exists():
        print(f"Data dictionary already exists. Skipping download.")
    else:
        print(f"Downloading PLACES data dictionary...")
        # The Socrata API returns JSON directly; use limit to get all records
        resp = requests.get(url, params={"$limit": 1000}, timeout=60)
        resp.raise_for_status()
        data = resp.json()

        with open(dest, "w") as f:
            json.dump(data, f, indent=2)

        print(f"  Saved {len(data)} measure records to {dest}")

    # Verify
    with open(dest) as f:
        data = json.load(f)
    print(f"\nData dictionary: {len(data)} records")
    if len(data) > 0:
        print(f"  Sample keys: {list(data[0].keys())}")
        # Show HRSN measures
        hrsn = [d for d in data if d.get("categoryid") == "SOCLNEED" or d.get("CategoryID") == "SOCLNEED"]
        print(f"  HRSN (SOCLNEED) measures found: {len(hrsn)}")

    return dest


if __name__ == "__main__":
    print("=" * 70)
    print("STEP 1.1: Download CDC PLACES 2024 Census Tract Data")
    print("=" * 70)
    download_places_data()

    print("\n" + "=" * 70)
    print("STEP 1.2: Download PLACES Data Dictionary")
    print("=" * 70)
    download_data_dictionary()

    print("\nPhase 1 Steps 1.1 & 1.2 complete.")
