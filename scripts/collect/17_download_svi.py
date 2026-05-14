"""
Download CDC/ATSDR Social Vulnerability Index (SVI) 2022 at census tract level.

The SVI ranks census tracts on 16 social factors grouped into 4 themes:
  Theme 1: Socioeconomic Status
  Theme 2: Household Characteristics & Disability
  Theme 3: Racial & Ethnic Minority Status
  Theme 4: Housing Type & Transportation

Output:
  - data/raw/svi/SVI_2022_US.csv
"""
import sys
import zipfile
import io
from pathlib import Path

import requests
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from hrsn_analysis.paths import PATHS, ensure_dirs
from hrsn_analysis.logging_utils import setup_logging, get_logger

setup_logging()
logger = get_logger(__name__)

# Known URL patterns for CDC SVI 2022
SVI_URLS = [
    "https://www.atsdr.cdc.gov/placeandhealth/svi/documentation/csv/SVI_2022_US.csv",
    "https://svi.cdc.gov/Documents/Data/2022/csv/SVI_2022_US.csv",
    "https://svi.cdc.gov/data/download/2022/SVI2022_US.csv",
]

# Key columns to retain
SVI_KEEP_COLS = [
    "FIPS",            # Tract FIPS
    "STATE",           # State name
    "ST_ABBR",         # State abbreviation
    "COUNTY",          # County name
    # Theme percentile rankings (0-1)
    "RPL_THEME1",      # Socioeconomic Status
    "RPL_THEME2",      # Household Characteristics & Disability
    "RPL_THEME3",      # Racial & Ethnic Minority Status
    "RPL_THEME4",      # Housing Type & Transportation
    "RPL_THEMES",      # Overall SVI ranking
    # Individual variable percentile rankings
    "EPL_POV150",      # Below 150% poverty
    "EPL_UNEMP",       # Unemployment
    "EPL_HBURD",       # Housing cost burden
    "EPL_NOHSDP",      # No high school diploma
    "EPL_UNINSUR",     # Uninsured
    "EPL_AGE65",       # Age 65+
    "EPL_AGE17",       # Age 17 or younger
    "EPL_DISABL",      # Disability
    "EPL_SNGPNT",      # Single parent
    "EPL_LIMENG",      # Limited English
    "EPL_MINRTY",      # Minority status
    "EPL_MUNIT",       # Multi-unit housing
    "EPL_MOBILE",      # Mobile homes
    "EPL_CROWD",       # Crowding
    "EPL_NOVEH",       # No vehicle
    "EPL_GROUPQ",      # Group quarters
]


def download_svi():
    """Download SVI 2022 data."""
    ensure_dirs()
    svi_dir = PATHS["raw"] / "svi"
    svi_dir.mkdir(parents=True, exist_ok=True)
    dest = svi_dir / "SVI_2022_US.csv"

    if dest.exists():
        logger.info(f"SVI data already exists at {dest}")
        df = pd.read_csv(dest, dtype={"FIPS": str})
        logger.info(f"SVI: {len(df):,} tracts")
        return df

    # Try each URL
    for url in SVI_URLS:
        logger.info(f"Trying: {url}")
        try:
            resp = requests.get(url, timeout=120, stream=True)
            if resp.status_code == 200:
                content_type = resp.headers.get("content-type", "")
                if "zip" in content_type or url.endswith(".zip"):
                    # Handle zip response
                    z = zipfile.ZipFile(io.BytesIO(resp.content))
                    csv_files = [f for f in z.namelist() if f.endswith(".csv")]
                    if csv_files:
                        with z.open(csv_files[0]) as f:
                            df = pd.read_csv(f, dtype={"FIPS": str})
                else:
                    df = pd.read_csv(io.StringIO(resp.text), dtype={"FIPS": str})

                logger.info(f"Downloaded SVI: {len(df):,} rows × {len(df.columns)} cols")

                # Save full file
                df.to_csv(dest, index=False)
                logger.info(f"Saved to {dest}")
                return df
            else:
                logger.warning(f"  HTTP {resp.status_code}")
        except Exception as e:
            logger.warning(f"  Failed: {e}")

    # If all URLs fail, try the ATSDR API
    logger.error("All download URLs failed.")
    logger.error("Please download SVI 2022 manually from:")
    logger.error("  https://www.atsdr.cdc.gov/placeandhealth/svi/data_documentation_download.html")
    logger.error(f"  Save to: {dest}")
    return None


if __name__ == "__main__":
    print("=" * 70)
    print("Downloading CDC SVI 2022")
    print("=" * 70)
    df = download_svi()
    if df is not None:
        print(f"\nDone. {len(df):,} tracts.")
        # Show available columns
        print(f"Columns: {len(df.columns)}")
        svi_cols = [c for c in SVI_KEEP_COLS if c in df.columns]
        missing = [c for c in SVI_KEEP_COLS if c not in df.columns]
        print(f"Key SVI columns found: {len(svi_cols)}/{len(SVI_KEEP_COLS)}")
        if missing:
            print(f"Missing columns: {missing}")
    else:
        print("\nFailed — see error above for manual download instructions.")
