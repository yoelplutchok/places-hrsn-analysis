"""
Step 32: Download EPA EJScreen 2024 Tract-Level Environmental Justice Data.

Downloads the EPA EJScreen dataset, which provides environmental exposure
indicators (PM2.5, ozone, diesel PM, traffic proximity, Superfund/hazardous
waste proximity) and demographic indices at the census tract level.

These data can be used as environmental confounders to test whether HRSN-disease
associations are robust to controlling for environmental exposures.

Source: https://www.epa.gov/ejscreen/download-ejscreen-data
Data: 2024 release (September 2024 version 2.32)

NOTE (2026-02-16): The EPA gaftp.epa.gov FTP URL below returns 404 as of Feb 2026.
  The EPA may have reorganized their download infrastructure. Check the source URL
  above for the current download location, or try the Zenodo archive at
  https://zenodo.org/records/14767363 for a community-maintained copy.

Output:
  - data/raw/ejscreen/ejscreen_2024_tracts.csv
"""
import sys
from pathlib import Path
import zipfile
import io

import pandas as pd
import requests

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from hrsn_analysis.paths import PATHS, ensure_dirs
from hrsn_analysis.logging_utils import setup_logging, get_logger

setup_logging()
logger = get_logger(__name__)

EJSCREEN_URL = (
    "https://gaftp.epa.gov/EJScreen/2024/"
    "2.32_September_2024/"
    "EJScreen_2024_Tract_with_AS_CNMI_GU_VI.csv.zip"
)

# Key columns to retain (environmental indicators + demographics)
EJSCREEN_KEEP_COLS = [
    "ID",              # Census tract FIPS (11-digit)
    "STATE_NAME",
    "ST_ABBREV",
    # Environmental indicators
    "PM25",            # Particulate matter 2.5
    "OZONE",           # Ozone
    "DSLPM",           # Diesel particulate matter
    "PTRAF",           # Traffic proximity
    "PNPL",            # Superfund (NPL) proximity
    "PRMP",            # RMP facility proximity
    "PTSDF",           # TSDF (hazardous waste) proximity
    "PWDIS",           # Wastewater discharge
    "UST",             # Underground storage tanks
    # Demographic indices
    "DEMOGIDX_5",      # Demographic index (5-factor)
    "DEMOGIDX_2",      # Supplemental demographic index (2-factor)
    # Population
    "ACSTOTPOP",       # Total population (ACS)
    "PEOPCOLORPCT",    # People of color percentage
    "LOWINCPCT",       # Low income percentage
    "OVER64PCT",       # Over 64 percentage
    "UNDER5PCT",       # Under 5 percentage
]


def _process_ejscreen(df, output_path):
    """Process raw EJScreen dataframe: subset columns, standardize FIPS, save."""
    available_cols = [c for c in EJSCREEN_KEEP_COLS if c in df.columns]
    missing_cols = [c for c in EJSCREEN_KEEP_COLS if c not in df.columns]
    if missing_cols:
        logger.warning(f"  Missing expected columns: {missing_cols}")

    df = df[available_cols].copy()

    # Standardize tract FIPS
    if "ID" in df.columns:
        df = df.rename(columns={"ID": "tract_fips"})
        df["tract_fips"] = df["tract_fips"].astype(str).str.zfill(11)

    # Convert environmental columns to numeric
    env_cols = ["PM25", "OZONE", "DSLPM", "PTRAF", "PNPL", "PRMP",
                "PTSDF", "PWDIS", "UST", "DEMOGIDX_5", "DEMOGIDX_2",
                "ACSTOTPOP", "PEOPCOLORPCT", "LOWINCPCT", "OVER64PCT", "UNDER5PCT"]
    for col in env_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Save
    df.to_csv(output_path, index=False)
    logger.info(f"\nSaved EJScreen data: {len(df):,} tracts, {len(df.columns)} columns")
    logger.info(f"  Output: {output_path}")

    # Summary statistics
    logger.info(f"\nMissing data rates:")
    for col in df.columns:
        if col not in ["tract_fips", "STATE_NAME", "ST_ABBREV"]:
            pct_missing = df[col].isna().mean() * 100
            if pct_missing > 0:
                logger.info(f"  {col:20s}: {pct_missing:.1f}% missing")

    return df


def download_ejscreen():
    """Download and process EPA EJScreen 2024 tract-level data."""
    ensure_dirs()

    output_dir = PATHS["raw"] / "ejscreen"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "ejscreen_2024_tracts.csv"

    # Check if already processed
    if output_path.exists():
        logger.info(f"EJScreen data already exists at {output_path}")
        df = pd.read_csv(output_path, dtype={"tract_fips": str})
        logger.info(f"  {len(df):,} tracts, {len(df.columns)} columns")
        return df

    # Check if user manually placed a raw CSV in the directory
    manual_csvs = list(output_dir.glob("EJScreen*.csv"))
    manual_zips = list(output_dir.glob("EJScreen*.zip"))
    if manual_csvs:
        logger.info(f"Found manually downloaded CSV: {manual_csvs[0].name}")
        df = pd.read_csv(manual_csvs[0], low_memory=False)
    elif manual_zips:
        logger.info(f"Found manually downloaded ZIP: {manual_zips[0].name}")
        with zipfile.ZipFile(manual_zips[0]) as zf:
            csv_files = [f for f in zf.namelist() if f.endswith(".csv")]
            with zf.open(csv_files[0]) as f:
                df = pd.read_csv(f, low_memory=False)
    else:
        df = None

    if df is not None:
        # Process the manually downloaded file (skip to processing below)
        logger.info(f"  Raw data: {len(df):,} rows, {len(df.columns)} columns")
        return _process_ejscreen(df, output_path)

    # Download ZIP file (streaming to handle large file)
    logger.info(f"Downloading EJScreen 2024 from EPA...")
    logger.info(f"  URL: {EJSCREEN_URL}")
    logger.info(f"  (This is ~200MB, may take a few minutes)")

    try:
        response = requests.get(EJSCREEN_URL, stream=True, timeout=300)
        response.raise_for_status()
    except requests.RequestException as e:
        logger.error(f"Download failed: {e}")
        logger.info("")
        logger.info("=" * 60)
        logger.info("MANUAL DOWNLOAD INSTRUCTIONS:")
        logger.info("=" * 60)
        logger.info("The EPA reorganized their FTP server. To download manually:")
        logger.info("")
        logger.info("  1. Google: 'EPA EJScreen download data 2024'")
        logger.info("  2. Go to the EPA EJScreen download page")
        logger.info("  3. Look for 'EJScreen 2024' tract-level CSV download")
        logger.info("     (filename like: EJScreen_2024_Tract_*.csv.zip)")
        logger.info("  4. Download the ZIP file (~200MB)")
        logger.info("  5. Extract the CSV and place it at:")
        logger.info(f"     {output_dir / 'EJScreen_2024_Tract.csv'}")
        logger.info("  6. Re-run this script to process the file")
        logger.info("")
        logger.info("  Alternative: Check Zenodo for a community archive:")
        logger.info("  https://zenodo.org/records/14767363")
        logger.info("=" * 60)
        return None

    # Read ZIP into memory and extract CSV
    logger.info("  Download complete. Extracting CSV from ZIP...")
    content = io.BytesIO(response.content)
    with zipfile.ZipFile(content) as zf:
        csv_files = [f for f in zf.namelist() if f.endswith(".csv")]
        if not csv_files:
            logger.error("No CSV found in ZIP file")
            return None

        csv_name = csv_files[0]
        logger.info(f"  Extracting: {csv_name}")

        with zf.open(csv_name) as f:
            df = pd.read_csv(f, low_memory=False)

    logger.info(f"  Raw data: {len(df):,} rows, {len(df.columns)} columns")
    return _process_ejscreen(df, output_path)


if __name__ == "__main__":
    print("=" * 70)
    print("Step 32: Download EPA EJScreen 2024")
    print("=" * 70)
    result = download_ejscreen()
    if result is not None:
        print(f"\nDone. {len(result):,} tracts downloaded.")
    else:
        print("\nDownload failed. See log for details.")
