"""
Step 33: Download USDA Food Access Research Atlas 2019 (Tract-Level).

Downloads the USDA Economic Research Service Food Access Research Atlas,
which provides tract-level indicators of food desert status, SNAP participation,
and low-access population shares.

These data can be used to validate HRSN food insecurity measures against
an independent, objective food-environment indicator.

Source: https://www.ers.usda.gov/data-products/food-access-research-atlas/
Data: 2019 release (based on 2019 ACS and 2020 Census geographies)

NOTE (2026-02-16): The direct USDA download URL below returns 404 as of Feb 2026.
  Updated URL format appears to be:
  https://www.ers.usda.gov/media/5627/food-access-research-atlas-data-download-2019.zip
  Check the source URL above → "Download the Data" for current links.

Output:
  - data/raw/food_access/food_access_2019_tracts.csv
"""
import sys
from pathlib import Path

import pandas as pd
import requests

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from hrsn_analysis.paths import PATHS, ensure_dirs
from hrsn_analysis.logging_utils import setup_logging, get_logger

setup_logging()
logger = get_logger(__name__)

FOOD_ACCESS_URLS = [
    # Updated URL format (discovered Feb 2026)
    "https://www.ers.usda.gov/media/5626/"
    "food-access-research-atlas-data-download-2019.xlsx",
    # Original URL (deprecated)
    "https://www.ers.usda.gov/webdocs/DataFiles/80591/"
    "FoodAccessResearchAtlasData2019.xlsx",
]

# Key columns to retain
FOOD_ACCESS_KEEP_COLS = [
    "CensusTract",       # 11-digit FIPS code
    "State",
    "County",
    "Urban",             # 1 = urban, 0 = rural
    "PovertyRate",       # Tract poverty rate
    "MedianFamilyIncome",
    # Low Income Low Access (LILA) flags
    "LILATracts_1And10", # LILA using 1-mile (urban) / 10-mile (rural)
    "LILATracts_halfAnd10", # LILA using 0.5-mile (urban) / 10-mile (rural)
    "LILATracts_1And20", # LILA using 1-mile (urban) / 20-mile (rural)
    # Low-access population shares
    "lapop1share",       # Share of pop >1 mile from supermarket
    "lapop10share",      # Share of pop >10 miles from supermarket
    "lapop05share",      # Share of pop >0.5 miles from supermarket
    "lapop20share",      # Share of pop >20 miles from supermarket
    # Low-access low-income population
    "lalowi1share",      # Low-income share >1 mile
    "lalowi10share",     # Low-income share >10 miles
    # SNAP
    "TractSNAP",         # SNAP households in tract
    "TractHUNV",         # Housing units without vehicle
    # Population
    "Pop2010",           # 2010 Census population
    "OHU2010",           # Occupied housing units
]


def download_food_access():
    """Download and process USDA Food Access Research Atlas 2019."""
    ensure_dirs()

    output_dir = PATHS["raw"] / "food_access"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "food_access_2019_tracts.csv"

    # Check if already downloaded
    if output_path.exists():
        logger.info(f"Food Access data already exists at {output_path}")
        df = pd.read_csv(output_path, dtype={"tract_fips": str})
        logger.info(f"  {len(df):,} tracts, {len(df.columns)} columns")
        return df

    # Download Excel file — try multiple URLs
    logger.info("Downloading USDA Food Access Research Atlas 2019...")
    logger.info("  (This is ~25-80MB Excel file, may take a minute)")

    response = None
    for url in FOOD_ACCESS_URLS:
        logger.info(f"  Trying: {url}")
        try:
            response = requests.get(url, timeout=300)
            response.raise_for_status()
            logger.info("  Download succeeded.")
            break
        except requests.RequestException as e:
            logger.warning(f"  Failed: {e}")
            response = None

    if response is None:
        logger.error("All download URLs failed.")
        logger.info("You can manually download from: "
                     "https://www.ers.usda.gov/data-products/food-access-research-atlas/")
        return None

    # Read Excel — the main data is in the "Food Access Research Atlas" sheet
    logger.info("  Download complete. Reading Excel file...")
    try:
        df = pd.read_excel(
            pd.io.common.BytesIO(response.content),
            sheet_name="Food Access Research Atlas",
            engine="openpyxl",
        )
    except Exception:
        # Try first sheet if named sheet not found
        logger.info("  Named sheet not found, trying first sheet...")
        df = pd.read_excel(
            pd.io.common.BytesIO(response.content),
            sheet_name=0,
            engine="openpyxl",
        )

    logger.info(f"  Raw data: {len(df):,} rows, {len(df.columns)} columns")

    # Subset to key columns that exist
    available_cols = [c for c in FOOD_ACCESS_KEEP_COLS if c in df.columns]
    missing_cols = [c for c in FOOD_ACCESS_KEEP_COLS if c not in df.columns]
    if missing_cols:
        logger.warning(f"  Missing expected columns: {missing_cols}")

    df = df[available_cols].copy()

    # Standardize tract FIPS
    if "CensusTract" in df.columns:
        df = df.rename(columns={"CensusTract": "tract_fips"})
        df["tract_fips"] = df["tract_fips"].astype(str).str.zfill(11)

    # Convert numeric columns
    numeric_cols = [c for c in df.columns
                    if c not in ["tract_fips", "State", "County"]]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Save
    df.to_csv(output_path, index=False)
    logger.info(f"\nSaved Food Access data: {len(df):,} tracts, {len(df.columns)} columns")
    logger.info(f"  Output: {output_path}")

    # Summary statistics
    if "LILATracts_1And10" in df.columns:
        n_lila = df["LILATracts_1And10"].sum()
        logger.info(f"\n  Food deserts (LILA 1&10): {n_lila:,.0f} / {len(df):,} tracts "
                     f"({n_lila / len(df) * 100:.1f}%)")

    if "Urban" in df.columns:
        n_urban = df["Urban"].sum()
        logger.info(f"  Urban tracts: {n_urban:,.0f} ({n_urban / len(df) * 100:.1f}%)")

    logger.info(f"\nMissing data rates:")
    for col in df.columns:
        if col not in ["tract_fips", "State", "County"]:
            pct_missing = df[col].isna().mean() * 100
            if pct_missing > 0:
                logger.info(f"  {col:25s}: {pct_missing:.1f}% missing")

    return df


if __name__ == "__main__":
    print("=" * 70)
    print("Step 33: Download USDA Food Access Research Atlas 2019")
    print("=" * 70)
    result = download_food_access()
    if result is not None:
        print(f"\nDone. {len(result):,} tracts downloaded.")
    else:
        print("\nDownload failed. See log for details.")
