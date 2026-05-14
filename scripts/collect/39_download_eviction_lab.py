"""
Phase E, Step 39: Download Eviction Lab County Data.

Downloads county-level eviction filing data from the Princeton Eviction Lab
V1 dataset (2000-2018), hosted on AWS S3. This uses court-issued eviction
records — administrative data completely independent of BRFSS.

Computes eviction filing rate (filings per 100 renter households) as a
non-BRFSS proxy for housing insecurity in the V5 validation analysis.

Source: Eviction Lab, Princeton University.
        https://evictionlab.org/
        S3: eviction-lab-data-downloads/data-for-analysis/

Output: data/raw/eviction_lab/eviction_county.csv
"""
import sys
from pathlib import Path

import pandas as pd
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from hrsn_analysis.paths import PATHS, ensure_dirs
from hrsn_analysis.io_utils import save_csv
from hrsn_analysis.logging_utils import setup_logging, get_logger

setup_logging()
logger = get_logger(__name__)

# Eviction Lab V1 county-level court-issued data (publicly available on S3)
EVICTION_URL = (
    "https://eviction-lab-data-downloads.s3.amazonaws.com/"
    "data-for-analysis/county_court-issued_2000_2018.csv"
)

MANUAL_INSTRUCTIONS = """
=== MANUAL DOWNLOAD INSTRUCTIONS ===
If the automatic download fails:
1. Go to: https://data-downloads.evictionlab.org/
2. Navigate to data-for-analysis/
3. Download county_court-issued_2000_2018.csv
4. Save to: data/raw/eviction_lab/county_court-issued_2000_2018.csv
========================================
"""

# Use the most recent years (2016-2018) for robust averaging
TARGET_YEARS = [2016, 2017, 2018]


def download_eviction_data():
    """Download and process Eviction Lab county data."""
    ensure_dirs()

    out_dir = PATHS["raw"] / "eviction_lab"
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_path = out_dir / "county_court-issued_2000_2018.csv"
    out_path = out_dir / "eviction_county.csv"

    # Try automatic download
    logger.info("Downloading Eviction Lab county data from S3...")
    try:
        df = pd.read_csv(EVICTION_URL)
        logger.info(f"Downloaded: {len(df):,} rows × {len(df.columns)} cols")
        df.to_csv(raw_path, index=False)
    except Exception as e:
        logger.error(f"Download failed: {e}")
        logger.info(MANUAL_INSTRUCTIONS)

        if raw_path.exists():
            logger.info(f"Found manually downloaded file at {raw_path}")
            df = pd.read_csv(raw_path)
        else:
            raise FileNotFoundError(
                f"Could not download and no manual file at {raw_path}"
            )

    logger.info(f"Columns: {list(df.columns)}")
    logger.info(f"Years available: {sorted(df['year'].unique())}")

    # Build 5-digit county FIPS
    df["county_fips"] = df["fips_county"].astype(str).str.zfill(5)

    # Filter to target years
    df_recent = df[df["year"].isin(TARGET_YEARS)].copy()
    logger.info(f"Rows in {TARGET_YEARS}: {len(df_recent):,}")

    # Compute eviction filing rate: filings / renting_hh * 100
    df_recent["filing_rate"] = np.where(
        (df_recent["renting_hh"].notna()) & (df_recent["renting_hh"] > 0),
        df_recent["filings_observed"] / df_recent["renting_hh"] * 100,
        np.nan,
    )

    # Also compute eviction threat rate
    df_recent["threat_rate"] = np.where(
        (df_recent["renting_hh"].notna()) & (df_recent["renting_hh"] > 0),
        df_recent["hh_threat_observed"].fillna(0) / df_recent["renting_hh"] * 100,
        np.nan,
    )

    # Average across target years per county
    county_avg = (
        df_recent.groupby("county_fips")
        .agg(
            eviction_filing_rate=("filing_rate", "mean"),
            eviction_threat_rate=("threat_rate", "mean"),
            renting_hh=("renting_hh", "mean"),
            filings_total=("filings_observed", "sum"),
            n_years=("year", "count"),
        )
        .reset_index()
    )

    # Keep counties with at least 2 years of data and nonzero renters
    county_avg = county_avg[
        (county_avg["n_years"] >= 2) & (county_avg["renting_hh"] > 0)
    ].copy()

    logger.info(f"Processed: {len(county_avg):,} counties with eviction data")
    logger.info(f"Eviction filing rate — mean: {county_avg['eviction_filing_rate'].mean():.2f}%, "
                f"median: {county_avg['eviction_filing_rate'].median():.2f}%")
    logger.info(f"Counties with valid filing rate: "
                f"{county_avg['eviction_filing_rate'].notna().sum():,}")

    save_csv(county_avg, out_path)
    logger.info(f"Saved to {out_path}")

    return county_avg


if __name__ == "__main__":
    print("=" * 70)
    print("PHASE E: Download Eviction Lab County Data")
    print("=" * 70)
    df = download_eviction_data()
    print(f"\nDone. {len(df):,} counties with eviction measures.")
