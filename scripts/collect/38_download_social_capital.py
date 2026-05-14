"""
Phase E, Step 38: Download Social Capital Atlas Data.

Downloads the Chetty et al. (2022) Social Capital Atlas county-level data
from the Humanitarian Data Exchange (HDX). This dataset provides measures
of social capital derived from Facebook friendship patterns, including:
  - Economic connectedness (cross-SES friendships)
  - Social cohesiveness (clustering)
  - Civic engagement (volunteering rate, civic organizations)

These serve as non-BRFSS proxies for social isolation / emotional support
constructs in the V5 validation analysis.

Source: Chetty, R., Jackson, M.O., Kuchler, T. et al. (2022).
        Social capital I & II. Nature, 608.
        https://data.humdata.org/dataset/social-capital-atlas

Output: data/raw/social_capital/social_capital_county.csv
"""
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from hrsn_analysis.paths import PATHS, ensure_dirs
from hrsn_analysis.io_utils import save_csv
from hrsn_analysis.logging_utils import setup_logging, get_logger

setup_logging()
logger = get_logger(__name__)

# Social Capital Atlas county CSV (Chetty et al. 2022, via HDX)
SCA_URL = (
    "https://data.humdata.org/dataset/85ee8e10-0c66-4635-b997-79b6fad44c71/"
    "resource/ec896b64-c922-4737-b759-e4bd7f73b8cc/download/"
    "social_capital_county.csv"
)

MANUAL_INSTRUCTIONS = """
=== MANUAL DOWNLOAD INSTRUCTIONS ===
If the automatic download fails:
1. Go to: https://data.humdata.org/dataset/social-capital-atlas
2. Click "Download" for "Social Capital Atlas - US Counties.csv"
3. Save the file to: data/raw/social_capital/social_capital_county.csv
========================================
"""

# Columns to keep and rename
KEEP_COLS = {
    "county": "county_fips",
    "ec_county": "economic_connectedness",       # Cross-SES friendships (higher = more connected)
    "clustering_county": "clustering",            # Social cohesiveness
    "support_ratio_county": "support_ratio",      # Ratio of supportive friends
    "volunteering_rate_county": "volunteering_rate",  # Civic engagement
    "civic_organizations_county": "civic_organizations",  # Density of civic orgs
    "pop2018": "pop2018",
}


def download_social_capital():
    """Download and process Social Capital Atlas county data."""
    ensure_dirs()

    out_dir = PATHS["raw"] / "social_capital"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "social_capital_county.csv"

    # Try automatic download
    logger.info("Downloading Social Capital Atlas county data from HDX...")
    try:
        df = pd.read_csv(SCA_URL)
        logger.info(f"Downloaded: {len(df):,} rows × {len(df.columns)} cols")
    except Exception as e:
        logger.error(f"Download failed: {e}")
        logger.info(MANUAL_INSTRUCTIONS)

        # Check if file was manually placed
        if out_path.exists():
            logger.info(f"Found manually downloaded file at {out_path}")
            df = pd.read_csv(out_path)
        else:
            raise FileNotFoundError(
                f"Could not download and no manual file at {out_path}"
            )

    # Select and rename columns
    available = {k: v for k, v in KEEP_COLS.items() if k in df.columns}
    df = df[list(available.keys())].rename(columns=available)

    # Clean county FIPS — ensure 5-digit zero-padded
    df["county_fips"] = df["county_fips"].astype(str).str.zfill(5)

    # Drop rows with missing FIPS or all-NaN social capital
    sc_cols = [c for c in df.columns if c not in ("county_fips", "pop2018")]
    df = df.dropna(subset=["county_fips"])
    df = df[df[sc_cols].notna().any(axis=1)]

    # Compute composite social capital index (mean of z-scored components)
    for col in sc_cols:
        s = df[col].dropna()
        if len(s) > 0 and s.std() > 0:
            df[f"{col}_z"] = (df[col] - s.mean()) / s.std()

    z_cols = [c for c in df.columns if c.endswith("_z")]
    if z_cols:
        df["social_capital_index"] = df[z_cols].mean(axis=1)
        # Drop intermediate z-score columns
        df = df.drop(columns=z_cols)

    logger.info(f"Processed: {len(df):,} counties with social capital data")
    logger.info(f"Columns: {list(df.columns)}")
    logger.info(f"Social Capital Index — mean: {df['social_capital_index'].mean():.3f}, "
                f"std: {df['social_capital_index'].std():.3f}")

    save_csv(df, out_path)
    logger.info(f"Saved to {out_path}")

    return df


if __name__ == "__main__":
    print("=" * 70)
    print("PHASE E: Download Social Capital Atlas Data")
    print("=" * 70)
    df = download_social_capital()
    print(f"\nDone. {len(df):,} counties with social capital measures.")
