"""
Step 2.2: Process Census ACS covariates.

Merges the main ACS download with supplementary age/education data,
then computes derived percentage variables:
  - pct_black, pct_hispanic, pct_white
  - pct_poverty, median_income
  - pct_no_hs, pct_college
  - pct_65plus, median_age
  - total_population

Output: data/processed/census_covariates.parquet
"""
import sys
from pathlib import Path

import pandas as pd
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from hrsn_analysis.paths import PATHS, ensure_dirs
from hrsn_analysis.io_utils import save_parquet
from hrsn_analysis.logging_utils import setup_logging, get_logger

setup_logging()
logger = get_logger(__name__)


def process_census():
    """Load raw Census ACS data and compute covariate percentages."""
    ensure_dirs()

    # ---- Load main demographic data ----
    raw_path = PATHS["census_raw"] / "acs_tract_demographics_2022.csv"
    logger.info(f"Loading Census ACS data from {raw_path}")
    df = pd.read_csv(raw_path, dtype={"STATE": str, "COUNTY": str, "TRACT": str})
    logger.info(f"Main Census data: {len(df):,} tracts × {len(df.columns)} columns")

    # ---- Load supplementary age/education data ----
    supp_path = PATHS["census_raw"] / "acs_supplementary_2022.csv"
    logger.info(f"Loading supplementary ACS data from {supp_path}")
    supp = pd.read_csv(supp_path, dtype={"STATE": str, "COUNTY": str, "TRACT": str})
    logger.info(f"Supplementary data: {len(supp):,} tracts × {len(supp.columns)} columns")

    # ---- Build tract FIPS for both ----
    for d in [df, supp]:
        d["tract_fips"] = (
            d["STATE"].str.zfill(2)
            + d["COUNTY"].str.zfill(3)
            + d["TRACT"].str.zfill(6)
        )

    # ---- Merge on tract_fips ----
    supp_cols = [c for c in supp.columns if c.startswith("B") and c not in df.columns]
    merged = df.merge(supp[["tract_fips"] + supp_cols], on="tract_fips", how="left")
    logger.info(f"Merged: {len(merged):,} tracts × {len(merged.columns)} columns")

    # ---- Extract raw variables ----
    total_pop = merged["B01003_001E"].astype(float)
    median_age = merged["B01002_001E"].astype(float)
    white_alone = merged["B02001_002E"].astype(float)
    black_alone = merged["B02001_003E"].astype(float)
    hispanic = merged["B03003_003E"].astype(float)
    below_poverty = merged["B17001_002E"].astype(float)
    median_income = merged["B19013_001E"].astype(float)
    pop_25plus = merged["B15003_001E"].astype(float)
    bachelors = merged["B15003_022E"].astype(float)
    masters = merged["B15003_023E"].astype(float)
    professional = merged["B15003_024E"].astype(float)
    doctorate = merged["B15003_025E"].astype(float)

    # Age 65+ from supplementary data
    # Males 65+: B01001_020E through B01001_025E
    # Females 65+: B01001_044E through B01001_049E
    male_65plus_cols = [f"B01001_{i:03d}E" for i in range(20, 26)]
    female_65plus_cols = [f"B01001_{i:03d}E" for i in range(44, 50)]
    pop_65plus = merged[male_65plus_cols + female_65plus_cols].astype(float).sum(axis=1)

    # Education below HS: B15003_002E through B15003_016E
    below_hs_cols = [f"B15003_{i:03d}E" for i in range(2, 17)]
    # Check which columns exist
    available_below_hs = [c for c in below_hs_cols if c in merged.columns]
    if len(available_below_hs) == len(below_hs_cols):
        pop_below_hs = merged[available_below_hs].astype(float).sum(axis=1)
        has_no_hs = True
    else:
        logger.warning(f"Only {len(available_below_hs)}/{len(below_hs_cols)} below-HS columns available")
        has_no_hs = False

    # ---- Compute derived variables ----
    out = pd.DataFrame()
    out["tract_fips"] = merged["tract_fips"]
    out["total_population"] = total_pop

    # Race/ethnicity percentages
    out["pct_white"] = np.where(total_pop > 0, (white_alone / total_pop) * 100, np.nan)
    out["pct_black"] = np.where(total_pop > 0, (black_alone / total_pop) * 100, np.nan)
    out["pct_hispanic"] = np.where(total_pop > 0, (hispanic / total_pop) * 100, np.nan)

    # Socioeconomic — use poverty universe (B17001_001E) as denominator when available
    if "B17001_001E" in merged.columns:
        poverty_universe = merged["B17001_001E"].astype(float)
        out["pct_poverty"] = np.where(poverty_universe > 0, (below_poverty / poverty_universe) * 100, np.nan)
        logger.info("Using B17001_001E (poverty universe) as denominator for pct_poverty")
    else:
        out["pct_poverty"] = np.where(total_pop > 0, (below_poverty / total_pop) * 100, np.nan)
        logger.warning("B17001_001E not available; using total_pop as poverty denominator (less accurate)")
    out["median_income"] = median_income

    # Education
    out["pct_college"] = np.where(
        pop_25plus > 0,
        ((bachelors + masters + professional + doctorate) / pop_25plus) * 100,
        np.nan
    )
    if has_no_hs:
        out["pct_no_hs"] = np.where(
            pop_25plus > 0,
            (pop_below_hs / pop_25plus) * 100,
            np.nan
        )
        logger.info("Computed pct_no_hs from B15003 subcategories 002-016")
    else:
        out["pct_no_hs"] = np.nan
        logger.warning("Could not compute pct_no_hs")

    # Age
    out["pct_65plus"] = np.where(total_pop > 0, (pop_65plus / total_pop) * 100, np.nan)
    out["median_age"] = median_age
    logger.info("Computed pct_65plus from B01001 age subcategories")

    # ---- Direct social needs proxies (for validation framework) ----
    # SNAP receipt (proxy for food stamps/food insecurity)
    if "B22001_001E" in merged.columns and "B22001_002E" in merged.columns:
        snap_universe = merged["B22001_001E"].astype(float)
        snap_received = merged["B22001_002E"].astype(float)
        out["pct_snap"] = np.where(snap_universe > 0, (snap_received / snap_universe) * 100, np.nan)
        logger.info("Computed pct_snap (SNAP receipt) from B22001")
    else:
        out["pct_snap"] = np.nan
        logger.warning("B22001 variables not available; pct_snap not computed")

    # Housing cost burden >=30% among renters (proxy for housing insecurity)
    if "B25070_001E" in merged.columns:
        rent_total = merged["B25070_001E"].astype(float)
        rent_cols = ["B25070_007E", "B25070_008E", "B25070_009E", "B25070_010E"]
        available_rent = [c for c in rent_cols if c in merged.columns]
        if len(available_rent) == 4:
            rent_burdened = merged[available_rent].astype(float).sum(axis=1)
            out["pct_rent_burden"] = np.where(rent_total > 0, (rent_burdened / rent_total) * 100, np.nan)
            logger.info("Computed pct_rent_burden (>=30% income on rent) from B25070")
        else:
            out["pct_rent_burden"] = np.nan
            logger.warning(f"Only {len(available_rent)}/4 B25070 burden columns available")
    else:
        out["pct_rent_burden"] = np.nan
        logger.warning("B25070 variables not available; pct_rent_burden not computed")

    # No vehicle access (proxy for transportation barriers)
    if "B08201_001E" in merged.columns and "B08201_002E" in merged.columns:
        vehicle_universe = merged["B08201_001E"].astype(float)
        no_vehicle = merged["B08201_002E"].astype(float)
        out["pct_no_vehicle"] = np.where(vehicle_universe > 0, (no_vehicle / vehicle_universe) * 100, np.nan)
        logger.info("Computed pct_no_vehicle from B08201")
    else:
        out["pct_no_vehicle"] = np.nan
        logger.warning("B08201 variables not available; pct_no_vehicle not computed")

    # Living alone (proxy for social isolation/loneliness)
    if "B11001_001E" in merged.columns and "B11001_008E" in merged.columns:
        hh_total = merged["B11001_001E"].astype(float)
        living_alone = merged["B11001_008E"].astype(float)
        out["pct_living_alone"] = np.where(hh_total > 0, (living_alone / hh_total) * 100, np.nan)
        logger.info("Computed pct_living_alone from B11001")
    else:
        out["pct_living_alone"] = np.nan
        logger.warning("B11001 variables not available; pct_living_alone not computed")

    # ---- Validate ranges ----
    logger.info("\n--- Census Covariate Summary ---")
    for col in ["total_population", "pct_white", "pct_black", "pct_hispanic",
                "pct_poverty", "median_income", "pct_college", "pct_no_hs",
                "pct_65plus", "median_age",
                "pct_snap", "pct_rent_burden", "pct_no_vehicle", "pct_living_alone"]:
        vals = out[col].dropna()
        if len(vals) > 0:
            logger.info(f"  {col}: n={len(vals):,}, mean={vals.mean():.1f}, median={vals.median():.1f}, "
                         f"min={vals.min():.1f}, max={vals.max():.1f}, missing={out[col].isna().sum():,}")

    # Flag any out-of-range percentages
    pct_cols = ["pct_white", "pct_black", "pct_hispanic", "pct_poverty",
                "pct_college", "pct_no_hs", "pct_65plus"]
    for col in pct_cols:
        vals = out[col].dropna()
        if len(vals) > 0:
            over_100 = (vals > 100).sum()
            under_0 = (vals < 0).sum()
            if over_100 > 0 or under_0 > 0:
                logger.warning(f"  {col}: {over_100} values > 100, {under_0} values < 0")

    # ---- Save ----
    dest = PATHS["processed"] / "census_covariates.parquet"
    save_parquet(out, dest)
    logger.info(f"Saved to {dest} ({dest.stat().st_size / 1e6:.1f} MB)")

    return out


if __name__ == "__main__":
    print("=" * 70)
    print("STEP 2.2: Process Census ACS Covariates")
    print("=" * 70)
    df = process_census()
    print(f"\nFinal dataset: {len(df):,} tracts × {len(df.columns)} columns")
    print(f"Columns: {list(df.columns)}")
    print("\nStep 2.2 complete.")
