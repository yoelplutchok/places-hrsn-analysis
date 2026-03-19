"""
Step 2.3: Merge all datasets into master analysis file.

Merges:
  - PLACES wide (HRSN + outcomes, filtered to HRSN states)
  - Census covariates (demographics)

On: tract_fips (11-digit GEOID)

Applies exclusions:
  1. Drop tracts with total_population < 500
  2. Drop tracts with missing values on primary HRSN or outcome measures
  3. Drop tracts with missing covariates

Output: data/processed/merged_tracts.parquet
"""
import sys
from pathlib import Path

import pandas as pd
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from hrsn_analysis.paths import PATHS, ensure_dirs
from hrsn_analysis.io_utils import load_parquet, save_parquet, load_params
from hrsn_analysis.logging_utils import setup_logging, get_logger

setup_logging()
logger = get_logger(__name__)


def merge_datasets():
    """Merge PLACES and Census data, apply exclusions, create master dataset."""
    ensure_dirs()
    params = load_params()

    # ---- Load processed data ----
    places = load_parquet(PATHS["processed"] / "places_wide.parquet")
    census = load_parquet(PATHS["processed"] / "census_covariates.parquet")

    # ---- Merge on tract_fips ----
    logger.info(f"Merging PLACES ({len(places):,} tracts) with Census ({len(census):,} tracts)")
    df = places.merge(census, on="tract_fips", how="inner")
    logger.info(f"After inner join: {len(df):,} tracts")

    unmatched_places = len(places) - len(df)
    unmatched_census = len(census) - len(df)
    logger.info(f"  PLACES tracts not in Census: {unmatched_places:,}")
    logger.info(f"  Census tracts not in PLACES: {unmatched_census:,}")

    # ---- Define key column groups ----
    hrsn_cols = [m["id"].lower() for m in params["hrsn_measures"]]
    outcome_cols = [m["id"].lower() for m in params["outcome_measures"]]
    # Only check analysis covariates for NaN exclusion — median_income is
    # retained in the dataset but not used in regressions (collinear with
    # pct_poverty), so missing median_income should not exclude tracts.
    covariate_cols = ["pct_black", "pct_hispanic", "pct_poverty",
                      "pct_college", "pct_65plus", "median_age"]

    # ---- Exclusion cascade (document counts at each step) ----
    exclusions = []
    n_start = len(df)

    # Step 1: Drop tracts with low population
    min_pop = params["analysis"]["min_tract_population"]
    # Use total_population from Census (more reliable than PLACES total_pop)
    mask_pop = df["total_population"] >= min_pop
    n_low_pop = (~mask_pop).sum()
    df = df[mask_pop].copy()
    exclusions.append(("Population < 500", n_low_pop))
    logger.info(f"Excluded {n_low_pop:,} tracts with population < {min_pop}")

    # Step 2: Drop tracts missing any primary HRSN measure
    mask_hrsn = df[hrsn_cols].notna().all(axis=1)
    n_miss_hrsn = (~mask_hrsn).sum()
    df = df[mask_hrsn].copy()
    exclusions.append(("Missing HRSN measure(s)", n_miss_hrsn))
    logger.info(f"Excluded {n_miss_hrsn:,} tracts with missing HRSN measures")

    # Step 3: Drop tracts missing any primary outcome
    mask_outcome = df[outcome_cols].notna().all(axis=1)
    n_miss_outcome = (~mask_outcome).sum()
    df = df[mask_outcome].copy()
    exclusions.append(("Missing outcome measure(s)", n_miss_outcome))
    logger.info(f"Excluded {n_miss_outcome:,} tracts with missing outcome measures")

    # Step 4: Drop tracts missing key covariates
    mask_cov = df[covariate_cols].notna().all(axis=1)
    n_miss_cov = (~mask_cov).sum()
    df = df[mask_cov].copy()
    exclusions.append(("Missing covariate(s)", n_miss_cov))
    logger.info(f"Excluded {n_miss_cov:,} tracts with missing covariates")

    n_final = len(df)
    n_excluded = n_start - n_final
    logger.info(f"\n--- Exclusion Summary ---")
    logger.info(f"Started with: {n_start:,} tracts")
    for reason, count in exclusions:
        logger.info(f"  {reason}: -{count:,}")
    logger.info(f"Final: {n_final:,} tracts ({n_excluded:,} excluded, {n_excluded/n_start*100:.1f}%)")

    # ---- Add derived columns ----
    df["county_fips"] = df["tract_fips"].str[:5]
    df["state_fips"] = df["tract_fips"].str[:2]

    # ---- Final quality checks ----
    logger.info(f"\n--- Final Dataset Quality Checks ---")
    logger.info(f"Shape: {df.shape}")
    logger.info(f"States: {df['state_abbr'].nunique()} ({', '.join(sorted(df['state_abbr'].unique()))})")
    logger.info(f"Counties: {df['county_fips'].nunique():,}")

    # Check no remaining missing in key columns
    key_cols = hrsn_cols + outcome_cols + covariate_cols
    for col in key_cols:
        n_miss = df[col].isna().sum()
        if n_miss > 0:
            logger.warning(f"  {col}: still has {n_miss:,} missing values!")

    # Summary of all variables
    logger.info(f"\n--- Variable Summaries ---")
    logger.info("HRSN Measures:")
    for col in hrsn_cols:
        v = df[col]
        logger.info(f"  {col}: mean={v.mean():.1f}, sd={v.std():.1f}, min={v.min():.1f}, max={v.max():.1f}")

    logger.info("Outcomes:")
    for col in outcome_cols:
        v = df[col]
        logger.info(f"  {col}: mean={v.mean():.1f}, sd={v.std():.1f}, min={v.min():.1f}, max={v.max():.1f}")

    logger.info("Covariates:")
    for col in covariate_cols:
        v = df[col]
        logger.info(f"  {col}: mean={v.mean():.1f}, sd={v.std():.1f}, min={v.min():.1f}, max={v.max():.1f}")

    # ---- Save ----
    dest = PATHS["processed"] / "merged_tracts.parquet"
    save_parquet(df, dest)
    logger.info(f"Saved to {dest} ({dest.stat().st_size / 1e6:.1f} MB)")

    # Also save exclusion summary
    excl_df = pd.DataFrame(exclusions, columns=["Reason", "Count"])
    excl_df.loc[len(excl_df)] = ["FINAL SAMPLE", n_final]
    excl_path = PATHS["tables"] / "exclusion_cascade.csv"
    excl_df.to_csv(excl_path, index=False)
    logger.info(f"Exclusion cascade saved to {excl_path}")

    return df


if __name__ == "__main__":
    print("=" * 70)
    print("STEP 2.3: Merge All Datasets")
    print("=" * 70)
    df = merge_datasets()
    print(f"\nMASTER DATASET: {len(df):,} tracts × {len(df.columns)} columns")
    print(f"Columns: {list(df.columns)}")
    print("\nStep 2.3 complete. Master analysis dataset ready.")
