"""
Step 2.1: Reshape and clean PLACES data.

Since we used the GIS-friendly format, data is already wide.
This script:
  1. Loads the raw PLACES CSV
  2. Extracts relevant columns (HRSN, outcomes, secondary outcomes)
  3. Renames to short analysis-friendly names
  4. Filters to tracts with HRSN data (drops states without HRSN measures)
  5. Saves as parquet

Output: data/processed/places_wide.parquet
"""
import sys
from pathlib import Path

import pandas as pd
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from hrsn_analysis.paths import PATHS, ensure_dirs
from hrsn_analysis.io_utils import load_params, save_parquet
from hrsn_analysis.logging_utils import setup_logging, get_logger

setup_logging()
logger = get_logger(__name__)


def reshape_places():
    """Load, clean, and reshape PLACES data."""
    ensure_dirs()
    params = load_params()

    # ---- Load raw data ----
    raw_path = PATHS["places_raw"] / "places_tract_2024.csv"
    logger.info(f"Loading PLACES data from {raw_path}")
    df = pd.read_csv(raw_path, dtype={"TractFIPS": str, "CountyFIPS": str})
    logger.info(f"Raw PLACES: {len(df):,} tracts × {len(df.columns)} columns")

    # ---- Build column mapping ----
    # HRSN measures
    hrsn_map = {}
    for m in params["hrsn_measures"]:
        col = f"{m['id']}_CrudePrev"
        short = m["id"].lower()
        hrsn_map[col] = short

    # Primary outcomes
    outcome_map = {}
    for m in params["outcome_measures"]:
        col = f"{m['id']}_CrudePrev"
        short = m["id"].lower()
        outcome_map[col] = short

    # Secondary outcomes
    secondary_map = {}
    for m in params["secondary_outcomes"]:
        col = f"{m['id']}_CrudePrev"
        short = m["id"].lower()
        secondary_map[col] = short

    # Metadata columns
    meta_cols = {
        "TractFIPS": "tract_fips",
        "StateAbbr": "state_abbr",
        "CountyFIPS": "county_fips",
        "CountyName": "county_name",
        "TotalPopulation": "total_pop_places",
    }

    # ---- Check all expected columns exist ----
    all_measure_cols = {**hrsn_map, **outcome_map, **secondary_map}
    missing = [c for c in list(meta_cols.keys()) + list(all_measure_cols.keys()) if c not in df.columns]
    if missing:
        logger.warning(f"Missing columns: {missing}")
        # Remove missing from maps
        for m in missing:
            all_measure_cols.pop(m, None)
            meta_cols.pop(m, None)

    # ---- Select and rename ----
    keep_cols = list(meta_cols.keys()) + list(all_measure_cols.keys())
    df_out = df[keep_cols].copy()
    df_out.rename(columns={**meta_cols, **all_measure_cols}, inplace=True)

    logger.info(f"Selected {len(df_out.columns)} columns")

    # ---- Ensure tract_fips is string, zero-padded to 11 digits ----
    df_out["tract_fips"] = df_out["tract_fips"].astype(str).str.zfill(11)
    df_out["county_fips"] = df_out["tract_fips"].str[:5]
    df_out["state_fips"] = df_out["tract_fips"].str[:2]

    # ---- Filter to tracts with HRSN data ----
    hrsn_cols = list(hrsn_map.values())
    n_before = len(df_out)

    # Count non-null HRSN values per tract
    hrsn_available = df_out[hrsn_cols].notna().sum(axis=1)
    df_out = df_out[hrsn_available > 0].copy()

    n_after = len(df_out)
    n_dropped = n_before - n_after
    logger.info(f"Filtered to tracts with HRSN data: {n_before:,} → {n_after:,} ({n_dropped:,} dropped)")

    # ---- Document state coverage ----
    states_with_hrsn = df_out["state_abbr"].nunique()
    state_list = sorted(df_out["state_abbr"].unique())
    logger.info(f"States with HRSN data: {states_with_hrsn}")
    logger.info(f"  States: {', '.join(state_list)}")

    # ---- Validate ranges ----
    measure_cols = list(all_measure_cols.values())
    for col in measure_cols:
        if col in df_out.columns:
            vals = df_out[col].dropna()
            if len(vals) > 0:
                if vals.min() < 0 or vals.max() > 100:
                    logger.warning(f"  {col}: out of range [0,100] — min={vals.min():.1f}, max={vals.max():.1f}")

    # ---- Summary stats ----
    logger.info("\n--- HRSN Measure Summary ---")
    for col in hrsn_cols:
        vals = df_out[col].dropna()
        logger.info(f"  {col}: n={len(vals):,}, mean={vals.mean():.1f}, median={vals.median():.1f}, "
                     f"min={vals.min():.1f}, max={vals.max():.1f}, missing={df_out[col].isna().sum():,}")

    outcome_cols = list(outcome_map.values())
    logger.info("\n--- Outcome Measure Summary ---")
    for col in outcome_cols:
        vals = df_out[col].dropna()
        logger.info(f"  {col}: n={len(vals):,}, mean={vals.mean():.1f}, median={vals.median():.1f}, "
                     f"min={vals.min():.1f}, max={vals.max():.1f}, missing={df_out[col].isna().sum():,}")

    # ---- Save ----
    dest = PATHS["processed"] / "places_wide.parquet"
    save_parquet(df_out, dest)
    logger.info(f"Saved to {dest} ({dest.stat().st_size / 1e6:.1f} MB)")

    return df_out


if __name__ == "__main__":
    print("=" * 70)
    print("STEP 2.1: Reshape and Clean PLACES Data")
    print("=" * 70)
    df = reshape_places()
    print(f"\nFinal dataset: {len(df):,} tracts × {len(df.columns)} columns")
    print(f"Columns: {list(df.columns)}")
    print("\nStep 2.1 complete.")
