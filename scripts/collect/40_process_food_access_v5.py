"""
Phase E, Step 40: Process Food Access Data for V5 Validation.

Two independent food-related exposure proxies:

1. USDA Food Access Atlas (already downloaded at tract level):
   - Aggregates to county level: LILA tract share, SNAP per capita,
     low-access population share
   - SNAP data from HUD-USDA administrative geocoding (NOT ACS/BRFSS)

2. Map the Meal Gap (Feeding America, CPS-modeled):
   - County-level food insecurity rate from CPS Food Security Supplement
   - Completely independent of BRFSS
   - 3,144 counties, 2019-2023 data (use 2022 to match PLACES 2024)

Sources:
  USDA: https://www.ers.usda.gov/data-products/food-access-research-atlas/
  MMG:  https://www.feedingamerica.org/research/map-the-meal-gap/

Output: data/raw/food_access/food_access_county.csv
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


def process_food_access_atlas():
    """Aggregate tract-level Food Access Atlas to county level."""
    fa_path = PATHS["raw"] / "food_access" / "food_access_2019_tracts.csv"
    if not fa_path.exists():
        raise FileNotFoundError(f"Food Access Atlas not found at {fa_path}")

    df = pd.read_csv(fa_path, dtype={"tract_fips": str})
    df["tract_fips"] = df["tract_fips"].str.zfill(11)
    df["county_fips"] = df["tract_fips"].str[:5]
    logger.info(f"Loaded Food Access Atlas: {len(df):,} tracts")

    # Ensure numeric types
    for col in ["LILATracts_1And10", "TractSNAP", "Pop2010", "lapop1share"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Aggregate to county level
    records = []
    for fips, grp in df.groupby("county_fips"):
        n_tracts = len(grp)
        n_lila = grp["LILATracts_1And10"].sum() if "LILATracts_1And10" in grp.columns else np.nan

        # LILA tract share
        pct_lila = n_lila / n_tracts if n_tracts > 0 else np.nan

        # SNAP per capita (TractSNAP = housing units receiving SNAP, from HUD-USDA)
        total_snap = grp["TractSNAP"].sum() if "TractSNAP" in grp.columns else np.nan
        total_pop = grp["Pop2010"].sum() if "Pop2010" in grp.columns else np.nan
        snap_per_capita = total_snap / total_pop if total_pop and total_pop > 0 else np.nan

        # Population-weighted low-access share
        if "lapop1share" in grp.columns and "Pop2010" in grp.columns:
            valid = grp["lapop1share"].notna() & grp["Pop2010"].notna() & (grp["Pop2010"] > 0)
            if valid.sum() > 0:
                low_access = np.average(grp.loc[valid, "lapop1share"],
                                        weights=grp.loc[valid, "Pop2010"])
            else:
                low_access = np.nan
        else:
            low_access = np.nan

        records.append({
            "county_fips": fips,
            "pct_lila_tracts": round(pct_lila, 4) if not np.isnan(pct_lila) else np.nan,
            "county_snap_per_capita": round(snap_per_capita, 4) if not np.isnan(snap_per_capita) else np.nan,
            "county_low_access_share": round(low_access / 100, 4) if not np.isnan(low_access) else np.nan,
            "n_tracts": n_tracts,
            "total_pop": total_pop,
        })

    fa_county = pd.DataFrame(records)
    logger.info(f"Food Access Atlas: {len(fa_county):,} counties")
    logger.info(f"  pct_lila_tracts — mean: {fa_county['pct_lila_tracts'].mean():.3f}")
    logger.info(f"  county_snap_per_capita — mean: {fa_county['county_snap_per_capita'].mean():.4f}")
    logger.info(f"  county_low_access_share — mean: {fa_county['county_low_access_share'].mean():.3f}")

    return fa_county


def process_map_the_meal_gap():
    """Extract county-level food insecurity rates from Map the Meal Gap xlsx."""
    mmg_path = (PROJECT_ROOT / "data" / "MMG2025_Data_ToShare" /
                "MMG2025_2019-2023_Data_To_Share.xlsx")
    if not mmg_path.exists():
        raise FileNotFoundError(f"Map the Meal Gap xlsx not found at {mmg_path}")

    logger.info(f"Reading Map the Meal Gap xlsx: {mmg_path}")
    df = pd.read_excel(mmg_path, sheet_name="County")
    logger.info(f"Loaded MMG: {len(df):,} rows, years: {sorted(df['Year'].unique())}")

    # Filter to 2022 (matches PLACES 2024 release / ACS 2022)
    df_2022 = df[df["Year"] == 2022].copy()
    logger.info(f"MMG 2022: {len(df_2022):,} counties")

    # Clean FIPS
    df_2022["county_fips"] = df_2022["FIPS"].astype(str).str.zfill(5)

    # Extract key columns
    mmg_county = df_2022[["county_fips"]].copy()
    mmg_county["mmg_food_insecurity_rate"] = pd.to_numeric(
        df_2022["Overall Food Insecurity Rate"], errors="coerce"
    )

    # Child food insecurity if available
    if "Child Food Insecurity Rate" in df_2022.columns:
        mmg_county["mmg_child_food_insecurity_rate"] = pd.to_numeric(
            df_2022["Child Food Insecurity Rate"], errors="coerce"
        )

    mmg_county = mmg_county.dropna(subset=["mmg_food_insecurity_rate"])
    logger.info(f"MMG processed: {len(mmg_county):,} counties")
    logger.info(f"  Food insecurity rate — mean: {mmg_county['mmg_food_insecurity_rate'].mean():.3f}, "
                f"range: [{mmg_county['mmg_food_insecurity_rate'].min():.3f}, "
                f"{mmg_county['mmg_food_insecurity_rate'].max():.3f}]")

    return mmg_county


def process_food_access_v5():
    """Process both food access data sources and merge."""
    ensure_dirs()

    out_dir = PATHS["raw"] / "food_access"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "food_access_county.csv"

    # Process both sources
    fa_county = process_food_access_atlas()
    mmg_county = process_map_the_meal_gap()

    # Merge
    merged = fa_county.merge(mmg_county, on="county_fips", how="outer")
    logger.info(f"\nMerged food access data: {len(merged):,} counties")
    logger.info(f"  Both FA + MMG: {merged[['pct_lila_tracts', 'mmg_food_insecurity_rate']].notna().all(axis=1).sum():,}")
    logger.info(f"  FA only: {merged['pct_lila_tracts'].notna().sum():,}")
    logger.info(f"  MMG only: {merged['mmg_food_insecurity_rate'].notna().sum():,}")

    save_csv(merged, out_path)
    logger.info(f"Saved to {out_path}")

    return merged


if __name__ == "__main__":
    print("=" * 70)
    print("PHASE E: Process Food Access Data for V5 Validation")
    print("=" * 70)
    df = process_food_access_v5()
    print(f"\nDone. {len(df):,} counties with food access measures.")
