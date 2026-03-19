"""
Step 22: Process independent validation outcome data.

Processes three independent data sources for the validation framework:

1. CMS Mapping Medicare Disparities (MMD) — county-level disease PREVALENCE
   - Diabetes, Depression, COPD, Ischemic Heart Disease, Asthma, Obesity, Stroke
   - Source: 100% Medicare FFS claims (ICD codes, NOT modeled from BRFSS)
   - Files: data/raw/cms/mmd_data*.csv (manually downloaded from MMD tool)

2. CMS Medicare Geographic Variation PUF — county-level UTILIZATION
   - HCC Risk Score, Spending, ER visits, Hospitalizations
   - Source: 100% Medicare FFS claims (NOT modeled from BRFSS)

3. County Health Rankings 2024 — county-level MORTALITY
   - Premature age-adjusted mortality rate (from NVSS death certificates)
   - Source: National Vital Statistics System (NOT modeled from BRFSS)

Output:
  - data/processed/validation_outcomes.csv (merged, cleaned validation outcomes)
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import io

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from hrsn_analysis.paths import PATHS, ensure_dirs
from hrsn_analysis.logging_utils import setup_logging, get_logger

setup_logging()
logger = get_logger(__name__)

# --- CMS MMD Disease Prevalence ---
# Mapping from MMD condition names to standardized column names
MMD_CONDITION_MAP = {
    "Diabetes": "cms_diabetes_prev",
    "Depression": "cms_depression_prev",
    "Chronic Obstructive Pulmonary Disease": "cms_copd_prev",
    "Ischemic Heart Disease": "cms_chd_prev",
    "Asthma": "cms_asthma_prev",
    "Obesity": "cms_obesity_prev",
    "Stroke/Transient Ischemic Attack": "cms_stroke_prev",
}

# Mapping from CMS disease outcomes back to PLACES disease outcome names
CMS_TO_PLACES = {
    "cms_diabetes_prev": "diabetes",
    "cms_depression_prev": "depression",
    "cms_copd_prev": "copd",
    "cms_chd_prev": "chd",
    "cms_asthma_prev": "casthma",
    "cms_obesity_prev": "obesity",
    "cms_stroke_prev": "stroke",
}

# --- CMS Geographic Variation PUF ---
GV_PUF_CSV_URL = (
    "https://data.cms.gov/sites/default/files/2025-03/"
    "a40ac71d-9f80-4d99-92d2-fd149433d7d8/"
    "2014-2023%20Medicare%20Fee-for-Service%20Geographic%20Variation%20"
    "Public%20Use%20File.csv"
)

GV_PUF_COLS = {
    "BENE_GEO_CD": "county_fips",
    "BENE_AVG_RISK_SCRE": "medicare_risk_score",
    "TOT_MDCR_STDZD_PYMT_PC": "medicare_spending_pc",
    "ER_VISITS_PER_1000_BENES": "medicare_er_rate",
    "IP_CVRD_STAYS_PER_1000_BENES": "medicare_hosp_rate",
    "IP_CVRD_DAYS_PER_1000_BENES": "medicare_hosp_days",
    "ACUTE_HOSP_READMSN_PCT": "medicare_readmission_pct",
    "BENES_FFS_CNT": "medicare_ffs_benes",
    "BENE_DUAL_PCT": "medicare_dual_pct",
}

# --- County Health Rankings ---
CHR_CSV_URL = (
    "https://www.countyhealthrankings.org/sites/default/files/"
    "media/document/analytic_data2024.csv"
)

CHR_COLS = {
    "5-digit FIPS Code": "county_fips",
    "Premature Age-Adjusted Mortality raw value": "chr_premature_mortality",
    "Drug Overdose Deaths raw value": "chr_overdose_mortality",
    "Injury Deaths raw value": "chr_injury_mortality",
}


def process_mmd_files():
    """Process CMS Mapping Medicare Disparities disease prevalence files."""
    cms_dir = PATHS["raw"] / "cms"
    mmd_files = sorted(cms_dir.glob("mmd_data*.csv"))

    if not mmd_files:
        logger.warning(f"No MMD files found in {cms_dir}.")
        logger.warning("Download from https://data.cms.gov/mapping-medicare-disparities")
        return None

    logger.info(f"Found {len(mmd_files)} MMD files in {cms_dir}")

    all_county_data = {}  # fips -> {col: value}

    for fpath in mmd_files:
        df = pd.read_csv(fpath, dtype=str, low_memory=False)
        logger.info(f"  {fpath.name}: {len(df)} rows")

        if "condition" not in df.columns or "analysis_value" not in df.columns:
            logger.warning(f"  Skipping {fpath.name}: missing expected columns")
            continue

        condition = df["condition"].iloc[0].strip('"').strip()
        col_name = MMD_CONDITION_MAP.get(condition)

        if col_name is None:
            logger.warning(f"  Unknown condition: {condition}")
            continue

        logger.info(f"  Condition: {condition} → {col_name}")

        for _, row in df.iterrows():
            fips = str(row["fips"]).zfill(5)
            val = row["analysis_value"]
            try:
                val = float(val)
            except (ValueError, TypeError):
                val = np.nan

            if fips not in all_county_data:
                all_county_data[fips] = {"county_fips": fips}
            all_county_data[fips][col_name] = val

    if not all_county_data:
        return None

    mmd_df = pd.DataFrame(list(all_county_data.values()))
    logger.info(f"\nMMD disease prevalence: {len(mmd_df)} counties, "
                f"{len(mmd_df.columns) - 1} conditions")

    for col in sorted(mmd_df.columns):
        if col == "county_fips":
            continue
        n_valid = mmd_df[col].notna().sum()
        mean_val = mmd_df[col].mean()
        logger.info(f"  {col}: n={n_valid}, mean={mean_val:.1f}%")

    return mmd_df


def download_gv_puf():
    """Download CMS Geographic Variation PUF county-level data."""
    ensure_dirs()
    cms_dir = PATHS["raw"] / "cms"
    cms_dir.mkdir(parents=True, exist_ok=True)
    raw_dest = cms_dir / "cms_geographic_variation_puf.csv"

    if raw_dest.exists():
        logger.info(f"GV PUF already exists at {raw_dest}. Loading...")
        df = pd.read_csv(raw_dest, dtype=str, low_memory=False)
        logger.info(f"  Loaded: {len(df):,} rows")
        return df

    logger.info("Downloading CMS Geographic Variation PUF (~53 MB)...")
    try:
        resp = requests.get(GV_PUF_CSV_URL, timeout=300)
        resp.raise_for_status()
    except Exception as e:
        logger.error(f"GV PUF download failed: {e}")
        return None

    df = pd.read_csv(io.StringIO(resp.text), dtype=str, low_memory=False)
    logger.info(f"  Downloaded {len(df):,} rows, {len(df.columns)} columns")

    df = df[df["BENE_GEO_LVL"] == "County"].copy()
    logger.info(f"  County-level rows: {len(df):,}")

    df.to_csv(raw_dest, index=False)
    logger.info(f"  Saved to {raw_dest} ({raw_dest.stat().st_size / 1e6:.1f} MB)")
    return df


def download_chr():
    """Download County Health Rankings 2024 data."""
    raw_dest = PATHS["raw"] / "chr_county_health_rankings_2024.csv"

    if raw_dest.exists():
        logger.info(f"CHR data already exists at {raw_dest}. Loading...")
        df = pd.read_csv(raw_dest, dtype=str, low_memory=False)
        logger.info(f"  Loaded: {len(df):,} rows")
        return df

    logger.info("Downloading County Health Rankings 2024 (~13 MB)...")
    try:
        resp = requests.get(CHR_CSV_URL, timeout=120)
        resp.raise_for_status()
    except Exception as e:
        logger.error(f"CHR download failed: {e}")
        return None

    full_df = pd.read_csv(io.StringIO(resp.text), header=0, low_memory=False)
    logger.info(f"  Downloaded {len(full_df):,} rows, {len(full_df.columns)} columns")

    available = {k: v for k, v in CHR_COLS.items() if k in full_df.columns}
    chr_df = full_df[list(available.keys())].rename(columns=available).copy()

    chr_df.to_csv(raw_dest, index=False)
    logger.info(f"  Saved {len(chr_df)} rows to {raw_dest}")
    return chr_df


def process_gv_puf(gv_df):
    """Extract utilization measures from GV PUF."""
    gv = gv_df[(gv_df["YEAR"] == gv_df["YEAR"].max()) &
               (gv_df["BENE_AGE_LVL"] == "All")].copy()
    year_used = gv["YEAR"].iloc[0] if len(gv) > 0 else "?"
    logger.info(f"GV PUF: using year {year_used}, {len(gv):,} counties")

    medicare = pd.DataFrame()
    medicare["county_fips"] = gv["BENE_GEO_CD"].astype(str).str.zfill(5)

    for src_col, dst_col in GV_PUF_COLS.items():
        if src_col == "BENE_GEO_CD":
            continue
        if src_col in gv.columns:
            medicare[dst_col] = pd.to_numeric(
                gv[src_col].replace("*", np.nan), errors="coerce"
            ).values

    logger.info(f"  Medicare utilization: {len(medicare):,} counties")
    for col in medicare.columns[1:]:
        n_valid = medicare[col].notna().sum()
        if n_valid > 0:
            logger.info(f"    {col}: n={n_valid:,}, mean={medicare[col].mean():.2f}")

    return medicare


def process_chr(chr_df):
    """Extract mortality measures from CHR."""
    chr_out = chr_df.copy()
    chr_out["county_fips"] = chr_out["county_fips"].astype(str).str.zfill(5)
    for col in chr_out.columns[1:]:
        chr_out[col] = pd.to_numeric(chr_out[col], errors="coerce")

    logger.info(f"  CHR mortality: {len(chr_out):,} counties")
    for col in chr_out.columns[1:]:
        n_valid = chr_out[col].notna().sum()
        if n_valid > 0:
            logger.info(f"    {col}: n={n_valid:,}, mean={chr_out[col].mean():.3f}")

    return chr_out


def merge_validation_outcomes(mmd_df, medicare_df, chr_df):
    """Merge all validation outcome datasets."""
    logger.info("\n=== Merging validation outcomes ===")

    # Start with MMD (the primary independent outcomes)
    if mmd_df is not None:
        merged = mmd_df.copy()
        logger.info(f"  MMD disease prevalence: {len(merged)} counties")
    else:
        merged = pd.DataFrame(columns=["county_fips"])

    # Merge GV PUF utilization
    if medicare_df is not None:
        if len(merged) > 0:
            merged = merged.merge(medicare_df, on="county_fips", how="outer")
        else:
            merged = medicare_df.copy()
        logger.info(f"  After GV PUF merge: {len(merged)} counties")

    # Merge CHR mortality
    if chr_df is not None:
        if len(merged) > 0:
            merged = merged.merge(chr_df, on="county_fips", how="outer")
        else:
            merged = chr_df.copy()
        logger.info(f"  After CHR merge: {len(merged)} counties")

    logger.info(f"\n  Final merged: {len(merged):,} counties, "
                f"{len(merged.columns)} columns")

    # Summary
    cms_cols = [c for c in merged.columns if c.startswith("cms_")]
    med_cols = [c for c in merged.columns if c.startswith("medicare_")]
    chr_cols = [c for c in merged.columns if c.startswith("chr_")]

    logger.info(f"\n  CMS disease prevalence ({len(cms_cols)} conditions):")
    for col in sorted(cms_cols):
        n = merged[col].notna().sum()
        logger.info(f"    {col}: {n:,} counties")

    logger.info(f"\n  Medicare utilization ({len(med_cols)} measures):")
    for col in sorted(med_cols):
        n = merged[col].notna().sum()
        logger.info(f"    {col}: {n:,} counties")

    logger.info(f"\n  CHR mortality ({len(chr_cols)} measures):")
    for col in sorted(chr_cols):
        n = merged[col].notna().sum()
        logger.info(f"    {col}: {n:,} counties")

    # Save
    dest = PATHS["processed"] / "validation_outcomes.csv"
    dest.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(dest, index=False)
    logger.info(f"\n  Saved to {dest}")

    return merged


if __name__ == "__main__":
    print("=" * 70)
    print("Step 22: Process Independent Validation Outcome Data")
    print("=" * 70)

    # 1. Process MMD disease prevalence (primary — manually downloaded)
    mmd_df = process_mmd_files()

    # 2. Download/load GV PUF (supplementary utilization)
    gv_raw = download_gv_puf()
    medicare_df = process_gv_puf(gv_raw) if gv_raw is not None else None

    # 3. Download/load CHR mortality (supplementary mortality)
    chr_raw = download_chr()
    chr_df = process_chr(chr_raw) if chr_raw is not None else None

    # 4. Merge all
    if mmd_df is not None or medicare_df is not None or chr_df is not None:
        merged = merge_validation_outcomes(mmd_df, medicare_df, chr_df)
        print(f"\nDone. Validation outcomes: {len(merged):,} counties.")
    else:
        print("\nNo data available. See instructions above.")
