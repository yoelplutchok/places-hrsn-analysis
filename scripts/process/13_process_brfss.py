"""
Phase B, Step 13: Process BRFSS 2023 for individual-level analysis.

Filters to SDOH/HRSN module respondents, recodes HRSN exposures, chronic
disease outcomes, and demographic covariates. Retains survey design variables.

Output:
  - data/processed/brfss_analytic.parquet
  - outputs/tables/brfss_sample_description.csv
"""
import sys
from pathlib import Path

import pandas as pd
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from hrsn_analysis.paths import PATHS, ensure_dirs
from hrsn_analysis.io_utils import load_params, save_parquet, save_csv
from hrsn_analysis.logging_utils import setup_logging, get_logger
from hrsn_analysis.survey_utils import recode_brfss_binary, recode_demographics

setup_logging()
logger = get_logger(__name__)


def process_brfss():
    """Process raw BRFSS XPT file into analysis-ready parquet."""
    ensure_dirs()
    params = load_params()
    brfss_params = params["brfss"]

    xpt_path = PATHS["brfss_raw"] / brfss_params["xpt_filename"]
    logger.info(f"Loading BRFSS from: {xpt_path}")

    # ---- Step 1: Load XPT, select needed columns ----
    # Identify all columns we need
    needed_cols = set()

    # Survey design
    needed_cols.add(brfss_params["weight_var"])
    needed_cols.add(brfss_params["strata_var"])
    needed_cols.add(brfss_params["psu_var"])

    # HRSN variables
    hrsn_map = {}
    for hrsn_name, info in brfss_params["hrsn_variables"].items():
        needed_cols.add(info["brfss_var"])
        hrsn_map[hrsn_name] = info

    # Outcome variables
    outcome_map = {}
    for out_name, info in brfss_params["outcome_variables"].items():
        needed_cols.add(info["brfss_var"])
        outcome_map[out_name] = info

    # Demographics
    demo_vars = brfss_params["demographic_vars"]
    for var in demo_vars.values():
        needed_cols.add(var)

    # State FIPS for filtering
    needed_cols.add("_STATE")

    logger.info(f"Selecting {len(needed_cols)} columns from BRFSS file...")

    # Read full file (SAS XPT)
    df_raw = pd.read_sas(xpt_path, format="xport")
    logger.info(f"Raw BRFSS: {len(df_raw):,} respondents × {len(df_raw.columns)} variables")

    # Keep only needed columns (case-insensitive matching)
    raw_cols_upper = {c.upper(): c for c in df_raw.columns}
    cols_to_keep = []
    col_rename = {}
    for needed in needed_cols:
        upper = needed.upper()
        if upper in raw_cols_upper:
            cols_to_keep.append(raw_cols_upper[upper])
            col_rename[raw_cols_upper[upper]] = needed
        else:
            logger.warning(f"  Column not found: {needed}")

    df = df_raw[cols_to_keep].copy()
    df = df.rename(columns=col_rename)
    logger.info(f"Selected columns: {len(df.columns)}")

    # ---- Step 2: Filter to SD/HE module respondents ----
    # Respondents who received the SDOH module will have non-blank HRSN variables
    # Check for at least one non-missing HRSN variable
    hrsn_brfss_vars = [info["brfss_var"] for info in hrsn_map.values() if info["brfss_var"] in df.columns]

    if hrsn_brfss_vars:
        has_hrsn = df[hrsn_brfss_vars].notna().any(axis=1)
        # Also exclude rows where ALL HRSN vars are 0 (likely not administered)
        n_before = len(df)
        df = df[has_hrsn].copy()
        logger.info(f"Filtered to SDOH module respondents: {len(df):,} "
                     f"(dropped {n_before - len(df):,})")
    else:
        logger.warning("No HRSN variables found — cannot filter to SDOH module")

    # ---- Step 3: Recode HRSN to binary ----
    logger.info("\n=== Recoding HRSN variables ===")
    hrsn_recoded = []
    for hrsn_name, info in hrsn_map.items():
        brfss_var = info["brfss_var"]
        recode_type = info["recode"]

        if brfss_var not in df.columns:
            logger.warning(f"  {hrsn_name}: variable {brfss_var} not in data — skipping")
            continue

        df[hrsn_name] = recode_brfss_binary(df[brfss_var], recode_type)
        n_valid = df[hrsn_name].notna().sum()
        prevalence = df[hrsn_name].mean() * 100 if n_valid > 0 else 0
        logger.info(f"  {hrsn_name:15s}: n={n_valid:,}, prevalence={prevalence:.1f}%")
        hrsn_recoded.append(hrsn_name)

    # ---- Step 4: Recode outcomes to binary ----
    logger.info("\n=== Recoding outcome variables ===")
    outcome_recoded = []
    for out_name, info in outcome_map.items():
        brfss_var = info["brfss_var"]
        recode_type = info["recode"]

        if brfss_var not in df.columns:
            logger.warning(f"  {out_name}: variable {brfss_var} not in data — skipping")
            continue

        df[out_name] = recode_brfss_binary(df[brfss_var], recode_type)
        n_valid = df[out_name].notna().sum()
        prevalence = df[out_name].mean() * 100 if n_valid > 0 else 0
        logger.info(f"  {out_name:15s}: n={n_valid:,}, prevalence={prevalence:.1f}%")
        outcome_recoded.append(out_name)

    # ---- Step 5: Recode demographics ----
    logger.info("\n=== Recoding demographics ===")
    demo_df = recode_demographics(df)
    for col in demo_df.columns:
        df[col] = demo_df[col]
        n_valid = df[col].notna().sum()
        logger.info(f"  {col:20s}: n_valid={n_valid:,}")

    # ---- Step 6: Build analytic dataset ----
    # Keep: recoded HRSN, recoded outcomes, demographics, survey design
    keep_cols = (
        hrsn_recoded
        + outcome_recoded
        + list(demo_df.columns)
        + [brfss_params["weight_var"], brfss_params["strata_var"], brfss_params["psu_var"]]
        + ["_STATE"]
    )
    keep_cols = [c for c in keep_cols if c in df.columns]

    analytic = df[keep_cols].copy()

    # Drop rows with zero or missing weight
    weight_var = brfss_params["weight_var"]
    if weight_var in analytic.columns:
        n_before = len(analytic)
        analytic = analytic[analytic[weight_var].notna() & (analytic[weight_var] > 0)]
        logger.info(f"Dropped {n_before - len(analytic):,} rows with missing/zero weights")

    logger.info(f"\nFinal analytic dataset: {len(analytic):,} respondents × {len(analytic.columns)} columns")

    # Save
    save_parquet(analytic, PATHS["processed"] / "brfss_analytic.parquet")

    # ---- Sample description table ----
    desc_rows = []
    desc_rows.append({"variable": "N (total)", "value": f"{len(analytic):,}"})

    for col in hrsn_recoded:
        if col in analytic.columns:
            n = analytic[col].notna().sum()
            prev = analytic[col].mean() * 100
            desc_rows.append({"variable": f"HRSN: {col}", "value": f"n={n:,}, prev={prev:.1f}%"})

    for col in outcome_recoded:
        if col in analytic.columns:
            n = analytic[col].notna().sum()
            prev = analytic[col].mean() * 100
            desc_rows.append({"variable": f"Outcome: {col}", "value": f"n={n:,}, prev={prev:.1f}%"})

    if "female" in analytic.columns:
        pct_female = analytic["female"].mean() * 100
        desc_rows.append({"variable": "% Female", "value": f"{pct_female:.1f}%"})

    desc_df = pd.DataFrame(desc_rows)
    save_csv(desc_df, PATHS["tables"] / "brfss_sample_description.csv")

    return analytic


if __name__ == "__main__":
    print("=" * 70)
    print("PHASE B: Process BRFSS 2023")
    print("=" * 70)
    analytic = process_brfss()
    print(f"\nDone. Analytic dataset: {len(analytic):,} respondents.")
