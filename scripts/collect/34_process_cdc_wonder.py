"""
Step 34: Process CDC WONDER County-Level Cause-Specific Mortality Data.

Reads the 5 manually downloaded tab-delimited text files from CDC WONDER
(Underlying Cause of Death, 2018-2023, Single Race, grouped by County)
and merges them into a single county-level mortality dataset.

Manual download instructions:
  1. Go to https://wonder.cdc.gov/ucd-icd10-expanded.html (database D158)
  2. Section 1: Group Results By = County; UNCHECK Age Adjusted Rate
     (age-adjusted rates not available at county level); keep Crude Rate checked
  3. Section 2-5: Leave defaults (All states, All ages, All dates, etc.)
  4. Section 6: Select ICD-10 codes for each condition (one query per condition):
     - Diabetes: E10-E14
     - IHD: I20-I25
     - Stroke: I60-I69
     - COPD: J40-J47
     - Asthma: J45-J46
  5. Section 7: Export Results = checked, Tab-Delimited Text,
     Show Suppressed Values + Show Zero Values
  6. Save as: wonder_diabetes.txt, wonder_ihd.txt, wonder_stroke.txt,
              wonder_copd.txt, wonder_asthma.txt
  7. Place in data/raw/cdc_wonder/

Output:
  - data/raw/cdc_wonder/wonder_county_mortality.csv
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

# Mapping of file names to condition labels
WONDER_FILES = {
    "wonder_diabetes.txt": "diabetes",
    "wonder_ihd.txt": "ihd",
    "wonder_stroke.txt": "stroke",
    "wonder_copd.txt": "copd",
    "wonder_asthma.txt": "asthma",
    # Also check for .tsv extension (CDC WONDER sometimes exports as .tsv)
    "wonder_diabetes.tsv": "diabetes",
    "wonder_ihd.tsv": "ihd",
    "wonder_stroke.tsv": "stroke",
    "wonder_copd.tsv": "copd",
    "wonder_asthma.tsv": "asthma",
}

# Column mapping for CDC WONDER output
# Note: Age Adjusted Rate is NOT available at the county level in WONDER;
# we use Crude Rate instead (our regressions control for age demographics).
WONDER_COLS = {
    "County Code": "county_fips",
    "County": "county_name",
    "Deaths": "deaths",
    "Population": "population",
    "Crude Rate": "crude_rate",
}


def _read_wonder_file(filepath, condition):
    """Read a single CDC WONDER tab-delimited export file.

    CDC WONDER text files have:
    - Header rows with column names
    - Data rows (tab-separated)
    - Footer rows starting with '---' containing notes/caveats
    - 'Suppressed' or 'Unreliable' in rate columns for small counts
    """
    logger.info(f"  Reading {filepath.name} ({condition})...")

    # Read tab-delimited file; the header starts with "Notes" and data rows
    # have an empty Notes field. The footer notes section starts with "---".
    rows = []
    header = None
    with open(filepath, "r") as f:
        for line in f:
            line = line.rstrip("\n")
            if line.startswith("---"):
                break
            if header is not None and line.startswith('"Notes"'):
                # Footer notes section (not the header)
                break
            if header is None:
                header = line.split("\t")
                # Clean quotes from header
                header = [h.strip('"') for h in header]
            else:
                if line.strip():
                    fields = line.split("\t")
                    fields = [f.strip('"') for f in fields]
                    rows.append(fields)

    if header is None or len(rows) == 0:
        logger.warning(f"    No data found in {filepath.name}")
        return pd.DataFrame()

    df = pd.DataFrame(rows, columns=header[:len(rows[0])])

    # Rename columns
    rename_map = {}
    for old_name, new_name in WONDER_COLS.items():
        if old_name in df.columns:
            rename_map[old_name] = new_name
    df = df.rename(columns=rename_map)

    # Standardize county FIPS
    if "county_fips" in df.columns:
        df["county_fips"] = df["county_fips"].astype(str).str.zfill(5)

    # Convert numeric columns, handling 'Suppressed' and 'Unreliable' values
    for col in ["deaths", "population", "crude_rate"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Add condition label
    df["condition"] = condition

    # Drop rows without county FIPS (total rows, etc.)
    if "county_fips" in df.columns:
        df = df[df["county_fips"].str.len() == 5].copy()

    n_total = len(df)
    n_suppressed = df["crude_rate"].isna().sum()
    logger.info(f"    {n_total:,} counties, {n_suppressed} suppressed "
                f"({n_suppressed / n_total * 100:.1f}%)")

    return df


def process_wonder_files():
    """Process all CDC WONDER files and merge into a single dataset."""
    ensure_dirs()

    input_dir = PATHS["raw"] / "cdc_wonder"
    output_path = input_dir / "wonder_county_mortality.csv"

    # Check if already processed
    if output_path.exists():
        logger.info(f"CDC WONDER data already processed at {output_path}")
        df = pd.read_csv(output_path, dtype={"county_fips": str})
        logger.info(f"  {len(df):,} rows, {df['county_fips'].nunique()} counties")
        return df

    # Check which files exist
    found_files = {}
    missing_files = []
    for filename, condition in WONDER_FILES.items():
        filepath = input_dir / filename
        if filepath.exists():
            found_files[filename] = condition
        else:
            missing_files.append(filename)

    if not found_files:
        logger.error("No CDC WONDER files found in data/raw/cdc_wonder/")
        logger.info("")
        logger.info("=" * 60)
        logger.info("MANUAL DOWNLOAD INSTRUCTIONS:")
        logger.info("=" * 60)
        logger.info("1. Go to: https://wonder.cdc.gov/ucd-icd10-expanded.html")
        logger.info("2. Section 1: Group Results By = County")
        logger.info("   Check 'Age Adjusted Rate' under Measures")
        logger.info("3. Section 6: Select ICD-10 codes (one query per condition):")
        logger.info("   - Diabetes: E10-E14  -> save as wonder_diabetes.txt")
        logger.info("   - IHD: I20-I25       -> save as wonder_ihd.txt")
        logger.info("   - Stroke: I60-I69    -> save as wonder_stroke.txt")
        logger.info("   - COPD: J40-J47      -> save as wonder_copd.txt")
        logger.info("   - Asthma: J45-J46    -> save as wonder_asthma.txt")
        logger.info("4. Section 7: Export = Tab-Delimited Text")
        logger.info("   Show Suppressed Values = checked")
        logger.info(f"5. Place files in: {input_dir}")
        logger.info("6. Re-run this script")
        logger.info("=" * 60)
        return None

    if missing_files:
        logger.warning(f"Missing files (will proceed with available): {missing_files}")

    # Read and concatenate all files
    logger.info(f"Processing {len(found_files)} CDC WONDER files...")
    all_dfs = []
    for filename, condition in found_files.items():
        filepath = input_dir / filename
        df = _read_wonder_file(filepath, condition)
        if len(df) > 0:
            all_dfs.append(df)

    if not all_dfs:
        logger.error("No valid data found in any file")
        return None

    # Stack long-format
    long_df = pd.concat(all_dfs, ignore_index=True)
    logger.info(f"\nCombined: {len(long_df):,} rows across {long_df['condition'].nunique()} conditions")

    # Save long format
    save_csv(long_df, output_path)

    # Also pivot to wide format for easier merging with validation data
    wide_df = long_df.pivot_table(
        index="county_fips",
        columns="condition",
        values="crude_rate",
        aggfunc="first",
    ).reset_index()
    wide_df.columns = ["county_fips"] + [
        f"mortality_{c}" for c in wide_df.columns[1:]
    ]

    wide_path = input_dir / "wonder_county_mortality_wide.csv"
    save_csv(wide_df, wide_path)

    # Summary
    logger.info(f"\n=== CDC WONDER Mortality Summary ===")
    logger.info(f"Counties: {wide_df['county_fips'].nunique():,}")
    for col in wide_df.columns[1:]:
        n_valid = wide_df[col].notna().sum()
        if n_valid > 0:
            logger.info(f"  {col:30s}: {n_valid:,} counties with data, "
                        f"mean={wide_df[col].mean():.1f} per 100K")

    return long_df


if __name__ == "__main__":
    print("=" * 70)
    print("Step 34: Process CDC WONDER County-Level Mortality")
    print("=" * 70)
    result = process_wonder_files()
    if result is not None:
        print(f"\nDone. {len(result):,} mortality records processed.")
    else:
        print("\nNo data found. See instructions above.")
