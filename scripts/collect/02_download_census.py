"""
Step 1.3: Download Census ACS 5-Year (2022) demographic data at census tract level.

Uses censusdis to pull ACS variables for all US census tracts.

Variables — Demographics:
  B01003_001E  - Total population
  B01002_001E  - Median age
  B02001_002E  - White alone
  B02001_003E  - Black/African American alone
  B03003_003E  - Hispanic/Latino
  B17001_001E  - Poverty universe total
  B17001_002E  - Below poverty level
  B19013_001E  - Median household income
  B15003_001E  - Pop 25+ total (for education denominator)
  B15003_017E  - High school diploma
  B15003_018E  - GED
  B15003_022E  - Bachelor's degree
  B15003_023E  - Master's degree
  B15003_024E  - Professional degree
  B15003_025E  - Doctorate

Variables — Direct Social Needs Proxies (for validation framework):
  B22001_001E  - Total households (SNAP universe)
  B22001_002E  - Household received Food Stamps/SNAP
  B25070_001E  - Total renter-occupied units (rent burden universe)
  B25070_007E  - Gross rent 30.0-34.9% of income
  B25070_008E  - Gross rent 35.0-39.9% of income
  B25070_009E  - Gross rent 40.0-49.9% of income
  B25070_010E  - Gross rent 50.0%+ of income
  B08201_001E  - Total households (vehicle universe)
  B08201_002E  - No vehicle available
  B11001_001E  - Total households (living arrangement universe)
  B11001_008E  - Nonfamily households: householder living alone

Output: data/raw/census/acs_tract_demographics_2022.csv
"""
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from hrsn_analysis.paths import PATHS, ensure_dirs


# All 50 states + DC FIPS codes
STATE_FIPS = [
    "01", "02", "04", "05", "06", "08", "09", "10", "11", "12",
    "13", "15", "16", "17", "18", "19", "20", "21", "22", "23",
    "24", "25", "26", "27", "28", "29", "30", "31", "32", "33",
    "34", "35", "36", "37", "38", "39", "40", "41", "42", "44",
    "45", "46", "47", "48", "49", "50", "51", "53", "54", "55",
    "56",
]


def download_census_data():
    """Download ACS 5-year 2022 tract-level data using censusdis."""
    import censusdis.data as ced
    from censusdis.datasets import ACS5

    ensure_dirs()
    dest = PATHS["census_raw"] / "acs_tract_demographics_2022.csv"

    if dest.exists():
        print(f"Census data already exists at {dest}. Skipping download.")
        df = pd.read_csv(dest, dtype={"state": str, "county": str, "tract": str})
        print(f"  Rows: {len(df):,}")
        return df

    # Variables to pull
    variables = [
        # Demographics
        "B01003_001E",  # Total population
        "B01002_001E",  # Median age
        "B02001_002E",  # White alone
        "B02001_003E",  # Black alone
        "B03003_003E",  # Hispanic/Latino
        "B17001_001E",  # Poverty universe total (denominator for poverty rate)
        "B17001_002E",  # Below poverty
        "B19013_001E",  # Median household income
        "B15003_001E",  # Pop 25+ total
        "B15003_017E",  # HS diploma
        "B15003_018E",  # GED
        "B15003_022E",  # Bachelor's
        "B15003_023E",  # Master's
        "B15003_024E",  # Professional
        "B15003_025E",  # Doctorate
        # Direct social needs proxies (validation framework)
        "B22001_001E",  # Total households (SNAP universe)
        "B22001_002E",  # Received Food Stamps/SNAP
        "B25070_001E",  # Total renters (rent burden universe)
        "B25070_007E",  # Rent 30.0-34.9% of income
        "B25070_008E",  # Rent 35.0-39.9% of income
        "B25070_009E",  # Rent 40.0-49.9% of income
        "B25070_010E",  # Rent 50.0%+ of income
        "B08201_001E",  # Total households (vehicle universe)
        "B08201_002E",  # No vehicle available
        "B11001_001E",  # Total households (living arrangement)
        "B11001_008E",  # Householder living alone
    ]

    print(f"Downloading ACS 5-Year 2022 data for {len(STATE_FIPS)} states...")
    print(f"  Variables: {len(variables)}")

    all_dfs = []
    for i, state_fips in enumerate(STATE_FIPS):
        try:
            df_state = ced.download(
                ACS5,
                2022,
                variables,
                state=state_fips,
                county="*",
                tract="*",
            )
            all_dfs.append(df_state)
            print(f"  [{i+1}/{len(STATE_FIPS)}] State {state_fips}: {len(df_state):,} tracts")
        except Exception as e:
            print(f"  [{i+1}/{len(STATE_FIPS)}] State {state_fips}: FAILED — {e}")

    df = pd.concat(all_dfs, ignore_index=True)
    print(f"\nTotal tracts downloaded: {len(df):,}")

    # Ensure FIPS codes are strings with proper padding
    for col in ["state", "county", "tract"]:
        if col in df.columns:
            df[col] = df[col].astype(str)

    # Save
    df.to_csv(dest, index=False)
    print(f"Saved to {dest}")
    print(f"  File size: {dest.stat().st_size / 1e6:.1f} MB")

    return df


def verify_census_data(df):
    """Basic verification of downloaded Census data."""
    print("\n--- Census Data Verification ---")
    print(f"Rows: {len(df):,}")
    print(f"Columns: {list(df.columns)}")
    print(f"\nSample (first 3 rows):")
    print(df.head(3).to_string())

    # Check ranges
    if "B01003_001E" in df.columns:
        pop = df["B01003_001E"]
        print(f"\nTotal population: min={pop.min()}, max={pop.max():,}, median={pop.median():,.0f}")
    if "B19013_001E" in df.columns:
        inc = df["B19013_001E"]
        print(f"Median income: min=${inc.min():,.0f}, max=${inc.max():,.0f}, median=${inc.median():,.0f}")

    print("--- Verification Complete ---")


if __name__ == "__main__":
    print("=" * 70)
    print("STEP 1.3: Download Census ACS 5-Year (2022) Tract Demographics")
    print("=" * 70)
    df = download_census_data()
    verify_census_data(df)
    print("\nStep 1.3 complete.")
