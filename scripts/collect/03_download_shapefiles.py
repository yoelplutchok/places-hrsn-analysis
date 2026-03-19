"""
Step 1.4: Download Census TIGER/Line 2022 tract shapefiles.

Uses censusdis to download tract geometries with GEOID for all US states,
then saves as a single GeoPackage file.

Output: data/raw/geo/us_tracts_2022.gpkg
"""
import sys
from pathlib import Path

import geopandas as gpd
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


def download_shapefiles():
    """Download tract boundaries using censusdis."""
    import censusdis.data as ced
    from censusdis.datasets import ACS5

    ensure_dirs()
    dest = PATHS["geo_raw"] / "us_tracts_2022.gpkg"

    if dest.exists():
        print(f"Shapefiles already exist at {dest}. Skipping download.")
        gdf = gpd.read_file(dest)
        print(f"  Tracts: {len(gdf):,}")
        return gdf

    print(f"Downloading tract geometries for {len(STATE_FIPS)} states...")
    print("  (This may take a while — fetching geometry for ~73,000 tracts)")

    all_gdfs = []
    for i, state_fips in enumerate(STATE_FIPS):
        try:
            # Download just total population with geometry
            gdf_state = ced.download(
                ACS5,
                2022,
                ["B01003_001E"],
                state=state_fips,
                county="*",
                tract="*",
                with_geometry=True,
            )
            # Keep only GEOID and geometry
            # Build GEOID from state + county + tract
            gdf_state["GEOID"] = (
                gdf_state["STATE"].astype(str).str.zfill(2)
                + gdf_state["COUNTY"].astype(str).str.zfill(3)
                + gdf_state["TRACT"].astype(str).str.zfill(6)
            )
            gdf_state = gdf_state[["GEOID", "geometry"]]
            all_gdfs.append(gdf_state)
            print(f"  [{i+1}/{len(STATE_FIPS)}] State {state_fips}: {len(gdf_state):,} tracts")
        except Exception as e:
            print(f"  [{i+1}/{len(STATE_FIPS)}] State {state_fips}: FAILED — {e}")

    gdf = gpd.GeoDataFrame(pd.concat(all_gdfs, ignore_index=True))
    print(f"\nTotal tract geometries: {len(gdf):,}")

    # Save as GeoPackage
    gdf.to_file(dest, driver="GPKG")
    print(f"Saved to {dest}")
    print(f"  File size: {dest.stat().st_size / 1e6:.1f} MB")

    return gdf


def verify_shapefiles(gdf):
    """Basic verification."""
    print("\n--- Shapefile Verification ---")
    print(f"Total tracts: {len(gdf):,}")
    print(f"CRS: {gdf.crs}")
    print(f"GEOID sample: {gdf['GEOID'].head(3).tolist()}")
    print(f"GEOID length check: all 11 chars = {(gdf['GEOID'].str.len() == 11).all()}")
    print(f"Unique states: {gdf['GEOID'].str[:2].nunique()}")
    print("--- Verification Complete ---")


if __name__ == "__main__":
    print("=" * 70)
    print("STEP 1.4: Download Census Tract Shapefiles (2022)")
    print("=" * 70)
    gdf = download_shapefiles()
    verify_shapefiles(gdf)
    print("\nStep 1.4 complete.")
