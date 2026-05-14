"""
Step 3.3: Geographic distribution maps for HRSN measures.

Creates choropleth maps of each HRSN measure at the census tract level
for the contiguous US.

Output: outputs/figures/maps/hrsn_{measure}_map.png (7 maps)
"""
import sys
from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import geopandas as gpd

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from hrsn_analysis.paths import PATHS, ensure_dirs
from hrsn_analysis.io_utils import load_parquet, load_params
from hrsn_analysis.logging_utils import setup_logging, get_logger

setup_logging()
logger = get_logger(__name__)


def create_hrsn_maps():
    """Create choropleth maps for each HRSN measure."""
    ensure_dirs()
    params = load_params()

    # Create maps subdirectory
    maps_dir = PATHS["figures"] / "maps"
    maps_dir.mkdir(parents=True, exist_ok=True)

    # ---- Load data ----
    logger.info("Loading master dataset...")
    df = load_parquet(PATHS["processed"] / "merged_tracts.parquet")

    logger.info("Loading tract geometries...")
    gdf_geo = gpd.read_file(PATHS["geo_raw"] / "us_tracts_2022.gpkg")
    logger.info(f"Geometries: {len(gdf_geo):,} tracts")

    # Merge data with geometries
    gdf = gdf_geo.merge(df, left_on="GEOID", right_on="tract_fips", how="inner")
    logger.info(f"Merged for mapping: {len(gdf):,} tracts")

    # Filter to contiguous US (exclude AK=02, HI=15)
    gdf_conus = gdf[~gdf["state_fips"].isin(["02", "15"])].copy()
    logger.info(f"Contiguous US: {len(gdf_conus):,} tracts")

    # ---- Create maps ----
    hrsn_measures = params["hrsn_measures"]

    for measure in hrsn_measures:
        col = measure["id"].lower()
        name = measure["name"]
        logger.info(f"Creating map for {name} ({col})...")

        fig, ax = plt.subplots(1, 1, figsize=(16, 10))

        gdf_conus.plot(
            column=col,
            cmap="YlOrRd",
            linewidth=0,
            ax=ax,
            legend=True,
            legend_kwds={
                "label": f"{name} — Crude Prevalence (%)",
                "orientation": "horizontal",
                "shrink": 0.6,
                "pad": 0.02,
            },
            missing_kwds={"color": "lightgrey", "label": "No data"},
        )

        ax.set_xlim(-125, -66)
        ax.set_ylim(24, 50)
        ax.set_axis_off()
        ax.set_title(f"{name}\nCensus Tract-Level Prevalence (N={len(gdf_conus):,} tracts)",
                      fontsize=14, fontweight="bold")

        dest = maps_dir / f"hrsn_{col}_map.png"
        fig.savefig(dest, dpi=300, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        logger.info(f"  Saved to {dest}")

    # ---- Also create a panel figure with all 7 ----
    logger.info("Creating combined HRSN panel map...")
    fig, axes = plt.subplots(3, 3, figsize=(24, 20))
    axes = axes.flatten()

    for i, measure in enumerate(hrsn_measures):
        col = measure["id"].lower()
        name = measure["name"]
        ax = axes[i]

        gdf_conus.plot(
            column=col,
            cmap="YlOrRd",
            linewidth=0,
            ax=ax,
            legend=True,
            legend_kwds={
                "orientation": "horizontal",
                "shrink": 0.8,
                "pad": 0.02,
                "label": "Prevalence (%)",
            },
        )
        ax.set_xlim(-125, -66)
        ax.set_ylim(24, 50)
        ax.set_axis_off()
        ax.set_title(name, fontsize=12, fontweight="bold")

    # Hide unused subplots
    for j in range(len(hrsn_measures), len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("Health-Related Social Needs: Census Tract-Level Prevalence",
                 fontsize=16, fontweight="bold", y=0.98)
    plt.tight_layout()

    dest_panel = PATHS["figures"] / "hrsn_maps_panel.png"
    fig.savefig(dest_panel, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    logger.info(f"Panel map saved to {dest_panel}")


if __name__ == "__main__":
    print("=" * 70)
    print("STEP 3.3: Geographic Distribution Maps")
    print("=" * 70)
    create_hrsn_maps()
    print("\nStep 3.3 complete.")
