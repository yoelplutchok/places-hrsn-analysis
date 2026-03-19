"""
Phase C, Step 15: Spatial Autocorrelation Analysis.

Tests for spatial dependence in HRSN and disease variables, and in OLS
residuals from the 49 regression models. Produces LISA cluster maps for
the strongest associations.

Steps:
  1. Merge data with geometries, filter to CONUS
  2. Build Queen contiguity weights
  3. Global Moran's I for all 14 variables
  4. Moran's I on OLS residuals for all 49 models
  5. Local Moran's I (LISA) → cluster maps

Output:
  - data/final/spatial_weights.gal
  - data/final/global_morans_i.csv
  - data/final/residual_morans_i.csv
  - outputs/figures/maps/lisa_*.png
"""
import sys
from pathlib import Path

import pandas as pd
import numpy as np
import geopandas as gpd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import statsmodels.api as sm
import libpysal
from esda.moran import Moran, Moran_Local

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from hrsn_analysis.paths import PATHS, ensure_dirs
from hrsn_analysis.io_utils import load_parquet, load_params, save_csv
from hrsn_analysis.logging_utils import setup_logging, get_logger

setup_logging()
logger = get_logger(__name__)


def standardize(series):
    """Z-score standardize."""
    return (series - series.mean()) / series.std()


def run_spatial_analysis():
    """Run spatial autocorrelation analysis."""
    ensure_dirs()
    params = load_params()
    spatial_params = params.get("spatial", {})

    maps_dir = PATHS["figures"] / "maps"
    maps_dir.mkdir(parents=True, exist_ok=True)

    hrsn_cols = [m["id"].lower() for m in params["hrsn_measures"]]
    outcome_cols = [m["id"].lower() for m in params["outcome_measures"]]
    hrsn_labels = {m["id"].lower(): m["name"] for m in params["hrsn_measures"]}
    outcome_labels = {m["id"].lower(): m["name"] for m in params["outcome_measures"]}

    covariate_cols = ["pct_black", "pct_hispanic", "pct_poverty",
                      "pct_college", "pct_65plus", "median_age"]

    exclude_states = spatial_params.get("exclude_states", ["02", "15"])

    # ================================================================
    # Step 1: Load and merge data with geometries
    # ================================================================
    logger.info("=== Step 1: Load data and geometries ===")

    df = load_parquet(PATHS["processed"] / "merged_tracts.parquet")
    gdf_geo = gpd.read_file(PATHS["geo_raw"] / "us_tracts_2022.gpkg")
    logger.info(f"Data: {len(df):,} tracts, Geometries: {len(gdf_geo):,} tracts")

    gdf = gdf_geo.merge(df, left_on="GEOID", right_on="tract_fips", how="inner")
    logger.info(f"Merged: {len(gdf):,} tracts")

    # Filter to CONUS
    gdf = gdf[~gdf["state_fips"].isin(exclude_states)].copy()
    gdf = gdf[gdf.geometry.notna()].copy()
    gdf = gdf.reset_index(drop=True)
    logger.info(f"CONUS tracts: {len(gdf):,}")

    # ================================================================
    # Step 2: Build Queen contiguity weights
    # ================================================================
    logger.info("\n=== Step 2: Build Spatial Weights ===")

    w = libpysal.weights.Queen.from_dataframe(gdf, use_index=False)

    # Handle islands (tracts with no neighbors)
    n_islands = sum(1 for v in w.cardinalities.values() if v == 0)
    if n_islands > 0:
        logger.warning(f"  {n_islands} island tracts (no neighbors) — will be excluded from Moran's I")

    # Row-standardize
    w.transform = "R"

    mean_neighbors = np.mean(list(w.cardinalities.values()))
    logger.info(f"  Weight type: Queen contiguity")
    logger.info(f"  N observations: {w.n}")
    logger.info(f"  Mean neighbors: {mean_neighbors:.1f}")
    logger.info(f"  Islands: {n_islands}")

    # Save weights
    weights_path = PATHS["final"] / "spatial_weights.gal"
    w.to_file(str(weights_path))
    logger.info(f"  Saved weights to {weights_path}")

    # ================================================================
    # Step 3: Global Moran's I for all variables
    # ================================================================
    logger.info("\n=== Step 3: Global Moran's I ===")

    all_vars = hrsn_cols + outcome_cols
    all_labels = {**hrsn_labels, **outcome_labels}

    moran_results = []
    for var in all_vars:
        if var not in gdf.columns:
            continue

        y = gdf[var].values
        # Handle NaN by subsetting weights (not mean-fill, which biases toward zero)
        valid_mask = ~np.isnan(y)
        if valid_mask.sum() < 10:
            logger.warning(f"  {var}: too few valid observations for Moran's I")
            continue
        valid_indices = np.where(valid_mask)[0]
        w_sub = libpysal.weights.w_subset(w, valid_indices)
        w_sub.transform = "r"

        mi = Moran(y[valid_mask], w_sub)

        row = {
            "variable": var,
            "label": all_labels.get(var, var),
            "type": "hrsn" if var in hrsn_cols else "outcome",
            "morans_i": round(mi.I, 4),
            "expected_i": round(mi.EI, 4),
            "z_score": round(mi.z_sim, 4),
            "p_value": mi.p_sim,
            "significant_05": mi.p_sim < 0.05,
        }
        moran_results.append(row)

        sig = "***" if mi.p_sim < 0.001 else "**" if mi.p_sim < 0.01 else "*" if mi.p_sim < 0.05 else "ns"
        logger.info(f"  {all_labels.get(var, var):30s}: I={mi.I:.4f} (z={mi.z_sim:.2f}) {sig}")

    moran_df = pd.DataFrame(moran_results)
    save_csv(moran_df, PATHS["final"] / "global_morans_i.csv")

    # ================================================================
    # Step 4: Moran's I on OLS residuals
    # ================================================================
    logger.info("\n=== Step 4: Residual Spatial Autocorrelation (49 models) ===")

    # Standardize for regression
    gdf_std = gdf.copy()
    for col in hrsn_cols + outcome_cols + covariate_cols:
        if col in gdf_std.columns:
            gdf_std[col] = standardize(gdf[col])

    cluster_var = gdf["county_fips"]
    residual_results = []

    for hrsn in hrsn_cols:
        for outcome in outcome_cols:
            X_cols = [hrsn] + covariate_cols
            X = sm.add_constant(gdf_std[X_cols])
            y = gdf_std[outcome]

            mask = X.notna().all(axis=1) & y.notna()
            X_clean = X[mask]
            y_clean = y[mask]

            try:
                model = sm.OLS(y_clean, X_clean)
                res = model.fit()

                # Get residuals aligned to full dataset
                residuals = np.full(len(gdf), np.nan)
                residuals[mask.values] = res.resid

                # Moran's I on residuals — exclude tracts with NaN residuals
                # (filling NaN with 0 would artificially inflate spatial autocorrelation)
                valid_resid_mask = ~np.isnan(residuals)
                if valid_resid_mask.sum() < 10:
                    logger.warning(f"  {hrsn} × {outcome}: too few valid residuals")
                    continue
                valid_indices = np.where(valid_resid_mask)[0]
                w_sub = libpysal.weights.w_subset(w, valid_indices)
                w_sub.transform = "r"
                mi = Moran(residuals[valid_resid_mask], w_sub)

                row = {
                    "hrsn_measure": hrsn,
                    "outcome": outcome,
                    "residual_morans_i": round(mi.I, 4),
                    "z_score": round(mi.z_sim, 4),
                    "p_value": mi.p_sim,
                    "significant_05": mi.p_sim < 0.05,
                    "n_obs": int(res.nobs),
                    "r_squared": round(res.rsquared, 4),
                }
                residual_results.append(row)

            except Exception as e:
                logger.warning(f"  {hrsn} × {outcome}: residual Moran's I failed — {e}")

    resid_df = pd.DataFrame(residual_results)
    save_csv(resid_df, PATHS["final"] / "residual_morans_i.csv")

    n_sig_resid = resid_df["significant_05"].sum() if len(resid_df) > 0 else 0
    logger.info(f"\nResidual Moran's I: {n_sig_resid}/{len(resid_df)} models show "
                f"significant spatial autocorrelation in residuals")
    if len(resid_df) > 0:
        mean_i = resid_df["residual_morans_i"].mean()
        logger.info(f"Mean residual I: {mean_i:.4f}")

    # ================================================================
    # Step 5: LISA cluster maps for top associations
    # ================================================================
    logger.info("\n=== Step 5: LISA Cluster Maps ===")

    # Select top 5 associations by |residual Moran's I| (most spatial structure)
    if len(resid_df) > 0:
        top_assoc = resid_df.nlargest(5, "residual_morans_i")

        permutations = spatial_params.get("lisa", {}).get("permutations", 999)
        alpha = spatial_params.get("lisa", {}).get("alpha", 0.05)

        for _, assoc in top_assoc.iterrows():
            hrsn = assoc["hrsn_measure"]
            outcome = assoc["outcome"]

            logger.info(f"\n  LISA: {hrsn} × {outcome} (residual I = {assoc['residual_morans_i']:.4f})")

            # Compute bivariate values for mapping — subset to valid obs
            y_hrsn = gdf[hrsn].values
            y_outcome = gdf[outcome].values
            valid_lisa = ~np.isnan(y_hrsn) & ~np.isnan(y_outcome)
            if valid_lisa.sum() < 10:
                logger.warning(f"  LISA: too few valid obs for {hrsn} × {outcome}")
                continue
            valid_lisa_idx = np.where(valid_lisa)[0]
            w_lisa = libpysal.weights.w_subset(w, valid_lisa_idx)
            w_lisa.transform = "r"

            # LISA on the outcome variable (valid subset only)
            lisa = Moran_Local(y_outcome[valid_lisa], w_lisa, permutations=permutations)

            # Classify clusters: 1=HH, 2=LH, 3=LL, 4=HL, 0=not significant
            sig_mask = lisa.p_sim < alpha
            quadrant = lisa.q.copy()
            quadrant[~sig_mask] = 0

            # Plot LISA cluster map
            fig, ax = plt.subplots(1, 1, figsize=(16, 10))

            # Cluster colors: 0=NS (light gray), 1=HH (red), 2=LH (light blue), 3=LL (blue), 4=HL (light red)
            cluster_colors = {0: "#d9d9d9", 1: "#e41a1c", 2: "#377eb8", 3: "#4daf4a", 4: "#ff7f00"}
            cluster_names = {0: "Not Significant", 1: "High-High", 2: "Low-High", 3: "Low-Low", 4: "High-Low"}

            gdf_plot = gdf.copy()
            # Align quadrant (subset-length) to full GeoDataFrame
            full_quadrant = np.zeros(len(gdf), dtype=int)  # 0 = not significant
            full_quadrant[valid_lisa_idx] = quadrant
            gdf_plot["cluster"] = full_quadrant

            # Plot each cluster type
            legend_patches = []
            for cluster_val in sorted(cluster_colors.keys()):
                subset = gdf_plot[gdf_plot["cluster"] == cluster_val]
                if len(subset) > 0:
                    subset.plot(
                        ax=ax,
                        color=cluster_colors[cluster_val],
                        linewidth=0,
                    )
                    legend_patches.append(
                        Patch(facecolor=cluster_colors[cluster_val],
                              label=f"{cluster_names[cluster_val]} (n={len(subset):,})")
                    )

            ax.set_xlim(-125, -66)
            ax.set_ylim(24, 50)
            ax.set_axis_off()

            hrsn_name = hrsn_labels.get(hrsn, hrsn)
            outcome_name = outcome_labels.get(outcome, outcome)
            ax.set_title(f"LISA Cluster Map: {outcome_name}\n"
                         f"(from {hrsn_name} model, p < {alpha})",
                         fontsize=14, fontweight="bold")
            ax.legend(handles=legend_patches, loc="lower left", fontsize=9)

            dest = maps_dir / f"lisa_{hrsn}_{outcome}.png"
            fig.savefig(dest, dpi=300, bbox_inches="tight", facecolor="white")
            plt.close(fig)
            logger.info(f"  Saved LISA map to {dest}")

            n_hh = (quadrant == 1).sum()
            n_ll = (quadrant == 3).sum()
            n_sig_total = sig_mask.sum()
            logger.info(f"  Significant clusters: {n_sig_total:,} tracts "
                        f"(HH={n_hh:,}, LL={n_ll:,})")

    return moran_df, resid_df


if __name__ == "__main__":
    print("=" * 70)
    print("PHASE C: Spatial Autocorrelation Analysis")
    print("=" * 70)
    moran, resid = run_spatial_analysis()
    print(f"\nDone. {len(moran)} global Moran's I, {len(resid)} residual tests.")
