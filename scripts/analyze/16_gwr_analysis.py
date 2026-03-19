"""
Phase C, Step 16: Geographically Weighted Regression (GWR).

Fits GWR models for the top 2-3 HRSN-disease associations to examine
geographic heterogeneity in relationships. Uses subsampling (~12K tracts)
because full GWR on 55K+ tracts is computationally infeasible.

Fallback: If GWR fails, runs spatial lag/error models via spreg.

Steps:
  1. Select top associations from regression results
  2. Subsample tracts (stratified by state)
  3. Project to EPSG:5070, extract centroids
  4. GWR with adaptive bandwidth
  5. Map local coefficients and t-statistics

Output:
  - data/final/gwr_summary.csv
  - outputs/figures/maps/gwr_*_local_beta.png
  - outputs/figures/maps/gwr_*_local_tstat.png
"""
import sys
import warnings
from pathlib import Path

import pandas as pd
import numpy as np
import geopandas as gpd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import statsmodels.api as sm

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from hrsn_analysis.paths import PATHS, ensure_dirs
from hrsn_analysis.io_utils import load_parquet, load_params, save_csv
from hrsn_analysis.logging_utils import setup_logging, get_logger

setup_logging()
logger = get_logger(__name__)

# Suppress convergence warnings from GWR
warnings.filterwarnings("ignore", category=RuntimeWarning)


def standardize(series):
    """Z-score standardize."""
    return (series - series.mean()) / series.std()


def select_top_associations(params, n_top=3):
    """Select top N associations by |beta| from regression results."""
    try:
        results = pd.read_csv(PATHS["final"] / "results_matrix.csv")
        top = results.reindex(results["beta_std"].abs().sort_values(ascending=False).index).head(n_top)
        pairs = [(row["hrsn_measure"], row["outcome"]) for _, row in top.iterrows()]
        logger.info(f"Selected top {n_top} associations:")
        for hrsn, outcome in pairs:
            beta = results[(results["hrsn_measure"] == hrsn) & (results["outcome"] == outcome)]["beta_std"].values[0]
            logger.info(f"  {hrsn} × {outcome}: β = {beta:+.4f}")
        return pairs
    except FileNotFoundError:
        # Fallback: use commonly strong associations
        logger.warning("Results matrix not found — using default associations")
        return [
            ("shututility", "copd"),
            ("foodinsecu", "diabetes"),
            ("emotionspt", "depression"),
        ]


def subsample_stratified(gdf, n_target, stratify_col="state_fips", random_state=42):
    """Stratified subsample by state FIPS."""
    state_counts = gdf[stratify_col].value_counts()
    n_states = len(state_counts)

    # Proportional allocation
    fractions = state_counts / state_counts.sum()
    allocations = (fractions * n_target).round().astype(int)
    # Ensure at least 1 per state
    allocations = allocations.clip(lower=1)

    samples = []
    for state, n_alloc in allocations.items():
        state_df = gdf[gdf[stratify_col] == state]
        n_sample = min(n_alloc, len(state_df))
        samples.append(state_df.sample(n=n_sample, random_state=random_state))

    result = pd.concat(samples).reset_index(drop=True)
    logger.info(f"Subsampled: {len(result):,} tracts from {n_states} states "
                f"(target was {n_target:,})")
    return result


def run_gwr_analysis():
    """Run GWR analysis for top associations."""
    ensure_dirs()
    params = load_params()
    spatial_params = params.get("spatial", {})
    gwr_params = spatial_params.get("gwr", {})

    maps_dir = PATHS["figures"] / "maps"
    maps_dir.mkdir(parents=True, exist_ok=True)

    hrsn_labels = {m["id"].lower(): m["name"] for m in params["hrsn_measures"]}
    outcome_labels = {m["id"].lower(): m["name"] for m in params["outcome_measures"]}

    covariate_cols = ["pct_black", "pct_hispanic", "pct_poverty",
                      "pct_college", "pct_65plus", "median_age"]

    exclude_states = spatial_params.get("exclude_states", ["02", "15"])
    n_top = gwr_params.get("top_associations", 3)
    subsample_size = gwr_params.get("subsample_size", 12000)

    # ---- Select top associations ----
    top_pairs = select_top_associations(params, n_top=n_top)

    # ---- Load and prepare data ----
    logger.info("\n=== Loading data ===")
    df = load_parquet(PATHS["processed"] / "merged_tracts.parquet")
    gdf_geo = gpd.read_file(PATHS["geo_raw"] / "us_tracts_2022.gpkg")

    gdf = gdf_geo.merge(df, left_on="GEOID", right_on="tract_fips", how="inner")
    gdf = gdf[~gdf["state_fips"].isin(exclude_states)].copy()
    gdf = gdf[gdf.geometry.notna()].copy()
    logger.info(f"CONUS tracts: {len(gdf):,}")

    # Project to Albers Equal Area
    projection = spatial_params.get("projection", "EPSG:5070")
    gdf = gdf.to_crs(projection)
    logger.info(f"Projected to {projection}")

    # Try GWR, fall back to spatial regression
    gwr_available = False
    try:
        from mgwr.gwr import GWR
        from mgwr.sel_bw import Sel_BW
        gwr_available = True
        logger.info("mgwr package available — will run GWR")
    except ImportError:
        logger.warning("mgwr not available — falling back to spatial lag/error models")

    summary_results = []

    for hrsn, outcome in top_pairs:
        logger.info(f"\n{'='*60}")
        logger.info(f"GWR: {hrsn} × {outcome}")
        logger.info(f"{'='*60}")

        if gwr_available:
            result = _run_gwr_model(
                gdf, hrsn, outcome, covariate_cols, subsample_size,
                hrsn_labels, outcome_labels, maps_dir
            )
        else:
            result = _run_spatial_regression_fallback(
                gdf, hrsn, outcome, covariate_cols,
                hrsn_labels, outcome_labels
            )

        if result:
            summary_results.append(result)

    # Save summary
    if summary_results:
        summary_df = pd.DataFrame(summary_results)
        save_csv(summary_df, PATHS["final"] / "gwr_summary.csv")

    return summary_results


def _run_gwr_model(gdf, hrsn, outcome, covariate_cols, subsample_size,
                    hrsn_labels, outcome_labels, maps_dir):
    """Run a single GWR model."""
    from mgwr.gwr import GWR
    from mgwr.sel_bw import Sel_BW

    # Subsample
    gdf_sub = subsample_stratified(gdf, subsample_size)

    # Standardize
    all_vars = [hrsn] + covariate_cols + [outcome]
    for col in all_vars:
        if col in gdf_sub.columns:
            gdf_sub[col] = standardize(gdf_sub[col])

    # Drop NaN
    mask = gdf_sub[all_vars].notna().all(axis=1)
    gdf_clean = gdf_sub[mask].copy()
    logger.info(f"Clean subsample: {len(gdf_clean):,} tracts")

    # Extract coordinates
    coords = np.column_stack([
        gdf_clean.geometry.centroid.x.values,
        gdf_clean.geometry.centroid.y.values,
    ])

    # Prepare arrays
    y = gdf_clean[outcome].values.reshape(-1, 1)
    X_cols = [hrsn] + covariate_cols
    # Add intercept column — mgwr.GWR requires explicit constant
    X = np.column_stack([np.ones(len(gdf_clean)), gdf_clean[X_cols].values])

    try:
        # Bandwidth selection
        logger.info("Selecting adaptive bandwidth (this may take several minutes)...")
        selector = Sel_BW(coords, y, X, kernel="bisquare", fixed=False, constant=False)
        bw = selector.search()
        logger.info(f"Optimal bandwidth: {bw:.0f} nearest neighbors")

        # Fit GWR
        logger.info("Fitting GWR model...")
        gwr_model = GWR(coords, y, X, bw=bw, kernel="bisquare", fixed=False, constant=False)
        gwr_results = gwr_model.fit()

        # Compare global vs local
        # Global OLS for comparison
        X_const = sm.add_constant(gdf_clean[X_cols])
        ols_res = sm.OLS(gdf_clean[outcome], X_const).fit()

        result = {
            "hrsn_measure": hrsn,
            "outcome": outcome,
            "method": "GWR",
            "n_obs": len(gdf_clean),
            "bandwidth": bw,
            "global_r2": round(ols_res.rsquared, 4),
            "gwr_r2": round(gwr_results.R2, 4),
            "global_aic": round(ols_res.aic, 1),
            "gwr_aicc": round(gwr_results.aicc, 1),
            # params[:, 0] is the intercept; params[:, 1] is the HRSN coefficient
            "hrsn_beta_mean": round(gwr_results.params[:, 1].mean(), 4),
            "hrsn_beta_std": round(gwr_results.params[:, 1].std(), 4),
            "hrsn_beta_min": round(gwr_results.params[:, 1].min(), 4),
            "hrsn_beta_max": round(gwr_results.params[:, 1].max(), 4),
        }

        logger.info(f"  Global R²: {ols_res.rsquared:.4f}")
        logger.info(f"  GWR R²: {gwr_results.R2:.4f}")
        logger.info(f"  GWR AICc: {gwr_results.aicc:.1f}")
        logger.info(f"  Local β range: [{gwr_results.params[:, 1].min():.4f}, "
                     f"{gwr_results.params[:, 1].max():.4f}]")

        # Map local coefficients (index 1 = HRSN, skipping intercept at 0)
        _map_local_coefficients(
            gdf_clean, gwr_results.params[:, 1],
            hrsn, outcome, hrsn_labels, outcome_labels,
            maps_dir, "beta"
        )

        # Map local t-statistics (index 1 = HRSN, skipping intercept at 0)
        local_t = gwr_results.params[:, 1] / gwr_results.bse[:, 1]
        _map_local_coefficients(
            gdf_clean, local_t,
            hrsn, outcome, hrsn_labels, outcome_labels,
            maps_dir, "tstat"
        )

        return result

    except Exception as e:
        logger.error(f"GWR failed: {e}")
        logger.info("Falling back to spatial regression...")
        return _run_spatial_regression_fallback(
            gdf, hrsn, outcome, covariate_cols,
            hrsn_labels, outcome_labels
        )


def _run_spatial_regression_fallback(gdf, hrsn, outcome, covariate_cols,
                                      hrsn_labels, outcome_labels):
    """Fallback: spatial lag/error model via spreg."""
    try:
        import spreg
        import libpysal
    except ImportError:
        logger.error("spreg not available — cannot run spatial regression fallback")
        return None

    logger.info("Running spatial error model (spreg fallback)...")

    # Standardize
    gdf_std = gdf.copy()
    all_vars = [hrsn] + covariate_cols + [outcome]
    for col in all_vars:
        if col in gdf_std.columns:
            gdf_std[col] = standardize(gdf[col])

    mask = gdf_std[all_vars].notna().all(axis=1)
    gdf_clean = gdf_std[mask].reset_index(drop=True)

    # Build weights for this subset
    w = libpysal.weights.Queen.from_dataframe(gdf_clean, use_index=False)
    w.transform = "R"

    y = gdf_clean[[outcome]].values
    X = gdf_clean[covariate_cols + [hrsn]].values

    try:
        # Spatial error model (ML)
        sem = spreg.ML_Error(y, X, w,
                             name_y=outcome,
                             name_x=covariate_cols + [hrsn])

        # The HRSN coefficient is the last one
        hrsn_idx = len(covariate_cols)  # 0-indexed in X
        beta_hrsn = sem.betas[hrsn_idx + 1][0]  # +1 for constant
        se_hrsn = sem.std_err[hrsn_idx + 1]
        z_hrsn = sem.z_stat[hrsn_idx + 1][0]
        p_hrsn = sem.z_stat[hrsn_idx + 1][1]
        lambda_val = sem.lam

        result = {
            "hrsn_measure": hrsn,
            "outcome": outcome,
            "method": "Spatial_Error_ML",
            "n_obs": len(gdf_clean),
            "bandwidth": np.nan,
            "global_r2": round(sem.pr2, 4),
            "gwr_r2": np.nan,
            "global_aic": round(sem.aic, 1) if hasattr(sem, 'aic') else np.nan,
            "gwr_aicc": np.nan,
            "hrsn_beta_mean": round(beta_hrsn, 4),
            "hrsn_beta_std": np.nan,
            "hrsn_beta_min": np.nan,
            "hrsn_beta_max": np.nan,
            "lambda": round(lambda_val, 4),
            "hrsn_pvalue": p_hrsn,
        }

        sig = "***" if p_hrsn < 0.001 else "**" if p_hrsn < 0.01 else "*" if p_hrsn < 0.05 else "ns"
        logger.info(f"  β({hrsn}) = {beta_hrsn:+.4f} (SE={se_hrsn:.4f}) {sig}")
        logger.info(f"  λ (spatial error) = {lambda_val:.4f}")
        logger.info(f"  Pseudo R² = {sem.pr2:.4f}")

        return result

    except Exception as e:
        logger.error(f"Spatial error model failed: {e}")
        return None


def _map_local_coefficients(gdf, values, hrsn, outcome,
                             hrsn_labels, outcome_labels, maps_dir, value_type):
    """Map local GWR coefficients or t-statistics."""
    fig, ax = plt.subplots(1, 1, figsize=(16, 10))

    gdf_plot = gdf.copy()
    gdf_plot["local_value"] = values

    if value_type == "beta":
        cmap = "RdBu_r"
        vmax = max(abs(values.min()), abs(values.max()))
        vmin = -vmax
        label = "Local β"
        title_suffix = "Local Coefficients"
    else:  # tstat
        cmap = "RdBu_r"
        vmax = max(abs(values.min()), abs(values.max()))
        vmin = -vmax
        label = "Local t-statistic"
        title_suffix = "Local t-Statistics"

    gdf_plot.plot(
        column="local_value",
        cmap=cmap,
        linewidth=0,
        ax=ax,
        legend=True,
        vmin=vmin,
        vmax=vmax,
        legend_kwds={
            "label": label,
            "orientation": "horizontal",
            "shrink": 0.6,
            "pad": 0.02,
        },
    )

    # Set CONUS extent (in projected coordinates — approximate)
    ax.set_axis_off()

    hrsn_name = hrsn_labels.get(hrsn, hrsn)
    outcome_name = outcome_labels.get(outcome, outcome)
    ax.set_title(f"GWR {title_suffix}: {hrsn_name} → {outcome_name}\n"
                 f"(N = {len(gdf_plot):,} tracts)",
                 fontsize=14, fontweight="bold")

    dest = maps_dir / f"gwr_{hrsn}_{outcome}_local_{value_type}.png"
    fig.savefig(dest, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    logger.info(f"  Saved {value_type} map to {dest}")


if __name__ == "__main__":
    print("=" * 70)
    print("PHASE C: Geographically Weighted Regression")
    print("=" * 70)
    results = run_gwr_analysis()
    print(f"\nDone. {len(results)} GWR models completed.")
