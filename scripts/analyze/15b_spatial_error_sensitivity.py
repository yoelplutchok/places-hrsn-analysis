"""
Phase C, Step 15b: Spatial Error Model Sensitivity Analysis.

Re-runs all 49 ecological regressions using spatial error models (GM_Error_Het)
as a sensitivity check. Compares OLS betas with spatial error model betas to
demonstrate that associations are robust to spatial autocorrelation.

Uses GM_Error_Het (Heteroskedastic GM estimator) rather than ML_Error because:
  - More robust to heteroskedasticity (common in tract-level data)
  - Faster than ML on large datasets (~55K tracts)
  - Does not require normality assumption

Steps:
  1. Load merged data and spatial weights
  2. Run spatial error models for all 49 HRSN x outcome pairs
  3. Compare OLS vs spatial error betas
  4. Save comparison table

Output:
  - data/final/spatial_error_results.csv   (49 rows, spatial error model results)
  - data/final/ols_vs_spatial_comparison.csv (side-by-side OLS and spatial betas)
"""
import sys
from pathlib import Path

import pandas as pd
import numpy as np
import geopandas as gpd
import statsmodels.api as sm

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


def run_spatial_error_sensitivity():
    """Run spatial error models for all 49 HRSN x outcome pairs."""
    ensure_dirs()
    params = load_params()
    spatial_params = params.get("spatial", {})

    try:
        import spreg
        import libpysal
    except ImportError:
        logger.error("spreg/libpysal not installed. Install with: pip install spreg libpysal")
        return None

    hrsn_cols = [m["id"].lower() for m in params["hrsn_measures"]]
    outcome_cols = [m["id"].lower() for m in params["outcome_measures"]]
    hrsn_labels = {m["id"].lower(): m["name"] for m in params["hrsn_measures"]}
    outcome_labels = {m["id"].lower(): m["name"] for m in params["outcome_measures"]}

    covariate_cols = ["pct_black", "pct_hispanic", "pct_poverty",
                      "pct_college", "pct_65plus", "median_age"]

    exclude_states = spatial_params.get("exclude_states", ["02", "15"])

    # ================================================================
    # Step 1: Load data and spatial weights
    # ================================================================
    logger.info("=== Step 1: Load data and build spatial weights ===")

    df = load_parquet(PATHS["processed"] / "merged_tracts.parquet")
    gdf_geo = gpd.read_file(PATHS["geo_raw"] / "us_tracts_2022.gpkg")

    gdf = gdf_geo.merge(df, left_on="GEOID", right_on="tract_fips", how="inner")
    gdf = gdf[~gdf["state_fips"].isin(exclude_states)].copy()
    gdf = gdf[gdf.geometry.notna()].copy()
    gdf = gdf.reset_index(drop=True)
    logger.info(f"CONUS tracts: {len(gdf):,}")

    # Build weights (or load if available)
    weights_path = PATHS["final"] / "spatial_weights.gal"
    if weights_path.exists():
        logger.info(f"Loading existing weights from {weights_path}")
        w = libpysal.weights.W.from_file(str(weights_path))
        # Verify same size
        if w.n != len(gdf):
            logger.warning(f"Weight size mismatch ({w.n} vs {len(gdf)}), rebuilding...")
            w = libpysal.weights.Queen.from_dataframe(gdf, use_index=False)
    else:
        logger.info("Building Queen contiguity weights...")
        w = libpysal.weights.Queen.from_dataframe(gdf, use_index=False)

    w.transform = "R"
    logger.info(f"Weights: {w.n} observations, mean {np.mean(list(w.cardinalities.values())):.1f} neighbors")

    # ================================================================
    # Step 2: Standardize and run models
    # ================================================================
    logger.info("\n=== Step 2: Running 49 Spatial Error Models ===")

    gdf_std = gdf.copy()
    for col in hrsn_cols + outcome_cols + covariate_cols:
        if col in gdf_std.columns:
            gdf_std[col] = standardize(gdf[col])

    spatial_results = []
    n_success = 0
    n_fail = 0

    for hrsn in hrsn_cols:
        for outcome in outcome_cols:
            X_cols = [hrsn] + covariate_cols
            all_cols = X_cols + [outcome]

            mask = gdf_std[all_cols].notna().all(axis=1)
            gdf_clean = gdf_std[mask].reset_index(drop=True)

            if len(gdf_clean) < 100:
                logger.warning(f"  {hrsn} x {outcome}: too few observations ({len(gdf_clean)})")
                n_fail += 1
                continue

            # Subset weights to match clean data
            clean_indices = np.where(mask.values)[0]
            try:
                w_sub = libpysal.weights.w_subset(w, clean_indices)
                w_sub.transform = "r"
            except Exception:
                # If w_subset fails, rebuild for this subset
                w_sub = libpysal.weights.Queen.from_dataframe(
                    gdf[mask].reset_index(drop=True), use_index=False
                )
                w_sub.transform = "r"

            y = gdf_clean[[outcome]].values
            X = gdf_clean[X_cols].values

            try:
                # GM_Error_Het: heteroskedasticity-robust spatial error model
                sem = spreg.GM_Error_Het(
                    y, X, w_sub,
                    name_y=outcome,
                    name_x=X_cols,
                )

                # HRSN coefficient is index 1 (0 is constant)
                beta_hrsn = sem.betas[1][0]
                se_hrsn = sem.std_err[1]
                z_hrsn = sem.z_stat[1][0]
                p_hrsn = sem.z_stat[1][1]
                lambda_val = sem.betas[-1][0]  # spatial error parameter

                row = {
                    "hrsn_measure": hrsn,
                    "hrsn_label": hrsn_labels[hrsn],
                    "outcome": outcome,
                    "outcome_label": outcome_labels[outcome],
                    "sem_beta": round(beta_hrsn, 4),
                    "sem_se": round(se_hrsn, 4),
                    "sem_z": round(z_hrsn, 4),
                    "sem_pvalue": p_hrsn,
                    "sem_significant_05": p_hrsn < 0.05,
                    "lambda": round(lambda_val, 4),
                    "pseudo_r2": round(sem.pr2, 4),
                    "n_obs": len(gdf_clean),
                }
                spatial_results.append(row)
                n_success += 1

                sig = "***" if p_hrsn < 0.001 else "**" if p_hrsn < 0.01 else "*" if p_hrsn < 0.05 else "ns"
                logger.info(f"  {hrsn:15s} x {outcome:12s}: "
                            f"beta={beta_hrsn:+.4f} (SE={se_hrsn:.4f}) {sig}  "
                            f"lambda={lambda_val:.3f}")

            except Exception as e:
                logger.warning(f"  {hrsn} x {outcome}: FAILED - {e}")
                n_fail += 1

    logger.info(f"\nCompleted: {n_success} succeeded, {n_fail} failed")

    spatial_df = pd.DataFrame(spatial_results)
    save_csv(spatial_df, PATHS["final"] / "spatial_error_results.csv")

    # ================================================================
    # Step 3: Compare OLS vs Spatial Error betas
    # ================================================================
    logger.info("\n=== Step 3: OLS vs Spatial Error Comparison ===")

    ols_path = PATHS["final"] / "results_matrix.csv"
    if ols_path.exists() and len(spatial_df) > 0:
        ols_df = pd.read_csv(ols_path)

        comparison = ols_df[["hrsn_measure", "outcome", "beta_std", "se", "pvalue",
                             "significant_05"]].rename(columns={
            "beta_std": "ols_beta",
            "se": "ols_se",
            "pvalue": "ols_pvalue",
            "significant_05": "ols_significant",
        })

        comparison = comparison.merge(
            spatial_df[["hrsn_measure", "outcome", "sem_beta", "sem_se",
                        "sem_pvalue", "sem_significant_05", "lambda"]],
            on=["hrsn_measure", "outcome"],
            how="inner"
        )

        # Compute attenuation and direction consistency
        comparison["beta_change_pct"] = np.where(
            comparison["ols_beta"].abs() > 0.001,
            ((comparison["sem_beta"] - comparison["ols_beta"]) / comparison["ols_beta"].abs()) * 100,
            np.nan
        )
        comparison["same_direction"] = (
            np.sign(comparison["ols_beta"]) == np.sign(comparison["sem_beta"])
        )
        comparison["both_significant"] = (
            comparison["ols_significant"] & comparison["sem_significant_05"]
        )

        save_csv(comparison, PATHS["final"] / "ols_vs_spatial_comparison.csv")

        # Summary statistics
        n_same_dir = comparison["same_direction"].sum()
        n_both_sig = comparison["both_significant"].sum()
        mean_change = comparison["beta_change_pct"].mean()
        mean_lambda = comparison["lambda"].mean()

        logger.info(f"\n--- Comparison Summary ({len(comparison)} models) ---")
        logger.info(f"Same direction: {n_same_dir}/{len(comparison)} ({n_same_dir/len(comparison)*100:.0f}%)")
        logger.info(f"Both significant (p<.05): {n_both_sig}/{len(comparison)} ({n_both_sig/len(comparison)*100:.0f}%)")
        logger.info(f"Mean beta change: {mean_change:+.1f}%")
        logger.info(f"Mean lambda: {mean_lambda:.4f}")

        # Correlation between OLS and SEM betas
        corr = comparison["ols_beta"].corr(comparison["sem_beta"])
        logger.info(f"Correlation (OLS beta vs SEM beta): {corr:.4f}")

    return spatial_df


if __name__ == "__main__":
    print("=" * 70)
    print("PHASE C: Spatial Error Model Sensitivity Analysis")
    print("=" * 70)
    results = run_spatial_error_sensitivity()
    if results is not None:
        print(f"\nDone. {len(results)} spatial error models completed.")
    else:
        print("\nFailed — check logs for details.")
