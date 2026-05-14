"""
Path 4: Incremental Validity — Does HRSN Add Predictive Value Beyond SVI?

Compares three model specifications for each of 7 disease outcomes:
  Model A: disease ~ SVI themes + demographics
  Model B: disease ~ HRSN measures + demographics
  Model C: disease ~ SVI themes + HRSN measures + demographics

Tests whether HRSN measures add incremental R² beyond SVI, and vice versa.
Uses cluster-robust Wald tests for formal inference (with naive F-tests for reference).

Output:
  - data/final/svi_comparison_results.csv
  - data/final/svi_incremental_validity.csv
  - outputs/figures/svi_vs_hrsn_r2_comparison.png
"""
import sys
from pathlib import Path

import pandas as pd
import numpy as np
import statsmodels.api as sm
from scipy import stats

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from hrsn_analysis.paths import PATHS, ensure_dirs
from hrsn_analysis.io_utils import load_parquet, load_params, save_csv, load_csv
from hrsn_analysis.logging_utils import setup_logging, get_logger

setup_logging()
logger = get_logger(__name__)


def standardize(series):
    return (series - series.mean()) / series.std()


def partial_f_test(r2_restricted, r2_full, p_restricted, p_full, n):
    """Partial F-test for nested models.

    NOTE: This test assumes i.i.d. errors. When models use clustered SEs,
    p-values will be anti-conservative. Use as a rough guide alongside
    AIC and adjusted R² for model comparison.
    """
    df1 = p_full - p_restricted  # additional parameters
    df2 = n - p_full - 1
    if df1 <= 0 or df2 <= 0 or r2_full <= r2_restricted:
        return np.nan, np.nan
    f_stat = ((r2_full - r2_restricted) / df1) / ((1 - r2_full) / df2)
    p_value = 1 - stats.f.cdf(f_stat, df1, df2)
    return f_stat, p_value


def run_svi_comparison():
    """Compare SVI vs HRSN predictive models."""
    ensure_dirs()
    params = load_params()

    hrsn_cols = [m["id"].lower() for m in params["hrsn_measures"]]
    outcome_cols = [m["id"].lower() for m in params["outcome_measures"]]
    hrsn_labels = {m["id"].lower(): m["name"] for m in params["hrsn_measures"]}
    outcome_labels = {m["id"].lower(): m["name"] for m in params["outcome_measures"]}

    demo_cols = ["pct_black", "pct_hispanic", "pct_poverty",
                 "pct_college", "pct_65plus", "median_age"]

    # SVI theme columns (percentile rankings)
    svi_theme_cols = ["RPL_THEME1", "RPL_THEME2", "RPL_THEME3", "RPL_THEME4"]

    # ================================================================
    # Load data
    # ================================================================
    logger.info("=== Loading data ===")
    df = load_parquet(PATHS["processed"] / "merged_tracts.parquet")

    # Load SVI
    svi_path = PATHS["raw"] / "svi" / "SVI_2022_US.csv"
    if not svi_path.exists():
        logger.error(f"SVI data not found at {svi_path}")
        logger.error("Run: python scripts/collect/17_download_svi.py")
        return None

    svi = pd.read_csv(svi_path, dtype={"FIPS": str})
    logger.info(f"SVI: {len(svi):,} tracts")

    # SVI uses -999 for missing values
    for col in svi_theme_cols:
        if col in svi.columns:
            svi.loc[svi[col] < 0, col] = np.nan

    # Merge
    merged = df.merge(svi[["FIPS"] + svi_theme_cols],
                      left_on="tract_fips", right_on="FIPS", how="inner")
    logger.info(f"Merged: {len(merged):,} tracts with both HRSN and SVI")

    # Standardize all predictors
    for col in hrsn_cols + demo_cols + svi_theme_cols:
        if col in merged.columns:
            merged[col] = standardize(merged[col])
    for col in outcome_cols:
        if col in merged.columns:
            merged[col] = standardize(merged[col])

    cluster_var = merged["county_fips"]

    # ================================================================
    # Run three model specifications per outcome
    # ================================================================
    logger.info("\n=== Model Comparison: SVI vs HRSN ===")

    all_results = []
    incremental_results = []

    for outcome in outcome_cols:
        logger.info(f"\n--- {outcome_labels.get(outcome, outcome)} ---")

        y = merged[outcome]

        # Shared mask for all three models
        all_predictors = svi_theme_cols + hrsn_cols + demo_cols + [outcome]
        mask = merged[all_predictors].notna().all(axis=1)
        y_clean = y[mask]
        cluster_clean = cluster_var[mask]
        n = int(mask.sum())

        # Model A: SVI themes + demographics
        X_a = sm.add_constant(merged.loc[mask, svi_theme_cols + demo_cols])
        try:
            res_a = sm.OLS(y_clean, X_a).fit(
                cov_type="cluster", cov_kwds={"groups": cluster_clean})
            r2_a = res_a.rsquared
            adj_r2_a = res_a.rsquared_adj
            aic_a = res_a.aic
        except Exception as e:
            logger.warning(f"  Model A (SVI) failed: {e}")
            continue

        # Model B: HRSN + demographics
        X_b = sm.add_constant(merged.loc[mask, hrsn_cols + demo_cols])
        try:
            res_b = sm.OLS(y_clean, X_b).fit(
                cov_type="cluster", cov_kwds={"groups": cluster_clean})
            r2_b = res_b.rsquared
            adj_r2_b = res_b.rsquared_adj
            aic_b = res_b.aic
        except Exception as e:
            logger.warning(f"  Model B (HRSN) failed: {e}")
            continue

        # Model C: SVI + HRSN + demographics
        X_c = sm.add_constant(merged.loc[mask, svi_theme_cols + hrsn_cols + demo_cols])
        try:
            res_c = sm.OLS(y_clean, X_c).fit(
                cov_type="cluster", cov_kwds={"groups": cluster_clean})
            r2_c = res_c.rsquared
            adj_r2_c = res_c.rsquared_adj
            aic_c = res_c.aic
        except Exception as e:
            logger.warning(f"  Model C (Combined) failed: {e}")
            continue

        # Partial F-tests (naive, assumes i.i.d. — see caveat in partial_f_test)
        f_hrsn, p_hrsn = partial_f_test(
            r2_a, r2_c, X_a.shape[1] - 1, X_c.shape[1] - 1, n)
        f_svi, p_svi = partial_f_test(
            r2_b, r2_c, X_b.shape[1] - 1, X_c.shape[1] - 1, n)

        # Cluster-robust Wald test (preferred with clustered SEs)
        # Tests H0: all HRSN coefficients = 0 (in combined model)
        try:
            hrsn_idx = [X_c.columns.get_loc(h) for h in hrsn_cols if h in X_c.columns]
            R_hrsn = np.zeros((len(hrsn_idx), X_c.shape[1]))
            for i, idx in enumerate(hrsn_idx):
                R_hrsn[i, idx] = 1
            wald_hrsn = res_c.wald_test(R_hrsn, use_f=True)
            p_hrsn_wald = float(wald_hrsn.pvalue)

            svi_idx = [X_c.columns.get_loc(s) for s in svi_theme_cols if s in X_c.columns]
            R_svi = np.zeros((len(svi_idx), X_c.shape[1]))
            for i, idx in enumerate(svi_idx):
                R_svi[i, idx] = 1
            wald_svi = res_c.wald_test(R_svi, use_f=True)
            p_svi_wald = float(wald_svi.pvalue)
        except Exception:
            p_hrsn_wald = np.nan
            p_svi_wald = np.nan

        delta_r2_hrsn = r2_c - r2_a  # R² gained by adding HRSN to SVI
        delta_r2_svi = r2_c - r2_b   # R² gained by adding SVI to HRSN

        logger.info(f"  Model A (SVI only):     R²={r2_a:.4f}  AIC={aic_a:.0f}")
        logger.info(f"  Model B (HRSN only):    R²={r2_b:.4f}  AIC={aic_b:.0f}")
        logger.info(f"  Model C (SVI + HRSN):   R²={r2_c:.4f}  AIC={aic_c:.0f}")
        logger.info(f"  ΔR² adding HRSN to SVI: {delta_r2_hrsn:+.4f} (F={f_hrsn:.1f}, p={p_hrsn:.2e})")
        logger.info(f"  ΔR² adding SVI to HRSN: {delta_r2_svi:+.4f} (F={f_svi:.1f}, p={p_svi:.2e})")

        for model_name, r2, adj_r2, aic, res in [
            ("SVI_only", r2_a, adj_r2_a, aic_a, res_a),
            ("HRSN_only", r2_b, adj_r2_b, aic_b, res_b),
            ("Combined", r2_c, adj_r2_c, aic_c, res_c),
        ]:
            all_results.append({
                "outcome": outcome,
                "outcome_label": outcome_labels.get(outcome, outcome),
                "model": model_name,
                "r_squared": round(r2, 4),
                "adj_r_squared": round(adj_r2, 4),
                "aic": round(aic, 1),
                "n_predictors": res.df_model,
                "n_obs": n,
            })

        delta_adj_r2_hrsn = adj_r2_c - adj_r2_a
        delta_adj_r2_svi = adj_r2_c - adj_r2_b

        incremental_results.append({
            "outcome": outcome,
            "outcome_label": outcome_labels.get(outcome, outcome),
            "r2_svi_only": round(r2_a, 4),
            "r2_hrsn_only": round(r2_b, 4),
            "r2_combined": round(r2_c, 4),
            "adj_r2_svi_only": round(adj_r2_a, 4),
            "adj_r2_hrsn_only": round(adj_r2_b, 4),
            "adj_r2_combined": round(adj_r2_c, 4),
            "delta_r2_hrsn_over_svi": round(delta_r2_hrsn, 4),
            "delta_r2_svi_over_hrsn": round(delta_r2_svi, 4),
            "delta_adj_r2_hrsn_over_svi": round(delta_adj_r2_hrsn, 4),
            "delta_adj_r2_svi_over_hrsn": round(delta_adj_r2_svi, 4),
            "n_predictors_svi": X_a.shape[1] - 1,
            "n_predictors_hrsn": X_b.shape[1] - 1,
            "n_predictors_combined": X_c.shape[1] - 1,
            "f_stat_hrsn_naive": round(f_hrsn, 2) if not np.isnan(f_hrsn) else np.nan,
            "p_value_hrsn_naive": p_hrsn,
            "f_stat_svi_naive": round(f_svi, 2) if not np.isnan(f_svi) else np.nan,
            "p_value_svi_naive": p_svi,
            "p_value_hrsn_wald": p_hrsn_wald,
            "p_value_svi_wald": p_svi_wald,
            "hrsn_significant": p_hrsn_wald < 0.05 if not np.isnan(p_hrsn_wald) else (p_hrsn < 0.05 if not np.isnan(p_hrsn) else False),
            "svi_significant": p_svi_wald < 0.05 if not np.isnan(p_svi_wald) else (p_svi < 0.05 if not np.isnan(p_svi) else False),
            "aic_svi": round(aic_a, 1),
            "aic_hrsn": round(aic_b, 1),
            "aic_combined": round(aic_c, 1),
            "best_model_aic": min(
                [("SVI", aic_a), ("HRSN", aic_b), ("Combined", aic_c)],
                key=lambda x: x[1]
            )[0],
        })

    # Save results
    results_df = pd.DataFrame(all_results)
    save_csv(results_df, PATHS["final"] / "svi_comparison_results.csv")

    incr_df = pd.DataFrame(incremental_results)
    save_csv(incr_df, PATHS["final"] / "svi_incremental_validity.csv")

    # ================================================================
    # Visualization: R² comparison bar chart
    # ================================================================
    logger.info("\n=== Creating comparison figure ===")

    if len(incr_df) > 0:
        fig, axes = plt.subplots(1, 2, figsize=(16, 7))

        # Panel A: R² by model type
        ax = axes[0]
        x = np.arange(len(incr_df))
        width = 0.25
        ax.bar(x - width, incr_df["r2_svi_only"], width, label="SVI Only",
               color="#3498db", alpha=0.85)
        ax.bar(x, incr_df["r2_hrsn_only"], width, label="HRSN Only",
               color="#e74c3c", alpha=0.85)
        ax.bar(x + width, incr_df["r2_combined"], width, label="SVI + HRSN",
               color="#2ecc71", alpha=0.85)
        ax.set_xticks(x)
        ax.set_xticklabels([outcome_labels.get(o, o) for o in incr_df["outcome"]],
                           rotation=45, ha="right", fontsize=9)
        ax.set_ylabel("R²", fontsize=12)
        ax.set_title("A) Model R² Comparison", fontsize=13, fontweight="bold")
        ax.legend(fontsize=9)
        ax.set_ylim(0, 1)

        # Panel B: Incremental R²
        ax = axes[1]
        ax.bar(x - width/2, incr_df["delta_r2_hrsn_over_svi"], width,
               label="ΔR² adding HRSN to SVI", color="#e74c3c", alpha=0.85)
        ax.bar(x + width/2, incr_df["delta_r2_svi_over_hrsn"], width,
               label="ΔR² adding SVI to HRSN", color="#3498db", alpha=0.85)
        ax.set_xticks(x)
        ax.set_xticklabels([outcome_labels.get(o, o) for o in incr_df["outcome"]],
                           rotation=45, ha="right", fontsize=9)
        ax.set_ylabel("Incremental R²", fontsize=12)
        ax.set_title("B) Incremental Validity", fontsize=13, fontweight="bold")
        ax.legend(fontsize=9)
        ax.axhline(y=0, color="black", linewidth=0.5)

        # Add significance stars
        for i, row in incr_df.iterrows():
            if row.get("hrsn_significant", False):
                ax.text(i - width/2, row["delta_r2_hrsn_over_svi"] + 0.002,
                        "*", ha="center", fontsize=12, fontweight="bold")
            if row.get("svi_significant", False):
                ax.text(i + width/2, row["delta_r2_svi_over_hrsn"] + 0.002,
                        "*", ha="center", fontsize=12, fontweight="bold")

        plt.tight_layout()
        dest = PATHS["figures"] / "svi_vs_hrsn_r2_comparison.png"
        fig.savefig(dest, dpi=300, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        logger.info(f"Saved comparison figure to {dest}")

    # Summary
    logger.info("\n=== Summary ===")
    if len(incr_df) > 0:
        n_hrsn_adds = incr_df["hrsn_significant"].sum()
        n_svi_adds = incr_df["svi_significant"].sum()
        mean_delta_hrsn = incr_df["delta_r2_hrsn_over_svi"].mean()
        mean_delta_svi = incr_df["delta_r2_svi_over_hrsn"].mean()
        logger.info(f"HRSN adds significant R² beyond SVI: {n_hrsn_adds}/{len(incr_df)} outcomes")
        logger.info(f"SVI adds significant R² beyond HRSN: {n_svi_adds}/{len(incr_df)} outcomes")
        logger.info(f"Mean ΔR² from HRSN (over SVI): {mean_delta_hrsn:+.4f}")
        logger.info(f"Mean ΔR² from SVI (over HRSN): {mean_delta_svi:+.4f}")

        best_counts = incr_df["best_model_aic"].value_counts()
        logger.info(f"Best model by AIC: {dict(best_counts)}")

    return results_df, incr_df


if __name__ == "__main__":
    print("=" * 70)
    print("PATH 4: SVI vs HRSN — Incremental Validity Comparison")
    print("=" * 70)
    result = run_svi_comparison()
    if result:
        res_df, incr_df = result
        print(f"\nDone. {len(res_df)} model results, {len(incr_df)} outcomes compared.")
