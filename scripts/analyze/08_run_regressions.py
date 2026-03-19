"""
Phase 4, Steps 4.1–4.2: Run individually-adjusted regressions.

Model specification (for each HRSN × Disease pair):
  Disease_prev = β₀ + β₁(HRSN_measure) + β₂(pct_black) + β₃(pct_hispanic)
                + β₄(pct_poverty) + β₅(pct_college) + β₆(pct_65plus)
                + β₇(median_age) + ε

- All continuous variables z-score standardized → β₁ is comparable across models
- Clustered standard errors by county FIPS
- 7 HRSN × 7 outcomes = 49 models

Output:
  - data/final/results_matrix.csv (49 rows, full results)
  - data/final/results_heatmap.csv (7×7 pivoted matrix of standardized betas)
"""
import sys
from pathlib import Path

import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.stats.multitest import multipletests

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from hrsn_analysis.paths import PATHS, ensure_dirs
from hrsn_analysis.io_utils import load_parquet, load_params, save_csv
from hrsn_analysis.logging_utils import setup_logging, get_logger

setup_logging()
logger = get_logger(__name__)


def standardize(series):
    """Z-score standardize a pandas Series."""
    return (series - series.mean()) / series.std()


def run_individual_regressions():
    """Run all 49 individually-adjusted HRSN × Disease regressions."""
    ensure_dirs()
    params = load_params()

    # ---- Load data ----
    df = load_parquet(PATHS["processed"] / "merged_tracts.parquet")

    hrsn_cols = [m["id"].lower() for m in params["hrsn_measures"]]
    outcome_cols = [m["id"].lower() for m in params["outcome_measures"]]
    hrsn_labels = {m["id"].lower(): m["name"] for m in params["hrsn_measures"]}
    outcome_labels = {m["id"].lower(): m["name"] for m in params["outcome_measures"]}

    # Covariates: demographics to control for
    # Note: Using pct_poverty but NOT median_income simultaneously (collinear)
    # Using pct_college but NOT pct_no_hs simultaneously (collinear)
    covariate_cols = ["pct_black", "pct_hispanic", "pct_poverty",
                      "pct_college", "pct_65plus", "median_age"]

    cluster_var = df["county_fips"]

    logger.info(f"Running 49 individually-adjusted regressions")
    logger.info(f"  HRSN measures: {hrsn_cols}")
    logger.info(f"  Outcomes: {outcome_cols}")
    logger.info(f"  Covariates: {covariate_cols}")
    logger.info(f"  Clustering: county_fips ({df['county_fips'].nunique():,} clusters)")
    logger.info(f"  N observations: {len(df):,}")

    # ---- Standardize all variables ----
    df_std = df.copy()
    all_vars = hrsn_cols + outcome_cols + covariate_cols
    for col in all_vars:
        df_std[col] = standardize(df[col])

    # ---- Also run on unstandardized data for raw betas ----
    # We'll run both standardized and unstandardized in the same loop.

    # ---- Run regressions ----
    results = []
    for hrsn in hrsn_cols:
        for outcome in outcome_cols:
            # Build X matrix: HRSN + covariates (standardized)
            X_cols = [hrsn] + covariate_cols
            X = sm.add_constant(df_std[X_cols])
            y = df_std[outcome]

            # Drop any remaining NaN (shouldn't be any, but safety)
            mask = X.notna().all(axis=1) & y.notna()
            X_clean = X[mask]
            y_clean = y[mask]
            cluster_clean = cluster_var[mask]

            # Run OLS with clustered SEs (standardized)
            model = sm.OLS(y_clean, X_clean)
            try:
                res = model.fit(cov_type='cluster', cov_kwds={'groups': cluster_clean})

                beta = res.params[hrsn]
                se = res.bse[hrsn]
                pval = res.pvalues[hrsn]
                ci = res.conf_int().loc[hrsn]

                # Also run unstandardized for raw (original-unit) betas
                X_raw = sm.add_constant(df[X_cols].loc[mask])
                y_raw = df[outcome].loc[mask]
                res_raw = sm.OLS(y_raw, X_raw).fit(
                    cov_type='cluster', cov_kwds={'groups': cluster_clean})
                beta_raw = res_raw.params[hrsn]
                se_raw = res_raw.bse[hrsn]
                ci_raw = res_raw.conf_int().loc[hrsn]

                row = {
                    'hrsn_measure': hrsn,
                    'hrsn_label': hrsn_labels[hrsn],
                    'outcome': outcome,
                    'outcome_label': outcome_labels[outcome],
                    'beta_std': round(beta, 4),
                    'se_std': round(se, 4),
                    'beta_unstd': round(beta_raw, 6),
                    'se_unstd': round(se_raw, 6),
                    'ci_lower_unstd': round(ci_raw[0], 6),
                    'ci_upper_unstd': round(ci_raw[1], 6),
                    'pvalue': pval,
                    'ci_lower': round(ci[0], 4),
                    'ci_upper': round(ci[1], 4),
                    'r_squared': round(res.rsquared, 4),
                    'adj_r_squared': round(res.rsquared_adj, 4),
                    'n_obs': int(res.nobs),
                    'n_clusters': cluster_clean.nunique(),
                    'significant_05': pval < 0.05,
                    'significant_01': pval < 0.01,
                    'significant_001': pval < 0.001,
                }
                results.append(row)

                sig = "***" if pval < 0.001 else "**" if pval < 0.01 else "*" if pval < 0.05 else "ns"
                logger.info(f"  {hrsn:15s} × {outcome:12s}: β={beta:+.4f} (SE={se:.4f}) {sig}  R²={res.rsquared:.4f}")

            except Exception as e:
                logger.error(f"  FAILED: {hrsn} × {outcome}: {e}")
                results.append({
                    'hrsn_measure': hrsn, 'hrsn_label': hrsn_labels[hrsn],
                    'outcome': outcome, 'outcome_label': outcome_labels[outcome],
                    'beta_std': np.nan, 'se_std': np.nan,
                    'beta_unstd': np.nan, 'se_unstd': np.nan,
                    'ci_lower_unstd': np.nan, 'ci_upper_unstd': np.nan,
                    'pvalue': np.nan,
                    'ci_lower': np.nan, 'ci_upper': np.nan,
                    'r_squared': np.nan, 'adj_r_squared': np.nan,
                    'n_obs': 0, 'n_clusters': 0,
                    'significant_05': False, 'significant_01': False,
                    'significant_001': False,
                })

    # ---- FDR correction (Benjamini-Hochberg) ----
    results_df = pd.DataFrame(results)
    valid_pvals = results_df['pvalue'].dropna()
    if len(valid_pvals) > 0:
        reject, qvals, _, _ = multipletests(valid_pvals.values, method='fdr_bh')
        results_df.loc[valid_pvals.index, 'pvalue_fdr'] = qvals
        results_df.loc[valid_pvals.index, 'significant_fdr_05'] = reject
        results_df['pvalue_fdr'] = results_df['pvalue_fdr'].where(
            results_df['pvalue'].notna(), np.nan
        )
        results_df['significant_fdr_05'] = results_df['significant_fdr_05'].fillna(False)
    else:
        results_df['pvalue_fdr'] = np.nan
        results_df['significant_fdr_05'] = False

    n_sig_fdr = results_df['significant_fdr_05'].sum()
    logger.info(f"\n--- FDR Correction (Benjamini-Hochberg) ---")
    logger.info(f"Significant after FDR (q<0.05): {n_sig_fdr}/{len(results_df)}")

    # ---- Save full results ----
    dest = PATHS["final"] / "results_matrix.csv"
    dest.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(dest, index=False)
    logger.info(f"\nFull results saved to {dest} ({len(results_df)} models)")

    # ---- Create pivoted heatmap matrix ----
    heatmap = results_df.pivot(index='hrsn_measure', columns='outcome', values='beta_std')
    # Reorder to match params order
    heatmap = heatmap.reindex(index=hrsn_cols, columns=outcome_cols)
    heatmap_dest = PATHS["final"] / "results_heatmap.csv"
    heatmap.to_csv(heatmap_dest)
    logger.info(f"Heatmap matrix saved to {heatmap_dest}")

    # ---- Summary ----
    n_sig = results_df['significant_05'].sum()
    n_total = len(results_df)
    logger.info(f"\n--- Summary ---")
    logger.info(f"Total models: {n_total}")
    logger.info(f"Significant (p<0.05): {n_sig} ({n_sig/n_total*100:.0f}%)")
    logger.info(f"Significant (p<0.001): {results_df['significant_001'].sum()}")
    logger.info(f"Significant after FDR (q<0.05): {n_sig_fdr} ({n_sig_fdr/n_total*100:.0f}%)")

    # Strongest associations
    top5 = results_df.reindex(results_df['beta_std'].abs().sort_values(ascending=False).index).head(5)
    logger.info(f"\nTop 5 strongest associations (|β|):")
    for _, r in top5.iterrows():
        logger.info(f"  {r['hrsn_label']} × {r['outcome_label']}: β={r['beta_std']:+.4f} (p={r['pvalue']:.2e})")

    return results_df


def run_wls_sensitivity(df, df_std):
    """C4: Population-weighted WLS sensitivity analysis.

    Re-runs the 49 HRSN × disease regressions using WLS with total_population
    as weights. This gives more influence to tracts where PLACES prevalence
    estimates are more precisely measured (larger populations → smaller SAE SE).

    See docs/pipeline_rerun_decisions.md, Decision 1 for full rationale.
    """
    logger.info("\n" + "=" * 70)
    logger.info("SENSITIVITY: Population-Weighted WLS Regressions")
    logger.info("=" * 70)

    params = load_params()
    hrsn_cols = [m["id"].lower() for m in params["hrsn_measures"]]
    outcome_cols = [m["id"].lower() for m in params["outcome_measures"]]
    hrsn_labels = {m["id"].lower(): m["name"] for m in params["hrsn_measures"]}
    outcome_labels = {m["id"].lower(): m["name"] for m in params["outcome_measures"]}
    covariate_cols = ["pct_black", "pct_hispanic", "pct_poverty",
                      "pct_college", "pct_65plus", "median_age"]
    cluster_var = df["county_fips"]
    weights = df["total_population"]

    results = []
    for hrsn in hrsn_cols:
        for outcome in outcome_cols:
            X_cols = [hrsn] + covariate_cols
            X = sm.add_constant(df_std[X_cols])
            y = df_std[outcome]

            mask = X.notna().all(axis=1) & y.notna() & weights.notna() & (weights > 0)
            X_clean = X[mask]
            y_clean = y[mask]
            cluster_clean = cluster_var[mask]
            w_clean = weights[mask]

            try:
                model = sm.WLS(y_clean, X_clean, weights=w_clean)
                res = model.fit(cov_type='cluster', cov_kwds={'groups': cluster_clean})

                beta = res.params[hrsn]
                pval = res.pvalues[hrsn]

                results.append({
                    'hrsn_measure': hrsn,
                    'hrsn_label': hrsn_labels[hrsn],
                    'outcome': outcome,
                    'outcome_label': outcome_labels[outcome],
                    'beta_std': round(beta, 4),
                    'se_std': round(res.bse[hrsn], 4),
                    'pvalue': pval,
                    'r_squared': round(res.rsquared, 4),
                    'n_obs': int(res.nobs),
                    'significant_05': pval < 0.05,
                })

                sig = "***" if pval < 0.001 else "**" if pval < 0.01 else "*" if pval < 0.05 else "ns"
                logger.info(f"  {hrsn:15s} × {outcome:12s}: β={beta:+.4f} {sig}")

            except Exception as e:
                logger.warning(f"  WLS FAILED: {hrsn} × {outcome}: {e}")

    wls_df = pd.DataFrame(results)
    dest = PATHS["final"] / "results_matrix_wls.csv"
    save_csv(wls_df, dest)
    logger.info(f"\nWLS results saved to {dest} ({len(wls_df)} models)")

    # Compare with OLS
    ols_path = PATHS["final"] / "results_matrix.csv"
    if ols_path.exists():
        ols_df = pd.read_csv(ols_path)
        merged = ols_df.merge(wls_df, on=["hrsn_measure", "outcome"],
                              suffixes=("_ols", "_wls"))
        if len(merged) > 0:
            corr = merged["beta_std_ols"].corr(merged["beta_std_wls"])
            mean_diff = (merged["beta_std_wls"] - merged["beta_std_ols"]).mean()
            logger.info(f"\nOLS vs WLS comparison:")
            logger.info(f"  Beta correlation: r = {corr:.4f}")
            logger.info(f"  Mean beta difference (WLS - OLS): {mean_diff:+.4f}")
            n_sign_change = (
                (merged["beta_std_ols"] > 0) != (merged["beta_std_wls"] > 0)
            ).sum()
            logger.info(f"  Sign changes: {n_sign_change}/{len(merged)}")

    return wls_df


def run_sae_sensitivity(df, df_std):
    """C17: Sensitivity analysis dropping SAE-input covariates.

    PLACES estimates are generated via MRP using individual-level demographics
    (age, sex, race, education) and poststratified to tracts using ACS
    demographic composition. Our covariates pct_college, pct_65plus, and
    median_age directly overlap with PLACES MRP poststratification variables.

    This sensitivity re-runs the 49 regressions with ONLY:
      [pct_black, pct_hispanic, pct_poverty]
    Dropping: pct_college, pct_65plus, median_age.

    See docs/pipeline_rerun_decisions.md, Decision 2 for full rationale.
    """
    logger.info("\n" + "=" * 70)
    logger.info("SENSITIVITY: Dropping SAE-Input Covariates")
    logger.info("=" * 70)
    logger.info("Retained covariates: pct_black, pct_hispanic, pct_poverty")
    logger.info("Dropped covariates: pct_college, pct_65plus, median_age")

    params = load_params()
    hrsn_cols = [m["id"].lower() for m in params["hrsn_measures"]]
    outcome_cols = [m["id"].lower() for m in params["outcome_measures"]]
    hrsn_labels = {m["id"].lower(): m["name"] for m in params["hrsn_measures"]}
    outcome_labels = {m["id"].lower(): m["name"] for m in params["outcome_measures"]}
    reduced_covariates = ["pct_black", "pct_hispanic", "pct_poverty"]
    cluster_var = df["county_fips"]

    results = []
    for hrsn in hrsn_cols:
        for outcome in outcome_cols:
            X_cols = [hrsn] + reduced_covariates
            X = sm.add_constant(df_std[X_cols])
            y = df_std[outcome]

            mask = X.notna().all(axis=1) & y.notna()
            X_clean = X[mask]
            y_clean = y[mask]
            cluster_clean = cluster_var[mask]

            try:
                model = sm.OLS(y_clean, X_clean)
                res = model.fit(cov_type='cluster', cov_kwds={'groups': cluster_clean})

                beta = res.params[hrsn]
                pval = res.pvalues[hrsn]

                results.append({
                    'hrsn_measure': hrsn,
                    'hrsn_label': hrsn_labels[hrsn],
                    'outcome': outcome,
                    'outcome_label': outcome_labels[outcome],
                    'beta_std': round(beta, 4),
                    'se_std': round(res.bse[hrsn], 4),
                    'pvalue': pval,
                    'r_squared': round(res.rsquared, 4),
                    'n_obs': int(res.nobs),
                    'significant_05': pval < 0.05,
                    'covariates_used': ", ".join(reduced_covariates),
                })

                sig = "***" if pval < 0.001 else "**" if pval < 0.01 else "*" if pval < 0.05 else "ns"
                logger.info(f"  {hrsn:15s} × {outcome:12s}: β={beta:+.4f} {sig}")

            except Exception as e:
                logger.warning(f"  SAE-sensitivity FAILED: {hrsn} × {outcome}: {e}")

    sae_df = pd.DataFrame(results)
    dest = PATHS["final"] / "results_matrix_sae_sensitivity.csv"
    save_csv(sae_df, dest)
    logger.info(f"\nSAE sensitivity results saved to {dest} ({len(sae_df)} models)")

    # Compare with primary OLS
    ols_path = PATHS["final"] / "results_matrix.csv"
    if ols_path.exists():
        ols_df = pd.read_csv(ols_path)
        merged = ols_df.merge(sae_df, on=["hrsn_measure", "outcome"],
                              suffixes=("_primary", "_sae"))
        if len(merged) > 0:
            corr = merged["beta_std_primary"].corr(merged["beta_std_sae"])
            mean_diff = (merged["beta_std_sae"] - merged["beta_std_primary"]).mean()
            logger.info(f"\nPrimary vs SAE-sensitivity comparison:")
            logger.info(f"  Beta correlation: r = {corr:.4f}")
            logger.info(f"  Mean beta difference (SAE - Primary): {mean_diff:+.4f}")
            n_sign_change = (
                (merged["beta_std_primary"] > 0) != (merged["beta_std_sae"] > 0)
            ).sum()
            logger.info(f"  Sign changes: {n_sign_change}/{len(merged)}")

    return sae_df


if __name__ == "__main__":
    print("=" * 70)
    print("PHASE 4, Steps 4.1-4.2: Individually-Adjusted Regressions")
    print("=" * 70)
    results = run_individual_regressions()
    print(f"\nDone. {len(results)} models completed.")

    # Sensitivity analyses need the raw and standardized DataFrames
    from hrsn_analysis.io_utils import load_parquet, load_params as _lp
    _params = _lp()
    _df = load_parquet(PATHS["processed"] / "merged_tracts.parquet")
    _df_std = _df.copy()
    _all_vars = ([m["id"].lower() for m in _params["hrsn_measures"]]
                 + [m["id"].lower() for m in _params["outcome_measures"]]
                 + ["pct_black", "pct_hispanic", "pct_poverty",
                    "pct_college", "pct_65plus", "median_age"])
    for _col in _all_vars:
        if _col in _df_std.columns:
            _df_std[_col] = standardize(_df[_col])

    print("\n--- WLS Sensitivity ---")
    wls = run_wls_sensitivity(_df, _df_std)
    print(f"WLS: {len(wls)} models completed.")

    print("\n--- SAE Covariate Sensitivity ---")
    sae = run_sae_sensitivity(_df, _df_std)
    print(f"SAE: {len(sae)} models completed.")
