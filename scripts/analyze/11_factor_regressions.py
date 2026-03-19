"""
Phase A, Step 11: Re-run regressions with 2 factor scores instead of 7 HRSN measures.

Mirrors scripts 08/09 exactly but uses Material Hardship + Social Isolation
composite scores. Key expected outcome: VIF drops from ~97 to ~2-3, sign
reversals eliminated.

Model specification (individually-adjusted, 14 models = 2 factors × 7 outcomes):
  Disease_prev = β₀ + β₁(factor_score) + β₂(pct_black) + β₃(pct_hispanic)
                + β₄(pct_poverty) + β₅(pct_college) + β₆(pct_65plus)
                + β₇(median_age) + ε

Model specification (mutually-adjusted, 7 models):
  Disease_prev = β₀ + β₁(material_hardship) + β₂(social_isolation)
                + [covariates] + ε

Output:
  - data/final/factor_regression_results.csv   (14 individually-adjusted)
  - data/final/factor_mutually_adjusted_results.csv (7 mutually-adjusted)
  - data/final/factor_vif_table.csv
  - outputs/tables/model_comparison_table.csv   (original 7-HRSN vs 2-factor)
"""
import sys
from pathlib import Path

import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
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


def run_factor_regressions():
    """Run regressions using 2 factor scores instead of 7 HRSN measures."""
    ensure_dirs()
    params = load_params()

    # ---- Load data ----
    df = load_parquet(PATHS["processed"] / "merged_tracts.parquet")
    factor_scores = load_parquet(PATHS["final"] / "factor_scores.parquet")

    # Merge factor scores into main data
    df = df.merge(factor_scores[["tract_fips", "material_hardship", "social_isolation"]],
                  on="tract_fips", how="inner")
    logger.info(f"Merged dataset: {len(df):,} tracts with factor scores")

    outcome_cols = [m["id"].lower() for m in params["outcome_measures"]]
    outcome_labels = {m["id"].lower(): m["name"] for m in params["outcome_measures"]}
    factor_cols = ["material_hardship", "social_isolation"]
    factor_labels = {"material_hardship": "Material Hardship", "social_isolation": "Social Isolation"}

    covariate_cols = ["pct_black", "pct_hispanic", "pct_poverty",
                      "pct_college", "pct_65plus", "median_age"]

    cluster_var = df["county_fips"]

    # ---- Standardize ----
    df_std = df.copy()
    for col in factor_cols + outcome_cols + covariate_cols:
        df_std[col] = standardize(df[col])

    # ================================================================
    # Part 1: Individually-adjusted regressions (2 factors × 7 outcomes = 14)
    # ================================================================
    logger.info("\n=== Part 1: Individually-Adjusted Factor Regressions (14 models) ===")

    results = []
    for factor in factor_cols:
        for outcome in outcome_cols:
            X_cols = [factor] + covariate_cols
            X = sm.add_constant(df_std[X_cols])
            y = df_std[outcome]

            mask = X.notna().all(axis=1) & y.notna()
            X_clean = X[mask]
            y_clean = y[mask]
            cluster_clean = cluster_var[mask]

            model = sm.OLS(y_clean, X_clean)
            try:
                res = model.fit(cov_type='cluster', cov_kwds={'groups': cluster_clean})

                beta = res.params[factor]
                se = res.bse[factor]
                pval = res.pvalues[factor]
                ci = res.conf_int().loc[factor]

                row = {
                    'factor': factor,
                    'factor_label': factor_labels[factor],
                    'outcome': outcome,
                    'outcome_label': outcome_labels[outcome],
                    'beta_std': round(beta, 4),
                    'se': round(se, 4),
                    'pvalue': pval,
                    'ci_lower': round(ci[0], 4),
                    'ci_upper': round(ci[1], 4),
                    'r_squared': round(res.rsquared, 4),
                    'adj_r_squared': round(res.rsquared_adj, 4),
                    'n_obs': int(res.nobs),
                    'n_clusters': cluster_clean.nunique(),
                    'significant_05': pval < 0.05,
                    'significant_001': pval < 0.001,
                    'model_type': 'individually_adjusted',
                }
                results.append(row)

                sig = "***" if pval < 0.001 else "**" if pval < 0.01 else "*" if pval < 0.05 else "ns"
                logger.info(f"  {factor_labels[factor]:20s} × {outcome:12s}: "
                            f"β={beta:+.4f} (SE={se:.4f}) {sig}  R²={res.rsquared:.4f}")

            except Exception as e:
                logger.error(f"  FAILED: {factor} × {outcome}: {e}")
                results.append({
                    'factor': factor, 'factor_label': factor_labels[factor],
                    'outcome': outcome, 'outcome_label': outcome_labels[outcome],
                    'beta_std': np.nan, 'se': np.nan, 'pvalue': np.nan,
                    'ci_lower': np.nan, 'ci_upper': np.nan,
                    'r_squared': np.nan, 'adj_r_squared': np.nan,
                    'n_obs': 0, 'n_clusters': 0,
                    'significant_05': False, 'significant_001': False,
                    'model_type': 'individually_adjusted',
                })

    ind_df = pd.DataFrame(results)

    # Apply Benjamini-Hochberg FDR correction across all 14 factor tests
    valid_p = ind_df["pvalue"].dropna()
    if len(valid_p) > 0:
        reject, pvals_corrected, _, _ = multipletests(valid_p, alpha=0.05, method="fdr_bh")
        ind_df.loc[valid_p.index, "pvalue_fdr"] = pvals_corrected
        ind_df.loc[valid_p.index, "significant_05"] = reject

    save_csv(ind_df, PATHS["final"] / "factor_regression_results.csv")

    # ================================================================
    # Part 2: Mutually-adjusted regressions (both factors per disease = 7 models)
    # ================================================================
    logger.info("\n=== Part 2: Mutually-Adjusted Factor Regressions (7 models) ===")

    # VIF check first
    logger.info("\n--- VIF Check (2 factors + covariates) ---")
    X_vif = sm.add_constant(df_std[factor_cols + covariate_cols])
    vif_data = []
    for i, col in enumerate(X_vif.columns):
        if col == 'const':
            continue
        vif_val = variance_inflation_factor(X_vif.values, i)
        vif_data.append({'variable': col, 'VIF': round(vif_val, 2)})
        flag = " *** VERY HIGH" if vif_val > 10 else " ** HIGH" if vif_val > 5 else ""
        logger.info(f"  {col:25s}: VIF = {vif_val:.1f}{flag}")

    vif_df = pd.DataFrame(vif_data)
    save_csv(vif_df, PATHS["final"] / "factor_vif_table.csv")

    mut_results = []
    for outcome in outcome_cols:
        logger.info(f"\n  Model: {outcome_labels[outcome]}")

        X_cols = factor_cols + covariate_cols
        X = sm.add_constant(df_std[X_cols])
        y = df_std[outcome]

        mask = X.notna().all(axis=1) & y.notna()
        X_clean = X[mask]
        y_clean = y[mask]
        cluster_clean = cluster_var[mask]

        model = sm.OLS(y_clean, X_clean)
        res = model.fit(cov_type='cluster', cov_kwds={'groups': cluster_clean})

        for factor in factor_cols:
            beta = res.params[factor]
            se = res.bse[factor]
            pval = res.pvalues[factor]
            ci = res.conf_int().loc[factor]

            sig = "***" if pval < 0.001 else "**" if pval < 0.01 else "*" if pval < 0.05 else "ns"
            logger.info(f"    {factor_labels[factor]:20s}: β={beta:+.4f} (SE={se:.4f}) {sig}")

            mut_results.append({
                'factor': factor,
                'factor_label': factor_labels[factor],
                'outcome': outcome,
                'outcome_label': outcome_labels[outcome],
                'beta_std': round(beta, 4),
                'se': round(se, 4),
                'pvalue': pval,
                'ci_lower': round(ci[0], 4),
                'ci_upper': round(ci[1], 4),
                'r_squared': round(res.rsquared, 4),
                'adj_r_squared': round(res.rsquared_adj, 4),
                'n_obs': int(res.nobs),
                'significant_05': pval < 0.05,
                'model_type': 'mutually_adjusted',
            })

        logger.info(f"    R² = {res.rsquared:.4f} (adj = {res.rsquared_adj:.4f})")

    mut_df = pd.DataFrame(mut_results)
    save_csv(mut_df, PATHS["final"] / "factor_mutually_adjusted_results.csv")

    # ================================================================
    # Part 3: Model comparison table (original 7-HRSN vs 2-factor)
    # ================================================================
    logger.info("\n=== Part 3: Model Comparison (7-HRSN vs 2-Factor) ===")

    # Load original results for comparison
    try:
        orig_results = pd.read_csv(PATHS["final"] / "results_matrix.csv")
        orig_vif = pd.read_csv(PATHS["final"] / "vif_table.csv")

        comparison = []
        for outcome in outcome_cols:
            # Original: best single HRSN predictor R²
            orig_subset = orig_results[orig_results['outcome'] == outcome]
            orig_max_r2 = orig_subset['r_squared'].max() if len(orig_subset) > 0 else np.nan

            # Factor: both factors mutually adjusted R²
            fac_subset = mut_df[mut_df['outcome'] == outcome]
            fac_r2 = fac_subset['r_squared'].iloc[0] if len(fac_subset) > 0 else np.nan

            # Max VIF in original
            hrsn_vif = orig_vif[orig_vif['variable'].isin(
                [m["id"].lower() for m in params["hrsn_measures"]])]
            orig_max_vif = hrsn_vif['VIF'].max() if len(hrsn_vif) > 0 else np.nan

            # Max VIF in factor model
            fac_vif = vif_df[vif_df['variable'].isin(factor_cols)]
            fac_max_vif = fac_vif['VIF'].max() if len(fac_vif) > 0 else np.nan

            # Check sign consistency
            orig_signs = orig_subset['beta_std'].apply(lambda x: "+" if x > 0 else "-").unique()
            n_sign_reversals = len(orig_signs)  # If both + and -, that's a problem

            comparison.append({
                'outcome': outcome,
                'outcome_label': outcome_labels[outcome],
                'original_best_r2': round(orig_max_r2, 4),
                'factor_mutual_r2': round(fac_r2, 4),
                'r2_difference': round(fac_r2 - orig_max_r2, 4) if not np.isnan(fac_r2) else np.nan,
                'original_max_vif': round(orig_max_vif, 1),
                'factor_max_vif': round(fac_max_vif, 1),
                'original_sign_reversals': n_sign_reversals > 1,
            })

        comp_df = pd.DataFrame(comparison)
        dest = PATHS["tables"] / "model_comparison_table.csv"
        save_csv(comp_df, dest)

        logger.info("\n--- Model Comparison Summary ---")
        for _, row in comp_df.iterrows():
            logger.info(f"  {row['outcome_label']:25s}: "
                        f"R²_orig={row['original_best_r2']:.4f} vs R²_factor={row['factor_mutual_r2']:.4f} | "
                        f"VIF: {row['original_max_vif']:.0f} → {row['factor_max_vif']:.1f}")

    except FileNotFoundError:
        logger.warning("Original results not found — skipping comparison table")

    return ind_df, mut_df


if __name__ == "__main__":
    print("=" * 70)
    print("PHASE A: Factor-Based Regressions")
    print("=" * 70)
    ind, mut = run_factor_regressions()
    print(f"\nDone. {len(ind)} individual + {len(mut)} mutually-adjusted coefficients.")
