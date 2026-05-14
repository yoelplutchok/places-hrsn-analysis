"""
Phase 4, Step 4.3–4.4: Mutually-adjusted models and ranking.

For each disease outcome, run a single model with ALL 7 HRSN measures simultaneously:
  Disease = β₀ + β₁(loneliness) + β₂(foodstamp) + β₃(foodinsecu) + β₄(housinsecu)
           + β₅(shututility) + β₆(lacktrpt) + β₇(emotionspt) + [covariates] + ε

Also checks VIF for multicollinearity, and compiles rankings.

Output:
  - data/final/mutually_adjusted_results.csv
  - data/final/vif_table.csv
  - outputs/tables/ranked_associations.csv
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
    """Z-score standardize."""
    return (series - series.mean()) / series.std()


def compute_vif(df_std, hrsn_cols, covariate_cols):
    """Compute VIF for HRSN variables in the full model."""
    logger.info("=== VIF Check ===")
    X_cols = hrsn_cols + covariate_cols
    X = sm.add_constant(df_std[X_cols])

    vif_data = []
    for i, col in enumerate(X.columns):
        if col == 'const':
            continue
        vif_val = variance_inflation_factor(X.values, i)
        vif_data.append({'variable': col, 'VIF': round(vif_val, 2)})
        flag = " *** VERY HIGH" if vif_val > 10 else " ** HIGH" if vif_val > 5 else ""
        logger.info(f"  {col:20s}: VIF = {vif_val:.1f}{flag}")

    vif_df = pd.DataFrame(vif_data)
    dest = PATHS["final"] / "vif_table.csv"
    vif_df.to_csv(dest, index=False)
    logger.info(f"VIF table saved to {dest}")

    return vif_df


def run_mutually_adjusted(df, params):
    """Run mutually-adjusted models (all 7 HRSN in each disease model)."""
    logger.info("=== Step 4.3: Mutually-Adjusted Models ===")

    hrsn_cols = [m["id"].lower() for m in params["hrsn_measures"]]
    outcome_cols = [m["id"].lower() for m in params["outcome_measures"]]
    hrsn_labels = {m["id"].lower(): m["name"] for m in params["hrsn_measures"]}
    outcome_labels = {m["id"].lower(): m["name"] for m in params["outcome_measures"]}

    covariate_cols = ["pct_black", "pct_hispanic", "pct_poverty",
                      "pct_college", "pct_65plus", "median_age"]

    cluster_var = df["county_fips"]

    # Standardize
    df_std = df.copy()
    for col in hrsn_cols + outcome_cols + covariate_cols:
        df_std[col] = standardize(df[col])

    # VIF check
    compute_vif(df_std, hrsn_cols, covariate_cols)

    # Run one model per outcome with ALL HRSN
    results = []
    for outcome in outcome_cols:
        logger.info(f"\n  Model: {outcome_labels[outcome]}")

        X_cols = hrsn_cols + covariate_cols
        X = sm.add_constant(df_std[X_cols])
        y = df_std[outcome]

        mask = X.notna().all(axis=1) & y.notna()
        X_clean = X[mask]
        y_clean = y[mask]
        cluster_clean = cluster_var[mask]

        model = sm.OLS(y_clean, X_clean)
        res = model.fit(cov_type='cluster', cov_kwds={'groups': cluster_clean})

        for hrsn in hrsn_cols:
            beta = res.params[hrsn]
            se = res.bse[hrsn]
            pval = res.pvalues[hrsn]
            ci = res.conf_int().loc[hrsn]

            sig = "***" if pval < 0.001 else "**" if pval < 0.01 else "*" if pval < 0.05 else "ns"
            logger.info(f"    {hrsn:15s}: β={beta:+.4f} (SE={se:.4f}) {sig}")

            results.append({
                'hrsn_measure': hrsn,
                'hrsn_label': hrsn_labels[hrsn],
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

    results_df = pd.DataFrame(results)

    # FDR correction (Benjamini-Hochberg) across all mutually-adjusted p-values
    valid_pvals = results_df['pvalue'].dropna()
    if len(valid_pvals) > 0:
        reject, qvals, _, _ = multipletests(valid_pvals.values, method='fdr_bh')
        results_df.loc[valid_pvals.index, 'pvalue_fdr'] = qvals
        results_df.loc[valid_pvals.index, 'significant_fdr_05'] = reject
        results_df['significant_fdr_05'] = results_df['significant_fdr_05'].fillna(False)
    else:
        results_df['pvalue_fdr'] = np.nan
        results_df['significant_fdr_05'] = False

    n_sig_fdr = results_df['significant_fdr_05'].sum()
    logger.info(f"\n  FDR correction: {n_sig_fdr}/{len(results_df)} significant (q<0.05)")

    dest = PATHS["final"] / "mutually_adjusted_results.csv"
    results_df.to_csv(dest, index=False)
    logger.info(f"\nMutually-adjusted results saved to {dest}")

    return results_df


def compile_rankings(params):
    """Step 4.4: Compile and rank associations."""
    logger.info("\n=== Step 4.4: Compile and Rank Results ===")

    hrsn_cols = [m["id"].lower() for m in params["hrsn_measures"]]
    outcome_cols = [m["id"].lower() for m in params["outcome_measures"]]
    hrsn_labels = {m["id"].lower(): m["name"] for m in params["hrsn_measures"]}
    outcome_labels = {m["id"].lower(): m["name"] for m in params["outcome_measures"]}

    # Load individual results
    ind = pd.read_csv(PATHS["final"] / "results_matrix.csv")
    mut = pd.read_csv(PATHS["final"] / "mutually_adjusted_results.csv")

    rankings = []

    # For each disease: rank HRSN measures by |β| (individual models)
    logger.info("\n--- Rankings by Disease (Individual Models) ---")
    for outcome in outcome_cols:
        subset = ind[ind['outcome'] == outcome].copy()
        subset['abs_beta'] = subset['beta_std'].abs()
        subset = subset.sort_values('abs_beta', ascending=False)

        for rank, (_, row) in enumerate(subset.iterrows(), 1):
            sig = "***" if row['pvalue'] < 0.001 else "**" if row['pvalue'] < 0.01 else "*" if row['pvalue'] < 0.05 else ""
            rankings.append({
                'perspective': 'by_disease',
                'disease': outcome,
                'disease_label': outcome_labels[outcome],
                'hrsn': row['hrsn_measure'],
                'hrsn_label': row['hrsn_label'],
                'rank': rank,
                'beta_individual': row['beta_std'],
                'pvalue_individual': row['pvalue'],
            })

        top = subset.iloc[0]
        logger.info(f"  {outcome_labels[outcome]:30s}: #1 = {top['hrsn_label']} "
                     f"(β={top['beta_std']:+.4f})")

    # For each HRSN: rank diseases by |β|
    logger.info("\n--- Rankings by HRSN Measure (Individual Models) ---")
    for hrsn in hrsn_cols:
        subset = ind[ind['hrsn_measure'] == hrsn].copy()
        subset['abs_beta'] = subset['beta_std'].abs()
        subset = subset.sort_values('abs_beta', ascending=False)

        for rank, (_, row) in enumerate(subset.iterrows(), 1):
            # Find existing ranking entry and add to it
            for r in rankings:
                if r['hrsn'] == hrsn and r['disease'] == row['outcome'] and r['perspective'] == 'by_disease':
                    r['rank_by_hrsn'] = rank
                    break

        top = subset.iloc[0]
        logger.info(f"  {hrsn_labels[hrsn]:30s}: #1 = {top['outcome_label']} "
                     f"(β={top['beta_std']:+.4f})")

    # Merge with mutually-adjusted results
    for r in rankings:
        mut_match = mut[(mut['hrsn_measure'] == r['hrsn']) & (mut['outcome'] == r['disease'])]
        if len(mut_match) > 0:
            r['beta_mutually_adjusted'] = mut_match.iloc[0]['beta_std']
            r['pvalue_mutually_adjusted'] = mut_match.iloc[0]['pvalue']

    rankings_df = pd.DataFrame(rankings)
    dest = PATHS["tables"] / "ranked_associations.csv"
    rankings_df.to_csv(dest, index=False)
    logger.info(f"\nRankings saved to {dest}")

    return rankings_df


if __name__ == "__main__":
    print("=" * 70)
    print("PHASE 4, Steps 4.3-4.4: Mutually-Adjusted Models & Rankings")
    print("=" * 70)

    ensure_dirs()
    params = load_params()
    df = load_parquet(PATHS["processed"] / "merged_tracts.parquet")

    mut_results = run_mutually_adjusted(df, params)
    rankings = compile_rankings(params)

    print(f"\nDone. {len(mut_results)} mutually-adjusted coefficients, {len(rankings)} rankings.")
