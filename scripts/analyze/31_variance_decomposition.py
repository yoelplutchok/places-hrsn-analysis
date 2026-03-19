"""
Step 31: Variance Decomposition — Between-County vs Within-County.

Uses multilevel models (random county intercepts) to decompose variance in
HRSN and disease variables into between-county and within-county components.

Phase 1 — Null ICCs (14 models):
  For each of the 7 HRSN and 7 disease variables, fit y ~ 1 + (1|county).
  ICC = sigma_between^2 / (sigma_between^2 + sigma_within^2).

Phase 2 — Conditional models (49 models):
  For each HRSN x disease pair, fit outcome ~ hrsn + covariates + (1|county).
  Compare conditional vs null variance components to compute R-squared at
  each level (between-county, within-county, total).

This reveals whether HRSN-disease covariation is driven by between-county
geographic sorting or within-county local neighborhood effects — a critical
distinction for policy (county-level vs tract-level intervention targeting).

Output:
  - data/final/variance_decomposition.csv (63 rows: 14 null + 49 conditional)
"""
import sys
from pathlib import Path

import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.regression.mixed_linear_model import MixedLM

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from hrsn_analysis.paths import PATHS, ensure_dirs
from hrsn_analysis.io_utils import load_parquet, load_params, save_csv
from hrsn_analysis.logging_utils import setup_logging, get_logger

setup_logging()
logger = get_logger(__name__)


def compute_null_icc(y, groups):
    """Fit null mixed model y ~ 1 + (1|group) and compute ICC.

    ICC = sigma_between^2 / (sigma_between^2 + sigma_within^2)

    Returns dict with variance components and ICC.
    """
    try:
        model = MixedLM(y, np.ones(len(y)), groups=groups)
        result = model.fit(reml=True, method="lbfgs")

        sigma_between = float(result.cov_re.iloc[0, 0])
        sigma_within = float(result.scale)
        icc = sigma_between / (sigma_between + sigma_within) if (sigma_between + sigma_within) > 0 else np.nan

        return {
            "sigma_between": round(sigma_between, 6),
            "sigma_within": round(sigma_within, 6),
            "icc": round(icc, 4),
            "n_obs": int(result.nobs),
            "n_groups": int(result.nobs_re.sum() if hasattr(result, 'nobs_re') else groups.nunique()),
            "converged": result.converged,
        }
    except Exception as e:
        logger.warning(f"    Null ICC failed: {e}")
        # Try Powell optimizer as fallback
        try:
            result = model.fit(reml=True, method="powell")
            sigma_between = float(result.cov_re.iloc[0, 0])
            sigma_within = float(result.scale)
            icc = sigma_between / (sigma_between + sigma_within) if (sigma_between + sigma_within) > 0 else np.nan
            return {
                "sigma_between": round(sigma_between, 6),
                "sigma_within": round(sigma_within, 6),
                "icc": round(icc, 4),
                "n_obs": int(result.nobs),
                "n_groups": int(groups.nunique()),
                "converged": result.converged,
            }
        except Exception as e2:
            logger.warning(f"    Null ICC (Powell fallback) also failed: {e2}")
            return {
                "sigma_between": np.nan,
                "sigma_within": np.nan,
                "icc": np.nan,
                "n_obs": len(y),
                "n_groups": groups.nunique(),
                "converged": False,
            }


def compute_conditional_model(y, X, groups):
    """Fit conditional mixed model y ~ X + (1|group).

    Returns dict with variance components.
    """
    try:
        model = MixedLM(y, sm.add_constant(X), groups=groups)
        result = model.fit(reml=True, method="lbfgs")

        sigma_between = float(result.cov_re.iloc[0, 0])
        sigma_within = float(result.scale)

        return {
            "sigma_between_cond": round(sigma_between, 6),
            "sigma_within_cond": round(sigma_within, 6),
            "converged": result.converged,
        }
    except Exception as e:
        logger.warning(f"    Conditional model failed: {e}")
        try:
            result = model.fit(reml=True, method="powell")
            sigma_between = float(result.cov_re.iloc[0, 0])
            sigma_within = float(result.scale)
            return {
                "sigma_between_cond": round(sigma_between, 6),
                "sigma_within_cond": round(sigma_within, 6),
                "converged": result.converged,
            }
        except Exception as e2:
            logger.warning(f"    Conditional (Powell fallback) also failed: {e2}")
            return {
                "sigma_between_cond": np.nan,
                "sigma_within_cond": np.nan,
                "converged": False,
            }


def run_variance_decomposition():
    """Compute ICCs and conditional variance decomposition."""
    ensure_dirs()
    params = load_params()

    hrsn_cols = [m["id"].lower() for m in params["hrsn_measures"]]
    outcome_cols = [m["id"].lower() for m in params["outcome_measures"]]
    hrsn_labels = {m["id"].lower(): m["name"] for m in params["hrsn_measures"]}
    outcome_labels = {m["id"].lower(): m["name"] for m in params["outcome_measures"]}
    covariate_cols = ["pct_black", "pct_hispanic", "pct_poverty",
                      "pct_college", "pct_65plus", "median_age"]

    # Load and prepare data
    df = load_parquet(PATHS["processed"] / "merged_tracts.parquet")

    # Drop rows with any missing in analysis variables
    all_vars = hrsn_cols + outcome_cols + covariate_cols
    df = df.dropna(subset=all_vars + ["county_fips"])
    logger.info(f"Complete cases: {len(df):,} tracts, {df['county_fips'].nunique()} counties")

    # Z-standardize all analysis variables
    df_std = df.copy()
    for col in all_vars:
        s = df[col]
        df_std[col] = (s - s.mean()) / s.std()

    groups = df_std["county_fips"]
    results = []

    # --- Phase 1: Null ICCs ---
    logger.info("\n=== Phase 1: Null ICCs ===")
    null_iccs = {}  # Store for use in Phase 2

    for var in hrsn_cols + outcome_cols:
        var_type = "hrsn" if var in hrsn_cols else "outcome"
        label = hrsn_labels.get(var, outcome_labels.get(var, var))

        icc_result = compute_null_icc(df_std[var], groups)
        null_iccs[var] = icc_result

        logger.info(f"  {label:30s}: ICC={icc_result['icc']:.4f}  "
                    f"(between={icc_result['sigma_between']:.4f}, "
                    f"within={icc_result['sigma_within']:.4f})")

        results.append({
            "model_type": "null_icc",
            "variable": var,
            "variable_label": label,
            "variable_type": var_type,
            "hrsn_measure": None,
            "sigma_between": icc_result["sigma_between"],
            "sigma_within": icc_result["sigma_within"],
            "icc": icc_result["icc"],
            "r2_between": np.nan,
            "r2_within": np.nan,
            "r2_total": np.nan,
            "n_obs": icc_result["n_obs"],
            "n_groups": icc_result["n_groups"],
            "converged": icc_result["converged"],
        })

    # --- Phase 2: Conditional Models ---
    logger.info("\n=== Phase 2: Conditional Variance Decomposition ===")
    logger.info("  Fitting outcome ~ hrsn + covariates + (1|county)")

    for hrsn in hrsn_cols:
        for outcome in outcome_cols:
            X = df_std[[hrsn] + covariate_cols]
            y = df_std[outcome]

            cond = compute_conditional_model(y, X, groups)

            # Compute R-squared at each level
            null = null_iccs.get(outcome, {})
            sig_b_null = null.get("sigma_between", np.nan)
            sig_w_null = null.get("sigma_within", np.nan)
            sig_b_cond = cond.get("sigma_between_cond", np.nan)
            sig_w_cond = cond.get("sigma_within_cond", np.nan)

            r2_between = np.nan
            r2_within = np.nan
            r2_total = np.nan

            if not any(np.isnan(v) for v in [sig_b_null, sig_w_null, sig_b_cond, sig_w_cond]):
                if sig_b_null > 0:
                    r2_between = 1 - sig_b_cond / sig_b_null
                if sig_w_null > 0:
                    r2_within = 1 - sig_w_cond / sig_w_null
                total_null = sig_b_null + sig_w_null
                total_cond = sig_b_cond + sig_w_cond
                if total_null > 0:
                    r2_total = 1 - total_cond / total_null

            hrsn_label = hrsn_labels.get(hrsn, hrsn)
            out_label = outcome_labels.get(outcome, outcome)

            logger.info(f"  {hrsn_label:25s} x {out_label:12s}: "
                        f"R²_between={r2_between:.4f}, R²_within={r2_within:.4f}, "
                        f"R²_total={r2_total:.4f}"
                        if not np.isnan(r2_total)
                        else f"  {hrsn_label:25s} x {out_label:12s}: FAILED")

            results.append({
                "model_type": "conditional",
                "variable": outcome,
                "variable_label": out_label,
                "variable_type": "outcome",
                "hrsn_measure": hrsn,
                "sigma_between": sig_b_cond,
                "sigma_within": sig_w_cond,
                "icc": null.get("icc", np.nan),
                "r2_between": round(r2_between, 4) if not np.isnan(r2_between) else np.nan,
                "r2_within": round(r2_within, 4) if not np.isnan(r2_within) else np.nan,
                "r2_total": round(r2_total, 4) if not np.isnan(r2_total) else np.nan,
                "n_obs": null.get("n_obs", len(y)),
                "n_groups": null.get("n_groups", groups.nunique()),
                "converged": cond.get("converged", False),
            })

    results_df = pd.DataFrame(results)
    save_csv(results_df, PATHS["final"] / "variance_decomposition.csv")

    # --- Summary ---
    logger.info("\n=== Summary ===")

    # ICC summary
    null_rows = results_df[results_df["model_type"] == "null_icc"]
    hrsn_iccs = null_rows[null_rows["variable_type"] == "hrsn"]
    outcome_iccs = null_rows[null_rows["variable_type"] == "outcome"]

    logger.info(f"\nHRSN ICCs (% variance between counties):")
    for _, r in hrsn_iccs.iterrows():
        logger.info(f"  {r['variable_label']:30s}: {r['icc']:.3f} "
                    f"({r['icc'] * 100:.1f}% between-county)")

    logger.info(f"\nDisease ICCs:")
    for _, r in outcome_iccs.iterrows():
        logger.info(f"  {r['variable_label']:30s}: {r['icc']:.3f} "
                    f"({r['icc'] * 100:.1f}% between-county)")

    # Conditional model summary
    cond_rows = results_df[results_df["model_type"] == "conditional"]
    if len(cond_rows) > 0:
        converged = cond_rows["converged"].sum()
        logger.info(f"\nConditional models: {converged}/{len(cond_rows)} converged")
        valid = cond_rows[cond_rows["converged"] == True]
        if len(valid) > 0:
            logger.info(f"  Mean R²_between: {valid['r2_between'].mean():.4f}")
            logger.info(f"  Mean R²_within:  {valid['r2_within'].mean():.4f}")
            logger.info(f"  Mean R²_total:   {valid['r2_total'].mean():.4f}")

    return results_df


if __name__ == "__main__":
    print("=" * 70)
    print("Step 31: Variance Decomposition (Between/Within County)")
    print("=" * 70)
    results = run_variance_decomposition()
    print(f"\nDone. {len(results)} variance decomposition results.")
