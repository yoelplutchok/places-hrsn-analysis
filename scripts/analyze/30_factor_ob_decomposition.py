"""
Step 30: Factor-Level Oaxaca-Blinder Decomposition.

Runs the same Neumark/pooled OB decomposition as script 19, but using
2 factor scores (Material Hardship + Social Isolation) instead of
7 individual HRSN measures. This resolves the multicollinearity problem
(VIF drops from 97.4 to 8.8) and provides stable, interpretable
per-factor decomposition contributions.

Addresses audit finding C3-3: individual HRSN contributions reported
as "unreliable" due to VIF=97 while still being interpreted.

The factor scores are pre-computed in script 10 (EFA with promax rotation)
and stored in data/final/factor_scores.parquet.

Output:
  - data/final/ob_decomposition_factor_level.csv
  - outputs/figures/factor_decomposition_comparison.png
"""
import sys
from pathlib import Path

import pandas as pd
import numpy as np
import statsmodels.api as sm

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from hrsn_analysis.paths import PATHS, ensure_dirs
from hrsn_analysis.io_utils import load_parquet, load_params, save_csv
from hrsn_analysis.logging_utils import setup_logging, get_logger

setup_logging()
logger = get_logger(__name__)


# --- OB decomposition functions (duplicated from script 19 for self-containment) ---

def _ob_point_estimate(y_a, X_a, y_b, X_b, var_names):
    """Core OB computation returning (raw_gap, endowment_contributions dict)."""
    X_all = pd.concat([X_a, X_b])
    y_all = pd.concat([y_a, y_b])
    group_indicator = pd.Series(
        [0] * len(X_a) + [1] * len(X_b),
        index=X_all.index, name="group_B"
    )
    X_all_with_group = pd.concat([X_all, group_indicator], axis=1)
    X_all_c = sm.add_constant(X_all_with_group)
    res_pooled = sm.OLS(y_all, X_all_c).fit()
    beta_pooled = res_pooled.params

    mean_a = X_a.mean()
    mean_b = X_b.mean()
    raw_gap = y_b.mean() - y_a.mean()

    endowment_contributions = {}
    total_endowment = 0
    for var in var_names:
        if var in beta_pooled.index:
            contrib = (mean_b[var] - mean_a[var]) * beta_pooled[var]
            endowment_contributions[var] = contrib
            total_endowment += contrib

    total_explained = sum(endowment_contributions.values())
    unexplained = raw_gap - total_explained

    return {
        "raw_gap": raw_gap,
        "total_explained": total_explained,
        "total_unexplained": unexplained,
        "pct_explained": (total_explained / raw_gap * 100) if raw_gap != 0 else np.nan,
        "endowment_contributions": endowment_contributions,
        "mean_a": dict(mean_a),
        "mean_b": dict(mean_b),
        "n_a": len(y_a),
        "n_b": len(y_b),
    }


def oaxaca_blinder(y_a, X_a, y_b, X_b, var_names, n_bootstrap=500, seed=42):
    """Twofold OB decomposition with bootstrap CIs. Identical to script 19."""
    result = _ob_point_estimate(y_a, X_a, y_b, X_b, var_names)

    rng = np.random.default_rng(seed)
    boot_contribs = {var: [] for var in var_names}
    boot_explained = []

    for _ in range(n_bootstrap):
        idx_a = rng.choice(len(y_a), size=len(y_a), replace=True)
        idx_b = rng.choice(len(y_b), size=len(y_b), replace=True)
        try:
            boot_result = _ob_point_estimate(
                y_a.iloc[idx_a].reset_index(drop=True),
                X_a.iloc[idx_a].reset_index(drop=True),
                y_b.iloc[idx_b].reset_index(drop=True),
                X_b.iloc[idx_b].reset_index(drop=True),
                var_names,
            )
            for var in var_names:
                boot_contribs[var].append(
                    boot_result["endowment_contributions"].get(var, 0)
                )
            boot_explained.append(boot_result["total_explained"])
        except Exception:
            continue

    ci = {}
    for var in var_names:
        if len(boot_contribs[var]) >= 50:
            ci[var] = (
                float(np.percentile(boot_contribs[var], 2.5)),
                float(np.percentile(boot_contribs[var], 97.5)),
            )
        else:
            ci[var] = (np.nan, np.nan)

    if len(boot_explained) >= 50:
        ci["total_explained"] = (
            float(np.percentile(boot_explained, 2.5)),
            float(np.percentile(boot_explained, 97.5)),
        )
    else:
        ci["total_explained"] = (np.nan, np.nan)

    result["bootstrap_ci"] = ci
    result["n_bootstrap"] = len(boot_explained)
    return result


# --- Main analysis ---

def run_factor_decomposition():
    """Run factor-level OB decomposition."""
    ensure_dirs()
    params = load_params()

    outcome_cols = [m["id"].lower() for m in params["outcome_measures"]]
    outcome_labels = {m["id"].lower(): m["name"] for m in params["outcome_measures"]}

    factor_cols = ["material_hardship", "social_isolation"]
    factor_labels = {
        "material_hardship": "Material Hardship (Factor 1)",
        "social_isolation": "Social Isolation (Factor 2)",
    }
    demo_cols = ["pct_poverty", "pct_college", "pct_65plus", "median_age"]

    # Load data
    df = load_parquet(PATHS["processed"] / "merged_tracts.parquet")
    factors = load_parquet(PATHS["final"] / "factor_scores.parquet")
    df = df.merge(factors, on="tract_fips", how="inner")
    logger.info(f"After factor merge: {len(df):,} tracts")

    # Define groups (same thresholds as script 19)
    white_tracts = df[df["pct_white"] >= 60].copy()
    black_tracts = df[df["pct_black"] >= 50].copy()
    hisp_tracts = df[df["pct_hispanic"] >= 50].copy()

    logger.info(f"  Majority-White tracts (>=60%): {len(white_tracts):,}")
    logger.info(f"  Majority-Black tracts (>=50%): {len(black_tracts):,}")
    logger.info(f"  Majority-Hispanic tracts (>=50%): {len(hisp_tracts):,}")

    predictor_cols = factor_cols + demo_cols
    results = []

    for comparison, group_b, group_b_name in [
        ("Black_vs_White", black_tracts, "Majority-Black"),
        ("Hispanic_vs_White", hisp_tracts, "Majority-Hispanic"),
    ]:
        logger.info(f"\n  --- {comparison} ---")

        for outcome in outcome_cols:
            all_cols = predictor_cols + [outcome]
            wa = white_tracts[all_cols].dropna()
            gb = group_b[all_cols].dropna()

            if len(wa) < 100 or len(gb) < 100:
                logger.warning(f"    {outcome}: insufficient data "
                               f"(White={len(wa)}, {group_b_name}={len(gb)})")
                continue

            result = oaxaca_blinder(
                y_a=wa[outcome], X_a=wa[predictor_cols],
                y_b=gb[outcome], X_b=gb[predictor_cols],
                var_names=predictor_cols,
            )

            raw_gap = result["raw_gap"]
            if abs(raw_gap) < 1e-10:
                continue

            pct_exp = result["pct_explained"]
            logger.info(f"    {outcome_labels.get(outcome, outcome):20s}: "
                        f"gap={raw_gap:+.3f}, explained={pct_exp:.1f}%")

            boot_ci = result.get("bootstrap_ci", {})

            # Factor contributions
            for var in factor_cols:
                contrib = result["endowment_contributions"].get(var, 0)
                pct_of_gap = (contrib / raw_gap * 100) if raw_gap != 0 else 0
                ci_lo, ci_hi = boot_ci.get(var, (np.nan, np.nan))

                results.append({
                    "comparison": comparison,
                    "outcome": outcome,
                    "outcome_label": outcome_labels.get(outcome, outcome),
                    "variable": var,
                    "variable_label": factor_labels.get(var, var),
                    "variable_type": "factor",
                    "raw_gap": round(raw_gap, 4),
                    "endowment_contribution": round(contrib, 4),
                    "endowment_ci_lower": round(ci_lo, 4) if not np.isnan(ci_lo) else np.nan,
                    "endowment_ci_upper": round(ci_hi, 4) if not np.isnan(ci_hi) else np.nan,
                    "pct_of_gap": round(pct_of_gap, 2),
                    "mean_white": round(result["mean_a"].get(var, 0), 4),
                    "mean_minority": round(result["mean_b"].get(var, 0), 4),
                    "n_white": result["n_a"],
                    "n_minority": result["n_b"],
                    "total_pct_explained": round(pct_exp, 2),
                    "level": "ecological",
                })

            # Demographic contributions
            for var in demo_cols:
                contrib = result["endowment_contributions"].get(var, 0)
                pct_of_gap = (contrib / raw_gap * 100) if raw_gap != 0 else 0
                ci_lo, ci_hi = boot_ci.get(var, (np.nan, np.nan))

                results.append({
                    "comparison": comparison,
                    "outcome": outcome,
                    "outcome_label": outcome_labels.get(outcome, outcome),
                    "variable": var,
                    "variable_label": var,
                    "variable_type": "demographic",
                    "raw_gap": round(raw_gap, 4),
                    "endowment_contribution": round(contrib, 4),
                    "endowment_ci_lower": round(ci_lo, 4) if not np.isnan(ci_lo) else np.nan,
                    "endowment_ci_upper": round(ci_hi, 4) if not np.isnan(ci_hi) else np.nan,
                    "pct_of_gap": round(pct_of_gap, 2),
                    "mean_white": round(result["mean_a"].get(var, 0), 4),
                    "mean_minority": round(result["mean_b"].get(var, 0), 4),
                    "n_white": result["n_a"],
                    "n_minority": result["n_b"],
                    "total_pct_explained": round(pct_exp, 2),
                    "level": "ecological",
                })

    results_df = pd.DataFrame(results)
    save_csv(results_df, PATHS["final"] / "ob_decomposition_factor_level.csv")

    # Compare with individual-HRSN decomposition
    _compare_with_individual(results_df, outcome_cols, outcome_labels)

    # Plot comparison
    _plot_comparison(results_df, outcome_cols, outcome_labels)

    return results_df


def _compare_with_individual(factor_df, outcome_cols, outcome_labels):
    """Compare factor-level vs individual-HRSN total explained percentages."""
    logger.info("\n=== Factor vs Individual-HRSN Decomposition Comparison ===")

    indiv_path = PATHS["final"] / "ob_decomposition_ecological.csv"
    if not indiv_path.exists():
        logger.warning("Individual-HRSN decomposition not found; skipping comparison")
        return

    indiv_df = pd.read_csv(indiv_path)

    for comparison in ["Black_vs_White", "Hispanic_vs_White"]:
        logger.info(f"\n  {comparison}:")
        logger.info(f"  {'Outcome':20s}  {'Indiv-HRSN %':>14s}  {'Factor %':>10s}  {'Diff':>8s}")

        for outcome in outcome_cols:
            # Individual HRSN total
            indiv_sub = indiv_df[
                (indiv_df["comparison"] == comparison) &
                (indiv_df["outcome"] == outcome) &
                (indiv_df["variable_type"] == "hrsn")
            ]
            indiv_total = indiv_sub["pct_of_gap"].sum() if len(indiv_sub) > 0 else np.nan

            # Factor total
            factor_sub = factor_df[
                (factor_df["comparison"] == comparison) &
                (factor_df["outcome"] == outcome) &
                (factor_df["variable_type"] == "factor")
            ]
            factor_total = factor_sub["pct_of_gap"].sum() if len(factor_sub) > 0 else np.nan

            diff = factor_total - indiv_total if not (np.isnan(indiv_total) or np.isnan(factor_total)) else np.nan
            label = outcome_labels.get(outcome, outcome)
            logger.info(f"  {label:20s}  {indiv_total:+14.1f}%  {factor_total:+10.1f}%  {diff:+8.1f}%"
                        if not np.isnan(diff)
                        else f"  {label:20s}  {'N/A':>14s}  {'N/A':>10s}  {'N/A':>8s}")


def _plot_comparison(factor_df, outcome_cols, outcome_labels):
    """Bar chart comparing individual-HRSN vs factor-level explained %."""
    indiv_path = PATHS["final"] / "ob_decomposition_ecological.csv"
    if not indiv_path.exists():
        return

    indiv_df = pd.read_csv(indiv_path)
    bw_factor = factor_df[factor_df["comparison"] == "Black_vs_White"]
    bw_indiv = indiv_df[indiv_df["comparison"] == "Black_vs_White"]

    if len(bw_factor) == 0:
        return

    outcomes = [o for o in outcome_cols
                if o in bw_factor["outcome"].values]
    labels = [outcome_labels.get(o, o) for o in outcomes]

    # Get total HRSN/factor explained for each outcome
    indiv_totals = []
    factor_totals = []
    mh_contribs = []
    si_contribs = []

    for outcome in outcomes:
        i_sub = bw_indiv[(bw_indiv["outcome"] == outcome) &
                         (bw_indiv["variable_type"] == "hrsn")]
        indiv_totals.append(i_sub["pct_of_gap"].sum() if len(i_sub) > 0 else 0)

        f_sub = bw_factor[(bw_factor["outcome"] == outcome) &
                          (bw_factor["variable_type"] == "factor")]
        factor_totals.append(f_sub["pct_of_gap"].sum() if len(f_sub) > 0 else 0)

        mh = f_sub[f_sub["variable"] == "material_hardship"]["pct_of_gap"].values
        mh_contribs.append(mh[0] if len(mh) > 0 else 0)

        si = f_sub[f_sub["variable"] == "social_isolation"]["pct_of_gap"].values
        si_contribs.append(si[0] if len(si) > 0 else 0)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Panel 1: Total explained comparison
    ax = axes[0]
    x = np.arange(len(outcomes))
    width = 0.35
    ax.bar(x - width / 2, indiv_totals, width, label="7 Individual HRSN",
           color="#3498db", alpha=0.8)
    ax.bar(x + width / 2, factor_totals, width, label="2 Factor Scores",
           color="#e74c3c", alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=9)
    ax.set_ylabel("% of B-W Gap Explained", fontsize=11)
    ax.set_title("Total HRSN Contribution:\n7-Variable vs 2-Factor", fontsize=12,
                 fontweight="bold")
    ax.legend(fontsize=9)
    ax.axhline(y=0, color="black", linewidth=0.5)

    # Panel 2: Factor-level decomposition (stacked)
    ax = axes[1]
    ax.bar(x, mh_contribs, width=0.5, label="Material Hardship",
           color="#e74c3c", alpha=0.8)
    ax.bar(x, si_contribs, width=0.5, bottom=mh_contribs,
           label="Social Isolation", color="#9b59b6", alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=9)
    ax.set_ylabel("% of B-W Gap Explained", fontsize=11)
    ax.set_title("Factor-Level Decomposition:\nMaterial Hardship vs Social Isolation",
                 fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)
    ax.axhline(y=0, color="black", linewidth=0.5)

    plt.tight_layout()
    dest = PATHS["figures"] / "factor_decomposition_comparison.png"
    fig.savefig(dest, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    logger.info(f"Saved comparison plot to {dest}")


if __name__ == "__main__":
    print("=" * 70)
    print("Step 30: Factor-Level Oaxaca-Blinder Decomposition")
    print("=" * 70)
    results = run_factor_decomposition()
    print(f"\nDone. {len(results)} decomposition results.")
