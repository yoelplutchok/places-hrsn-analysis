"""
Phase B, Step 14: BRFSS individual-level logistic regressions.

49 survey-weighted logistic regressions comparing individual-level HRSN-disease
associations with ecological (tract-level) results from scripts 08/09.

Model: Pr(disease=1) = logit(β₀ + β₁*HRSN + β₂*age + β₃*female + β₄*race
                              + β₅*income + β₆*education)

Output:
  - data/final/brfss_individual_results.csv
  - data/final/brfss_comparison_table.csv
  - outputs/figures/brfss_odds_ratio_forest.png
  - outputs/figures/ecological_vs_individual_comparison.png
"""
import sys
from pathlib import Path

import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.genmod.families import Binomial
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from hrsn_analysis.paths import PATHS, ensure_dirs
from hrsn_analysis.io_utils import load_parquet, load_params, save_csv
from hrsn_analysis.logging_utils import setup_logging, get_logger
from hrsn_analysis.survey_utils import weighted_logistic, extract_or

setup_logging()
logger = get_logger(__name__)


def run_brfss_regressions():
    """Run 49 survey-weighted logistic regressions on BRFSS data."""
    ensure_dirs()
    params = load_params()
    brfss_params = params["brfss"]

    # ---- Load data ----
    df = load_parquet(PATHS["processed"] / "brfss_analytic.parquet")
    logger.info(f"Loaded BRFSS analytic: {len(df):,} respondents")

    hrsn_cols = list(brfss_params["hrsn_variables"].keys())
    outcome_cols = list(brfss_params["outcome_variables"].keys())
    hrsn_labels = {m["id"].lower(): m["name"] for m in params["hrsn_measures"]}
    outcome_labels = {m["id"].lower(): m["name"] for m in params["outcome_measures"]}

    weight_var = brfss_params["weight_var"]
    psu_var = brfss_params.get("psu_var", "_PSU")
    strata_var = brfss_params.get("strata_var", "_STSTR")

    # Demographic covariates (from recoded variables)
    demo_cols = ["age_group", "female", "income_cat", "education_cat"]
    # Race as dummies
    if "race_cat" in df.columns:
        race_dummies = pd.get_dummies(df["race_cat"], prefix="race", drop_first=True, dtype=float)
        for col in race_dummies.columns:
            df[col] = race_dummies[col]
        race_dummy_cols = list(race_dummies.columns)
    else:
        race_dummy_cols = []

    covariate_cols = demo_cols + race_dummy_cols
    covariate_cols = [c for c in covariate_cols if c in df.columns]
    logger.info(f"Covariates: {covariate_cols}")

    # ---- Run 49 regressions ----
    logger.info(f"\n=== Running {len(hrsn_cols)} × {len(outcome_cols)} = "
                f"{len(hrsn_cols) * len(outcome_cols)} logistic regressions ===")

    results = []
    for hrsn in hrsn_cols:
        if hrsn not in df.columns:
            logger.warning(f"  {hrsn} not in data — skipping")
            continue

        for outcome in outcome_cols:
            if outcome not in df.columns:
                logger.warning(f"  {outcome} not in data — skipping")
                continue

            # Build model matrix — include survey design variables
            X_cols = [hrsn] + covariate_cols
            keep_cols = [outcome] + X_cols + [weight_var]
            if psu_var in df.columns:
                keep_cols.append(psu_var)
            if strata_var in df.columns:
                keep_cols.append(strata_var)
            model_df = df[keep_cols].dropna()

            if len(model_df) < 100:
                logger.warning(f"  {hrsn} × {outcome}: insufficient data (n={len(model_df)})")
                continue

            y = model_df[outcome]
            X = sm.add_constant(model_df[X_cols])
            weights = model_df[weight_var]
            psu = model_df[psu_var] if psu_var in model_df.columns else None
            strata = model_df[strata_var] if strata_var in model_df.columns else None

            try:
                res = weighted_logistic(y, X, weights, psu=psu, strata=strata)
                or_info = extract_or(res, hrsn)

                row = {
                    'hrsn_measure': hrsn,
                    'hrsn_label': hrsn_labels.get(hrsn, hrsn),
                    'outcome': outcome,
                    'outcome_label': outcome_labels.get(outcome, outcome),
                    'odds_ratio': round(or_info['or'], 3),
                    'or_lower': round(or_info['or_lower'], 3),
                    'or_upper': round(or_info['or_upper'], 3),
                    'coef': round(or_info['coef'], 4),
                    'se': round(or_info['se'], 4),
                    'pvalue': or_info['pvalue'],
                    'n_obs': len(model_df),
                    'n_cases': int(y.sum()),
                    'significant_05': or_info['pvalue'] < 0.05,
                    'significant_001': or_info['pvalue'] < 0.001,
                }
                results.append(row)

                sig = "***" if or_info['pvalue'] < 0.001 else "**" if or_info['pvalue'] < 0.01 \
                    else "*" if or_info['pvalue'] < 0.05 else "ns"
                logger.info(f"  {hrsn:15s} × {outcome:12s}: OR={or_info['or']:.2f} "
                            f"({or_info['or_lower']:.2f}-{or_info['or_upper']:.2f}) {sig}  "
                            f"n={len(model_df):,}")

            except Exception as e:
                logger.error(f"  FAILED: {hrsn} × {outcome}: {e}")

    results_df = pd.DataFrame(results)
    save_csv(results_df, PATHS["final"] / "brfss_individual_results.csv")

    # ---- Summary ----
    n_sig = results_df['significant_05'].sum() if len(results_df) > 0 else 0
    logger.info(f"\n--- Summary ---")
    logger.info(f"Total models: {len(results_df)}")
    logger.info(f"Significant (p<0.05): {n_sig} ({n_sig/max(len(results_df),1)*100:.0f}%)")

    # ---- Comparison with ecological results ----
    logger.info("\n=== Ecological vs Individual Comparison ===")
    try:
        ecological = pd.read_csv(PATHS["final"] / "results_matrix.csv")

        comparison = []
        for _, ind_row in results_df.iterrows():
            eco_match = ecological[
                (ecological['hrsn_measure'] == ind_row['hrsn_measure']) &
                (ecological['outcome'] == ind_row['outcome'])
            ]
            if len(eco_match) > 0:
                eco = eco_match.iloc[0]
                comparison.append({
                    'hrsn_measure': ind_row['hrsn_measure'],
                    'hrsn_label': ind_row['hrsn_label'],
                    'outcome': ind_row['outcome'],
                    'outcome_label': ind_row['outcome_label'],
                    'ecological_beta': eco['beta_std'],
                    'ecological_pvalue': eco['pvalue'],
                    'ecological_sig': eco['pvalue'] < 0.05,
                    'individual_or': ind_row['odds_ratio'],
                    'individual_log_or': ind_row['coef'],
                    'individual_pvalue': ind_row['pvalue'],
                    'individual_sig': ind_row['pvalue'] < 0.05,
                    'direction_match': (eco['beta_std'] > 0) == (ind_row['coef'] > 0),
                    'both_significant': (eco['pvalue'] < 0.05) and (ind_row['pvalue'] < 0.05),
                })

        comp_df = pd.DataFrame(comparison)
        save_csv(comp_df, PATHS["final"] / "brfss_comparison_table.csv")

        n_match = comp_df['direction_match'].sum() if len(comp_df) > 0 else 0
        n_both_sig = comp_df['both_significant'].sum() if len(comp_df) > 0 else 0
        logger.info(f"Direction match: {n_match}/{len(comp_df)} "
                     f"({n_match/max(len(comp_df),1)*100:.0f}%)")
        logger.info(f"Both significant: {n_both_sig}/{len(comp_df)}")

    except FileNotFoundError:
        logger.warning("Ecological results not found — skipping comparison")
        comp_df = pd.DataFrame()

    # ---- Forest plot of ORs ----
    if len(results_df) > 0:
        _create_forest_plot(results_df, hrsn_labels, outcome_labels)

    # ---- Ecological vs Individual scatter ----
    if len(comp_df) > 0:
        _create_comparison_plot(comp_df)

    return results_df


def _create_forest_plot(results_df, hrsn_labels, outcome_labels):
    """Create forest plot of odds ratios."""
    fig, ax = plt.subplots(figsize=(10, 14))

    # Sort by OR
    plot_df = results_df.sort_values("odds_ratio")
    y_pos = range(len(plot_df))

    # Color by significance
    colors = ["#c0392b" if sig else "#95a5a6" for sig in plot_df["significant_05"]]

    for i_row, (_, row) in enumerate(plot_df.iterrows()):
        color = colors[i_row]
        ax.errorbar(
            row["odds_ratio"], i_row,
            xerr=[[row["odds_ratio"] - row["or_lower"]],
                  [row["or_upper"] - row["odds_ratio"]]],
            fmt="o", color=color, ecolor=color,
            elinewidth=1.5, capsize=3, markersize=5,
        )

    # Reference line at OR=1
    ax.axvline(x=1, color="black", linestyle="--", alpha=0.5)

    # Labels
    labels = [f"{row['hrsn_label']} × {row['outcome_label']}" for _, row in plot_df.iterrows()]
    ax.set_yticks(list(y_pos))
    ax.set_yticklabels(labels, fontsize=7)
    ax.set_xlabel("Odds Ratio (95% CI)")
    ax.set_title("BRFSS Individual-Level: HRSN → Chronic Disease Odds Ratios")
    ax.grid(True, axis="x", alpha=0.3)

    dest = PATHS["figures"] / "brfss_odds_ratio_forest.png"
    fig.savefig(dest, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    logger.info(f"Forest plot saved to {dest}")


def _create_comparison_plot(comp_df):
    """Scatter plot comparing ecological betas vs individual log(OR)."""
    fig, ax = plt.subplots(figsize=(8, 8))

    colors = ["#2ecc71" if match else "#e74c3c" for match in comp_df["direction_match"]]

    ax.scatter(comp_df["ecological_beta"], comp_df["individual_log_or"],
               c=colors, s=50, alpha=0.7, edgecolors="black", linewidths=0.5)

    # Reference lines
    ax.axhline(y=0, color="gray", linestyle=":", alpha=0.5)
    ax.axvline(x=0, color="gray", linestyle=":", alpha=0.5)

    # 45-degree line (not expected to be exact but shows direction)
    lims = [
        min(ax.get_xlim()[0], ax.get_ylim()[0]),
        max(ax.get_xlim()[1], ax.get_ylim()[1]),
    ]
    ax.plot(lims, lims, "k--", alpha=0.3, label="y=x")

    # Correlation
    r = comp_df["ecological_beta"].corr(comp_df["individual_log_or"])
    ax.text(0.05, 0.95, f"r = {r:.3f}", transform=ax.transAxes, fontsize=12,
            verticalalignment="top", bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

    ax.set_xlabel("Ecological β (standardized, tract-level)")
    ax.set_ylabel("Individual log(OR) (BRFSS)")
    ax.set_title("Ecological vs Individual-Level Associations\n(Green = same direction, Red = opposite)")
    ax.grid(True, alpha=0.3)

    dest = PATHS["figures"] / "ecological_vs_individual_comparison.png"
    fig.savefig(dest, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    logger.info(f"Comparison plot saved to {dest}")


if __name__ == "__main__":
    print("=" * 70)
    print("PHASE B: BRFSS Individual-Level Regressions")
    print("=" * 70)
    results = run_brfss_regressions()
    print(f"\nDone. {len(results)} models completed.")
