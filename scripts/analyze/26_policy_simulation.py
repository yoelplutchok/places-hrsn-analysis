"""
Step 26: Policy Counterfactual Simulation.

IMPORTANT CAVEAT: This is an ASSOCIATIONAL projection, NOT a causal estimate.
The regression coefficients come from cross-sectional ecological data and
cannot support causal claims about intervention effects. All "estimated
associated case reductions" should be interpreted as illustrative upper
bounds that assume (1) the associations are causal, (2) the ecological
betas apply at the individual level, and (3) no confounding remains.
The primary betas are inflated ~2x relative to independent validation
(see attenuation analysis); adjusted estimates are also provided.

Uses regression coefficients to project:
  "If HRSN prevalence in high-need tracts were at the national median,
   how much lower would predicted chronic disease prevalence be?"

Identifies the top 500 "priority" tracts where HRSN-disease associations
are strongest, for illustrative geographic targeting.

Output:
  - data/final/policy_simulation_results.csv    (scenario-level)
  - data/final/priority_intervention_tracts.csv  (top 500 tracts)
  - outputs/figures/policy_simulation.png
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

# Key HRSN-disease pairs for simulation (strongest, most policy-relevant)
SIMULATION_PAIRS = [
    ("foodinsecu", "diabetes", "Reduce food insecurity"),
    ("foodinsecu", "copd", "Reduce food insecurity"),
    ("foodstamp", "casthma", "Increase SNAP access"),
    ("housinsecu", "stroke", "Reduce housing insecurity"),
    ("shututility", "copd", "Reduce utility shutoff threat"),
    ("shututility", "casthma", "Reduce utility shutoff threat"),
    ("lacktrpt", "casthma", "Reduce transportation barriers"),
]

# Threshold: tracts above this percentile get "treated"
TREATMENT_PERCENTILES = [90, 75]
# Target: reduce HRSN to this percentile
TARGET_PERCENTILE = 50


def run_policy_simulation():
    ensure_dirs()
    params = load_params()

    hrsn_cols = [m["id"].lower() for m in params["hrsn_measures"]]
    outcome_cols = [m["id"].lower() for m in params["outcome_measures"]]
    covariate_cols = ["pct_black", "pct_hispanic", "pct_poverty",
                      "pct_college", "pct_65plus", "median_age"]

    # Load data
    tract_df = load_parquet(PATHS["processed"] / "merged_tracts.parquet")
    logger.info(f"Tract data: {len(tract_df):,} tracts")

    # Fit regression models for all HRSN-disease pairs
    logger.info("\n=== Fitting regression models ===")
    models = {}
    for hrsn in hrsn_cols:
        if hrsn not in tract_df.columns:
            continue
        for disease in outcome_cols:
            if disease not in tract_df.columns:
                continue

            X_cols = [hrsn] + [c for c in covariate_cols if c in tract_df.columns]
            X = sm.add_constant(tract_df[X_cols])
            y = tract_df[disease]
            mask = X.notna().all(axis=1) & y.notna()

            if mask.sum() < 100:
                continue

            model = sm.OLS(y[mask], X[mask]).fit()
            models[(hrsn, disease)] = {
                "model": model,
                "beta": model.params[hrsn],
                "r2": model.rsquared,
            }

    logger.info(f"Fitted {len(models)} regression models")

    # Simulation scenarios
    logger.info("\n=== Running policy simulations ===")
    sim_results = []

    for hrsn in hrsn_cols:
        if hrsn not in tract_df.columns:
            continue

        hrsn_values = tract_df[hrsn].dropna()
        median_val = hrsn_values.quantile(0.50)

        for pct_threshold in TREATMENT_PERCENTILES:
            threshold = hrsn_values.quantile(pct_threshold / 100)
            treated_mask = tract_df[hrsn] > threshold
            n_treated = treated_mask.sum()

            if n_treated == 0:
                continue

            # Population in treated tracts
            pop_treated = tract_df.loc[treated_mask, "total_population"].sum()

            for disease in outcome_cols:
                key = (hrsn, disease)
                if key not in models:
                    continue

                beta = models[key]["beta"]

                # Predicted change in disease prevalence (pp)
                # For treated tracts: new_hrsn = median, old_hrsn = actual
                actual_hrsn = tract_df.loc[treated_mask, hrsn]
                delta_hrsn = actual_hrsn - median_val  # how much we reduce
                delta_disease = beta * delta_hrsn  # predicted prevalence change per tract

                mean_disease_change = delta_disease.mean()
                total_pp_reduction = mean_disease_change  # percentage points

                # Estimated associated case reduction (NOT causal associated case reductions)
                pop_in_treated = tract_df.loc[treated_mask, "total_population"]
                cases_reduction = (delta_disease * pop_in_treated / 100).sum()

                # Attenuation-adjusted estimate (dividing by median inflation ~2.0x)
                attenuation_factor = 2.0
                cases_reduction_adjusted = cases_reduction / attenuation_factor

                sim_results.append({
                    "hrsn_measure": hrsn,
                    "disease": disease,
                    "treatment_threshold_pct": pct_threshold,
                    "hrsn_threshold_value": round(threshold, 1),
                    "hrsn_target_value": round(median_val, 1),
                    "n_tracts_treated": n_treated,
                    "pop_treated": int(pop_treated),
                    "beta": round(beta, 4),
                    "mean_hrsn_reduction_pp": round(delta_hrsn.mean(), 1),
                    "mean_disease_reduction_pp": round(total_pp_reduction, 2),
                    "est_associated_case_reduction": int(cases_reduction),
                    "est_case_reduction_adjusted": int(cases_reduction_adjusted),
                })

    sim_df = pd.DataFrame(sim_results)
    save_csv(sim_df, PATHS["final"] / "policy_simulation_results.csv")

    # Log top scenarios
    logger.info("\n=== Top 10 Scenarios (by estimated associated case reductions, p90 threshold) ===")
    top = sim_df[sim_df["treatment_threshold_pct"] == 90].nlargest(10, "est_associated_case_reduction")
    for _, row in top.iterrows():
        logger.info(f"  {row['hrsn_measure']}→{row['disease']}: "
                   f"reduce {row['mean_hrsn_reduction_pp']:.0f}pp in "
                   f"{row['n_tracts_treated']:,} tracts → "
                   f"{row['mean_disease_reduction_pp']:.1f}pp disease reduction, "
                   f"~{row['est_associated_case_reduction']:,} associated case reductions")

    # Priority tracts: composite HRSN burden × population
    logger.info("\n=== Identifying Priority Intervention Tracts ===")
    priority_df = _identify_priority_tracts(tract_df, hrsn_cols, outcome_cols, models)
    save_csv(priority_df, PATHS["final"] / "priority_intervention_tracts.csv")

    # Plot
    _plot_simulation(sim_df, priority_df)

    return sim_df


def _identify_priority_tracts(tract_df, hrsn_cols, outcome_cols, models):
    """Identify top 500 tracts where HRSN intervention would yield largest health gains."""
    available_hrsn = [c for c in hrsn_cols if c in tract_df.columns]
    df = tract_df.copy()

    # Compute predicted health gain for each tract if all HRSN reduced to median
    df["predicted_gain"] = 0.0

    for hrsn in available_hrsn:
        median_val = df[hrsn].median()
        excess = (df[hrsn] - median_val).clip(lower=0)

        for disease in outcome_cols:
            key = (hrsn, disease)
            if key not in models:
                continue
            beta = models[key]["beta"]
            # Predicted reduction in disease prevalence (pp) × population
            df["predicted_gain"] += beta * excess * df["total_population"] / 100

    # Top 500 tracts — rank by predicted_gain directly (positive = beneficial
    # reduction in disease cases). Using .abs() would treat tracts where HRSN
    # reduction is predicted to WORSEN health as high-priority, which is misleading.
    top500 = df.nlargest(500, "predicted_gain")

    result = top500[["tract_fips", "state_abbr", "county_name", "total_population",
                     "predicted_gain"] + available_hrsn].copy()
    result["rank"] = range(1, len(result) + 1)

    logger.info(f"Top 500 priority tracts identified")
    logger.info(f"  Population covered: {result['total_population'].sum():,}")
    logger.info(f"  States represented: {result['state_abbr'].nunique()}")
    logger.info(f"  Top 5 states: {result['state_abbr'].value_counts().head().to_dict()}")

    return result


def _plot_simulation(sim_df, priority_df):
    """Plot simulation results."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Panel 1: Disease reduction by HRSN measure (p90 threshold)
    ax = axes[0]
    p90 = sim_df[sim_df["treatment_threshold_pct"] == 90].copy()

    if len(p90) > 0:
        # Aggregate: mean disease reduction across all diseases, by HRSN
        agg = p90.groupby("hrsn_measure").agg(
            mean_reduction=("mean_disease_reduction_pp", "mean"),
            total_cases=("est_associated_case_reduction", "sum"),
        ).sort_values("mean_reduction", ascending=True)

        colors = plt.cm.RdYlGn_r(np.linspace(0.2, 0.8, len(agg)))
        ax.barh(range(len(agg)), agg["mean_reduction"], color=colors, alpha=0.8)
        ax.set_yticks(range(len(agg)))
        ax.set_yticklabels(agg.index, fontsize=9)
        ax.set_xlabel("Mean Disease Prevalence Reduction (pp)", fontsize=11)
        ax.set_title("Predicted Impact: Reduce HRSN in Top-Decile\nTracts to National Median",
                    fontsize=12, fontweight="bold")

        # Add case counts as text
        for i, (_, row) in enumerate(agg.iterrows()):
            ax.text(row["mean_reduction"] + 0.02, i,
                   f"~{row['total_cases']:,.0f} cases",
                   va="center", fontsize=8, alpha=0.7)

    # Panel 2: Priority tract characteristics
    ax = axes[1]
    if len(priority_df) > 0:
        state_counts = priority_df["state_abbr"].value_counts().head(15)
        ax.barh(range(len(state_counts)), state_counts.values,
               color="#2c3e50", alpha=0.7)
        ax.set_yticks(range(len(state_counts)))
        ax.set_yticklabels(state_counts.index, fontsize=9)
        ax.set_xlabel("Number of Priority Tracts", fontsize=11)
        ax.set_title(f"Top 500 Priority Intervention Tracts\n"
                    f"({priority_df['total_population'].sum():,} people)",
                    fontsize=12, fontweight="bold")

    plt.tight_layout()
    dest = PATHS["figures"] / "policy_simulation.png"
    fig.savefig(dest, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    logger.info(f"Saved simulation plot to {dest}")


if __name__ == "__main__":
    print("=" * 70)
    print("Step 26: Policy Counterfactual Simulation")
    print("=" * 70)
    results = run_policy_simulation()
    print(f"\nDone. {len(results)} simulation scenarios.")
