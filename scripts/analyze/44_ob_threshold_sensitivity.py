"""
Phase B, Step 44: OB Decomposition Threshold Sensitivity Analysis.

Tests stability of Black-White Oaxaca-Blinder ecological decomposition
across different threshold definitions for "majority-Black" tracts:
  - 40% Black (less restrictive, larger group)
  - 50% Black (standard majority threshold — baseline)
  - 60% Black (more restrictive, smaller group)

Majority-White threshold is held constant at 60%.

Output:
  - data/final/ob_threshold_sensitivity.csv
  - outputs/figures/ob_threshold_sensitivity.png
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


def _ob_point_estimate(y_a, X_a, y_b, X_b, var_names):
    """Core OB computation (same as script 19)."""
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
    for var in var_names:
        if var in beta_pooled.index:
            endowment_contributions[var] = (mean_b[var] - mean_a[var]) * beta_pooled[var]

    total_explained = sum(endowment_contributions.values())

    return {
        "raw_gap": raw_gap,
        "total_explained": total_explained,
        "pct_explained": (total_explained / raw_gap * 100) if raw_gap != 0 else np.nan,
        "n_a": len(y_a),
        "n_b": len(y_b),
    }


def run_threshold_sensitivity():
    """Run OB decomposition at multiple majority-Black thresholds."""
    ensure_dirs()
    params = load_params()

    hrsn_cols = [m["id"].lower() for m in params["hrsn_measures"]]
    outcome_cols = [m["id"].lower() for m in params["outcome_measures"]]
    outcome_labels = {m["id"].lower(): m["name"] for m in params["outcome_measures"]}
    demo_cols = ["pct_poverty", "pct_college", "pct_65plus", "median_age"]
    predictor_cols = hrsn_cols + demo_cols

    df = load_parquet(PATHS["processed"] / "merged_tracts.parquet")

    # Fixed: majority-White threshold at 60%
    white_tracts = df[df["pct_white"] >= 60].copy()
    logger.info(f"Majority-White tracts (>=60%): {len(white_tracts):,}")

    thresholds = [40, 50, 60]
    results = []

    for thresh in thresholds:
        black_tracts = df[df["pct_black"] >= thresh].copy()
        logger.info(f"\n=== Majority-Black threshold: >={thresh}% (n={len(black_tracts):,}) ===")

        for outcome in outcome_cols:
            all_cols = predictor_cols + [outcome]
            wa = white_tracts[all_cols].dropna()
            gb = black_tracts[all_cols].dropna()

            if len(wa) < 100 or len(gb) < 100:
                logger.warning(f"  {outcome}: insufficient data (White={len(wa)}, Black={len(gb)})")
                continue

            result = _ob_point_estimate(
                y_a=wa[outcome], X_a=wa[predictor_cols],
                y_b=gb[outcome], X_b=gb[predictor_cols],
                var_names=predictor_cols,
            )

            # HRSN-only contribution
            hrsn_contrib = sum(
                result.get("endowment_contributions", {}).get(v, 0)
                for v in hrsn_cols
            )
            # Recompute HRSN-only from point estimate
            # Since _ob_point_estimate doesn't return individual contribs,
            # we need the full version
            X_all = pd.concat([wa[predictor_cols], gb[predictor_cols]])
            y_all = pd.concat([wa[outcome], gb[outcome]])
            group_ind = pd.Series(
                [0] * len(wa) + [1] * len(gb),
                index=X_all.index, name="group_B"
            )
            X_all_g = pd.concat([X_all, group_ind], axis=1)
            X_all_c = sm.add_constant(X_all_g)
            res = sm.OLS(y_all, X_all_c).fit()
            beta = res.params

            mean_w = wa[predictor_cols].mean()
            mean_b = gb[predictor_cols].mean()
            raw_gap = result["raw_gap"]

            hrsn_endow = sum(
                (mean_b[v] - mean_w[v]) * beta[v] for v in hrsn_cols if v in beta.index
            )
            hrsn_pct = (hrsn_endow / raw_gap * 100) if raw_gap != 0 else np.nan

            pct_exp = result["pct_explained"]
            logger.info(f"  {outcome_labels.get(outcome, outcome):20s}: "
                         f"gap={raw_gap:+.3f}, total={pct_exp:.1f}%, HRSN={hrsn_pct:.1f}%")

            results.append({
                "threshold_pct_black": thresh,
                "outcome": outcome,
                "outcome_label": outcome_labels.get(outcome, outcome),
                "n_white": len(wa),
                "n_black": len(gb),
                "raw_gap": round(raw_gap, 4),
                "pct_explained_total": round(pct_exp, 2),
                "pct_explained_hrsn": round(hrsn_pct, 2),
            })

    results_df = pd.DataFrame(results)
    save_csv(results_df, PATHS["final"] / "ob_threshold_sensitivity.csv")

    # Stability comparison
    logger.info("\n=== Stability Across Thresholds ===")
    logger.info(f"{'Outcome':20s} {'40%':>10s} {'50%':>10s} {'60%':>10s}")
    logger.info("-" * 52)

    for outcome in outcome_cols:
        vals = []
        for thresh in thresholds:
            row = results_df[(results_df["outcome"] == outcome) &
                             (results_df["threshold_pct_black"] == thresh)]
            if len(row) > 0:
                vals.append(f"{row.iloc[0]['pct_explained_hrsn']:.1f}%")
            else:
                vals.append("N/A")
        logger.info(f"{outcome_labels.get(outcome, outcome):20s} {vals[0]:>10s} {vals[1]:>10s} {vals[2]:>10s}")

    # Visualization
    _create_threshold_plot(results_df, outcome_labels)

    return results_df


def _create_threshold_plot(results_df, outcome_labels):
    """Line plot showing HRSN % explained across thresholds."""
    if len(results_df) == 0:
        return

    fig, ax = plt.subplots(figsize=(10, 6))

    outcomes = results_df["outcome"].unique()
    colors = plt.cm.Set1(np.linspace(0, 0.8, len(outcomes)))

    for i, outcome in enumerate(outcomes):
        subset = results_df[results_df["outcome"] == outcome].sort_values("threshold_pct_black")
        ax.plot(subset["threshold_pct_black"], subset["pct_explained_hrsn"],
                "o-", color=colors[i], linewidth=2, markersize=8,
                label=outcome_labels.get(outcome, outcome))

    ax.set_xlabel("Majority-Black Threshold (% Black)", fontsize=12)
    ax.set_ylabel("% of B-W Gap Explained by HRSN", fontsize=12)
    ax.set_title("OB Decomposition Sensitivity to Majority-Black Threshold\n"
                  "(Majority-White fixed at >=60%)", fontsize=13, fontweight="bold")
    ax.set_xticks([40, 50, 60])
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color="black", linewidth=0.5)

    plt.tight_layout()
    dest = PATHS["figures"] / "ob_threshold_sensitivity.png"
    fig.savefig(dest, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    logger.info(f"Saved threshold sensitivity plot to {dest}")


if __name__ == "__main__":
    print("=" * 70)
    print("OB Decomposition: Threshold Sensitivity Analysis")
    print("=" * 70)
    results = run_threshold_sensitivity()
    print(f"\nDone. {len(results)} decompositions across thresholds.")
