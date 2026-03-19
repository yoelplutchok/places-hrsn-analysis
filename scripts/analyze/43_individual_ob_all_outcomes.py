"""
Phase B, Step 43: Individual-Level Oaxaca-Blinder for All 7 Disease Outcomes.

Extends the existing BRFSS individual-level OB decomposition (script 19, which
computed only diabetes) to all 7 disease outcomes. Produces the key cross-level
comparison table for Paper 2 — quantifying "ecological amplification."

Comparison:
  Ecological OB (tract-level): HRSN explains 23-99% of Black-White gaps
  Individual OB (BRFSS):       HRSN explains ~13% (diabetes known)
  → "Ecological amplification" = spatial concentration of disadvantage

Output:
  - data/final/individual_ob_all_outcomes.csv
  - data/final/ecological_amplification_comparison.csv
  - outputs/figures/ecological_amplification_comparison.png
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


def oaxaca_blinder_individual(y_a, X_a, y_b, X_b, var_names, n_bootstrap=500,
                               seed=42):
    """
    Individual-level OB decomposition with bootstrap CIs.

    Uses Neumark/pooled specification with group indicator.
    """
    result = _ob_point_estimate(y_a, X_a, y_b, X_b, var_names)

    rng = np.random.default_rng(seed)
    boot_explained = []
    boot_hrsn_explained = []
    boot_pct = []

    hrsn_vars_in_model = [v for v in var_names if v in result["endowment_contributions"]]

    for i in range(n_bootstrap):
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
            boot_explained.append(boot_result["total_explained"])
            boot_pct.append(boot_result["pct_explained"])

            # HRSN-only contribution
            hrsn_contrib = sum(
                boot_result["endowment_contributions"].get(v, 0)
                for v in hrsn_vars_in_model
            )
            boot_hrsn_explained.append(hrsn_contrib)
        except Exception:
            continue

    ci = {}
    if len(boot_explained) >= 50:
        ci["total_explained"] = (
            float(np.percentile(boot_explained, 2.5)),
            float(np.percentile(boot_explained, 97.5)),
        )
        ci["pct_explained"] = (
            float(np.percentile(boot_pct, 2.5)),
            float(np.percentile(boot_pct, 97.5)),
        )
        ci["hrsn_explained"] = (
            float(np.percentile(boot_hrsn_explained, 2.5)),
            float(np.percentile(boot_hrsn_explained, 97.5)),
        )
    else:
        ci["total_explained"] = (np.nan, np.nan)
        ci["pct_explained"] = (np.nan, np.nan)
        ci["hrsn_explained"] = (np.nan, np.nan)

    result["bootstrap_ci"] = ci
    result["n_bootstrap_successful"] = len(boot_explained)
    return result


def run_individual_ob_all_outcomes():
    """Run individual-level OB decomposition for all 7 disease outcomes."""
    ensure_dirs()
    params = load_params()
    brfss_params = params["brfss"]

    # ---- Load data ----
    brfss_path = PATHS["processed"] / "brfss_analytic.parquet"
    if not brfss_path.exists():
        logger.error(f"BRFSS analytic data not found at {brfss_path}")
        return None

    brfss = load_parquet(brfss_path)
    logger.info(f"Loaded BRFSS: {len(brfss):,} respondents")

    hrsn_cols = list(brfss_params["hrsn_variables"].keys())
    outcome_cols = list(brfss_params["outcome_variables"].keys())
    hrsn_labels = {m["id"].lower(): m["name"] for m in params["hrsn_measures"]}
    outcome_labels = {m["id"].lower(): m["name"] for m in params["outcome_measures"]}

    if "race_cat" not in brfss.columns:
        logger.error("race_cat not found in BRFSS data")
        return None

    white = brfss[brfss["race_cat"] == "white"].copy()
    black = brfss[brfss["race_cat"] == "black"].copy()

    logger.info(f"White respondents: {len(white):,}")
    logger.info(f"Black respondents: {len(black):,}")

    # Individual covariates
    indiv_covs = ["age_group", "female", "income_cat", "education_cat"]
    avail_hrsn = [h for h in hrsn_cols if h in brfss.columns]
    avail_covs = [c for c in indiv_covs if c in brfss.columns]
    predictor_cols = avail_hrsn + avail_covs

    logger.info(f"HRSN variables available: {avail_hrsn}")
    logger.info(f"Covariates available: {avail_covs}")

    # ---- Run OB for all 7 outcomes ----
    logger.info(f"\n=== Individual-Level OB Decomposition (Black vs White) ===")
    logger.info(f"Running for {len(outcome_cols)} disease outcomes...")

    results = []

    for outcome in outcome_cols:
        if outcome not in brfss.columns:
            logger.warning(f"  {outcome} not in BRFSS data — skipping")
            continue

        all_cols = predictor_cols + [outcome]
        wa = white[all_cols].dropna()
        gb = black[all_cols].dropna()

        if len(wa) < 500 or len(gb) < 500:
            logger.warning(f"  {outcome}: insufficient data (White={len(wa)}, Black={len(gb)})")
            continue

        logger.info(f"\n  --- {outcome_labels.get(outcome, outcome)} ---")
        logger.info(f"    White n={len(wa):,}, Black n={len(gb):,}")
        logger.info(f"    Prevalence: White={wa[outcome].mean():.4f}, "
                     f"Black={gb[outcome].mean():.4f}")

        result = oaxaca_blinder_individual(
            y_a=wa[outcome], X_a=wa[predictor_cols],
            y_b=gb[outcome], X_b=gb[predictor_cols],
            var_names=predictor_cols,
            n_bootstrap=500,
        )

        raw_gap = result["raw_gap"]
        if abs(raw_gap) < 1e-10:
            logger.warning(f"    Gap is zero — skipping")
            continue

        pct_exp = result["pct_explained"]
        ci = result.get("bootstrap_ci", {})

        # HRSN-only contribution
        hrsn_contrib = sum(
            result["endowment_contributions"].get(v, 0) for v in avail_hrsn
        )
        hrsn_pct = (hrsn_contrib / raw_gap * 100) if raw_gap != 0 else np.nan

        # Covariate contribution
        cov_contrib = sum(
            result["endowment_contributions"].get(v, 0) for v in avail_covs
        )
        cov_pct = (cov_contrib / raw_gap * 100) if raw_gap != 0 else np.nan

        logger.info(f"    Gap: {raw_gap:+.4f} ({raw_gap*100:+.2f} pp)")
        logger.info(f"    Total explained: {pct_exp:.1f}%")
        logger.info(f"    HRSN contribution: {hrsn_pct:.1f}%")
        logger.info(f"    Covariate contribution: {cov_pct:.1f}%")
        logger.info(f"    Unexplained: {100 - pct_exp:.1f}%")

        pct_ci = ci.get("pct_explained", (np.nan, np.nan))
        logger.info(f"    95% CI for total explained: ({pct_ci[0]:.1f}%, {pct_ci[1]:.1f}%)")

        # Per-HRSN contributions
        logger.info(f"    Per-HRSN contributions:")
        for v in avail_hrsn:
            c = result["endowment_contributions"].get(v, 0)
            v_pct = (c / raw_gap * 100) if raw_gap != 0 else 0
            logger.info(f"      {hrsn_labels.get(v, v):30s}: {v_pct:+.1f}%")

        row = {
            "comparison": "Black_vs_White",
            "outcome": outcome,
            "outcome_label": outcome_labels.get(outcome, outcome),
            "n_white": len(wa),
            "n_black": len(gb),
            "prevalence_white": round(wa[outcome].mean(), 4),
            "prevalence_black": round(gb[outcome].mean(), 4),
            "raw_gap": round(raw_gap, 4),
            "raw_gap_pp": round(raw_gap * 100, 2),
            "total_explained": round(result["total_explained"], 4),
            "total_unexplained": round(result["total_unexplained"], 4),
            "pct_explained_total": round(pct_exp, 2),
            "pct_explained_hrsn": round(hrsn_pct, 2),
            "pct_explained_covariates": round(cov_pct, 2),
            "pct_unexplained": round(100 - pct_exp, 2),
            "ci_lower_pct": round(pct_ci[0], 2) if not np.isnan(pct_ci[0]) else np.nan,
            "ci_upper_pct": round(pct_ci[1], 2) if not np.isnan(pct_ci[1]) else np.nan,
            "n_bootstrap": result["n_bootstrap_successful"],
            "level": "individual",
        }

        # Add per-HRSN contributions
        for v in avail_hrsn:
            c = result["endowment_contributions"].get(v, 0)
            v_pct = (c / raw_gap * 100) if raw_gap != 0 else 0
            row[f"pct_{v}"] = round(v_pct, 2)

        results.append(row)

    results_df = pd.DataFrame(results)
    save_csv(results_df, PATHS["final"] / "individual_ob_all_outcomes.csv")

    # ================================================================
    # Cross-Level Comparison: Ecological vs Individual
    # ================================================================
    logger.info("\n=== Cross-Level Comparison: Ecological Amplification ===")

    eco_path = PATHS["final"] / "ob_decomposition_ecological.csv"
    if not eco_path.exists():
        logger.warning("Ecological OB results not found — skipping comparison")
        return results_df

    eco_df = pd.read_csv(eco_path)

    # Get ecological total % explained per outcome (Black_vs_White, HRSN only)
    eco_bw = eco_df[
        (eco_df["comparison"] == "Black_vs_White") &
        (eco_df["variable_type"] == "hrsn")
    ]

    comparison_rows = []
    for _, ind_row in results_df.iterrows():
        outcome = ind_row["outcome"]

        # Get ecological HRSN total for this outcome
        eco_outcome = eco_bw[eco_bw["outcome"] == outcome]
        if len(eco_outcome) == 0:
            continue

        eco_total_pct = eco_outcome["total_pct_explained"].iloc[0]
        eco_gap_pp = eco_outcome["raw_gap"].iloc[0] * 100  # already in prevalence units
        ind_pct = ind_row["pct_explained_hrsn"]
        ind_gap_pp = ind_row["raw_gap_pp"]

        # Amplification ratio
        if abs(ind_pct) > 0.1:
            amplification = eco_total_pct / ind_pct
        else:
            amplification = np.nan

        comparison_rows.append({
            "outcome": outcome,
            "outcome_label": ind_row["outcome_label"],
            "eco_gap_pp": round(eco_gap_pp, 2),
            "ind_gap_pp": round(ind_gap_pp, 2),
            "eco_pct_hrsn": round(eco_total_pct, 1),
            "ind_pct_hrsn": round(ind_pct, 1),
            "amplification_ratio": round(amplification, 1) if not np.isnan(amplification) else np.nan,
            "eco_n_white": int(eco_outcome["n_white"].iloc[0]),
            "eco_n_black": int(eco_outcome["n_minority"].iloc[0]),
            "ind_n_white": int(ind_row["n_white"]),
            "ind_n_black": int(ind_row["n_black"]),
        })

    comp_df = pd.DataFrame(comparison_rows)
    save_csv(comp_df, PATHS["final"] / "ecological_amplification_comparison.csv")

    logger.info("\nEcological Amplification Summary:")
    logger.info(f"{'Outcome':20s} {'Eco %':>8s} {'Ind %':>8s} {'Ratio':>8s}")
    logger.info("-" * 48)
    for _, row in comp_df.iterrows():
        amp = f"{row['amplification_ratio']:.1f}x" if not np.isnan(row.get("amplification_ratio", np.nan)) else "N/A"
        logger.info(f"{row['outcome_label']:20s} {row['eco_pct_hrsn']:>7.1f}% {row['ind_pct_hrsn']:>7.1f}% {amp:>8s}")

    if len(comp_df) > 0:
        valid_amps = comp_df["amplification_ratio"].dropna()
        if len(valid_amps) > 0:
            logger.info(f"\nMedian amplification ratio: {valid_amps.median():.1f}x")
            logger.info(f"Range: {valid_amps.min():.1f}x to {valid_amps.max():.1f}x")

    # ================================================================
    # Visualization
    # ================================================================
    _create_amplification_plot(comp_df)

    return results_df


def _create_amplification_plot(comp_df):
    """Bar chart comparing ecological vs individual % explained."""
    if len(comp_df) == 0:
        return

    fig, ax = plt.subplots(figsize=(12, 7))

    outcomes = comp_df["outcome_label"].values
    x = np.arange(len(outcomes))
    width = 0.35

    eco_vals = comp_df["eco_pct_hrsn"].values
    ind_vals = comp_df["ind_pct_hrsn"].values

    bars1 = ax.bar(x - width / 2, eco_vals, width, label="Ecological (tract-level)",
                    color="#c0392b", alpha=0.8, edgecolor="white")
    bars2 = ax.bar(x + width / 2, ind_vals, width, label="Individual (BRFSS)",
                    color="#2980b9", alpha=0.8, edgecolor="white")

    # Add amplification ratio labels
    for i, (eco, ind) in enumerate(zip(eco_vals, ind_vals)):
        if abs(ind) > 0.1:
            ratio = eco / ind
            ax.text(i, max(eco, ind) + 2, f"{ratio:.1f}x",
                    ha="center", va="bottom", fontsize=9, fontweight="bold",
                    color="#2c3e50")

    ax.set_xlabel("Disease Outcome", fontsize=12)
    ax.set_ylabel("% of Black-White Gap Explained by HRSN", fontsize=12)
    ax.set_title("Ecological Amplification: Community-Level vs Individual-Level\n"
                  "HRSN Contribution to Racial Health Disparities",
                  fontsize=13, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(outcomes, rotation=45, ha="right", fontsize=10)
    ax.legend(fontsize=11, loc="upper right")
    ax.axhline(y=0, color="black", linewidth=0.5)
    ax.grid(True, axis="y", alpha=0.3)

    plt.tight_layout()
    dest = PATHS["figures"] / "ecological_amplification_comparison.png"
    fig.savefig(dest, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    logger.info(f"Saved amplification comparison plot to {dest}")


if __name__ == "__main__":
    print("=" * 70)
    print("PHASE B: Individual-Level OB Decomposition — All 7 Outcomes")
    print("=" * 70)
    results = run_individual_ob_all_outcomes()
    if results is not None:
        print(f"\nDone. {len(results)} outcome decompositions completed.")
    else:
        print("\nFailed — check BRFSS data availability.")
