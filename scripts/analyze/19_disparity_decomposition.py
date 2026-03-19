"""
Path 2: Oaxaca-Blinder Decomposition of Racial Health Disparities.

Decomposes the gap in chronic disease prevalence between majority-White and
majority-Black/majority-Hispanic census tracts into portions attributable to
each HRSN measure. Also performs decomposition at the individual level using
BRFSS data.

Questions answered:
  - What fraction of the Black-White diabetes gap is explained by food
    insecurity vs housing insecurity vs transportation barriers?
  - Which HRSN measures explain the most racial health disparity?

Method:
  Oaxaca-Blinder twofold decomposition with Neumark/pooled coefficients:
    Gap = Endowment effect + Unexplained effect
  The endowment effect for each HRSN variable shows how much of the gap
  is "explained" by group differences in that social need.

Output:
  - data/final/ob_decomposition_ecological.csv
  - data/final/ob_decomposition_individual.csv
  - outputs/figures/disparity_decomposition_heatmap.png
  - outputs/figures/disparity_decomposition_bars.png
"""
import sys
from pathlib import Path

import pandas as pd
import numpy as np
import statsmodels.api as sm

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from hrsn_analysis.paths import PATHS, ensure_dirs
from hrsn_analysis.io_utils import load_parquet, load_params, save_csv
from hrsn_analysis.logging_utils import setup_logging, get_logger

setup_logging()
logger = get_logger(__name__)


def _ob_point_estimate(y_a, X_a, y_b, X_b, var_names):
    """Core OB computation returning (raw_gap, endowment_contributions dict)."""
    # Fit pooled model (combined data) with group indicator per Neumark (1988)
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

    # Fit group-specific models
    X_a_c = sm.add_constant(X_a)
    X_b_c = sm.add_constant(X_b)
    res_a = sm.OLS(y_a, X_a_c).fit()
    res_b = sm.OLS(y_b, X_b_c).fit()

    # Mean X for each group
    mean_a = X_a.mean()
    mean_b = X_b.mean()

    # Raw gap (B - A, positive = B is worse)
    raw_gap = y_b.mean() - y_a.mean()

    # Per-variable endowment contribution
    # Endowment = (mean_X_B - mean_X_A) * beta_pooled (excluding constant)
    endowment_contributions = {}
    total_endowment = 0
    for var in var_names:
        if var in beta_pooled.index:
            contrib = (mean_b[var] - mean_a[var]) * beta_pooled[var]
            endowment_contributions[var] = contrib
            total_endowment += contrib

    # Total coefficient effect (unexplained)
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
    """
    Twofold Oaxaca-Blinder decomposition using pooled (Neumark) coefficients.

    Groups: A (advantaged, e.g. White), B (disadvantaged, e.g. Black)
    Gap = mean(Y_B) - mean(Y_A) [positive = B is worse]

    Decomposition (Neumark/pooled twofold with group indicator):
      Gap = (X_B - X_A) @ beta_pooled     [Endowment: explained by X differences]
          + residual                       [Unexplained: gap - endowment]

    Returns per-variable endowment contributions with bootstrap 95% CIs.
    """
    # Point estimate
    result = _ob_point_estimate(y_a, X_a, y_b, X_b, var_names)

    # Bootstrap CIs
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

    # Compute 95% CIs from bootstrap distribution
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


def run_ecological_decomposition(df, hrsn_cols, outcome_cols, demo_cols,
                                  hrsn_labels, outcome_labels):
    """Decompose ecological (tract-level) racial health disparities."""
    logger.info("\n=== Ecological Decomposition (Tract-Level) ===")

    # Define racial composition groups
    # Majority-White: pct_white >= 60
    # Majority-Black: pct_black >= 50 (standard majority threshold)
    # Majority-Hispanic: pct_hispanic >= 50
    white_tracts = df[df["pct_white"] >= 60].copy()
    black_tracts = df[df["pct_black"] >= 50].copy()
    hisp_tracts = df[df["pct_hispanic"] >= 50].copy()

    logger.info(f"  Majority-White tracts (≥60%): {len(white_tracts):,}")
    logger.info(f"  Majority-Black tracts (≥50%): {len(black_tracts):,}")
    logger.info(f"  Majority-Hispanic tracts (≥50%): {len(hisp_tracts):,}")

    results = []
    predictor_cols = hrsn_cols + demo_cols

    for comparison, group_b, group_b_name in [
        ("Black_vs_White", black_tracts, "Majority-Black"),
        ("Hispanic_vs_White", hisp_tracts, "Majority-Hispanic"),
    ]:
        logger.info(f"\n  --- {comparison} ---")

        for outcome in outcome_cols:
            # Clean data
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

            # Log summary
            pct_exp = result["pct_explained"]
            logger.info(f"    {outcome_labels.get(outcome, outcome):20s}: "
                        f"gap={raw_gap:+.3f}, explained={pct_exp:.1f}%")

            boot_ci = result.get("bootstrap_ci", {})

            # Per-HRSN contributions
            for var in hrsn_cols:
                contrib = result["endowment_contributions"].get(var, 0)
                pct_of_gap = (contrib / raw_gap * 100) if raw_gap != 0 else 0
                ci_lo, ci_hi = boot_ci.get(var, (np.nan, np.nan))

                results.append({
                    "comparison": comparison,
                    "outcome": outcome,
                    "outcome_label": outcome_labels.get(outcome, outcome),
                    "variable": var,
                    "variable_label": hrsn_labels.get(var, var),
                    "variable_type": "hrsn",
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

            # Also report demographic contributions
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

    return pd.DataFrame(results)


def run_individual_decomposition(brfss, hrsn_cols, outcome_cols,
                                  hrsn_labels, outcome_labels):
    """Decompose individual-level (BRFSS) racial health disparities."""
    logger.info("\n=== Individual-Level Decomposition (BRFSS) ===")

    if "race_cat" not in brfss.columns:
        logger.warning("  race_cat not found in BRFSS data — skipping")
        return pd.DataFrame()

    white = brfss[brfss["race_cat"] == "white"].copy()
    black = brfss[brfss["race_cat"] == "black"].copy()
    hisp = brfss[brfss["race_cat"] == "hispanic"].copy()

    logger.info(f"  White respondents: {len(white):,}")
    logger.info(f"  Black respondents: {len(black):,}")
    logger.info(f"  Hispanic respondents: {len(hisp):,}")

    # Individual covariates
    indiv_covs = ["age_group", "female", "income_cat", "education_cat"]
    # Available HRSN (check which exist)
    avail_hrsn = [h for h in hrsn_cols if h in brfss.columns]
    avail_covs = [c for c in indiv_covs if c in brfss.columns]
    predictor_cols = avail_hrsn + avail_covs

    results = []

    for comparison, group_b, group_b_name in [
        ("Black_vs_White", black, "Black"),
        ("Hispanic_vs_White", hisp, "Hispanic"),
    ]:
        logger.info(f"\n  --- {comparison} ---")

        for outcome in outcome_cols:
            if outcome not in brfss.columns:
                continue

            all_cols = predictor_cols + [outcome]
            wa = white[all_cols].dropna()
            gb = group_b[all_cols].dropna()

            if len(wa) < 500 or len(gb) < 500:
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
                        f"gap={raw_gap:+.4f}, explained={pct_exp:.1f}%")

            for var in avail_hrsn:
                contrib = result["endowment_contributions"].get(var, 0)
                pct_of_gap = (contrib / raw_gap * 100) if raw_gap != 0 else 0

                results.append({
                    "comparison": comparison,
                    "outcome": outcome,
                    "outcome_label": outcome_labels.get(outcome, outcome),
                    "variable": var,
                    "variable_label": hrsn_labels.get(var, var),
                    "variable_type": "hrsn",
                    "raw_gap": round(raw_gap, 4),
                    "endowment_contribution": round(contrib, 4),
                    "pct_of_gap": round(pct_of_gap, 2),
                    "mean_white": round(result["mean_a"].get(var, 0), 4),
                    "mean_minority": round(result["mean_b"].get(var, 0), 4),
                    "n_white": result["n_a"],
                    "n_minority": result["n_b"],
                    "total_pct_explained": round(pct_exp, 2),
                    "level": "individual",
                })

    return pd.DataFrame(results)


def plot_decomposition_heatmap(eco_df, hrsn_cols, hrsn_labels,
                                outcome_cols, outcome_labels):
    """Heatmap of % of gap explained by each HRSN measure."""
    logger.info("\n=== Creating decomposition heatmap ===")

    fig, axes = plt.subplots(1, 2, figsize=(18, 7))

    for idx, comparison in enumerate(["Black_vs_White", "Hispanic_vs_White"]):
        ax = axes[idx]
        subset = eco_df[(eco_df["comparison"] == comparison) &
                        (eco_df["variable_type"] == "hrsn")]

        if len(subset) == 0:
            continue

        # Pivot: rows = HRSN measures, columns = outcomes
        pivot = subset.pivot_table(
            index="variable", columns="outcome",
            values="pct_of_gap", aggfunc="first"
        )

        # Reorder
        row_order = [h for h in hrsn_cols if h in pivot.index]
        col_order = [o for o in outcome_cols if o in pivot.columns]
        pivot = pivot.reindex(index=row_order, columns=col_order)

        # Rename for display
        pivot.index = [hrsn_labels.get(h, h) for h in pivot.index]
        pivot.columns = [outcome_labels.get(o, o) for o in pivot.columns]

        # Plot
        vmax = max(abs(pivot.min().min()), abs(pivot.max().max()))
        vmax = max(vmax, 10)  # minimum scale
        im = ax.imshow(pivot.values, cmap="RdBu_r", vmin=-vmax, vmax=vmax,
                        aspect="auto")

        ax.set_xticks(range(len(pivot.columns)))
        ax.set_xticklabels(pivot.columns, rotation=45, ha="right", fontsize=9)
        ax.set_yticks(range(len(pivot.index)))
        ax.set_yticklabels(pivot.index, fontsize=9)

        # Annotate cells
        for i in range(len(pivot.index)):
            for j in range(len(pivot.columns)):
                val = pivot.iloc[i, j]
                if not np.isnan(val):
                    color = "white" if abs(val) > vmax * 0.6 else "black"
                    ax.text(j, i, f"{val:.1f}%", ha="center", va="center",
                            fontsize=8, color=color)

        comp_label = comparison.replace("_vs_", " vs ")
        ax.set_title(f"{comp_label}\n% of Disease Gap Explained by HRSN",
                     fontsize=12, fontweight="bold")

        plt.colorbar(im, ax=ax, shrink=0.8, label="% of gap explained")

    plt.tight_layout()
    dest = PATHS["figures"] / "disparity_decomposition_heatmap.png"
    fig.savefig(dest, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    logger.info(f"Saved heatmap to {dest}")


def plot_decomposition_bars(eco_df, hrsn_cols, hrsn_labels,
                             outcome_labels):
    """Stacked bar chart: contribution of each HRSN to disparity."""
    logger.info("=== Creating decomposition bar chart ===")

    bw = eco_df[(eco_df["comparison"] == "Black_vs_White") &
                (eco_df["variable_type"] == "hrsn")]

    if len(bw) == 0:
        logger.warning("No Black_vs_White decomposition data — skipping bar chart")
        return

    outcomes = bw["outcome"].unique()
    hrsn_ordered = [h for h in hrsn_cols if h in bw["variable"].values]

    fig, ax = plt.subplots(figsize=(14, 7))

    x = np.arange(len(outcomes))
    bottom_pos = np.zeros(len(outcomes))
    bottom_neg = np.zeros(len(outcomes))

    colors = plt.cm.Set2(np.linspace(0, 1, len(hrsn_ordered)))

    for i, hrsn in enumerate(hrsn_ordered):
        vals = []
        for outcome in outcomes:
            row = bw[(bw["outcome"] == outcome) & (bw["variable"] == hrsn)]
            vals.append(row["pct_of_gap"].values[0] if len(row) > 0 else 0)
        vals = np.array(vals)

        pos_vals = np.where(vals > 0, vals, 0)
        neg_vals = np.where(vals < 0, vals, 0)

        if pos_vals.any():
            ax.bar(x, pos_vals, bottom=bottom_pos, label=hrsn_labels.get(hrsn, hrsn),
                   color=colors[i], edgecolor="white", linewidth=0.5)
            bottom_pos += pos_vals
        if neg_vals.any():
            ax.bar(x, neg_vals, bottom=bottom_neg, color=colors[i],
                   edgecolor="white", linewidth=0.5)
            bottom_neg += neg_vals

    ax.set_xticks(x)
    ax.set_xticklabels([outcome_labels.get(o, o) for o in outcomes],
                       rotation=45, ha="right", fontsize=10)
    ax.set_ylabel("% of Black-White Gap Explained", fontsize=12)
    ax.set_title("Oaxaca-Blinder Decomposition:\nHRSN Contributions to Racial Health Disparities",
                 fontsize=13, fontweight="bold")
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=9)
    ax.axhline(y=0, color="black", linewidth=0.5)

    plt.tight_layout()
    dest = PATHS["figures"] / "disparity_decomposition_bars.png"
    fig.savefig(dest, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    logger.info(f"Saved bar chart to {dest}")


def run_decomposition():
    """Run full disparity decomposition analysis."""
    ensure_dirs()
    params = load_params()

    hrsn_cols = [m["id"].lower() for m in params["hrsn_measures"]]
    outcome_cols = [m["id"].lower() for m in params["outcome_measures"]]
    hrsn_labels = {m["id"].lower(): m["name"] for m in params["hrsn_measures"]}
    outcome_labels = {m["id"].lower(): m["name"] for m in params["outcome_measures"]}

    demo_cols = ["pct_poverty", "pct_college", "pct_65plus", "median_age"]
    # Note: don't include pct_black/pct_hispanic as controls here —
    # they define the groups, not covariates

    # ---- Ecological decomposition ----
    df = load_parquet(PATHS["processed"] / "merged_tracts.parquet")
    eco_results = run_ecological_decomposition(
        df, hrsn_cols, outcome_cols, demo_cols, hrsn_labels, outcome_labels
    )
    if len(eco_results) > 0:
        save_csv(eco_results, PATHS["final"] / "ob_decomposition_ecological.csv")

    # ---- Individual-level decomposition ----
    brfss_path = PATHS["processed"] / "brfss_analytic.parquet"
    indiv_results = pd.DataFrame()
    if brfss_path.exists():
        brfss = load_parquet(brfss_path)
        indiv_results = run_individual_decomposition(
            brfss, hrsn_cols, outcome_cols, hrsn_labels, outcome_labels
        )
        if len(indiv_results) > 0:
            save_csv(indiv_results, PATHS["final"] / "ob_decomposition_individual.csv")
    else:
        logger.warning("BRFSS analytic data not found — skipping individual decomposition")

    # ---- Visualizations ----
    if len(eco_results) > 0:
        plot_decomposition_heatmap(
            eco_results, hrsn_cols, hrsn_labels, outcome_cols, outcome_labels
        )
        plot_decomposition_bars(
            eco_results, hrsn_cols, hrsn_labels, outcome_labels
        )

    # ---- Summary ----
    logger.info("\n=== Final Summary ===")
    if len(eco_results) > 0:
        hrsn_eco = eco_results[eco_results["variable_type"] == "hrsn"]
        bw_hrsn = hrsn_eco[hrsn_eco["comparison"] == "Black_vs_White"]
        if len(bw_hrsn) > 0:
            # Total HRSN contribution per outcome
            by_outcome = bw_hrsn.groupby("outcome")["pct_of_gap"].sum()
            logger.info("\nBlack-White gap: total HRSN contribution per outcome:")
            for outcome, pct in by_outcome.items():
                logger.info(f"  {outcome_labels.get(outcome, outcome):25s}: "
                            f"{pct:+.1f}% explained by HRSN")

            # Most important HRSN measure overall
            by_hrsn = bw_hrsn.groupby("variable")["pct_of_gap"].mean()
            logger.info("\nMean |%| explained across outcomes (Black-White):")
            for var in by_hrsn.abs().sort_values(ascending=False).index:
                logger.info(f"  {hrsn_labels.get(var, var):30s}: "
                            f"{by_hrsn[var]:+.1f}%")

    return eco_results, indiv_results


if __name__ == "__main__":
    print("=" * 70)
    print("PATH 2: Oaxaca-Blinder Disparity Decomposition")
    print("=" * 70)
    eco, indiv = run_decomposition()
    n_eco = len(eco) if eco is not None else 0
    n_indiv = len(indiv) if indiv is not None else 0
    print(f"\nDone. {n_eco} ecological + {n_indiv} individual decomposition results.")
