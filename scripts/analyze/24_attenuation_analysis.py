"""
Step 24: Attenuation / Calibration Analysis.

Quantifies how much the shared BRFSS/MRP modeling framework inflates
primary ecological associations by comparing beta magnitudes between:
  - Primary analysis: PLACES HRSN → PLACES Disease (both modeled)
  - V3 validation: ACS Direct → CMS Claims Disease (both independently measured)

For each matched pair (e.g., foodstamp→diabetes vs pct_snap→cms_diabetes_prev),
computes the ratio of primary beta to V3 beta (the "inflation factor").

If ratios are consistently > 1, this quantifies how much shared modeling inflates
the ecological associations. If the correlation between primary and V3 betas is
high, this confirms that the DIRECTION and RELATIVE MAGNITUDE are preserved
even though absolute magnitudes are inflated.

Output:
  - data/final/attenuation_analysis.csv
  - outputs/figures/attenuation_scatter.png
"""
import sys
from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from hrsn_analysis.paths import PATHS, ensure_dirs
from hrsn_analysis.io_utils import save_csv
from hrsn_analysis.logging_utils import setup_logging, get_logger

setup_logging()
logger = get_logger(__name__)

# Mapping: (primary exposure, primary outcome) → (V3 exposure, V3 outcome)
HRSN_TO_ACS = {
    "foodstamp": "pct_snap",
    "foodinsecu": "pct_snap",
    "housinsecu": "pct_rent_burden",
    "lacktrpt": "pct_no_vehicle",
    "loneliness": "pct_living_alone",
    "emotionspt": "pct_living_alone",
}

PLACES_TO_CMS = {
    "diabetes": "cms_diabetes_prev",
    "depression": "cms_depression_prev",
    "copd": "cms_copd_prev",
    "chd": "cms_chd_prev",
    "casthma": "cms_asthma_prev",
    "obesity": "cms_obesity_prev",
    "stroke": "cms_stroke_prev",
}


def run_attenuation_analysis():
    ensure_dirs()

    # Load primary results
    primary_path = PATHS["final"] / "results_matrix.csv"
    primary_df = pd.read_csv(primary_path)
    primary_df = primary_df.rename(columns={"hrsn_measure": "exposure", "beta_std": "beta"})
    logger.info(f"Primary results: {len(primary_df)} associations")

    # Load V3 results
    v3_path = PATHS["final"] / "validation_v3_acs_cms.csv"
    v3_df = pd.read_csv(v3_path)
    logger.info(f"V3 results: {len(v3_df)} associations")

    # Also load V2 for comparison
    v2_path = PATHS["final"] / "validation_v2_hrsn_cms.csv"
    v2_df = pd.read_csv(v2_path)

    # Match primary → V3 pairs
    records = []
    for _, prow in primary_df.iterrows():
        hrsn = prow["exposure"]
        disease = prow["outcome"]
        acs_proxy = HRSN_TO_ACS.get(hrsn)
        cms_outcome = PLACES_TO_CMS.get(disease)

        if acs_proxy is None or cms_outcome is None:
            continue

        # Find V3 match
        v3_match = v3_df[(v3_df["exposure"] == acs_proxy) &
                         (v3_df["outcome"] == cms_outcome)]
        # Find V2 match
        v2_match = v2_df[(v2_df["exposure"] == hrsn) &
                         (v2_df["outcome"] == cms_outcome)]

        record = {
            "hrsn_measure": hrsn,
            "disease": disease,
            "acs_proxy": acs_proxy,
            "cms_outcome": cms_outcome,
            "primary_beta": prow["beta"],
            "primary_se": prow.get("se", np.nan),
            "primary_pvalue": prow.get("pvalue", np.nan),
        }

        if len(v3_match) > 0:
            v3r = v3_match.iloc[0]
            record["v3_beta"] = v3r["beta"]
            record["v3_se"] = v3r["se"]
            record["v3_pvalue"] = v3r["pvalue"]
            record["same_direction"] = np.sign(prow["beta"]) == np.sign(v3r["beta"])

            # Only compute inflation ratio when V3 beta is meaningfully
            # different from zero (|beta| >= 0.05 and p < 0.05).
            # Near-zero denominators produce meaningless ratios (e.g., 96x).
            v3_sig = v3r.get("pvalue", 1.0) < 0.05
            if abs(v3r["beta"]) >= 0.05 and v3_sig:
                record["inflation_ratio"] = abs(prow["beta"]) / abs(v3r["beta"])
            else:
                record["inflation_ratio"] = np.nan

        if len(v2_match) > 0:
            v2r = v2_match.iloc[0]
            record["v2_beta"] = v2r["beta"]
            record["v2_same_direction"] = np.sign(prow["beta"]) == np.sign(v2r["beta"])

        records.append(record)

    att_df = pd.DataFrame(records)
    logger.info(f"\nMatched pairs: {len(att_df)}")

    # Summary statistics
    valid = att_df[att_df["v3_beta"].notna()]
    same_dir = valid["same_direction"].sum()
    logger.info(f"Same direction: {same_dir}/{len(valid)} ({same_dir/len(valid)*100:.0f}%)")

    # Filter transparency: report how many pairs are excluded from ratio computation
    n_total_with_v3 = len(valid)
    n_opposite_dir = (valid["same_direction"] == False).sum()
    n_small_v3_beta = ((valid["v3_beta"].abs() < 0.05) & valid["same_direction"]).sum()
    v3_pvals = valid.get("v3_pvalue", pd.Series(dtype=float))
    n_insig_v3 = ((v3_pvals >= 0.05) & valid["same_direction"] & (valid["v3_beta"].abs() >= 0.05)).sum() if len(v3_pvals) > 0 else 0
    n_with_ratio = att_df["inflation_ratio"].notna().sum()
    logger.info(f"\nInflation ratio filter transparency:")
    logger.info(f"  Total matched pairs with V3 beta: {n_total_with_v3}")
    logger.info(f"  Excluded — opposite direction: {n_opposite_dir}")
    logger.info(f"  Excluded — |V3 beta| < 0.05 (near-zero): {n_small_v3_beta}")
    logger.info(f"  Excluded — V3 p >= 0.05 (not significant): {n_insig_v3}")
    logger.info(f"  Pairs with valid inflation ratio: {n_with_ratio}")

    valid_ratios = att_df[att_df["inflation_ratio"].notna() & att_df["same_direction"]]
    if len(valid_ratios) > 0:
        median_ratio = valid_ratios["inflation_ratio"].median()
        mean_ratio = valid_ratios["inflation_ratio"].mean()
        logger.info(f"\nInflation ratios (same-direction pairs only):")
        logger.info(f"  Median: {median_ratio:.2f}x")
        logger.info(f"  Mean: {mean_ratio:.2f}x")
        logger.info(f"  Range: {valid_ratios['inflation_ratio'].min():.2f}x - "
                    f"{valid_ratios['inflation_ratio'].max():.2f}x")

    # Correlation between primary and V3 betas
    both_valid = att_df[att_df["v3_beta"].notna()]
    if len(both_valid) > 3:
        r, p = stats.pearsonr(both_valid["primary_beta"], both_valid["v3_beta"])
        rho, rho_p = stats.spearmanr(both_valid["primary_beta"], both_valid["v3_beta"])
        logger.info(f"\nPrimary vs V3 beta correlation:")
        logger.info(f"  Pearson r = {r:.3f} (p = {p:.2e})")
        logger.info(f"  Spearman rho = {rho:.3f} (p = {rho_p:.2e})")

    # By HRSN measure
    logger.info("\nInflation by HRSN measure:")
    for hrsn in att_df["hrsn_measure"].unique():
        sub = att_df[(att_df["hrsn_measure"] == hrsn) &
                     att_df["inflation_ratio"].notna() &
                     att_df["same_direction"]]
        if len(sub) > 0:
            logger.info(f"  {hrsn}: median {sub['inflation_ratio'].median():.2f}x "
                       f"(n={len(sub)} same-direction pairs)")
        else:
            sub_all = att_df[(att_df["hrsn_measure"] == hrsn) & att_df["v3_beta"].notna()]
            n_same = sub_all["same_direction"].sum() if len(sub_all) > 0 else 0
            logger.info(f"  {hrsn}: {n_same}/{len(sub_all)} same direction (no ratio)")

    save_csv(att_df, PATHS["final"] / "attenuation_analysis.csv")

    # Plot
    _plot_attenuation(att_df)

    return att_df


def _plot_attenuation(att_df):
    """Scatter: primary beta vs V3 beta with regression line."""
    valid = att_df[att_df["v3_beta"].notna()].copy()
    if len(valid) < 3:
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Panel 1: Primary beta vs V3 beta
    ax = axes[0]
    colors = {"pct_snap": "#e74c3c", "pct_rent_burden": "#3498db",
              "pct_no_vehicle": "#2ecc71", "pct_living_alone": "#9b59b6"}

    for acs_proxy, color in colors.items():
        sub = valid[valid["acs_proxy"] == acs_proxy]
        if len(sub) > 0:
            ax.scatter(sub["primary_beta"], sub["v3_beta"],
                      c=color, s=50, alpha=0.7, label=acs_proxy,
                      edgecolors="white", linewidth=0.5, zorder=3)

    # Regression line
    slope, intercept, r, p, se = stats.linregress(valid["primary_beta"], valid["v3_beta"])
    x_line = np.linspace(valid["primary_beta"].min(), valid["primary_beta"].max(), 100)
    ax.plot(x_line, slope * x_line + intercept, "k-", alpha=0.5, linewidth=1.5,
            label=f"OLS: slope={slope:.2f}")

    # Identity line
    lims = [min(valid["primary_beta"].min(), valid["v3_beta"].min()) - 0.05,
            max(valid["primary_beta"].max(), valid["v3_beta"].max()) + 0.05]
    ax.plot(lims, lims, "k--", alpha=0.3, linewidth=1)
    ax.axhline(0, color="gray", linewidth=0.5, alpha=0.5)
    ax.axvline(0, color="gray", linewidth=0.5, alpha=0.5)

    rho, _ = stats.spearmanr(valid["primary_beta"], valid["v3_beta"])
    ax.text(0.05, 0.95,
            f"Pearson r = {r:.2f}\nSpearman ρ = {rho:.2f}\nn = {len(valid)}\n"
            f"Slope = {slope:.2f}",
            transform=ax.transAxes, fontsize=10, va="top",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

    ax.set_xlabel("Primary β (PLACES HRSN → PLACES Disease)", fontsize=11)
    ax.set_ylabel("V3 β (ACS Direct → CMS Claims Disease)", fontsize=11)
    ax.set_title("Attenuation: Modeled vs. Independent Betas", fontsize=12,
                fontweight="bold")
    ax.legend(fontsize=8, loc="lower right")

    # Panel 2: Inflation ratios for same-direction pairs
    ax = axes[1]
    same_dir = att_df[att_df["same_direction"] == True].copy()
    same_dir = same_dir[same_dir["inflation_ratio"].notna()]

    if len(same_dir) > 0:
        same_dir_sorted = same_dir.sort_values("inflation_ratio")
        labels = [f"{r['hrsn_measure']}→{r['disease']}" for _, r in same_dir_sorted.iterrows()]
        ratios = same_dir_sorted["inflation_ratio"].values

        y_pos = range(len(labels))
        bar_colors = ["#e74c3c" if r > 2 else "#f39c12" if r > 1.5 else "#27ae60"
                     for r in ratios]
        ax.barh(y_pos, ratios, color=bar_colors, alpha=0.7, edgecolor="white")
        ax.set_yticks(y_pos)
        ax.set_yticklabels(labels, fontsize=7)
        ax.axvline(1, color="black", linewidth=1, linestyle="--", alpha=0.7)
        ax.axvline(same_dir["inflation_ratio"].median(), color="red",
                  linewidth=1.5, linestyle="-", alpha=0.7,
                  label=f"Median = {same_dir['inflation_ratio'].median():.1f}x")
        ax.set_xlabel("Inflation Ratio (|Primary β| / |V3 β|)", fontsize=11)
        ax.set_title("Modeling Inflation by Association\n(same-direction pairs only)",
                    fontsize=12, fontweight="bold")
        ax.legend(fontsize=9)
    else:
        ax.text(0.5, 0.5, "No same-direction pairs", ha="center", va="center")

    plt.tight_layout()
    dest = PATHS["figures"] / "attenuation_scatter.png"
    fig.savefig(dest, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    logger.info(f"Saved attenuation plot to {dest}")


if __name__ == "__main__":
    print("=" * 70)
    print("Step 24: Attenuation / Calibration Analysis")
    print("=" * 70)
    results = run_attenuation_analysis()
    print(f"\nDone. {len(results)} matched pairs analyzed.")
