"""
Step 37: Validation 4 — HRSN → CDC WONDER Cause-Specific Mortality.

Tests whether PLACES HRSN measures predict county-level cause-specific
mortality from CDC WONDER (death certificates, 2018-2023). This provides
a 4th validation source that is fully independent of BRFSS/PLACES:
death certificates are administrative records from vital statistics,
not survey-based or model-based.

Mapping of PLACES disease outcomes to WONDER mortality conditions:
  - diabetes → diabetes mortality (E10-E14)
  - chd → IHD mortality (I20-I25)
  - copd → COPD mortality (J40-J47)
  - stroke → stroke mortality (I60-I69)
  - casthma → asthma mortality (J45-J46) [high suppression, flagged]

No direct mortality match for depression or obesity.

Approach:
  1. Aggregate HRSN from tract to county (population-weighted means)
  2. Merge with CDC WONDER crude mortality rates
  3. Estimate OLS regressions: mortality ~ HRSN + county covariates
  4. Compare direction/significance with primary ecological analysis
  5. Compute concordance statistics

Output:
  - data/final/mortality_validation_results.csv
  - data/final/mortality_validation_summary.csv
  - outputs/figures/mortality_validation.png
"""
import sys
from pathlib import Path

import pandas as pd
import numpy as np
import statsmodels.api as sm
from scipy import stats
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

# Mapping: PLACES outcome → WONDER mortality condition
DISEASE_MORTALITY_MAP = {
    "diabetes": "mortality_diabetes",
    "chd": "mortality_ihd",
    "copd": "mortality_copd",
    "stroke": "mortality_stroke",
    "casthma": "mortality_asthma",
}

DISEASE_LABELS = {
    "diabetes": "Diabetes",
    "chd": "Coronary Heart Disease",
    "copd": "COPD",
    "stroke": "Stroke",
    "casthma": "Asthma",
}


def run_mortality_validation():
    """Validate HRSN associations against cause-specific mortality."""
    ensure_dirs()
    params = load_params()

    hrsn_cols = [m["id"].lower() for m in params["hrsn_measures"]]
    hrsn_labels = {m["id"].lower(): m["name"] for m in params["hrsn_measures"]}
    covariate_cols = ["pct_black", "pct_hispanic", "pct_poverty",
                      "pct_college", "pct_65plus", "median_age"]

    # --- Load and aggregate tract data to county level ---
    tract_df = load_parquet(PATHS["processed"] / "merged_tracts.parquet")

    # Population-weighted county means for HRSN and covariates
    logger.info("Aggregating tract data to county level (population-weighted)...")

    # Check for population column
    pop_col = None
    for candidate in ["total_population", "total_pop_places", "total_pop", "population"]:
        if candidate in tract_df.columns:
            pop_col = candidate
            break

    if pop_col is None:
        # Try to compute from ACS columns or use equal weights
        logger.warning("No population column found; using equal-weighted county means")
        county_df = tract_df.groupby("county_fips")[hrsn_cols + covariate_cols].mean()
    else:
        logger.info(f"  Using '{pop_col}' for population weighting")
        agg_cols = hrsn_cols + covariate_cols
        weighted_dfs = []
        for col in agg_cols:
            sub = tract_df[["county_fips", col, pop_col]].dropna()
            sub["weighted"] = sub[col] * sub[pop_col]
            grouped = sub.groupby("county_fips").agg(
                wsum=(("weighted", "sum")),
                popsum=((pop_col, "sum")),
            )
            grouped[col] = grouped["wsum"] / grouped["popsum"]
            weighted_dfs.append(grouped[[col]])

        county_df = pd.concat(weighted_dfs, axis=1)

    county_df = county_df.reset_index()
    logger.info(f"  {len(county_df):,} counties with HRSN data")

    # --- Load CDC WONDER mortality ---
    wonder_path = PATHS["raw"] / "cdc_wonder" / "wonder_county_mortality_wide.csv"
    if not wonder_path.exists():
        logger.error(f"CDC WONDER data not found at {wonder_path}")
        logger.info("Run script 34 first to process CDC WONDER files.")
        return None

    mortality_df = pd.read_csv(wonder_path, dtype={"county_fips": str})
    mortality_df["county_fips"] = mortality_df["county_fips"].str.zfill(5)
    logger.info(f"  {len(mortality_df):,} counties with mortality data")

    # --- Load primary analysis results for comparison ---
    primary_results = pd.read_csv(PATHS["final"] / "results_matrix.csv")

    # --- Merge ---
    merged = county_df.merge(mortality_df, on="county_fips", how="inner")
    logger.info(f"  {len(merged):,} counties after merge")

    # --- Standardize HRSN and covariates ---
    for col in hrsn_cols + covariate_cols:
        if col in merged.columns:
            s = merged[col]
            merged[col] = (s - s.mean()) / s.std()

    # --- Run regressions ---
    results = []

    for disease, mort_col in DISEASE_MORTALITY_MAP.items():
        disease_label = DISEASE_LABELS.get(disease, disease)

        if mort_col not in merged.columns:
            logger.warning(f"  Skipping {disease}: {mort_col} not in data")
            continue

        # Check data availability
        valid = merged[mort_col].notna()
        n_valid = valid.sum()
        pct_suppressed = (1 - n_valid / len(merged)) * 100

        logger.info(f"\n--- {disease_label} mortality ---")
        logger.info(f"  {n_valid:,} counties with data ({pct_suppressed:.1f}% suppressed)")

        if n_valid < 100:
            logger.warning(f"  Too few counties ({n_valid}), skipping")
            continue

        # Standardize mortality outcome
        sub = merged[valid].copy()
        y_raw = sub[mort_col]
        sub[mort_col] = (y_raw - y_raw.mean()) / y_raw.std()

        for hrsn in hrsn_cols:
            hrsn_label = hrsn_labels.get(hrsn, hrsn)

            # Regression: mortality ~ HRSN + covariates
            X_cols = [hrsn] + covariate_cols
            X = sub[X_cols].dropna()
            y = sub.loc[X.index, mort_col]
            X = sm.add_constant(X)

            try:
                model = sm.OLS(y, X).fit(cov_type="HC1")
                beta = model.params[hrsn]
                se = model.bse[hrsn]
                pval = model.pvalues[hrsn]
                ci_low = model.conf_int().loc[hrsn, 0]
                ci_high = model.conf_int().loc[hrsn, 1]
                r2 = model.rsquared
            except Exception as e:
                logger.warning(f"  {hrsn} x {disease}: regression failed: {e}")
                continue

            # Get primary analysis result for comparison
            primary_row = primary_results[
                (primary_results["hrsn_measure"] == hrsn) &
                (primary_results["outcome"] == disease)
            ]
            if len(primary_row) > 0:
                primary_beta = float(primary_row["beta_std"].iloc[0])
                primary_sig = float(primary_row["pvalue"].iloc[0]) < 0.05
            else:
                primary_beta = np.nan
                primary_sig = False

            # Concordance: same sign?
            concordant = (np.sign(beta) == np.sign(primary_beta)) if not np.isnan(primary_beta) else False

            sig = pval < 0.05

            results.append({
                "hrsn_measure": hrsn,
                "hrsn_label": hrsn_label,
                "disease": disease,
                "disease_label": disease_label,
                "mortality_col": mort_col,
                "n_counties": len(y),
                "pct_suppressed": round(pct_suppressed, 1),
                "beta_std": round(beta, 4),
                "se": round(se, 4),
                "p_value": round(pval, 6),
                "ci_lower": round(ci_low, 4),
                "ci_upper": round(ci_high, 4),
                "r2": round(r2, 4),
                "significant": sig,
                "primary_beta": round(primary_beta, 4) if not np.isnan(primary_beta) else np.nan,
                "primary_significant": primary_sig,
                "concordant": concordant,
                "both_significant": sig and primary_sig,
            })

            direction = "+" if beta > 0 else "-"
            primary_dir = "+" if primary_beta > 0 else "-" if not np.isnan(primary_beta) else "?"
            match = "MATCH" if concordant else "DIFF"
            logger.info(f"  {hrsn_label:30s}: beta={beta:+.4f} (p={pval:.4f}) "
                        f"[primary={primary_beta:+.4f}] {match}")

    if not results:
        logger.error("No results produced")
        return None

    results_df = pd.DataFrame(results)
    save_csv(results_df, PATHS["final"] / "mortality_validation_results.csv")

    # --- Summary statistics ---
    logger.info("\n" + "=" * 60)
    logger.info("=== Mortality Validation Summary (V4) ===")
    logger.info("=" * 60)

    total = len(results_df)
    n_sig = results_df["significant"].sum()
    n_concordant = results_df["concordant"].sum()
    n_both_sig = results_df["both_significant"].sum()

    # Exclude asthma from concordance (too much suppression)
    non_asthma = results_df[results_df["disease"] != "casthma"]
    n_concordant_no_asthma = non_asthma["concordant"].sum()
    total_no_asthma = len(non_asthma)

    logger.info(f"\nAll diseases ({total} models):")
    logger.info(f"  Significant (p<0.05): {n_sig}/{total} ({n_sig/total*100:.1f}%)")
    logger.info(f"  Directional concordance with primary: {n_concordant}/{total} ({n_concordant/total*100:.1f}%)")
    logger.info(f"  Both significant & concordant: {n_both_sig}/{total} ({n_both_sig/total*100:.1f}%)")

    logger.info(f"\nExcluding asthma ({total_no_asthma} models):")
    logger.info(f"  Concordance: {n_concordant_no_asthma}/{total_no_asthma} "
                f"({n_concordant_no_asthma/total_no_asthma*100:.1f}%)")

    # By disease
    logger.info("\nBy disease:")
    summary_rows = []
    for disease in DISEASE_MORTALITY_MAP:
        sub = results_df[results_df["disease"] == disease]
        if len(sub) == 0:
            continue
        n = len(sub)
        n_sig_d = sub["significant"].sum()
        n_conc = sub["concordant"].sum()
        mean_r2 = sub["r2"].mean()
        n_counties = sub["n_counties"].iloc[0]
        pct_supp = sub["pct_suppressed"].iloc[0]

        # Correlation of betas with primary
        valid_pairs = sub.dropna(subset=["primary_beta"])
        if len(valid_pairs) >= 3:
            r_pearson, p_pearson = stats.pearsonr(valid_pairs["beta_std"], valid_pairs["primary_beta"])
            r_spearman, p_spearman = stats.spearmanr(valid_pairs["beta_std"], valid_pairs["primary_beta"])
        else:
            r_pearson = r_spearman = np.nan

        logger.info(f"  {DISEASE_LABELS.get(disease, disease):25s}: "
                    f"{n_sig_d}/{n} sig, {n_conc}/{n} concordant, "
                    f"R²={mean_r2:.3f}, r(betas)={r_pearson:.3f}, "
                    f"n={n_counties} counties ({pct_supp:.0f}% suppressed)")

        summary_rows.append({
            "disease": disease,
            "disease_label": DISEASE_LABELS.get(disease, disease),
            "n_counties": n_counties,
            "pct_suppressed": pct_supp,
            "n_models": n,
            "n_significant": int(n_sig_d),
            "n_concordant": int(n_conc),
            "pct_concordant": round(n_conc / n * 100, 1),
            "mean_r2": round(mean_r2, 4),
            "pearson_r_betas": round(r_pearson, 4) if not np.isnan(r_pearson) else np.nan,
            "spearman_r_betas": round(r_spearman, 4) if not np.isnan(r_spearman) else np.nan,
        })

    # By HRSN measure
    logger.info("\nBy HRSN measure:")
    for hrsn in hrsn_cols:
        sub = results_df[results_df["hrsn_measure"] == hrsn]
        if len(sub) == 0:
            continue
        n_conc = sub["concordant"].sum()
        n = len(sub)
        logger.info(f"  {hrsn_labels.get(hrsn, hrsn):30s}: {n_conc}/{n} concordant")

    # Overall beta correlation
    valid = results_df.dropna(subset=["primary_beta"])
    if len(valid) >= 3:
        r_all, p_all = stats.pearsonr(valid["beta_std"], valid["primary_beta"])
        rho_all, p_rho = stats.spearmanr(valid["beta_std"], valid["primary_beta"])
        logger.info(f"\nOverall beta correlation (mortality vs primary):")
        logger.info(f"  Pearson r = {r_all:.3f} (p = {p_all:.4f})")
        logger.info(f"  Spearman rho = {rho_all:.3f} (p = {p_rho:.4f})")

        summary_rows.append({
            "disease": "OVERALL",
            "disease_label": "All diseases",
            "n_counties": int(results_df["n_counties"].mean()),
            "pct_suppressed": round(results_df["pct_suppressed"].mean(), 1),
            "n_models": total,
            "n_significant": int(n_sig),
            "n_concordant": int(n_concordant),
            "pct_concordant": round(n_concordant / total * 100, 1),
            "mean_r2": round(results_df["r2"].mean(), 4),
            "pearson_r_betas": round(r_all, 4),
            "spearman_r_betas": round(rho_all, 4),
        })

    summary_df = pd.DataFrame(summary_rows)
    save_csv(summary_df, PATHS["final"] / "mortality_validation_summary.csv")

    # --- Plot ---
    _plot_mortality_validation(results_df, hrsn_labels)

    return results_df


def _plot_mortality_validation(results_df, hrsn_labels):
    """Plot mortality validation: primary beta vs mortality beta."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    # Panel A: Scatter of primary vs mortality betas
    ax = axes[0]
    valid = results_df.dropna(subset=["primary_beta"]).copy()

    # Color by disease
    colors = {"diabetes": "#e74c3c", "chd": "#3498db", "copd": "#2ecc71",
              "stroke": "#9b59b6", "casthma": "#95a5a6"}
    for disease, color in colors.items():
        sub = valid[valid["disease"] == disease]
        if len(sub) == 0:
            continue
        label = DISEASE_LABELS.get(disease, disease)
        marker = "x" if disease == "casthma" else "o"
        alpha = 0.4 if disease == "casthma" else 0.8
        ax.scatter(sub["primary_beta"], sub["beta_std"],
                   c=color, label=label, s=60, alpha=alpha, marker=marker)

    # Reference line
    lims = [min(ax.get_xlim()[0], ax.get_ylim()[0]),
            max(ax.get_xlim()[1], ax.get_ylim()[1])]
    ax.plot(lims, lims, "--", color="gray", alpha=0.5, linewidth=1)
    ax.axhline(0, color="gray", linewidth=0.5)
    ax.axvline(0, color="gray", linewidth=0.5)

    r, p = stats.pearsonr(valid["primary_beta"], valid["beta_std"])
    ax.set_xlabel("Primary Analysis (PLACES → PLACES)\nStandardized Beta", fontsize=11)
    ax.set_ylabel("Mortality Validation (HRSN → WONDER Mortality)\nStandardized Beta", fontsize=11)
    ax.set_title(f"Primary vs Mortality Betas\n(r = {r:.3f}, p = {p:.4f})", fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)

    # Panel B: Concordance by disease
    ax = axes[1]
    diseases = [d for d in DISEASE_MORTALITY_MAP if d in results_df["disease"].values]
    concordance = []
    labels = []
    n_counties_list = []
    for d in diseases:
        sub = results_df[results_df["disease"] == d]
        conc = sub["concordant"].mean() * 100
        concordance.append(conc)
        labels.append(DISEASE_LABELS.get(d, d))
        n_counties_list.append(sub["n_counties"].iloc[0])

    bar_colors = [colors.get(d, "#333") for d in diseases]
    bars = ax.barh(range(len(diseases)), concordance, color=bar_colors, alpha=0.8)
    ax.set_yticks(range(len(diseases)))
    ax.set_yticklabels([f"{l}\n(n={n:,})" for l, n in zip(labels, n_counties_list)], fontsize=10)
    ax.set_xlabel("Directional Concordance with Primary (%)", fontsize=11)
    ax.set_title("Concordance by Disease\n(HRSN → Mortality vs HRSN → PLACES Disease)", fontsize=12, fontweight="bold")
    ax.axvline(50, color="red", linestyle="--", alpha=0.5, label="Chance (50%)")
    ax.set_xlim(0, 100)
    ax.legend(fontsize=9)

    # Add percentage labels on bars
    for bar, val in zip(bars, concordance):
        ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height() / 2,
                f"{val:.0f}%", va="center", fontsize=10)

    plt.suptitle("Validation 4: HRSN Predicts Cause-Specific Mortality\n"
                 "(CDC WONDER death certificates, 2018-2023)",
                 fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    dest = PATHS["figures"] / "mortality_validation.png"
    fig.savefig(dest, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    logger.info(f"\nSaved mortality validation plot to {dest}")


if __name__ == "__main__":
    print("=" * 70)
    print("Step 37: Mortality Validation (HRSN → CDC WONDER Mortality)")
    print("=" * 70)
    results = run_mortality_validation()
    if results is not None:
        print(f"\nDone. {len(results)} mortality validation models estimated.")
    else:
        print("\nFailed. See log for details.")
