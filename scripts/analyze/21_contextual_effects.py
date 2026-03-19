"""
Contextual Effects Analysis: MMSA-Level HRSN → Individual Disease Outcomes.

Links publicly available SMART BRFSS (individual disease outcomes with MMSA
geography) to MMSA-level aggregated HRSN from PLACES. This creates a
two-level structure:
  Level 1: Individual BRFSS respondents (N≈229K) with disease outcomes + demographics
  Level 2: MMSAs with aggregate HRSN prevalence from PLACES

Question: Does living in an MMSA with high food insecurity predict YOUR
individual diabetes risk, after controlling for your age, sex, race,
income, and education?

Steps:
  1. Load SMART BRFSS 2023, recode outcomes + demographics
  2. Build county→CBSA crosswalk from Census delineation file
  3. Aggregate PLACES HRSN to MMSA level (population-weighted tract means)
  4. Merge individual BRFSS with MMSA-level HRSN
  5. Run contextual effects logistic regressions (7 HRSN × 7 outcomes = 49)
  6. Compare with ecological-only results

Output:
  - data/final/mmsa_hrsn_aggregated.csv
  - data/final/contextual_effects_results.csv
  - data/final/contextual_vs_ecological.csv
  - outputs/figures/contextual_effects_forest.png
"""
import sys
import zipfile
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
from hrsn_analysis.survey_utils import recode_brfss_binary, recode_demographics
from hrsn_analysis.logging_utils import setup_logging, get_logger

setup_logging()
logger = get_logger(__name__)


def load_smart_brfss(params):
    """Load and process SMART BRFSS 2023."""
    smart_dir = PATHS["raw"] / "smart_brfss"
    zip_path = smart_dir / "MMSA2023_XPT.zip"

    # Extract XPT from zip
    xpt_path = smart_dir / "MMSA2023.xpt"
    if not xpt_path.exists():
        logger.info("Extracting SMART BRFSS from zip...")
        with zipfile.ZipFile(zip_path) as z:
            z.extractall(smart_dir)

    logger.info("Loading SMART BRFSS 2023...")
    df = pd.read_sas(xpt_path, format="xport")
    logger.info(f"Raw SMART BRFSS: {len(df):,} respondents × {len(df.columns)} variables")

    # Keep MMSA identifier
    df["mmsa_code"] = df["_MMSA"].astype(int)
    logger.info(f"Unique MMSAs: {df['mmsa_code'].nunique()}")

    # Recode outcomes
    brfss_cfg = params.get("brfss", {})
    outcome_vars = brfss_cfg.get("outcome_variables", {})

    for outcome_name, cfg in outcome_vars.items():
        brfss_var = cfg["brfss_var"]
        recode = cfg["recode"]
        if brfss_var in df.columns:
            df[outcome_name] = recode_brfss_binary(df[brfss_var], recode)
            n_valid = df[outcome_name].notna().sum()
            prev = df[outcome_name].mean() * 100 if n_valid > 0 else 0
            logger.info(f"  {outcome_name:15s}: n={n_valid:,}, prevalence={prev:.1f}%")
        else:
            logger.warning(f"  {outcome_name}: variable {brfss_var} not found")

    # Recode demographics
    demo_df = recode_demographics(df)
    for col in demo_df.columns:
        df[col] = demo_df[col]

    # Survey weights
    weight_col = "_MMSAWT"
    if weight_col in df.columns:
        df["weight"] = df[weight_col]
        df = df[df["weight"].notna() & (df["weight"] > 0)]
        logger.info(f"After weight filter: {len(df):,} respondents")

    return df


def build_county_to_cbsa_crosswalk():
    """Build county FIPS → CBSA code mapping from Census delineation file."""
    xls_path = PATHS["raw"] / "smart_brfss" / "cbsa_delineation_2023.xlsx"
    delin = pd.read_excel(xls_path, header=2)

    # Drop rows with missing FIPS codes
    delin = delin.dropna(subset=["FIPS State Code", "FIPS County Code", "CBSA Code"])

    # Create 5-digit county FIPS
    delin["county_fips"] = (
        delin["FIPS State Code"].astype(int).astype(str).str.zfill(2) +
        delin["FIPS County Code"].astype(int).astype(str).str.zfill(3)
    )
    delin["cbsa_code"] = delin["CBSA Code"].astype(int)

    crosswalk = delin[["county_fips", "cbsa_code", "CBSA Title"]].copy()
    crosswalk = crosswalk.drop_duplicates(subset=["county_fips"])

    logger.info(f"County→CBSA crosswalk: {len(crosswalk):,} counties → "
                f"{crosswalk['cbsa_code'].nunique()} CBSAs")
    return crosswalk


def aggregate_hrsn_to_mmsa(tracts, crosswalk, hrsn_cols):
    """Aggregate tract-level HRSN to MMSA level (population-weighted means)."""
    # Map tracts to CBSAs via county FIPS
    tracts_with_cbsa = tracts.merge(
        crosswalk[["county_fips", "cbsa_code"]],
        on="county_fips", how="inner"
    )
    logger.info(f"Tracts mapped to CBSAs: {len(tracts_with_cbsa):,} "
                f"({tracts_with_cbsa['cbsa_code'].nunique()} CBSAs)")

    # Population-weighted mean of HRSN measures per CBSA
    agg_results = []
    for cbsa, group in tracts_with_cbsa.groupby("cbsa_code"):
        pop = group["total_population"].values
        total_pop = pop.sum()
        if total_pop == 0:
            continue

        row = {"mmsa_code": int(cbsa), "mmsa_total_pop": int(total_pop),
               "mmsa_n_tracts": len(group)}

        for col in hrsn_cols:
            vals = group[col].values
            mask = ~np.isnan(vals) & ~np.isnan(pop)
            if mask.sum() > 0:
                row[f"mmsa_{col}"] = np.average(vals[mask], weights=pop[mask])
            else:
                row[f"mmsa_{col}"] = np.nan

        agg_results.append(row)

    mmsa_hrsn = pd.DataFrame(agg_results)
    logger.info(f"MMSA-level HRSN: {len(mmsa_hrsn)} MMSAs with aggregated measures")
    return mmsa_hrsn


def run_contextual_models(merged, hrsn_cols, outcome_cols,
                           hrsn_labels, outcome_labels):
    """Run contextual effects logistic regressions."""
    logger.info("\n=== Contextual Effects Models ===")
    logger.info("Model: Pr(disease=1) = logit(MMSA_HRSN + individual demographics)")

    demo_cols = ["age_group", "female", "income_cat", "education_cat"]
    # Add race dummies — ensure numeric dtype
    if "race_cat" in merged.columns:
        race_dummies = pd.get_dummies(merged["race_cat"], prefix="race",
                                       drop_first=True, dtype=float)
        for col in race_dummies.columns:
            merged[col] = race_dummies[col].astype(float)
        demo_cols += [c for c in race_dummies.columns]

    avail_demos = [c for c in demo_cols if c in merged.columns]

    # Ensure all demographic columns are float
    for col in avail_demos:
        merged[col] = pd.to_numeric(merged[col], errors="coerce")

    results = []

    for hrsn in hrsn_cols:
        mmsa_col = f"mmsa_{hrsn}"
        if mmsa_col not in merged.columns:
            continue

        # Standardize MMSA-level HRSN (z-score across MMSAs)
        merged[f"{mmsa_col}_z"] = (
            (merged[mmsa_col] - merged[mmsa_col].mean()) / merged[mmsa_col].std()
        ).astype(float)

        for outcome in outcome_cols:
            if outcome not in merged.columns:
                continue

            predictor_cols = [f"{mmsa_col}_z"] + avail_demos
            all_cols = predictor_cols + [outcome, "weight"]
            mask = merged[all_cols].notna().all(axis=1) & (merged[outcome].isin([0, 1]))

            if mask.sum() < 1000:
                continue

            X = sm.add_constant(merged.loc[mask, predictor_cols])
            y = merged.loc[mask, outcome]
            weights = merged.loc[mask, "weight"]

            try:
                # Normalize weights to sum to sample size (prevents inflated N)
                n = len(y)
                w_sum = weights.sum()
                scaled_weights = weights * n / w_sum if w_sum > 0 else weights
                model = sm.GLM(y, X, family=Binomial(), freq_weights=scaled_weights)
                # Cluster SEs by MMSA to account for within-MMSA correlation
                mmsa_cluster = merged.loc[mask, "mmsa_code"]
                res = model.fit(cov_type="cluster", cov_kwds={"groups": mmsa_cluster})

                coef = res.params[f"{mmsa_col}_z"]
                se = res.bse[f"{mmsa_col}_z"]
                pval = res.pvalues[f"{mmsa_col}_z"]
                ci = res.conf_int().loc[f"{mmsa_col}_z"]
                odds_ratio = np.exp(coef)
                or_lower = np.exp(ci[0])
                or_upper = np.exp(ci[1])

                sig = "***" if pval < 0.001 else "**" if pval < 0.01 else "*" if pval < 0.05 else "ns"
                logger.info(f"  {hrsn:15s} × {outcome:12s}: "
                            f"OR={odds_ratio:.3f} ({or_lower:.3f}-{or_upper:.3f}) {sig}  "
                            f"n={int(mask.sum()):,}")

                results.append({
                    "hrsn_measure": hrsn,
                    "hrsn_label": hrsn_labels.get(hrsn, hrsn),
                    "outcome": outcome,
                    "outcome_label": outcome_labels.get(outcome, outcome),
                    "coef": round(coef, 4),
                    "se": round(se, 4),
                    "odds_ratio": round(odds_ratio, 4),
                    "or_lower": round(or_lower, 4),
                    "or_upper": round(or_upper, 4),
                    "pvalue": pval,
                    "significant_05": pval < 0.05,
                    "n_individuals": int(mask.sum()),
                    "n_mmsas": int(merged.loc[mask, "mmsa_code"].nunique()),
                    "analysis_level": "contextual",
                })

            except Exception as e:
                logger.warning(f"  {hrsn} × {outcome}: failed — {e}")

    return pd.DataFrame(results)


def compare_with_ecological(contextual_df, hrsn_labels, outcome_labels):
    """Compare contextual ORs with ecological betas."""
    logger.info("\n=== Contextual vs Ecological Comparison ===")

    try:
        eco = pd.read_csv(PATHS["final"] / "results_matrix.csv")
    except FileNotFoundError:
        logger.warning("Ecological results not found — skipping comparison")
        return pd.DataFrame()

    comparison = []
    for _, ctx_row in contextual_df.iterrows():
        eco_row = eco[(eco["hrsn_measure"] == ctx_row["hrsn_measure"]) &
                      (eco["outcome"] == ctx_row["outcome"])]
        if len(eco_row) == 0:
            continue

        eco_beta = eco_row["beta_std"].values[0]
        eco_sig = eco_row["significant_05"].values[0]
        ctx_coef = ctx_row["coef"]
        ctx_sig = ctx_row["significant_05"]

        # Direction match
        same_direction = (eco_beta > 0) == (ctx_coef > 0)

        comparison.append({
            "hrsn_measure": ctx_row["hrsn_measure"],
            "hrsn_label": ctx_row["hrsn_label"],
            "outcome": ctx_row["outcome"],
            "outcome_label": ctx_row["outcome_label"],
            "ecological_beta": round(eco_beta, 4),
            "ecological_significant": bool(eco_sig),
            "contextual_or": ctx_row["odds_ratio"],
            "contextual_log_or": round(ctx_coef, 4),
            "contextual_significant": bool(ctx_sig),
            "direction_match": same_direction,
        })

    comp_df = pd.DataFrame(comparison)

    if len(comp_df) > 0:
        n_match = comp_df["direction_match"].sum()
        n_both_sig = (comp_df["ecological_significant"] & comp_df["contextual_significant"]).sum()
        logger.info(f"Direction match: {n_match}/{len(comp_df)} ({n_match/len(comp_df)*100:.1f}%)")
        logger.info(f"Both significant: {n_both_sig}/{len(comp_df)}")

    return comp_df


def plot_contextual_forest(results_df, hrsn_labels, outcome_labels):
    """Forest plot of contextual ORs."""
    if len(results_df) == 0:
        return

    # Sort by OR
    df = results_df.sort_values("odds_ratio").copy()
    df["label"] = df["hrsn_label"] + " × " + df["outcome_label"]

    fig, ax = plt.subplots(figsize=(12, max(8, len(df) * 0.22)))
    y_pos = range(len(df))

    colors = ["#e74c3c" if sig else "#95a5a6" for sig in df["significant_05"]]

    for i, (_, row) in enumerate(df.iterrows()):
        ax.plot([row["or_lower"], row["or_upper"]], [i, i],
                color=colors[i], linewidth=1.5, alpha=0.7)
        ax.plot(row["odds_ratio"], i, "o", color=colors[i],
                markersize=5, alpha=0.9)

    ax.axvline(x=1, color="black", linewidth=0.8, linestyle="--")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(df["label"], fontsize=7)
    ax.set_xlabel("Odds Ratio (per 1 SD increase in MMSA-level HRSN)", fontsize=11)
    ax.set_title("Contextual Effects: MMSA-Level HRSN → Individual Disease Risk\n"
                 "(Adjusted for age, sex, race, income, education)",
                 fontsize=12, fontweight="bold")

    # Add legend
    from matplotlib.patches import Patch
    legend_patches = [
        Patch(facecolor="#e74c3c", alpha=0.7, label="Significant (p<0.05)"),
        Patch(facecolor="#95a5a6", alpha=0.7, label="Not significant"),
    ]
    ax.legend(handles=legend_patches, loc="lower right", fontsize=9)

    plt.tight_layout()
    dest = PATHS["figures"] / "contextual_effects_forest.png"
    fig.savefig(dest, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    logger.info(f"Saved forest plot to {dest}")


def run_multilevel_mmsa():
    """Run full contextual effects analysis."""
    ensure_dirs()
    params = load_params()

    hrsn_cols = [m["id"].lower() for m in params["hrsn_measures"]]
    outcome_cols = [m["id"].lower() for m in params["outcome_measures"]]
    hrsn_labels = {m["id"].lower(): m["name"] for m in params["hrsn_measures"]}
    outcome_labels = {m["id"].lower(): m["name"] for m in params["outcome_measures"]}

    # ================================================================
    # Step 1: Load SMART BRFSS
    # ================================================================
    logger.info("=== Step 1: Load SMART BRFSS 2023 ===")
    smart = load_smart_brfss(params)

    # ================================================================
    # Step 2: Build county→CBSA crosswalk
    # ================================================================
    logger.info("\n=== Step 2: County → CBSA Crosswalk ===")
    crosswalk = build_county_to_cbsa_crosswalk()

    # ================================================================
    # Step 3: Aggregate PLACES HRSN to MMSA level
    # ================================================================
    logger.info("\n=== Step 3: Aggregate HRSN to MMSA Level ===")
    tracts = load_parquet(PATHS["processed"] / "merged_tracts.parquet")
    mmsa_hrsn = aggregate_hrsn_to_mmsa(tracts, crosswalk, hrsn_cols)
    save_csv(mmsa_hrsn, PATHS["final"] / "mmsa_hrsn_aggregated.csv")

    # Show MMSA-level HRSN summary
    for col in hrsn_cols:
        mmsa_col = f"mmsa_{col}"
        if mmsa_col in mmsa_hrsn.columns:
            vals = mmsa_hrsn[mmsa_col].dropna()
            logger.info(f"  {hrsn_labels.get(col, col):30s}: "
                        f"mean={vals.mean():.2f}, range=[{vals.min():.2f}, {vals.max():.2f}]")

    # ================================================================
    # Step 4: Merge individual BRFSS with MMSA-level HRSN
    # ================================================================
    logger.info("\n=== Step 4: Merge Individual + MMSA-Level Data ===")

    # Check MMSA overlap
    smart_mmsas = set(smart["mmsa_code"].unique())
    hrsn_mmsas = set(mmsa_hrsn["mmsa_code"].unique())
    overlap = smart_mmsas & hrsn_mmsas
    logger.info(f"SMART BRFSS MMSAs: {len(smart_mmsas)}")
    logger.info(f"HRSN MMSAs: {len(hrsn_mmsas)}")
    logger.info(f"Overlap: {len(overlap)} MMSAs")

    merged = smart.merge(mmsa_hrsn, on="mmsa_code", how="inner")
    logger.info(f"Merged: {len(merged):,} individuals in {merged['mmsa_code'].nunique()} MMSAs")

    # ================================================================
    # Step 5: Run contextual effects models
    # ================================================================
    results = run_contextual_models(
        merged, hrsn_cols, outcome_cols, hrsn_labels, outcome_labels
    )
    if len(results) > 0:
        save_csv(results, PATHS["final"] / "contextual_effects_results.csv")

    # ================================================================
    # Step 6: Compare with ecological results
    # ================================================================
    comparison = compare_with_ecological(results, hrsn_labels, outcome_labels)
    if len(comparison) > 0:
        save_csv(comparison, PATHS["final"] / "contextual_vs_ecological.csv")

    # ================================================================
    # Visualization
    # ================================================================
    if len(results) > 0:
        plot_contextual_forest(results, hrsn_labels, outcome_labels)

    # ================================================================
    # Summary
    # ================================================================
    logger.info("\n=== Final Summary ===")
    if len(results) > 0:
        n_sig = results["significant_05"].sum()
        logger.info(f"Contextual models: {n_sig}/{len(results)} significant")
        logger.info(f"Mean OR: {results['odds_ratio'].mean():.3f}")
        logger.info(f"OR range: [{results['odds_ratio'].min():.3f}, "
                    f"{results['odds_ratio'].max():.3f}]")

        # Strongest contextual effects
        results["abs_or_minus1"] = (results["odds_ratio"] - 1).abs()
        top = results.nlargest(5, "abs_or_minus1")
        logger.info("\nStrongest contextual effects:")
        for _, row in top.iterrows():
            logger.info(f"  {row['hrsn_label']:30s} × {row['outcome_label']:12s}: "
                        f"OR={row['odds_ratio']:.3f}")

    return results, comparison


if __name__ == "__main__":
    print("=" * 70)
    print("Contextual Effects: MMSA-Level HRSN → Individual Disease Outcomes")
    print("=" * 70)
    results, comparison = run_multilevel_mmsa()
    if results is not None:
        print(f"\nDone. {len(results)} contextual effect models.")
