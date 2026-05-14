"""
Phase D, Step 23: Independent Validation Framework.

Addresses the modeled-on-modeled limitation by running three validation
analyses with independently-measured data, then comparing all four analyses
for convergence.

Analysis Grid:
  Primary  : PLACES HRSN (modeled)  → PLACES Disease (modeled)  [tract]
  Valid. 1 : ACS Direct (measured)  → PLACES Disease (modeled)  [tract]
  Valid. 2 : PLACES HRSN (modeled)  → CMS Disease (claims)     [county]
  Valid. 3 : ACS Direct (measured)  → CMS Disease (claims)     [county]

The key innovation: V2 and V3 use Medicare claims-derived disease prevalence
from the CMS Mapping Medicare Disparities tool. Each PLACES disease outcome
(e.g., diabetes from BRFSS/MRP) has a direct CMS claims counterpart
(e.g., diabetes prevalence among Medicare FFS beneficiaries from ICD codes).

If associations converge across all 4 analyses — especially between
BRFSS-modeled outcomes (Primary) and claims-based outcomes (V2/V3) —
this provides strong evidence that findings are not artifacts of shared
BRFSS/MRP modeling.

ACS Direct Exposure Proxies:
  pct_snap        → proxy for FOODSTAMP / FOODINSECU
  pct_rent_burden → proxy for HOUSINSECU
  pct_no_vehicle  → proxy for LACKTRPT
  pct_living_alone → proxy for LONELINESS / EMOTIONSPT

CMS Claims-Based Disease Prevalence (county-level):
  cms_diabetes_prev    → PLACES diabetes
  cms_depression_prev  → PLACES depression
  cms_copd_prev        → PLACES copd
  cms_chd_prev         → PLACES chd
  cms_asthma_prev      → PLACES casthma
  cms_obesity_prev     → PLACES obesity
  cms_stroke_prev      → PLACES stroke

Supplementary Independent Outcomes (county-level):
  medicare_risk_score, medicare_spending_pc, medicare_er_rate,
  medicare_hosp_rate, chr_premature_mortality

Output:
  - data/final/validation_v1_acs_places.csv    (tract-level)
  - data/final/validation_v2_hrsn_cms.csv      (county-level)
  - data/final/validation_v3_acs_cms.csv       (county-level)
  - data/final/validation_convergence.csv      (all 4 analyses compared)
  - outputs/figures/validation_convergence_plot.png
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


# Mapping from ACS direct proxies to PLACES HRSN measures
ACS_TO_HRSN = {
    "pct_snap": ["foodstamp", "foodinsecu"],
    "pct_rent_burden": ["housinsecu"],
    "pct_no_vehicle": ["lacktrpt"],
    "pct_living_alone": ["loneliness", "emotionspt"],
}

# Reverse: which ACS proxy best represents each HRSN measure
HRSN_TO_ACS = {
    "foodstamp": "pct_snap",
    "foodinsecu": "pct_snap",
    "housinsecu": "pct_rent_burden",
    "shututility": None,  # No direct ACS proxy
    "lacktrpt": "pct_no_vehicle",
    "loneliness": "pct_living_alone",
    "emotionspt": "pct_living_alone",
}

# Direct mapping: PLACES disease outcome → CMS claims-based disease outcome
PLACES_TO_CMS = {
    "diabetes": "cms_diabetes_prev",
    "depression": "cms_depression_prev",
    "copd": "cms_copd_prev",
    "chd": "cms_chd_prev",
    "casthma": "cms_asthma_prev",
    "obesity": "cms_obesity_prev",
    "stroke": "cms_stroke_prev",
}

# Supplementary independent outcomes (utilization + mortality)
SUPPLEMENTARY_OUTCOMES = [
    "medicare_risk_score",
    "medicare_spending_pc",
    "medicare_er_rate",
    "medicare_hosp_rate",
    "chr_premature_mortality",
]


def standardize(series):
    """Z-score standardize."""
    s = series.dropna()
    if len(s) == 0 or s.std() == 0:
        return series - series.mean() if len(s) > 0 else series
    return (series - series.mean()) / series.std()


def run_ols_battery(df, exposure_cols, outcome_cols, covariate_cols,
                    cluster_col=None, weight_col=None, label=""):
    """Run a battery of OLS regressions for all exposure x outcome pairs."""
    results = []
    for exp in exposure_cols:
        if exp not in df.columns:
            continue
        for out in outcome_cols:
            if out not in df.columns:
                continue

            X_cols = [exp] + [c for c in covariate_cols if c in df.columns]
            X = sm.add_constant(df[X_cols])
            y = df[out]

            mask = X.notna().all(axis=1) & y.notna()
            X_clean = X[mask]
            y_clean = y[mask]

            if len(y_clean) < 30:
                continue

            try:
                model = sm.OLS(y_clean, X_clean)
                if cluster_col is not None and cluster_col in df.columns:
                    cluster_clean = df.loc[mask, cluster_col]
                    res = model.fit(cov_type="cluster",
                                    cov_kwds={"groups": cluster_clean})
                else:
                    res = model.fit(cov_type="HC1")

                beta = res.params[exp]
                pval = res.pvalues[exp]

                results.append({
                    "exposure": exp,
                    "outcome": out,
                    "beta": round(beta, 4),
                    "se": round(res.bse[exp], 4),
                    "pvalue": pval,
                    "significant": pval < 0.05,
                    "r_squared": round(res.rsquared, 4),
                    "n_obs": int(res.nobs),
                    "analysis": label,
                })
            except Exception as e:
                logger.warning(f"  {label}: {exp} x {out} failed — {e}")

    return pd.DataFrame(results)


def run_validation_framework():
    """Run the full 4-analysis validation framework."""
    ensure_dirs()
    params = load_params()

    hrsn_cols = [m["id"].lower() for m in params["hrsn_measures"]]
    outcome_cols = [m["id"].lower() for m in params["outcome_measures"]]
    covariate_cols = ["pct_black", "pct_hispanic", "pct_poverty",
                      "pct_college", "pct_65plus", "median_age"]
    acs_direct_cols = ["pct_snap", "pct_rent_burden", "pct_no_vehicle",
                       "pct_living_alone"]

    # CMS disease outcome columns
    cms_disease_cols = list(PLACES_TO_CMS.values())

    # ================================================================
    # Load data
    # ================================================================
    logger.info("=== Loading datasets ===")

    # Tract-level merged data (PLACES HRSN + disease + Census covariates)
    tract_df = load_parquet(PATHS["processed"] / "merged_tracts.parquet")
    logger.info(f"Tract data: {len(tract_df):,} tracts")

    # Independent validation outcomes (county-level)
    val_path = PATHS["processed"] / "validation_outcomes.csv"
    has_val = val_path.exists()
    if has_val:
        val_df = pd.read_csv(val_path, dtype={"county_fips": str})
        val_df["county_fips"] = val_df["county_fips"].str.zfill(5)

        available_cms = [c for c in cms_disease_cols if c in val_df.columns
                         and val_df[c].notna().sum() > 50]
        available_supp = [c for c in SUPPLEMENTARY_OUTCOMES if c in val_df.columns
                          and val_df[c].notna().sum() > 50]
        logger.info(f"Validation outcomes: {len(val_df):,} counties")
        logger.info(f"  CMS disease prevalence: {available_cms}")
        logger.info(f"  Supplementary outcomes: {available_supp}")
    else:
        logger.warning(f"Validation outcomes not found at {val_path} — "
                       "skipping V2 & V3. Run 22_download_cms_medicare.py first.")
        val_df = None
        available_cms = []
        available_supp = []

    # Check if ACS direct proxies are available
    available_acs = [c for c in acs_direct_cols if c in tract_df.columns]
    if not available_acs:
        logger.error("No ACS direct social needs proxies found in tract data.")
        return None
    logger.info(f"ACS direct proxies available: {available_acs}")

    # ================================================================
    # Standardize tract-level data
    # ================================================================
    df_std = tract_df.copy()
    all_std_cols = hrsn_cols + outcome_cols + covariate_cols + available_acs
    for col in all_std_cols:
        if col in df_std.columns:
            df_std[col] = standardize(tract_df[col])

    # ================================================================
    # ANALYSIS 0 (Primary): PLACES HRSN → PLACES Disease [tract]
    # ================================================================
    logger.info("\n=== Analysis 0 (Primary): PLACES HRSN → PLACES Disease ===")

    primary_path = PATHS["final"] / "results_matrix.csv"
    if primary_path.exists():
        primary_df = pd.read_csv(primary_path)
        primary_df = primary_df.rename(columns={
            "hrsn_measure": "exposure",
            "outcome": "outcome",
            "beta_std": "beta",
        })
        primary_df["analysis"] = "primary_places_hrsn_to_places_disease"
        if "significant_05" in primary_df.columns:
            primary_df["significant"] = primary_df["significant_05"]
        elif "pvalue" in primary_df.columns:
            primary_df["significant"] = primary_df["pvalue"] < 0.05
        keep_cols = ["exposure", "outcome", "beta", "se", "pvalue",
                     "r_squared", "n_obs", "analysis", "significant"]
        available_keep = [c for c in keep_cols if c in primary_df.columns]
        primary_df = primary_df[available_keep]
        logger.info(f"Loaded {len(primary_df)} primary results from {primary_path}")
    else:
        primary_df = run_ols_battery(
            df_std, hrsn_cols, outcome_cols, covariate_cols,
            cluster_col="county_fips",
            label="primary_places_hrsn_to_places_disease"
        )
        logger.info(f"Primary: {len(primary_df)} models, "
                    f"{primary_df['significant'].sum()} significant")

    # ================================================================
    # VALIDATION 1: ACS Direct → PLACES Disease [tract]
    # ================================================================
    logger.info("\n=== Validation 1: ACS Direct → PLACES Disease [tract] ===")

    v1_df = run_ols_battery(
        df_std, available_acs, outcome_cols, covariate_cols,
        cluster_col="county_fips",
        label="v1_acs_direct_to_places_disease"
    )
    logger.info(f"V1: {len(v1_df)} models, {v1_df['significant'].sum()} significant")
    save_csv(v1_df, PATHS["final"] / "validation_v1_acs_places.csv")

    # ================================================================
    # VALIDATION 2 & 3: County-level with CMS disease prevalence
    # ================================================================
    v2_df = pd.DataFrame()
    v3_df = pd.DataFrame()

    if has_val and len(available_cms) > 0:
        # Aggregate tract data to county level
        logger.info("\n=== Aggregating tract data to county level ===")

        tract_df["county_fips"] = tract_df["tract_fips"].str[:5]
        county_agg_cols = hrsn_cols + available_acs + covariate_cols
        county_agg_cols = [c for c in county_agg_cols if c in tract_df.columns]

        # Population-weighted mean
        county_records = []
        for fips, grp in tract_df.groupby("county_fips"):
            record = {"county_fips": fips}
            pop = grp["total_population"]
            for col in county_agg_cols:
                vals = grp[col]
                valid = vals.notna() & pop.notna() & (pop > 0)
                if valid.sum() > 0:
                    record[col] = np.average(vals[valid], weights=pop[valid])
                else:
                    record[col] = np.nan
            county_records.append(record)

        county_df = pd.DataFrame(county_records)
        logger.info(f"County aggregation: {len(county_df):,} counties")

        # Merge with validation outcomes
        county_merged = county_df.merge(val_df, on="county_fips", how="inner")
        logger.info(f"After merging with validation data: {len(county_merged):,} counties")

        if len(county_merged) > 100:
            # Standardize county-level data
            county_std = county_merged.copy()
            all_county_cols = (hrsn_cols + available_acs + covariate_cols
                               + available_cms + available_supp)
            for col in all_county_cols:
                if col in county_std.columns:
                    county_std[col] = standardize(county_merged[col])

            # VALIDATION 2: PLACES HRSN → CMS Disease Prevalence [county]
            logger.info("\n=== Validation 2: PLACES HRSN → CMS Disease [county] ===")

            v2_available_hrsn = [c for c in hrsn_cols if c in county_std.columns]
            v2_covariates = [c for c in covariate_cols if c in county_std.columns]
            v2_df = run_ols_battery(
                county_std, v2_available_hrsn, available_cms,
                v2_covariates,
                label="v2_places_hrsn_to_cms_disease"
            )
            logger.info(f"V2: {len(v2_df)} models, "
                        f"{v2_df['significant'].sum()} significant")
            save_csv(v2_df, PATHS["final"] / "validation_v2_hrsn_cms.csv")

            # VALIDATION 3: ACS Direct → CMS Disease Prevalence [county]
            logger.info("\n=== Validation 3: ACS Direct → CMS Disease [county] ===")

            v3_available_acs = [c for c in available_acs if c in county_std.columns]
            v3_df = run_ols_battery(
                county_std, v3_available_acs, available_cms,
                v2_covariates,
                label="v3_acs_direct_to_cms_disease"
            )
            logger.info(f"V3: {len(v3_df)} models, "
                        f"{v3_df['significant'].sum()} significant")
            save_csv(v3_df, PATHS["final"] / "validation_v3_acs_cms.csv")
        else:
            logger.warning(f"Only {len(county_merged)} counties after merge — "
                           "too few for county-level analyses")

    # ================================================================
    # Convergence Analysis
    # ================================================================
    logger.info("\n=== Convergence Analysis ===")

    convergence_records = []

    for _, primary_row in primary_df.iterrows():
        hrsn = primary_row["exposure"]
        outcome = primary_row["outcome"]
        primary_beta = primary_row["beta"]
        primary_sig = primary_row.get("significant", False)

        record = {
            "hrsn_measure": hrsn,
            "outcome": outcome,
            "primary_beta": primary_beta,
            "primary_significant": bool(primary_sig),
        }

        # V1: ACS proxy → same PLACES disease outcome
        acs_proxy = HRSN_TO_ACS.get(hrsn)
        if acs_proxy and acs_proxy in available_acs and len(v1_df) > 0:
            v1_match = v1_df[(v1_df["exposure"] == acs_proxy) &
                             (v1_df["outcome"] == outcome)]
            if len(v1_match) > 0:
                v1_row = v1_match.iloc[0]
                record["v1_acs_proxy"] = acs_proxy
                record["v1_beta"] = v1_row["beta"]
                record["v1_significant"] = bool(v1_row["significant"])
                record["v1_same_direction"] = (
                    np.sign(primary_beta) == np.sign(v1_row["beta"])
                )

        # V2: Same HRSN → CMS claims version of SAME disease
        cms_outcome = PLACES_TO_CMS.get(outcome)
        if cms_outcome and len(v2_df) > 0:
            v2_match = v2_df[(v2_df["exposure"] == hrsn) &
                             (v2_df["outcome"] == cms_outcome)]
            if len(v2_match) > 0:
                v2_row = v2_match.iloc[0]
                record["v2_cms_outcome"] = cms_outcome
                record["v2_beta"] = v2_row["beta"]
                record["v2_significant"] = bool(v2_row["significant"])
                record["v2_same_direction"] = (
                    np.sign(primary_beta) == np.sign(v2_row["beta"])
                )

        # V3: ACS proxy → CMS claims version of SAME disease
        if acs_proxy and acs_proxy in available_acs and cms_outcome and len(v3_df) > 0:
            v3_match = v3_df[(v3_df["exposure"] == acs_proxy) &
                             (v3_df["outcome"] == cms_outcome)]
            if len(v3_match) > 0:
                v3_row = v3_match.iloc[0]
                record["v3_beta"] = v3_row["beta"]
                record["v3_significant"] = bool(v3_row["significant"])
                record["v3_same_direction"] = (
                    np.sign(primary_beta) == np.sign(v3_row["beta"])
                )

        convergence_records.append(record)

    conv_df = pd.DataFrame(convergence_records)

    # ================================================================
    # Deduplication Flagging (C5)
    # ================================================================
    # Multiple HRSN measures map to the same ACS proxy, creating duplicate
    # validation pairs in V1/V3. Flag these for transparent reporting.
    # Keep the conceptually closest match as "primary" per proxy:
    #   pct_snap → foodinsecu (primary), foodstamp (duplicate)
    #   pct_living_alone → emotionspt (primary), loneliness (duplicate)
    PRIMARY_FOR_ACS = {
        "pct_snap": "foodinsecu",
        "pct_living_alone": "emotionspt",
    }

    conv_df["acs_proxy"] = conv_df["hrsn_measure"].map(HRSN_TO_ACS)
    conv_df["duplicate_acs_pair"] = False
    for acs_proxy, primary_hrsn in PRIMARY_FOR_ACS.items():
        mask = (conv_df["acs_proxy"] == acs_proxy) & (conv_df["hrsn_measure"] != primary_hrsn)
        conv_df.loc[mask, "duplicate_acs_pair"] = True

    n_dup = conv_df["duplicate_acs_pair"].sum()
    n_unique = (~conv_df["duplicate_acs_pair"]).sum()
    logger.info(f"\nDeduplication: {n_dup} duplicate ACS proxy pairs flagged, "
                f"{n_unique} unique pairs retained")

    # ================================================================
    # Summary Statistics
    # ================================================================
    logger.info("\n=== Convergence Summary (All Pairs) ===")
    logger.info(f"Total primary associations: {len(conv_df)}")

    for v_col, v_label in [
        ("v1_same_direction", "V1 (ACS→PLACES Disease)"),
        ("v2_same_direction", "V2 (HRSN→CMS Disease)"),
        ("v3_same_direction", "V3 (ACS→CMS Disease)"),
    ]:
        if v_col in conv_df.columns:
            valid = conv_df[v_col].dropna().astype(bool)
            n_same = int(valid.sum())
            n_total = len(valid)
            pct = n_same / n_total * 100 if n_total > 0 else 0
            logger.info(f"  {v_label}: {n_same}/{n_total} same direction ({pct:.0f}%)")

    # Count full convergence
    direction_cols = [c for c in conv_df.columns if c.endswith("_same_direction")]
    if direction_cols:
        conv_df["n_validations"] = conv_df[direction_cols].notna().sum(axis=1)
        conv_df["n_converge"] = sum(
            conv_df[c].map({True: 1, False: 0}).fillna(0).astype(int)
            for c in direction_cols
        )
        conv_df["all_converge"] = conv_df["n_converge"] == conv_df["n_validations"]

        has_any = conv_df["n_validations"] > 0
        n_full_converge = int(conv_df.loc[has_any, "all_converge"].sum())
        n_with_validations = int(has_any.sum())
        pct_converge = n_full_converge / n_with_validations * 100 if n_with_validations > 0 else 0
        logger.info(f"\n  Full convergence (all validations agree): "
                    f"{n_full_converge}/{n_with_validations} ({pct_converge:.0f}%)")

        # Deduplicated convergence
        dedup = conv_df[~conv_df["duplicate_acs_pair"]]
        has_any_d = dedup["n_validations"] > 0
        n_full_d = int(dedup.loc[has_any_d, "all_converge"].sum())
        n_with_d = int(has_any_d.sum())
        pct_d = n_full_d / n_with_d * 100 if n_with_d > 0 else 0
        logger.info(f"  Full convergence (DEDUPLICATED): "
                    f"{n_full_d}/{n_with_d} ({pct_d:.0f}%)")

        # By HRSN measure
        logger.info("\n  Convergence by HRSN measure:")
        for hrsn in conv_df["hrsn_measure"].unique():
            sub = conv_df[conv_df["hrsn_measure"] == hrsn]
            n_conv = int(sub["all_converge"].sum())
            dup_flag = " [duplicate proxy]" if sub["duplicate_acs_pair"].any() else ""
            logger.info(f"    {hrsn}: {n_conv}/{len(sub)} outcomes converge{dup_flag}")

    save_csv(conv_df, PATHS["final"] / "validation_convergence.csv")

    # ================================================================
    # Convergence Plot
    # ================================================================
    _plot_convergence(conv_df, primary_df, v1_df, v2_df, v3_df)

    return conv_df


def _plot_convergence(conv_df, primary_df, v1_df, v2_df, v3_df):
    """Create a convergence comparison plot."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # --- Panel 1: V1 — Primary β vs ACS→PLACES β ---
    ax = axes[0]
    if len(v1_df) > 0:
        matched_primary, matched_val = [], []
        for _, v_row in v1_df.iterrows():
            hrsn_matches = [h for h, a in HRSN_TO_ACS.items() if a == v_row["exposure"]]
            for hrsn in hrsn_matches:
                p_match = primary_df[(primary_df["exposure"] == hrsn) &
                                     (primary_df["outcome"] == v_row["outcome"])]
                if len(p_match) > 0:
                    matched_primary.append(p_match.iloc[0]["beta"])
                    matched_val.append(v_row["beta"])

        if matched_primary:
            _scatter_with_stats(ax, matched_primary, matched_val)
    else:
        ax.text(0.5, 0.5, "Data not available", ha="center", va="center")
    ax.set_title("V1: ACS Direct → PLACES Disease", fontsize=11, fontweight="bold")
    ax.set_xlabel("Primary Analysis (β)", fontsize=10)
    ax.set_ylabel("V1 Validation (β)", fontsize=10)

    # --- Panel 2: V2 — Primary β vs HRSN→CMS Disease β (direct match) ---
    ax = axes[1]
    if len(v2_df) > 0 and "v2_beta" in conv_df.columns:
        v2_valid = conv_df[conv_df["v2_beta"].notna()]
        if len(v2_valid) > 0:
            _scatter_with_stats(ax,
                                v2_valid["primary_beta"].tolist(),
                                v2_valid["v2_beta"].tolist())
    else:
        ax.text(0.5, 0.5, "Data not available", ha="center", va="center")
    ax.set_title("V2: PLACES HRSN → CMS Claims Disease", fontsize=11,
                 fontweight="bold")
    ax.set_xlabel("Primary β (HRSN → PLACES Disease)", fontsize=10)
    ax.set_ylabel("V2 β (HRSN → CMS Claims Disease)", fontsize=10)

    # --- Panel 3: V3 — Primary β vs ACS→CMS Disease β (fully independent) ---
    ax = axes[2]
    if len(v3_df) > 0 and "v3_beta" in conv_df.columns:
        v3_valid = conv_df[conv_df["v3_beta"].notna()]
        if len(v3_valid) > 0:
            _scatter_with_stats(ax,
                                v3_valid["primary_beta"].tolist(),
                                v3_valid["v3_beta"].tolist())
    else:
        ax.text(0.5, 0.5, "Data not available", ha="center", va="center")
    ax.set_title("V3: ACS Direct → CMS Claims Disease\n(Fully Independent)",
                 fontsize=11, fontweight="bold")
    ax.set_xlabel("Primary β (HRSN → PLACES Disease)", fontsize=10)
    ax.set_ylabel("V3 β (ACS → CMS Claims Disease)", fontsize=10)

    for ax in axes:
        ax.axhline(0, color="gray", linewidth=0.5, alpha=0.5)
        ax.axvline(0, color="gray", linewidth=0.5, alpha=0.5)

    fig.suptitle("Validation Framework: BRFSS-Modeled vs. Medicare Claims-Based Outcomes",
                 fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()

    dest = PATHS["figures"] / "validation_convergence_plot.png"
    dest.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(dest, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    logger.info(f"\nSaved convergence plot to {dest}")


def _scatter_with_stats(ax, x_vals, y_vals):
    """Add scatter points, reference line, and stats to axis."""
    ax.scatter(x_vals, y_vals, alpha=0.6, s=40, c="#2c3e50",
               edgecolors="white", linewidth=0.5, zorder=3)
    _add_reference_line(ax, x_vals, y_vals)
    corr = np.corrcoef(x_vals, y_vals)[0, 1]
    same_dir = sum(np.sign(x) == np.sign(y) for x, y in zip(x_vals, y_vals))
    ax.text(0.05, 0.95,
            f"r = {corr:.2f}\nn = {len(x_vals)}\nSame dir: {same_dir}/{len(x_vals)} "
            f"({same_dir/len(x_vals)*100:.0f}%)",
            transform=ax.transAxes, fontsize=10, va="top",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))


def _add_reference_line(ax, x_vals, y_vals):
    """Add diagonal reference line."""
    all_vals = list(x_vals) + list(y_vals)
    margin = (max(all_vals) - min(all_vals)) * 0.1
    lims = [min(all_vals) - margin, max(all_vals) + margin]
    ax.plot(lims, lims, "k--", alpha=0.3, linewidth=1)


if __name__ == "__main__":
    print("=" * 70)
    print("PHASE D: Independent Validation Framework")
    print("=" * 70)
    results = run_validation_framework()
    if results is not None:
        print(f"\nDone. {len(results)} associations assessed for convergence.")
    else:
        print("\nValidation framework failed — see logs.")
