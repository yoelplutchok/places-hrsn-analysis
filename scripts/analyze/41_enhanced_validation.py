"""
Phase E, Step 41: Enhanced Validation (V5) with Non-BRFSS Exposure Proxies.

Addresses the shared-modeling problem using domain-specific exposure proxies
from USDA, court records, CPS, and Facebook/administrative data — none
derived from BRFSS — paired with CMS Medicare claims-based disease outcomes.

New Exposure Proxies → HRSN Mapping:
  mmg_food_insecurity_rate    → foodinsecu, foodstamp (CPS-modeled, Feeding America)
  pct_lila_tracts             → foodinsecu, foodstamp (USDA admin data)
  county_low_access_share     → foodinsecu           (USDA admin data)
  eviction_filing_rate        → housinsecu            (court records)
  social_capital_index        → loneliness, emotionspt (Facebook/admin, INVERSE)

CMS Claims-Based Outcomes (county-level, from Medicare FFS ICD codes):
  cms_diabetes_prev, cms_depression_prev, cms_copd_prev, cms_chd_prev,
  cms_asthma_prev, cms_obesity_prev, cms_stroke_prev

Design:
  - Each new proxy → each CMS outcome, with covariates, HC1 robust SEs
  - Compare beta direction/significance with primary analysis (results_matrix.csv)
  - Social Capital Index: INVERSE coding (higher SCI → less isolation → negative beta)
  - Binomial test against 50% chance concordance

Output:
  - data/final/enhanced_validation_results.csv   — all V5 model results
  - data/final/enhanced_validation_summary.csv   — summary by proxy/disease
  - outputs/figures/enhanced_validation.png       — scatter + concordance bars
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
from hrsn_analysis.io_utils import load_params, save_csv
from hrsn_analysis.logging_utils import setup_logging, get_logger

setup_logging()
logger = get_logger(__name__)

# ================================================================
# Exposure proxy → HRSN measure mapping
# ================================================================
ENHANCED_TO_HRSN = {
    # Map the Meal Gap (CPS-modeled, Feeding America — strongest proxy)
    "mmg_food_insecurity_rate": ["foodinsecu", "foodstamp"],
    # Food Access Atlas (USDA admin data)
    "pct_lila_tracts": ["foodinsecu", "foodstamp"],
    "county_low_access_share": ["foodinsecu"],
    # Eviction Lab (court records)
    "eviction_filing_rate": ["housinsecu"],
    # Social Capital Atlas (Facebook/admin — INVERSE relationship)
    "social_capital_index": ["loneliness", "emotionspt"],
}

# Proxies where higher values mean LESS social need (require direction flip)
INVERSE_PROXIES = {"social_capital_index"}

# CMS claims → PLACES disease mapping
PLACES_TO_CMS = {
    "diabetes": "cms_diabetes_prev",
    "depression": "cms_depression_prev",
    "copd": "cms_copd_prev",
    "chd": "cms_chd_prev",
    "casthma": "cms_asthma_prev",
    "obesity": "cms_obesity_prev",
    "stroke": "cms_stroke_prev",
}

COVARIATES = ["pct_black", "pct_hispanic", "pct_poverty",
              "pct_college", "pct_65plus", "median_age"]


def standardize(series):
    """Z-score standardize."""
    s = series.dropna()
    if len(s) == 0 or s.std() == 0:
        return series - series.mean() if len(s) > 0 else series
    return (series - series.mean()) / series.std()


def load_v5_data():
    """Load and merge all V5 data sources."""
    logger.info("=== Loading V5 data sources ===")

    # 1. CMS validation outcomes
    val_path = PATHS["processed"] / "validation_outcomes.csv"
    if not val_path.exists():
        raise FileNotFoundError(f"Validation outcomes not found at {val_path}")
    val_df = pd.read_csv(val_path, dtype={"county_fips": str})
    val_df["county_fips"] = val_df["county_fips"].str.zfill(5)
    logger.info(f"CMS validation outcomes: {len(val_df):,} counties")

    # 2. Food access county data (FA Atlas + Map the Meal Gap)
    fa_path = PATHS["raw"] / "food_access" / "food_access_county.csv"
    if fa_path.exists():
        fa_df = pd.read_csv(fa_path, dtype={"county_fips": str})
        fa_df["county_fips"] = fa_df["county_fips"].str.zfill(5)
        logger.info(f"Food Access data: {len(fa_df):,} counties")
    else:
        logger.warning(f"Food Access data not found at {fa_path}")
        fa_df = None

    # 3. Eviction Lab
    ev_path = PATHS["raw"] / "eviction_lab" / "eviction_county.csv"
    if ev_path.exists():
        ev_df = pd.read_csv(ev_path, dtype={"county_fips": str})
        ev_df["county_fips"] = ev_df["county_fips"].str.zfill(5)
        logger.info(f"Eviction Lab data: {len(ev_df):,} counties")
    else:
        logger.warning(f"Eviction Lab data not found at {ev_path}")
        ev_df = None

    # 4. Social Capital Atlas
    sc_path = PATHS["raw"] / "social_capital" / "social_capital_county.csv"
    if sc_path.exists():
        sc_df = pd.read_csv(sc_path, dtype={"county_fips": str})
        sc_df["county_fips"] = sc_df["county_fips"].str.zfill(5)
        logger.info(f"Social Capital data: {len(sc_df):,} counties")
    else:
        logger.warning(f"Social Capital data not found at {sc_path}")
        sc_df = None

    # 5. County-level covariates (aggregate from tract data)
    tract_path = PATHS["processed"] / "merged_tracts.parquet"
    tract_df = pd.read_parquet(tract_path)
    tract_df["county_fips"] = tract_df["tract_fips"].str[:5]
    logger.info(f"Tract data for covariates: {len(tract_df):,} tracts")

    # Population-weighted county means for covariates
    cov_cols = [c for c in COVARIATES if c in tract_df.columns]
    county_records = []
    for fips, grp in tract_df.groupby("county_fips"):
        record = {"county_fips": fips}
        pop = grp["total_population"]
        for col in cov_cols:
            vals = grp[col]
            valid = vals.notna() & pop.notna() & (pop > 0)
            if valid.sum() > 0:
                record[col] = np.average(vals[valid], weights=pop[valid])
            else:
                record[col] = np.nan
        county_records.append(record)
    cov_df = pd.DataFrame(county_records)
    logger.info(f"County covariates: {len(cov_df):,} counties")

    # Merge everything on county_fips
    merged = val_df.copy()
    merged = merged.merge(cov_df, on="county_fips", how="inner")
    if fa_df is not None:
        merged = merged.merge(fa_df[["county_fips", "mmg_food_insecurity_rate",
                                      "pct_lila_tracts", "county_low_access_share",
                                      "county_snap_per_capita"]],
                               on="county_fips", how="left")
    if ev_df is not None:
        merged = merged.merge(ev_df[["county_fips", "eviction_filing_rate"]],
                               on="county_fips", how="left")
    if sc_df is not None:
        merged = merged.merge(sc_df[["county_fips", "social_capital_index"]],
                               on="county_fips", how="left")

    logger.info(f"\nMerged V5 dataset: {len(merged):,} counties")

    # Report coverage
    for proxy in ENHANCED_TO_HRSN:
        if proxy in merged.columns:
            n = merged[proxy].notna().sum()
            logger.info(f"  {proxy}: {n:,} counties ({n/len(merged)*100:.1f}%)")
        else:
            logger.warning(f"  {proxy}: NOT IN DATA")

    return merged


def run_v5_battery(df):
    """Run OLS battery: each V5 proxy → each CMS disease outcome."""
    logger.info("\n=== Running V5 OLS battery ===")

    cms_cols = list(PLACES_TO_CMS.values())
    available_cms = [c for c in cms_cols if c in df.columns and df[c].notna().sum() > 50]
    available_cov = [c for c in COVARIATES if c in df.columns]

    logger.info(f"CMS outcomes available: {available_cms}")
    logger.info(f"Covariates: {available_cov}")

    # Standardize
    df_std = df.copy()
    all_std_cols = (list(ENHANCED_TO_HRSN.keys()) + available_cms + available_cov)
    for col in all_std_cols:
        if col in df_std.columns:
            df_std[col] = standardize(df[col])

    results = []
    for proxy, hrsn_list in ENHANCED_TO_HRSN.items():
        if proxy not in df_std.columns:
            logger.warning(f"  Skipping {proxy} — not in data")
            continue

        for cms_col in available_cms:
            # Find which PLACES disease this CMS outcome corresponds to
            places_disease = None
            for pd_name, cms_name in PLACES_TO_CMS.items():
                if cms_name == cms_col:
                    places_disease = pd_name
                    break

            X_cols = [proxy] + available_cov
            X = sm.add_constant(df_std[X_cols])
            y = df_std[cms_col]

            mask = X.notna().all(axis=1) & y.notna()
            X_clean = X[mask]
            y_clean = y[mask]

            if len(y_clean) < 30:
                continue

            try:
                model = sm.OLS(y_clean, X_clean)
                res = model.fit(cov_type="HC1")

                beta = res.params[proxy]
                pval = res.pvalues[proxy]

                results.append({
                    "v5_proxy": proxy,
                    "hrsn_equivalents": ", ".join(hrsn_list),
                    "cms_outcome": cms_col,
                    "places_disease": places_disease,
                    "beta": round(beta, 4),
                    "se": round(res.bse[proxy], 4),
                    "pvalue": pval,
                    "significant": pval < 0.05,
                    "r_squared": round(res.rsquared, 4),
                    "n_obs": int(res.nobs),
                    "inverse_proxy": proxy in INVERSE_PROXIES,
                    "data_source": _get_data_source(proxy),
                })
            except Exception as e:
                logger.warning(f"  {proxy} × {cms_col} failed — {e}")

    results_df = pd.DataFrame(results)
    logger.info(f"\nV5 battery: {len(results_df)} models completed")
    logger.info(f"  Significant (p<0.05): {results_df['significant'].sum()}")

    return results_df


def _get_data_source(proxy):
    """Return data source label for each proxy."""
    sources = {
        "mmg_food_insecurity_rate": "Feeding America / CPS",
        "pct_lila_tracts": "USDA Food Access Atlas",
        "county_low_access_share": "USDA Food Access Atlas",
        "eviction_filing_rate": "Eviction Lab / court records",
        "social_capital_index": "Social Capital Atlas / Meta-Harvard",
    }
    return sources.get(proxy, "Unknown")


def compare_with_primary(v5_df):
    """Compare V5 results with primary analysis for concordance."""
    logger.info("\n=== Comparing V5 with Primary Analysis ===")

    primary_path = PATHS["final"] / "results_matrix.csv"
    if not primary_path.exists():
        logger.error(f"Primary results not found at {primary_path}")
        return v5_df

    primary = pd.read_csv(primary_path)

    # For each V5 result, find the matched primary result(s)
    concordance_records = []

    for _, v5_row in v5_df.iterrows():
        proxy = v5_row["v5_proxy"]
        hrsn_list = [h.strip() for h in v5_row["hrsn_equivalents"].split(",")]
        places_disease = v5_row["places_disease"]
        v5_beta = v5_row["beta"]
        is_inverse = v5_row["inverse_proxy"]

        for hrsn in hrsn_list:
            # Find matching primary result
            p_match = primary[
                (primary["hrsn_measure"] == hrsn) &
                (primary["outcome"] == places_disease)
            ]
            if len(p_match) == 0:
                continue

            p_row = p_match.iloc[0]
            primary_beta = p_row["beta_std"]

            # For inverse proxies: higher social capital → less disease
            # So V5 beta should be NEGATIVE where primary beta is POSITIVE
            # Concordance = opposite signs for inverse proxies
            if is_inverse:
                same_direction = np.sign(v5_beta) != np.sign(primary_beta)
            else:
                same_direction = np.sign(v5_beta) == np.sign(primary_beta)

            concordance_records.append({
                "v5_proxy": proxy,
                "hrsn_equivalent": hrsn,
                "places_disease": places_disease,
                "cms_outcome": v5_row["cms_outcome"],
                "v5_beta": v5_beta,
                "v5_pvalue": v5_row["pvalue"],
                "v5_significant": v5_row["significant"],
                "primary_beta": primary_beta,
                "primary_pvalue": p_row["pvalue"],
                "primary_significant": p_row.get("significant_05", p_row["pvalue"] < 0.05),
                "concordant": same_direction,
                "inverse_proxy": is_inverse,
                "data_source": v5_row["data_source"],
                "n_obs": v5_row["n_obs"],
            })

    conc_df = pd.DataFrame(concordance_records)

    if len(conc_df) == 0:
        logger.warning("No matched pairs for concordance analysis")
        return v5_df

    # ================================================================
    # Summary statistics
    # ================================================================
    n_concordant = conc_df["concordant"].sum()
    n_total = len(conc_df)
    pct = n_concordant / n_total * 100

    # Binomial test against 50% (chance)
    binom_p = stats.binomtest(n_concordant, n_total, 0.5).pvalue

    logger.info(f"\n=== V5 Concordance Summary ===")
    logger.info(f"Total matched pairs: {n_total}")
    logger.info(f"Concordant: {n_concordant}/{n_total} ({pct:.1f}%)")
    logger.info(f"Binomial test vs 50%: p = {binom_p:.4f}")

    # Among significant V5 results only
    sig_mask = conc_df["v5_significant"]
    if sig_mask.sum() > 0:
        n_sig_conc = conc_df.loc[sig_mask, "concordant"].sum()
        n_sig = sig_mask.sum()
        logger.info(f"Among V5-significant: {n_sig_conc}/{n_sig} "
                    f"({n_sig_conc/n_sig*100:.1f}%) concordant")

    # By proxy
    logger.info("\nBy proxy:")
    for proxy in conc_df["v5_proxy"].unique():
        sub = conc_df[conc_df["v5_proxy"] == proxy]
        n_c = sub["concordant"].sum()
        n_t = len(sub)
        inv = " [inverse]" if sub["inverse_proxy"].iloc[0] else ""
        logger.info(f"  {proxy}: {n_c}/{n_t} ({n_c/n_t*100:.0f}%) concordant{inv}")

    # By disease
    logger.info("\nBy CMS disease:")
    for disease in sorted(conc_df["places_disease"].unique()):
        sub = conc_df[conc_df["places_disease"] == disease]
        n_c = sub["concordant"].sum()
        n_t = len(sub)
        logger.info(f"  {disease}: {n_c}/{n_t} ({n_c/n_t*100:.0f}%) concordant")

    # Deduplicated concordance (one proxy per HRSN measure per disease)
    # Keep the conceptually strongest proxy for each HRSN:
    # foodinsecu → mmg_food_insecurity_rate (direct construct match)
    # foodstamp → mmg_food_insecurity_rate
    # housinsecu → eviction_filing_rate (only option)
    # loneliness → social_capital_index (only option)
    # emotionspt → social_capital_index (only option)
    PRIMARY_PROXY = {
        ("foodinsecu", "mmg_food_insecurity_rate"): True,
        ("foodstamp", "mmg_food_insecurity_rate"): True,
        ("housinsecu", "eviction_filing_rate"): True,
        ("loneliness", "social_capital_index"): True,
        ("emotionspt", "social_capital_index"): True,
    }
    conc_df["is_primary_pair"] = conc_df.apply(
        lambda r: (r["hrsn_equivalent"], r["v5_proxy"]) in PRIMARY_PROXY, axis=1
    )

    dedup = conc_df[conc_df["is_primary_pair"]]
    if len(dedup) > 0:
        n_dd_conc = dedup["concordant"].sum()
        n_dd = len(dedup)
        pct_dd = n_dd_conc / n_dd * 100
        binom_dd = stats.binomtest(n_dd_conc, n_dd, 0.5).pvalue
        logger.info(f"\nDeduplicated (primary pairs only): "
                    f"{n_dd_conc}/{n_dd} ({pct_dd:.1f}%) concordant "
                    f"(binomial p = {binom_dd:.4f})")

    # Correlation of V5 betas vs primary betas
    # For inverse proxies, flip V5 beta sign before correlation
    conc_df["v5_beta_aligned"] = conc_df.apply(
        lambda r: -r["v5_beta"] if r["inverse_proxy"] else r["v5_beta"],
        axis=1
    )
    valid = conc_df[["v5_beta_aligned", "primary_beta"]].dropna()
    if len(valid) >= 5:
        pearson_r = np.corrcoef(valid["v5_beta_aligned"], valid["primary_beta"])[0, 1]
        spearman_r, _ = stats.spearmanr(valid["v5_beta_aligned"], valid["primary_beta"])
        logger.info(f"\nBeta correlation (V5 aligned vs primary):")
        logger.info(f"  Pearson r = {pearson_r:.3f}")
        logger.info(f"  Spearman rho = {spearman_r:.3f}")

    # Verify independence: USDA SNAP vs ACS SNAP correlation check
    _check_snap_independence()

    return conc_df


def _check_snap_independence():
    """Verify USDA SNAP per capita is not identical to ACS pct_snap."""
    try:
        tract_df = pd.read_parquet(PATHS["processed"] / "merged_tracts.parquet")
        fa_county = pd.read_csv(PATHS["raw"] / "food_access" / "food_access_county.csv",
                                dtype={"county_fips": str})
        fa_county["county_fips"] = fa_county["county_fips"].str.zfill(5)

        tract_df["county_fips"] = tract_df["tract_fips"].str[:5]
        if "pct_snap" in tract_df.columns:
            acs_snap = tract_df.groupby("county_fips")["pct_snap"].mean().reset_index()
            check = acs_snap.merge(fa_county[["county_fips", "county_snap_per_capita"]],
                                    on="county_fips", how="inner")
            r = check[["pct_snap", "county_snap_per_capita"]].dropna().corr().iloc[0, 1]
            logger.info(f"\nIndependence check: ACS pct_snap vs USDA SNAP per capita r = {r:.3f}")
            if r >= 0.95:
                logger.warning("  WARNING: r >= 0.95 — proxies may not be independent!")
            else:
                logger.info("  OK: r < 0.95 — proxies are from different sources")
    except Exception as e:
        logger.warning(f"Independence check failed: {e}")


def create_summary(conc_df):
    """Create summary table by proxy and by disease."""
    rows = []

    # By proxy
    for proxy in conc_df["v5_proxy"].unique():
        sub = conc_df[conc_df["v5_proxy"] == proxy]
        n_c = int(sub["concordant"].sum())
        n_t = len(sub)
        n_sig = int(sub["v5_significant"].sum())
        rows.append({
            "group_by": "proxy",
            "group_value": proxy,
            "n_concordant": n_c,
            "n_total": n_t,
            "pct_concordant": round(n_c / n_t * 100, 1),
            "n_significant": n_sig,
            "data_source": sub["data_source"].iloc[0],
            "inverse_proxy": sub["inverse_proxy"].iloc[0],
        })

    # By disease
    for disease in sorted(conc_df["places_disease"].unique()):
        sub = conc_df[conc_df["places_disease"] == disease]
        n_c = int(sub["concordant"].sum())
        n_t = len(sub)
        n_sig = int(sub["v5_significant"].sum())
        rows.append({
            "group_by": "disease",
            "group_value": disease,
            "n_concordant": n_c,
            "n_total": n_t,
            "pct_concordant": round(n_c / n_t * 100, 1),
            "n_significant": n_sig,
            "data_source": "all",
            "inverse_proxy": False,
        })

    # Overall
    n_c = int(conc_df["concordant"].sum())
    n_t = len(conc_df)
    binom_p = stats.binomtest(n_c, n_t, 0.5).pvalue
    rows.append({
        "group_by": "overall",
        "group_value": "all",
        "n_concordant": n_c,
        "n_total": n_t,
        "pct_concordant": round(n_c / n_t * 100, 1),
        "n_significant": int(conc_df["v5_significant"].sum()),
        "data_source": "all",
        "inverse_proxy": False,
    })

    return pd.DataFrame(rows)


def create_v5_plot(conc_df):
    """Create V5 validation plot: scatter + concordance bars."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # --- Panel 1: Scatter of aligned V5 beta vs primary beta ---
    ax = axes[0]
    conc_df["v5_beta_aligned"] = conc_df.apply(
        lambda r: -r["v5_beta"] if r["inverse_proxy"] else r["v5_beta"],
        axis=1
    )

    # Color by proxy type
    proxy_colors = {
        "mmg_food_insecurity_rate": "#e74c3c",
        "pct_lila_tracts": "#e67e22",
        "county_low_access_share": "#f39c12",
        "eviction_filing_rate": "#2ecc71",
        "social_capital_index": "#3498db",
    }

    for proxy, color in proxy_colors.items():
        sub = conc_df[conc_df["v5_proxy"] == proxy]
        if len(sub) == 0:
            continue
        label = proxy.replace("_", " ").title()
        if proxy in INVERSE_PROXIES:
            label += " (inv.)"
        ax.scatter(sub["primary_beta"], sub["v5_beta_aligned"],
                   c=color, label=label, alpha=0.7, s=50,
                   edgecolors="white", linewidth=0.5, zorder=3)

    # Reference line and stats
    valid = conc_df[["v5_beta_aligned", "primary_beta"]].dropna()
    if len(valid) >= 3:
        all_vals = list(valid["v5_beta_aligned"]) + list(valid["primary_beta"])
        margin = (max(all_vals) - min(all_vals)) * 0.1
        lims = [min(all_vals) - margin, max(all_vals) + margin]
        ax.plot(lims, lims, "k--", alpha=0.3, linewidth=1)

        r = np.corrcoef(valid["v5_beta_aligned"], valid["primary_beta"])[0, 1]
        n_same = int(conc_df["concordant"].sum())
        n_total = len(conc_df)
        ax.text(0.05, 0.95,
                f"r = {r:.2f}\nn = {n_total}\nConcordant: {n_same}/{n_total} "
                f"({n_same/n_total*100:.0f}%)",
                transform=ax.transAxes, fontsize=10, va="top",
                bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

    ax.axhline(0, color="gray", linewidth=0.5, alpha=0.5)
    ax.axvline(0, color="gray", linewidth=0.5, alpha=0.5)
    ax.set_xlabel("Primary Analysis Beta (standardized)", fontsize=11)
    ax.set_ylabel("V5 Beta (aligned, standardized)", fontsize=11)
    ax.set_title("V5: Non-BRFSS Proxies → CMS Claims Disease", fontsize=12, fontweight="bold")
    ax.legend(fontsize=8, loc="lower right")

    # --- Panel 2: Concordance by proxy ---
    ax = axes[1]
    proxy_names = []
    proxy_pcts = []
    proxy_ns = []
    bar_colors = []

    for proxy in conc_df["v5_proxy"].unique():
        sub = conc_df[conc_df["v5_proxy"] == proxy]
        n_c = sub["concordant"].sum()
        n_t = len(sub)
        pct = n_c / n_t * 100

        name = proxy.replace("_", " ").replace("mmg ", "MMG ").title()
        if proxy in INVERSE_PROXIES:
            name += " (inv.)"
        proxy_names.append(name)
        proxy_pcts.append(pct)
        proxy_ns.append(f"{int(n_c)}/{n_t}")
        bar_colors.append(proxy_colors.get(proxy, "#7f8c8d"))

    y_pos = range(len(proxy_names))
    ax.barh(y_pos, proxy_pcts, color=bar_colors, alpha=0.8, edgecolor="white")
    ax.axvline(50, color="red", linestyle="--", alpha=0.5, label="Chance (50%)")

    for i, (pct, ns) in enumerate(zip(proxy_pcts, proxy_ns)):
        ax.text(pct + 1, i, f"{pct:.0f}% ({ns})", va="center", fontsize=9)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(proxy_names, fontsize=9)
    ax.set_xlabel("Directional Concordance (%)", fontsize=11)
    ax.set_title("Concordance by Proxy Source", fontsize=12, fontweight="bold")
    ax.set_xlim(0, 110)
    ax.legend(fontsize=9)

    fig.suptitle("V5 Enhanced Validation: Domain-Specific Non-BRFSS Proxies\n"
                 "vs. Medicare Claims-Based Disease Prevalence",
                 fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()

    dest = PATHS["figures"] / "enhanced_validation.png"
    dest.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(dest, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    logger.info(f"\nSaved V5 plot to {dest}")


def run_enhanced_validation():
    """Run the full V5 enhanced validation analysis."""
    ensure_dirs()

    # Load and merge data
    merged = load_v5_data()

    # Run V5 OLS battery
    v5_results = run_v5_battery(merged)
    save_csv(v5_results, PATHS["final"] / "enhanced_validation_results.csv")

    # Compare with primary analysis
    conc_df = compare_with_primary(v5_results)
    save_csv(conc_df, PATHS["final"] / "enhanced_validation_concordance.csv")

    # Summary
    summary = create_summary(conc_df)
    save_csv(summary, PATHS["final"] / "enhanced_validation_summary.csv")

    # Plot
    create_v5_plot(conc_df)

    return conc_df


if __name__ == "__main__":
    print("=" * 70)
    print("PHASE E: Enhanced Validation (V5) with Non-BRFSS Exposure Proxies")
    print("=" * 70)
    results = run_enhanced_validation()
    print(f"\nDone. {len(results)} matched pairs assessed for concordance.")
