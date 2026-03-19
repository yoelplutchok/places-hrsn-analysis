"""
Path 3: What Explains Geographic Heterogeneity in HRSN-Disease Associations?

Uses interaction models to test whether HRSN-disease associations are
moderated by policy context:
  - Medicaid expansion status
  - CMS 1115 HRSN waiver status
  - Urbanicity (population density proxy)
  - Healthcare supply proxy (uninsured rate from SVI if available)

If HRSN-disease associations weaken in Medicaid expansion states or states
with HRSN screening waivers, this is evidence that policy buffers the health
effects of social needs.

Also runs stratified regressions and maps coefficient heterogeneity by state.

Output:
  - data/final/interaction_results.csv
  - data/final/stratified_coefficients.csv
  - data/final/state_level_coefficients.csv
  - outputs/figures/policy_moderation_forest.png
  - outputs/figures/maps/state_coefficient_map.png
"""
import sys
from pathlib import Path

import pandas as pd
import numpy as np
import geopandas as gpd
import statsmodels.api as sm
from statsmodels.stats.multitest import multipletests

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from hrsn_analysis.paths import PATHS, ensure_dirs
from hrsn_analysis.io_utils import load_parquet, load_params, save_csv
from hrsn_analysis.logging_utils import setup_logging, get_logger

setup_logging()
logger = get_logger(__name__)


def standardize(series):
    return (series - series.mean()) / series.std()


# ====================================================================
# Policy data (hardcoded from CMS/KFF as of end-2023)
# ====================================================================

# States that had NOT expanded Medicaid as of the 2022 BRFSS data vintage
# (NC and SD expanded in 2023, after data collection)
# Source: KFF Medicaid expansion tracker
NON_EXPANSION_FIPS = {
    "01",  # Alabama
    "12",  # Florida
    "13",  # Georgia
    "20",  # Kansas
    "28",  # Mississippi
    "37",  # North Carolina (expanded Dec 2023, after 2022 data)
    "45",  # South Carolina
    "46",  # South Dakota (expanded Jul 2023, after 2022 data)
    "47",  # Tennessee
    "48",  # Texas
    "55",  # Wisconsin (partial, to 100% FPL but not full ACA expansion)
    "56",  # Wyoming
}

# States with approved CMS 1115 waivers that include HRSN services
# approved BEFORE the 2022 BRFSS data collection period.
# Excludes NJ (approved 2023), NY (approved 2024), WA (approved 2023).
HRSN_WAIVER_FIPS = {
    "04",  # Arizona
    "05",  # Arkansas
    "06",  # California (CalAIM)
    "25",  # Massachusetts
    "37",  # North Carolina
    "41",  # Oregon
}

# Population density thresholds for urbanicity classification
URBAN_POP_THRESHOLD = 2500  # tracts above this are "urban"
RURAL_POP_THRESHOLD = 1000  # tracts below this are "rural"


def create_policy_variables(df):
    """Add policy context variables to the dataset."""
    df = df.copy()

    # Medicaid expansion (1 = expanded)
    df["medicaid_expanded"] = (~df["state_fips"].isin(NON_EXPANSION_FIPS)).astype(int)

    # HRSN waiver (1 = has waiver)
    df["hrsn_waiver"] = df["state_fips"].isin(HRSN_WAIVER_FIPS).astype(int)

    # Urbanicity (simple proxy based on tract population)
    df["urban"] = (df["total_population"] >= URBAN_POP_THRESHOLD).astype(int)
    df["rural"] = (df["total_population"] < RURAL_POP_THRESHOLD).astype(int)
    df["log_population"] = np.log1p(df["total_population"])

    n_expanded = df["medicaid_expanded"].sum()
    n_waiver = df["hrsn_waiver"].sum()
    n_urban = df["urban"].sum()
    logger.info(f"  Medicaid expansion: {n_expanded:,}/{len(df):,} tracts "
                f"({n_expanded/len(df)*100:.1f}%)")
    logger.info(f"  HRSN waiver: {n_waiver:,}/{len(df):,} tracts "
                f"({n_waiver/len(df)*100:.1f}%)")
    logger.info(f"  Urban (pop≥{URBAN_POP_THRESHOLD}): {n_urban:,}/{len(df):,} tracts "
                f"({n_urban/len(df)*100:.1f}%)")

    return df


def run_interaction_models(df, hrsn_cols, outcome_cols, demo_cols,
                            hrsn_labels, outcome_labels):
    """Test HRSN × policy interactions for all 49 HRSN-outcome pairs."""
    logger.info("\n=== Interaction Models: HRSN × Policy Context ===")

    moderators = [
        ("medicaid_expanded", "Medicaid Expansion"),
        ("hrsn_waiver", "HRSN Waiver"),
        ("urban", "Urban"),
    ]

    results = []
    cluster_var = df["county_fips"]

    for hrsn in hrsn_cols:
        for outcome in outcome_cols:
            for mod_var, mod_label in moderators:
                # Model: outcome ~ hrsn + moderator + hrsn*moderator + demographics
                interaction_term = f"{hrsn}_x_{mod_var}"
                df[interaction_term] = df[hrsn] * df[mod_var]

                X_cols = [hrsn, mod_var, interaction_term] + demo_cols
                X = sm.add_constant(df[X_cols])
                y = df[outcome]

                mask = X.notna().all(axis=1) & y.notna()
                X_clean = X[mask]
                y_clean = y[mask]
                cluster_clean = cluster_var[mask]

                try:
                    res = sm.OLS(y_clean, X_clean).fit(
                        cov_type="cluster", cov_kwds={"groups": cluster_clean})

                    interaction_beta = res.params[interaction_term]
                    interaction_se = res.bse[interaction_term]
                    interaction_p = res.pvalues[interaction_term]
                    main_beta = res.params[hrsn]
                    main_p = res.pvalues[hrsn]

                    results.append({
                        "hrsn_measure": hrsn,
                        "hrsn_label": hrsn_labels.get(hrsn, hrsn),
                        "outcome": outcome,
                        "outcome_label": outcome_labels.get(outcome, outcome),
                        "moderator": mod_var,
                        "moderator_label": mod_label,
                        "main_beta": round(main_beta, 4),
                        "main_p": main_p,
                        "interaction_beta": round(interaction_beta, 4),
                        "interaction_se": round(interaction_se, 4),
                        "interaction_p": interaction_p,
                        "interaction_significant": interaction_p < 0.05,
                        "r_squared": round(res.rsquared, 4),
                        "n_obs": int(res.nobs),
                    })

                except Exception as e:
                    logger.warning(f"  {hrsn}×{outcome} mod={mod_var}: failed — {e}")

                # Clean up
                df.drop(columns=[interaction_term], inplace=True, errors="ignore")

    return pd.DataFrame(results)


def run_stratified_regressions(df, hrsn_cols, outcome_cols, demo_cols,
                                 hrsn_labels, outcome_labels):
    """Run regressions stratified by Medicaid expansion status."""
    logger.info("\n=== Stratified Regressions: Expansion vs Non-Expansion ===")

    results = []

    for stratum_var, stratum_label in [("medicaid_expanded", "Medicaid Expansion")]:
        for stratum_val, stratum_name in [(1, "Expanded"), (0, "Non-expanded")]:
            sub = df[df[stratum_var] == stratum_val]
            cluster_var = sub["county_fips"]
            logger.info(f"\n  --- {stratum_name}: {len(sub):,} tracts ---")

            for hrsn in hrsn_cols:
                for outcome in outcome_cols:
                    X_cols = [hrsn] + demo_cols
                    X = sm.add_constant(sub[X_cols])
                    y = sub[outcome]

                    mask = X.notna().all(axis=1) & y.notna()
                    X_clean = X[mask]
                    y_clean = y[mask]
                    cluster_clean = cluster_var[mask]

                    try:
                        res = sm.OLS(y_clean, X_clean).fit(
                            cov_type="cluster", cov_kwds={"groups": cluster_clean})

                        results.append({
                            "stratum_var": stratum_var,
                            "stratum_label": stratum_label,
                            "stratum_value": stratum_val,
                            "stratum_name": stratum_name,
                            "hrsn_measure": hrsn,
                            "hrsn_label": hrsn_labels.get(hrsn, hrsn),
                            "outcome": outcome,
                            "outcome_label": outcome_labels.get(outcome, outcome),
                            "beta": round(res.params[hrsn], 4),
                            "se": round(res.bse[hrsn], 4),
                            "pvalue": res.pvalues[hrsn],
                            "ci_lower": round(res.conf_int().loc[hrsn, 0], 4),
                            "ci_upper": round(res.conf_int().loc[hrsn, 1], 4),
                            "r_squared": round(res.rsquared, 4),
                            "n_obs": int(res.nobs),
                        })
                    except Exception as e:
                        logger.warning(f"  {stratum_name} {hrsn}×{outcome}: failed — {e}")

    return pd.DataFrame(results)


def run_state_level_coefficients(df, top_pairs, demo_cols,
                                  hrsn_labels, outcome_labels):
    """Estimate HRSN-disease coefficients per state for mapping."""
    logger.info("\n=== State-Level Coefficients ===")

    results = []

    for hrsn, outcome in top_pairs:
        logger.info(f"\n  {hrsn} × {outcome}")

        for state_fips, state_df in df.groupby("state_fips"):
            if len(state_df) < 50:
                continue

            X_cols = [hrsn] + demo_cols
            X = sm.add_constant(state_df[X_cols])
            y = state_df[outcome]

            mask = X.notna().all(axis=1) & y.notna()
            if mask.sum() < 50:
                continue

            X_clean = X[mask]
            y_clean = y[mask]

            try:
                res = sm.OLS(y_clean, X_clean).fit(cov_type="HC1")
                results.append({
                    "state_fips": state_fips,
                    "state_abbr": state_df["state_abbr"].iloc[0],
                    "hrsn_measure": hrsn,
                    "outcome": outcome,
                    "beta": round(res.params[hrsn], 4),
                    "se": round(res.bse[hrsn], 4),
                    "pvalue": res.pvalues[hrsn],
                    "r_squared": round(res.rsquared, 4),
                    "n_tracts": int(res.nobs),
                    "medicaid_expanded": int(state_fips not in NON_EXPANSION_FIPS),
                    "hrsn_waiver": int(state_fips in HRSN_WAIVER_FIPS),
                })
            except Exception as e:
                logger.warning(f"  State {state_fips} {hrsn}×{outcome}: failed — {e}")

    return pd.DataFrame(results)


def plot_policy_moderation(interaction_df, hrsn_labels, outcome_labels):
    """Forest plot showing interaction effects by moderator."""
    logger.info("\n=== Creating policy moderation figure ===")

    sig = interaction_df[interaction_df["interaction_significant"]].copy()
    if len(sig) == 0:
        logger.info("No significant interactions — skipping forest plot")
        # Still create plot showing all interactions for one moderator
        sig = interaction_df[interaction_df["moderator"] == "medicaid_expanded"].copy()
        if len(sig) == 0:
            return

    # Focus on Medicaid expansion interactions
    med_interactions = interaction_df[
        interaction_df["moderator"] == "medicaid_expanded"
    ].copy()
    med_interactions["label"] = (
        med_interactions["hrsn_label"] + " × " + med_interactions["outcome_label"]
    )

    # Sort by interaction beta
    med_interactions = med_interactions.sort_values("interaction_beta")

    fig, ax = plt.subplots(figsize=(12, max(8, len(med_interactions) * 0.25)))

    y_pos = range(len(med_interactions))
    colors = ["#e74c3c" if p < 0.05 else "#95a5a6"
              for p in med_interactions["interaction_p"]]

    ax.barh(y_pos, med_interactions["interaction_beta"],
            xerr=med_interactions["interaction_se"] * 1.96,
            color=colors, alpha=0.7, edgecolor="white", linewidth=0.5,
            capsize=2)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(med_interactions["label"], fontsize=7)
    ax.axvline(x=0, color="black", linewidth=0.8)
    ax.set_xlabel("Interaction β (HRSN × Medicaid Expansion)", fontsize=11)
    ax.set_title("Policy Moderation: Does Medicaid Expansion Buffer HRSN-Disease Associations?\n"
                 "(Negative = weaker association in expansion states)",
                 fontsize=12, fontweight="bold")

    legend_patches = [
        Patch(facecolor="#e74c3c", alpha=0.7, label="Significant (p<0.05)"),
        Patch(facecolor="#95a5a6", alpha=0.7, label="Not significant"),
    ]
    ax.legend(handles=legend_patches, loc="lower right", fontsize=9)

    plt.tight_layout()
    dest = PATHS["figures"] / "policy_moderation_forest.png"
    fig.savefig(dest, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    logger.info(f"Saved forest plot to {dest}")


def plot_state_coefficient_map(state_df, top_pair, hrsn_labels, outcome_labels):
    """Map state-level coefficients for one HRSN-outcome pair."""
    hrsn, outcome = top_pair
    pair_data = state_df[
        (state_df["hrsn_measure"] == hrsn) & (state_df["outcome"] == outcome)
    ].copy()

    if len(pair_data) == 0:
        return

    # Load state boundaries
    try:
        states_gdf = gpd.read_file(PATHS["geo_raw"] / "us_tracts_2022.gpkg")
        # Dissolve to state level
        states_gdf["state_fips"] = states_gdf["GEOID"].str[:2]
        states_gdf = states_gdf.dissolve(by="state_fips").reset_index()
    except Exception:
        logger.warning("Cannot load geometries for state map — trying alternative")
        try:
            # Try a simpler approach: use tract geometries but dissolve
            import warnings
            warnings.filterwarnings("ignore")
            states_gdf = gpd.read_file(PATHS["geo_raw"] / "us_tracts_2022.gpkg",
                                       columns=["GEOID", "geometry"])
            states_gdf["state_fips"] = states_gdf["GEOID"].str[:2]
            states_gdf = states_gdf.dissolve(by="state_fips").reset_index()
        except Exception as e:
            logger.warning(f"State map failed: {e}")
            return

    # Filter to CONUS
    exclude = {"02", "15", "60", "66", "69", "72", "78"}
    states_gdf = states_gdf[~states_gdf["state_fips"].isin(exclude)]

    merged = states_gdf.merge(pair_data, on="state_fips", how="left")

    fig, ax = plt.subplots(1, 1, figsize=(16, 10))

    vmax = max(abs(merged["beta"].min()), abs(merged["beta"].max()))
    merged.plot(
        column="beta", cmap="RdBu_r", linewidth=0.5, edgecolor="gray",
        ax=ax, legend=True, vmin=-vmax, vmax=vmax, missing_kwds={"color": "#f0f0f0"},
        legend_kwds={"label": "β coefficient", "orientation": "horizontal",
                     "shrink": 0.6, "pad": 0.02},
    )

    # Highlight waiver states
    waiver_states = merged[merged["hrsn_waiver"] == 1]
    if len(waiver_states) > 0:
        waiver_states.boundary.plot(ax=ax, linewidth=2, edgecolor="gold")

    # Highlight non-expansion states
    nonexp_states = merged[merged["medicaid_expanded"] == 0]
    if len(nonexp_states) > 0:
        nonexp_states.boundary.plot(ax=ax, linewidth=2, edgecolor="black",
                                      linestyle="--")

    ax.set_axis_off()
    hrsn_name = hrsn_labels.get(hrsn, hrsn)
    outcome_name = outcome_labels.get(outcome, outcome)
    ax.set_title(f"State-Level β: {hrsn_name} → {outcome_name}\n"
                 f"Gold border = HRSN waiver states | Dashed = non-expansion states",
                 fontsize=13, fontweight="bold")

    dest = PATHS["figures"] / "maps" / f"state_coefficients_{hrsn}_{outcome}.png"
    fig.savefig(dest, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    logger.info(f"  Saved state map to {dest}")


def run_heterogeneity_analysis():
    """Run full geographic heterogeneity analysis."""
    ensure_dirs()
    params = load_params()

    hrsn_cols = [m["id"].lower() for m in params["hrsn_measures"]]
    outcome_cols = [m["id"].lower() for m in params["outcome_measures"]]
    hrsn_labels = {m["id"].lower(): m["name"] for m in params["hrsn_measures"]}
    outcome_labels = {m["id"].lower(): m["name"] for m in params["outcome_measures"]}

    demo_cols = ["pct_black", "pct_hispanic", "pct_poverty",
                 "pct_college", "pct_65plus", "median_age"]

    # Top associations for state-level analysis
    try:
        results_matrix = pd.read_csv(PATHS["final"] / "results_matrix.csv")
        results_matrix["abs_beta"] = results_matrix["beta_std"].abs()
        top = results_matrix.nlargest(5, "abs_beta")
        top_pairs = [(row["hrsn_measure"], row["outcome"]) for _, row in top.iterrows()]
    except FileNotFoundError:
        top_pairs = [
            ("shututility", "copd"),
            ("foodinsecu", "diabetes"),
            ("emotionspt", "depression"),
        ]

    # ---- Load and prepare data ----
    logger.info("=== Loading data ===")
    df = load_parquet(PATHS["processed"] / "merged_tracts.parquet")

    # Add policy variables
    logger.info("\n=== Creating policy context variables ===")
    df = create_policy_variables(df)

    # Standardize HRSN and outcome variables
    for col in hrsn_cols + outcome_cols + demo_cols:
        if col in df.columns:
            df[col] = standardize(df[col])

    # ---- Interaction models ----
    interaction_df = run_interaction_models(
        df, hrsn_cols, outcome_cols, demo_cols, hrsn_labels, outcome_labels
    )
    if len(interaction_df) > 0:
        # Log pre-FDR significance for comparison
        n_sig_uncorrected = (interaction_df["interaction_p"] < 0.05).sum()
        n_total = len(interaction_df)
        logger.info(f"\nSignificant interactions before FDR: {n_sig_uncorrected}/{n_total} "
                    f"({n_sig_uncorrected/n_total*100:.1f}%)")

        # Apply Benjamini-Hochberg FDR correction across all interaction tests
        reject, pvals_corrected, _, _ = multipletests(
            interaction_df["interaction_p"], alpha=0.05, method="fdr_bh"
        )
        interaction_df["interaction_p_fdr"] = pvals_corrected
        interaction_df["interaction_significant"] = reject

        save_csv(interaction_df, PATHS["final"] / "interaction_results.csv")

        n_sig = interaction_df["interaction_significant"].sum()
        logger.info(f"\nSignificant interactions: {n_sig}/{n_total} "
                    f"({n_sig/n_total*100:.1f}%)")

        # Summarize by moderator
        for mod in interaction_df["moderator"].unique():
            mod_sub = interaction_df[interaction_df["moderator"] == mod]
            n_sig_mod = mod_sub["interaction_significant"].sum()
            mean_beta = mod_sub["interaction_beta"].mean()
            logger.info(f"  {mod}: {n_sig_mod}/{len(mod_sub)} significant, "
                        f"mean interaction β = {mean_beta:+.4f}")

    # ---- Stratified regressions ----
    strat_df = run_stratified_regressions(
        df, hrsn_cols, outcome_cols, demo_cols, hrsn_labels, outcome_labels
    )
    if len(strat_df) > 0:
        save_csv(strat_df, PATHS["final"] / "stratified_coefficients.csv")

        # Compare expansion vs non-expansion
        logger.info("\n=== Expansion vs Non-Expansion Summary ===")
        for hrsn in hrsn_cols[:3]:  # Top 3 for brevity
            for outcome in outcome_cols[:3]:
                exp = strat_df[(strat_df["hrsn_measure"] == hrsn) &
                               (strat_df["outcome"] == outcome) &
                               (strat_df["stratum_name"] == "Expanded")]
                non = strat_df[(strat_df["hrsn_measure"] == hrsn) &
                               (strat_df["outcome"] == outcome) &
                               (strat_df["stratum_name"] == "Non-expanded")]
                if len(exp) > 0 and len(non) > 0:
                    logger.info(f"  {hrsn:15s} × {outcome:12s}: "
                                f"Exp β={exp['beta'].values[0]:+.4f} | "
                                f"Non β={non['beta'].values[0]:+.4f}")

    # ---- State-level coefficients ----
    state_df = run_state_level_coefficients(
        df, top_pairs, demo_cols, hrsn_labels, outcome_labels
    )
    if len(state_df) > 0:
        save_csv(state_df, PATHS["final"] / "state_level_coefficients.csv")

        # Correlation between state beta and policy
        logger.info("\n=== State Beta vs Policy Correlations ===")
        for hrsn, outcome in top_pairs[:3]:
            pair = state_df[(state_df["hrsn_measure"] == hrsn) &
                            (state_df["outcome"] == outcome)]
            if len(pair) > 10:
                corr_exp = pair["beta"].corr(pair["medicaid_expanded"])
                corr_waiver = pair["beta"].corr(pair["hrsn_waiver"])
                logger.info(f"  {hrsn} × {outcome}: "
                            f"r(β, expansion)={corr_exp:+.3f}, "
                            f"r(β, waiver)={corr_waiver:+.3f}")

    # ---- Visualizations ----
    if len(interaction_df) > 0:
        plot_policy_moderation(interaction_df, hrsn_labels, outcome_labels)

    if len(state_df) > 0 and len(top_pairs) > 0:
        plot_state_coefficient_map(
            state_df, top_pairs[0], hrsn_labels, outcome_labels
        )

    return interaction_df, strat_df, state_df


if __name__ == "__main__":
    print("=" * 70)
    print("PATH 3: Geographic Heterogeneity — Policy Moderation Analysis")
    print("=" * 70)
    interact, strat, state = run_heterogeneity_analysis()
    print(f"\nDone. {len(interact)} interactions, {len(strat)} stratified, "
          f"{len(state)} state-level coefficients.")
