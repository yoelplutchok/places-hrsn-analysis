"""
Step 36: Mediation Analysis — SVI → HRSN → Disease.

Tests whether HRSN measures (Material Hardship and Social Isolation factors)
mediate the relationship between SVI themes and chronic disease prevalence.

This transforms the incremental validity finding (HRSN adds R² beyond SVI)
into a mechanistic insight: does SVI operate *through* social needs to
affect disease, or does it provide an independent pathway?

Uses the Baron & Kenny (1986) framework with bootstrap confidence intervals
for indirect effects (Preacher & Hayes, 2008):
  Path a: SVI → HRSN factor (exposure → mediator)
  Path b: HRSN factor → Disease (mediator → outcome, controlling for SVI)
  Path c: SVI → Disease (total effect)
  Path c': SVI → Disease controlling for HRSN factor (direct effect)
  Indirect effect = a × b (with bootstrap 95% CI)

Output:
  - data/final/mediation_results.csv (28 models: 4 SVI themes × 7 outcomes)
  - outputs/figures/mediation_effects.png
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

RANDOM_STATE = 42
N_BOOTSTRAP = 1000


def _bootstrap_indirect(X_exposure, mediator, y_outcome, covariates, n_boot, rng):
    """Compute bootstrap CI for indirect effect (a * b).

    a = coefficient of exposure on mediator (controlling for covariates)
    b = coefficient of mediator on outcome (controlling for exposure + covariates)
    indirect = a * b
    """
    n = len(y_outcome)
    indirect_effects = []

    for _ in range(n_boot):
        idx = rng.choice(n, size=n, replace=True)
        X_exp_b = X_exposure[idx]
        med_b = mediator[idx]
        y_b = y_outcome[idx]
        cov_b = covariates[idx] if covariates is not None else None

        try:
            # Path a: exposure -> mediator
            if cov_b is not None:
                Xa = sm.add_constant(np.column_stack([X_exp_b, cov_b]))
            else:
                Xa = sm.add_constant(X_exp_b)
            model_a = sm.OLS(med_b, Xa).fit()
            a = model_a.params[1]  # coefficient of exposure

            # Path b: mediator -> outcome (controlling for exposure)
            if cov_b is not None:
                Xb = sm.add_constant(np.column_stack([X_exp_b, med_b, cov_b]))
            else:
                Xb = sm.add_constant(np.column_stack([X_exp_b, med_b]))
            model_b = sm.OLS(y_b, Xb).fit()
            b = model_b.params[2]  # coefficient of mediator

            indirect_effects.append(a * b)
        except Exception:
            continue

    return np.array(indirect_effects)


def run_mediation_analysis():
    """Run mediation analysis: SVI → HRSN factor → Disease."""
    ensure_dirs()
    params = load_params()

    outcome_cols = [m["id"].lower() for m in params["outcome_measures"]]
    outcome_labels = {m["id"].lower(): m["name"] for m in params["outcome_measures"]}
    covariate_cols = ["pct_black", "pct_hispanic", "pct_poverty",
                      "pct_college", "pct_65plus", "median_age"]

    # Load tract data
    tract_df = load_parquet(PATHS["processed"] / "merged_tracts.parquet")

    # Load SVI
    svi_df = pd.read_csv(PATHS["raw"] / "svi" / "SVI_2022_US.csv",
                         usecols=["FIPS", "RPL_THEME1", "RPL_THEME2",
                                  "RPL_THEME3", "RPL_THEME4"],
                         dtype={"FIPS": str}, low_memory=False)
    svi_df = svi_df.rename(columns={"FIPS": "tract_fips"})
    svi_df["tract_fips"] = svi_df["tract_fips"].str.zfill(11)
    svi_cols = ["RPL_THEME1", "RPL_THEME2", "RPL_THEME3", "RPL_THEME4"]
    svi_labels = {
        "RPL_THEME1": "SVI: Socioeconomic Status",
        "RPL_THEME2": "SVI: Household Composition/Disability",
        "RPL_THEME3": "SVI: Minority Status/Language",
        "RPL_THEME4": "SVI: Housing Type/Transportation",
    }
    for col in svi_cols:
        svi_df[col] = pd.to_numeric(svi_df[col], errors="coerce")
    svi_df = svi_df.replace(-999, np.nan)

    # Load factor scores (mediators)
    factor_df = load_parquet(PATHS["final"] / "factor_scores.parquet")
    mediator_cols = ["material_hardship", "social_isolation"]

    # Merge
    merged = tract_df.merge(svi_df, on="tract_fips", how="inner")
    merged = merged.merge(factor_df, on="tract_fips", how="inner")

    # Drop incomplete cases
    all_vars = svi_cols + mediator_cols + outcome_cols + covariate_cols
    df = merged.dropna(subset=all_vars).copy()
    logger.info(f"Complete cases: {len(df):,} tracts")

    # Standardize all analysis variables
    for col in all_vars:
        s = df[col]
        df[col] = (s - s.mean()) / s.std()

    rng = np.random.default_rng(RANDOM_STATE)
    results = []

    # For each SVI theme × outcome, test mediation through Material Hardship
    # (Material Hardship is the dominant factor for physical disease outcomes)
    mediator = "material_hardship"
    logger.info(f"\n=== Mediation: SVI → {mediator} → Disease ===")
    logger.info(f"  Bootstrap: {N_BOOTSTRAP} resamples")

    for svi_col in svi_cols:
        svi_label = svi_labels.get(svi_col, svi_col)
        logger.info(f"\n  --- {svi_label} ---")

        for outcome in outcome_cols:
            out_label = outcome_labels.get(outcome, outcome)

            X_exp = df[svi_col].values
            med = df[mediator].values
            y = df[outcome].values
            cov = df[covariate_cols].values

            # Path c (total): outcome ~ SVI + covariates
            Xc = sm.add_constant(np.column_stack([X_exp, cov]))
            model_c = sm.OLS(y, Xc).fit()
            c_total = model_c.params[1]

            # Path a: mediator ~ SVI + covariates
            Xa = sm.add_constant(np.column_stack([X_exp, cov]))
            model_a = sm.OLS(med, Xa).fit()
            a = model_a.params[1]

            # Path c' (direct) and b: outcome ~ SVI + mediator + covariates
            Xcp = sm.add_constant(np.column_stack([X_exp, med, cov]))
            model_cp = sm.OLS(y, Xcp).fit()
            c_prime = model_cp.params[1]  # direct effect of SVI
            b = model_cp.params[2]  # effect of mediator

            # Indirect effect = a * b
            indirect = a * b
            # Proportion mediated
            prop_mediated = indirect / c_total if abs(c_total) > 0.001 else np.nan

            # Bootstrap CI for indirect effect
            boot_indirect = _bootstrap_indirect(X_exp, med, y, cov, N_BOOTSTRAP, rng)
            if len(boot_indirect) > 0:
                ci_lower = np.percentile(boot_indirect, 2.5)
                ci_upper = np.percentile(boot_indirect, 97.5)
                significant = not (ci_lower <= 0 <= ci_upper)
            else:
                ci_lower = ci_upper = np.nan
                significant = False

            logger.info(f"    {out_label:12s}: c={c_total:+.4f}, a={a:+.4f}, "
                        f"b={b:+.4f}, c'={c_prime:+.4f}, "
                        f"indirect={indirect:+.4f} [{ci_lower:+.4f}, {ci_upper:+.4f}] "
                        f"{'*' if significant else ''}")

            results.append({
                "svi_theme": svi_col,
                "svi_label": svi_label,
                "mediator": mediator,
                "outcome": outcome,
                "outcome_label": out_label,
                "path_a": round(a, 4),
                "path_b": round(b, 4),
                "path_c_total": round(c_total, 4),
                "path_c_prime_direct": round(c_prime, 4),
                "indirect_effect": round(indirect, 4),
                "indirect_ci_lower": round(ci_lower, 4),
                "indirect_ci_upper": round(ci_upper, 4),
                "proportion_mediated": round(prop_mediated, 4)
                    if not np.isnan(prop_mediated) else np.nan,
                "significant": significant,
                "r2_total": round(model_c.rsquared, 4),
                "r2_mediated": round(model_cp.rsquared, 4),
            })

    results_df = pd.DataFrame(results)
    save_csv(results_df, PATHS["final"] / "mediation_results.csv")

    # --- Summary ---
    sig = results_df[results_df["significant"]]
    logger.info(f"\n=== Summary ===")
    logger.info(f"Total mediation models: {len(results_df)}")
    logger.info(f"Significant indirect effects: {len(sig)}/{len(results_df)}")
    if len(sig) > 0:
        logger.info(f"Mean proportion mediated (significant): "
                    f"{sig['proportion_mediated'].mean():.3f}")
        logger.info(f"Range: {sig['proportion_mediated'].min():.3f} to "
                    f"{sig['proportion_mediated'].max():.3f}")

    # By SVI theme
    logger.info(f"\nBy SVI theme:")
    for svi_col in svi_cols:
        svi_label = svi_labels.get(svi_col, svi_col)
        sub = results_df[results_df["svi_theme"] == svi_col]
        n_sig = sub["significant"].sum()
        mean_prop = sub.loc[sub["significant"], "proportion_mediated"].mean()
        logger.info(f"  {svi_label:45s}: {n_sig}/7 significant, "
                    f"mean %mediated={mean_prop:.1%}" if n_sig > 0 else
                    f"  {svi_label:45s}: 0/7 significant")

    # Plot
    _plot_mediation(results_df, svi_labels)

    return results_df


def _plot_mediation(results_df, svi_labels):
    """Plot mediation effects."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    for i, svi_col in enumerate(["RPL_THEME1", "RPL_THEME2", "RPL_THEME3", "RPL_THEME4"]):
        ax = axes[i // 2][i % 2]
        sub = results_df[results_df["svi_theme"] == svi_col].sort_values("outcome_label")

        x = np.arange(len(sub))
        width = 0.35

        ax.bar(x - width / 2, sub["path_c_total"], width,
               label="Total effect (c)", color="#3498db", alpha=0.8)
        ax.bar(x + width / 2, sub["path_c_prime_direct"], width,
               label="Direct effect (c')", color="#e74c3c", alpha=0.8)

        # Mark significant mediation with stars
        for j, (_, row) in enumerate(sub.iterrows()):
            if row["significant"]:
                y_pos = max(abs(row["path_c_total"]), abs(row["path_c_prime_direct"])) + 0.02
                ax.text(j, y_pos, "*", ha="center", fontsize=14, fontweight="bold")

        ax.set_xticks(x)
        ax.set_xticklabels(sub["outcome_label"], rotation=30, ha="right", fontsize=9)
        ax.set_ylabel("Standardized Coefficient", fontsize=10)
        ax.set_title(svi_labels.get(svi_col, svi_col), fontsize=11, fontweight="bold")
        ax.legend(fontsize=8)
        ax.axhline(y=0, color="gray", linestyle="--", linewidth=0.5)

    plt.suptitle("Mediation of SVI → Disease Through Material Hardship\n"
                 "(* = significant indirect effect, bootstrap 95% CI)",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    dest = PATHS["figures"] / "mediation_effects.png"
    fig.savefig(dest, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    logger.info(f"Saved mediation plot to {dest}")


if __name__ == "__main__":
    print("=" * 70)
    print("Step 36: Mediation Analysis (SVI → HRSN → Disease)")
    print("=" * 70)
    results = run_mediation_analysis()
    print(f"\nDone. {len(results)} mediation models estimated.")
