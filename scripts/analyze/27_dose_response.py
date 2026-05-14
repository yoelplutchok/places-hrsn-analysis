"""
Step 27: Dose-Response and Threshold Analysis.

Tests whether HRSN-disease relationships are linear or exhibit
threshold/gradient effects using:

1. Quantile regression at the 10th, 25th, 50th, 75th, and 90th
   percentiles of disease prevalence.

2. If the coefficient increases at higher quantiles, the HRSN-disease
   relationship is STRONGER in communities with already-high disease
   burden — meaning interventions in the worst-off areas would have
   disproportionately large effects.

This informs policy about WHERE to focus HRSN interventions:
everywhere equally, or concentrated in the highest-burden areas.

Output:
  - data/final/quantile_regression_results.csv
  - outputs/figures/dose_response_quantile.png
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

QUANTILES = [0.10, 0.25, 0.50, 0.75, 0.90]

# Focus on key HRSN measures (one per factor plus strongest)
KEY_HRSN = ["foodinsecu", "shututility", "loneliness"]
KEY_DISEASES = ["diabetes", "depression", "copd", "obesity"]


def run_dose_response():
    ensure_dirs()
    params = load_params()

    hrsn_cols = [m["id"].lower() for m in params["hrsn_measures"]]
    outcome_cols = [m["id"].lower() for m in params["outcome_measures"]]
    covariate_cols = ["pct_black", "pct_hispanic", "pct_poverty",
                      "pct_college", "pct_65plus", "median_age"]

    tract_df = load_parquet(PATHS["processed"] / "merged_tracts.parquet")
    logger.info(f"Tract data: {len(tract_df):,} tracts")

    # Standardize
    df = tract_df.copy()
    all_cols = hrsn_cols + outcome_cols + covariate_cols
    for col in all_cols:
        if col in df.columns:
            s = df[col].dropna()
            if s.std() > 0:
                df[col] = (df[col] - s.mean()) / s.std()

    # Run quantile regressions for ALL HRSN x disease pairs
    logger.info("\n=== Quantile Regression Analysis ===")
    results = []

    for hrsn in hrsn_cols:
        if hrsn not in df.columns:
            continue
        for disease in outcome_cols:
            if disease not in df.columns:
                continue

            X_cols = [hrsn] + [c for c in covariate_cols if c in df.columns]
            X = sm.add_constant(df[X_cols])
            y = df[disease]
            mask = X.notna().all(axis=1) & y.notna()
            X_clean = X[mask]
            y_clean = y[mask]

            if len(y_clean) < 100:
                continue

            # OLS for reference
            try:
                ols = sm.OLS(y_clean, X_clean).fit()
                ols_beta = ols.params[hrsn]
            except Exception:
                ols_beta = np.nan

            # Quantile regressions
            for q in QUANTILES:
                try:
                    qr = sm.QuantReg(y_clean, X_clean).fit(q=q, max_iter=1000)
                    beta = qr.params[hrsn]
                    se = qr.bse[hrsn]
                    ci_low = beta - 1.96 * se
                    ci_high = beta + 1.96 * se

                    results.append({
                        "hrsn_measure": hrsn,
                        "disease": disease,
                        "quantile": q,
                        "beta": round(beta, 4),
                        "se": round(se, 4),
                        "ci_lower": round(ci_low, 4),
                        "ci_upper": round(ci_high, 4),
                        "ols_beta": round(ols_beta, 4),
                        "n_obs": int(len(y_clean)),
                    })
                except Exception as e:
                    logger.warning(f"  {hrsn}→{disease} q={q}: {e}")

    results_df = pd.DataFrame(results)
    logger.info(f"\nTotal quantile regression results: {len(results_df)}")

    # Analyze gradient (is effect stronger at higher quantiles?)
    logger.info("\n=== Gradient Analysis ===")
    gradient_records = []
    for hrsn in hrsn_cols:
        for disease in outcome_cols:
            sub = results_df[(results_df["hrsn_measure"] == hrsn) &
                            (results_df["disease"] == disease)]
            if len(sub) < 3:
                continue

            betas = sub.sort_values("quantile")["beta"].values
            quantiles = sub.sort_values("quantile")["quantile"].values

            # Descriptive slope of beta across quantiles
            # NOTE: With only 5 quantile points, formal significance testing
            # (linregress on n=5 correlated values) is not statistically valid.
            # We report the slope descriptively to characterize the pattern
            # (increasing/decreasing) without formal hypothesis testing.
            if len(betas) > 2:
                from scipy import stats as sp_stats
                slope, intercept, r, p, se = sp_stats.linregress(quantiles, betas)

                beta_10 = sub[sub["quantile"] == 0.10]["beta"].iloc[0] if 0.10 in sub["quantile"].values else np.nan
                beta_90 = sub[sub["quantile"] == 0.90]["beta"].iloc[0] if 0.90 in sub["quantile"].values else np.nan

                # Detect sign reversals: beta changes sign across quantile range
                sign_reversal = False
                if not (np.isnan(beta_10) or np.isnan(beta_90)):
                    sign_reversal = (beta_10 > 0) != (beta_90 > 0) and abs(beta_10) > 0.01 and abs(beta_90) > 0.01

                gradient_records.append({
                    "hrsn_measure": hrsn,
                    "disease": disease,
                    "beta_q10": beta_10,
                    "beta_q90": beta_90,
                    "gradient_slope": round(slope, 4),
                    "gradient_direction": "increasing" if slope > 0 else "decreasing",
                    "sign_reversal": sign_reversal,
                    "ratio_q90_q10": round(beta_90 / beta_10, 2) if beta_10 != 0 and not np.isnan(beta_10) else np.nan,
                })

    gradient_df = pd.DataFrame(gradient_records)

    n_increasing = (gradient_df["gradient_direction"] == "increasing").sum()
    n_decreasing = (gradient_df["gradient_direction"] == "decreasing").sum()
    n_sign_reversal = gradient_df["sign_reversal"].sum()
    n_total = len(gradient_df)
    logger.info(f"  Increasing gradient (stronger at high quantiles): {n_increasing}/{n_total}")
    logger.info(f"  Decreasing gradient (weaker at high quantiles): {n_decreasing}/{n_total}")
    logger.info(f"  Sign reversals (beta flips sign across quantiles): {n_sign_reversal}/{n_total}")
    if n_sign_reversal > 0:
        reversals = gradient_df[gradient_df["sign_reversal"]]
        for _, row in reversals.iterrows():
            logger.info(f"    {row['hrsn_measure']}→{row['disease']}: "
                        f"q10={row['beta_q10']:.3f}, q90={row['beta_q90']:.3f}")

    # Log key pairs
    for hrsn in KEY_HRSN:
        for disease in KEY_DISEASES:
            g = gradient_df[(gradient_df["hrsn_measure"] == hrsn) &
                           (gradient_df["disease"] == disease)]
            if len(g) > 0:
                row = g.iloc[0]
                logger.info(f"  {hrsn}→{disease}: q10={row['beta_q10']:.3f}, "
                          f"q90={row['beta_q90']:.3f}, gradient={row['gradient_slope']:.3f} "
                          f"({row['gradient_direction']})")

    save_csv(results_df, PATHS["final"] / "quantile_regression_results.csv")
    save_csv(gradient_df, PATHS["final"] / "quantile_gradient_analysis.csv")

    # Plot
    _plot_dose_response(results_df, gradient_df)

    return results_df


def _plot_dose_response(results_df, gradient_df):
    """Plot quantile regression coefficients across quantiles."""
    # Select key pairs for visualization
    pairs = [(h, d) for h in KEY_HRSN for d in KEY_DISEASES]
    available_pairs = [(h, d) for h, d in pairs
                       if len(results_df[(results_df["hrsn_measure"] == h) &
                                         (results_df["disease"] == d)]) >= 3]

    if not available_pairs:
        logger.warning("No pairs available for plotting")
        return

    n_pairs = min(len(available_pairs), 12)
    ncols = 4
    nrows = (n_pairs + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3.5 * nrows))
    if nrows == 1:
        axes = axes.reshape(1, -1)
    axes_flat = axes.flatten()

    for idx, (hrsn, disease) in enumerate(available_pairs[:n_pairs]):
        ax = axes_flat[idx]
        sub = results_df[(results_df["hrsn_measure"] == hrsn) &
                        (results_df["disease"] == disease)].sort_values("quantile")

        if len(sub) == 0:
            continue

        # Quantile regression betas with CI
        ax.fill_between(sub["quantile"], sub["ci_lower"], sub["ci_upper"],
                        alpha=0.2, color="#2c3e50")
        ax.plot(sub["quantile"], sub["beta"], "o-", color="#2c3e50",
                markersize=5, linewidth=1.5)

        # OLS reference line
        ols_beta = sub["ols_beta"].iloc[0]
        ax.axhline(ols_beta, color="#e74c3c", linestyle="--",
                   alpha=0.5, linewidth=1, label=f"OLS β={ols_beta:.2f}")

        ax.axhline(0, color="gray", linewidth=0.5, alpha=0.5)
        ax.set_title(f"{hrsn} → {disease}", fontsize=10, fontweight="bold")
        ax.set_xlabel("Quantile", fontsize=9)
        ax.set_ylabel("β coefficient", fontsize=9)
        ax.legend(fontsize=7, loc="best")

        # Add gradient annotation
        g = gradient_df[(gradient_df["hrsn_measure"] == hrsn) &
                       (gradient_df["disease"] == disease)]
        if len(g) > 0:
            row = g.iloc[0]
            direction = row["gradient_direction"]
            if direction == "increasing":
                ax.text(0.95, 0.05, "↑ Increasing", transform=ax.transAxes,
                       fontsize=8, ha="right", color="#e74c3c", fontweight="bold")
            elif direction == "decreasing":
                ax.text(0.95, 0.05, "↓ Decreasing", transform=ax.transAxes,
                       fontsize=8, ha="right", color="#3498db", fontweight="bold")

    # Hide unused axes
    for idx in range(n_pairs, len(axes_flat)):
        axes_flat[idx].set_visible(False)

    fig.suptitle("Quantile Heterogeneity: HRSN-Disease Associations Across Disease Prevalence Quantiles",
                fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()

    dest = PATHS["figures"] / "dose_response_quantile.png"
    fig.savefig(dest, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    logger.info(f"Saved dose-response plot to {dest}")


if __name__ == "__main__":
    print("=" * 70)
    print("Step 27: Dose-Response / Threshold Analysis")
    print("=" * 70)
    results = run_dose_response()
    print(f"\nDone. {len(results)} quantile regression results.")
