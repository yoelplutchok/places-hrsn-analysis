"""
Step 25: Out-of-Sample Cross-Validation.

Compares predictive validity of HRSN-only, SVI-only, and combined models
using random 10-fold cross-validation.

For each disease outcome, trains three model types on 90% of tracts and
predicts disease prevalence in the held-out 10%. Repeats 10 times.
Reports out-of-sample RMSE, MAE, and R² for each model type.

This addresses the concern that high in-sample R² (0.40-0.85) could
reflect overfitting or shared modeling artifacts rather than genuine
predictive power.

Output:
  - data/final/cross_validation_results.csv  (fold-level results)
  - data/final/cross_validation_summary.csv  (summary by model type)
  - outputs/figures/cross_validation_comparison.png
"""
import sys
from pathlib import Path

import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.model_selection import KFold
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

N_FOLDS = 10
RANDOM_STATE = 42


def run_cross_validation():
    ensure_dirs()
    params = load_params()

    hrsn_cols = [m["id"].lower() for m in params["hrsn_measures"]]
    outcome_cols = [m["id"].lower() for m in params["outcome_measures"]]
    covariate_cols = ["pct_black", "pct_hispanic", "pct_poverty",
                      "pct_college", "pct_65plus", "median_age"]

    # Load tract data
    tract_df = load_parquet(PATHS["processed"] / "merged_tracts.parquet")
    logger.info(f"Tract data: {len(tract_df):,} tracts")

    # Load SVI
    svi_df = pd.read_csv(PATHS["raw"] / "svi" / "SVI_2022_US.csv",
                         usecols=["FIPS", "RPL_THEME1", "RPL_THEME2",
                                  "RPL_THEME3", "RPL_THEME4"],
                         dtype={"FIPS": str}, low_memory=False)
    svi_df = svi_df.rename(columns={"FIPS": "tract_fips"})
    svi_df["tract_fips"] = svi_df["tract_fips"].str.zfill(11)
    svi_cols = ["RPL_THEME1", "RPL_THEME2", "RPL_THEME3", "RPL_THEME4"]
    for col in svi_cols:
        svi_df[col] = pd.to_numeric(svi_df[col], errors="coerce")
    # SVI uses -999 for missing
    svi_df = svi_df.replace(-999, np.nan)

    logger.info(f"SVI data: {len(svi_df):,} tracts")

    # Merge
    merged = tract_df.merge(svi_df, on="tract_fips", how="inner")
    logger.info(f"After SVI merge: {len(merged):,} tracts")

    # Drop rows with any missing in key columns
    all_features = hrsn_cols + svi_cols + covariate_cols
    available_features = [c for c in all_features if c in merged.columns]
    complete = merged.dropna(subset=available_features + outcome_cols)
    logger.info(f"Complete cases: {len(complete):,} tracts")

    # Standardize
    df = complete.copy()
    for col in available_features + outcome_cols:
        s = df[col]
        df[col] = (s - s.mean()) / s.std()

    # Define model specifications
    hrsn_features = [c for c in hrsn_cols if c in df.columns]
    svi_features = [c for c in svi_cols if c in df.columns]

    model_specs = {
        "HRSN + Covariates": hrsn_features + covariate_cols,
        "SVI + Covariates": svi_features + covariate_cols,
        "HRSN + SVI + Covariates": hrsn_features + svi_features + covariate_cols,
        "Covariates Only": covariate_cols,
    }

    # Run K-fold CV
    logger.info(f"\n=== {N_FOLDS}-Fold Cross-Validation ===")
    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    all_results = []
    for outcome in outcome_cols:
        if outcome not in df.columns:
            continue

        logger.info(f"\nOutcome: {outcome}")

        for model_name, features in model_specs.items():
            available = [f for f in features if f in df.columns]
            X_all = df[available].values
            y_all = df[outcome].values

            for fold_idx, (train_idx, test_idx) in enumerate(kf.split(X_all)):
                X_train = sm.add_constant(X_all[train_idx])
                X_test = sm.add_constant(X_all[test_idx])
                y_train = y_all[train_idx]
                y_test = y_all[test_idx]

                try:
                    model = sm.OLS(y_train, X_train).fit()
                    y_pred = model.predict(X_test)

                    residuals = y_test - y_pred
                    rmse = np.sqrt(np.mean(residuals ** 2))
                    mae = np.mean(np.abs(residuals))
                    ss_res = np.sum(residuals ** 2)
                    ss_tot = np.sum((y_test - y_test.mean()) ** 2)
                    r2_oos = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan

                    all_results.append({
                        "outcome": outcome,
                        "model": model_name,
                        "fold": fold_idx,
                        "n_train": len(train_idx),
                        "n_test": len(test_idx),
                        "rmse": rmse,
                        "mae": mae,
                        "r2_oos": r2_oos,
                        "r2_insample": model.rsquared,
                    })
                except Exception as e:
                    logger.warning(f"  {model_name} fold {fold_idx}: {e}")

        # Log summary for this outcome
        res_df = pd.DataFrame(all_results)
        for model_name in model_specs:
            sub = res_df[(res_df["outcome"] == outcome) & (res_df["model"] == model_name)]
            if len(sub) > 0:
                logger.info(f"  {model_name}: OOS R²={sub['r2_oos'].mean():.3f} "
                          f"(±{sub['r2_oos'].std():.3f}), "
                          f"RMSE={sub['rmse'].mean():.3f}")

    results_df = pd.DataFrame(all_results)
    save_csv(results_df, PATHS["final"] / "cross_validation_results.csv")

    # Summary table
    summary = results_df.groupby(["outcome", "model"]).agg(
        rmse_mean=("rmse", "mean"),
        rmse_std=("rmse", "std"),
        mae_mean=("mae", "mean"),
        r2_oos_mean=("r2_oos", "mean"),
        r2_oos_std=("r2_oos", "std"),
        r2_insample_mean=("r2_insample", "mean"),
    ).round(4).reset_index()
    save_csv(summary, PATHS["final"] / "cross_validation_summary.csv")

    # Overall summary
    logger.info("\n=== Overall Cross-Validation Summary ===")
    overall = results_df.groupby("model").agg(
        rmse_mean=("rmse", "mean"),
        r2_oos_mean=("r2_oos", "mean"),
        r2_insample_mean=("r2_insample", "mean"),
    ).round(4)
    for model_name, row in overall.iterrows():
        logger.info(f"  {model_name}: OOS R²={row['r2_oos_mean']:.3f}, "
                   f"In-sample R²={row['r2_insample_mean']:.3f}, "
                   f"RMSE={row['rmse_mean']:.3f}")

    # Plot
    _plot_cv_results(results_df, summary)

    return results_df


def _plot_cv_results(results_df, summary):
    """Create cross-validation comparison plots."""
    outcomes = results_df["outcome"].unique()
    models = ["Covariates Only", "SVI + Covariates",
              "HRSN + Covariates", "HRSN + SVI + Covariates"]
    model_colors = {
        "Covariates Only": "#95a5a6",
        "SVI + Covariates": "#3498db",
        "HRSN + Covariates": "#e74c3c",
        "HRSN + SVI + Covariates": "#2c3e50",
    }

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Panel 1: OOS R² by outcome and model
    ax = axes[0]
    x = np.arange(len(outcomes))
    width = 0.2
    for i, model_name in enumerate(models):
        means = []
        stds = []
        for outcome in outcomes:
            sub = summary[(summary["outcome"] == outcome) &
                         (summary["model"] == model_name)]
            if len(sub) > 0:
                means.append(sub.iloc[0]["r2_oos_mean"])
                stds.append(sub.iloc[0]["r2_oos_std"])
            else:
                means.append(0)
                stds.append(0)
        ax.bar(x + i * width, means, width, yerr=stds,
               label=model_name, color=model_colors[model_name],
               alpha=0.8, capsize=2)

    ax.set_xlabel("Disease Outcome", fontsize=11)
    ax.set_ylabel("Out-of-Sample R²", fontsize=11)
    ax.set_title("Predictive Validity: Out-of-Sample R²\n(10-Fold Cross-Validation)",
                fontsize=12, fontweight="bold")
    ax.set_xticks(x + 1.5 * width)
    ax.set_xticklabels(outcomes, rotation=30, ha="right", fontsize=9)
    ax.legend(fontsize=8, loc="upper left")
    ax.set_ylim(0, 1)

    # Panel 2: In-sample vs OOS R² (overfitting check)
    ax = axes[1]
    for model_name in models:
        sub = summary[summary["model"] == model_name]
        if len(sub) > 0:
            ax.scatter(sub["r2_insample_mean"], sub["r2_oos_mean"],
                      c=model_colors[model_name], s=60, alpha=0.7,
                      label=model_name, edgecolors="white", linewidth=0.5)

    lims = [0, 1]
    ax.plot(lims, lims, "k--", alpha=0.3, linewidth=1)
    ax.set_xlabel("In-Sample R²", fontsize=11)
    ax.set_ylabel("Out-of-Sample R²", fontsize=11)
    ax.set_title("Overfitting Check:\nIn-Sample vs. Out-of-Sample R²",
                fontsize=12, fontweight="bold")
    ax.legend(fontsize=8)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    plt.tight_layout()
    dest = PATHS["figures"] / "cross_validation_comparison.png"
    fig.savefig(dest, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    logger.info(f"Saved CV plot to {dest}")


if __name__ == "__main__":
    print("=" * 70)
    print("Step 25: Out-of-Sample Cross-Validation")
    print("=" * 70)
    results = run_cross_validation()
    print(f"\nDone. {len(results)} fold-level results.")
