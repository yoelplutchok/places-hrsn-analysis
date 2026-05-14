"""
Step 28: Spatially Blocked Cross-Validation.

Extends the random 10-fold CV (script 25) with two spatial blocking strategies
to detect spatial information leakage in OOS R-squared estimates:

1. Random baseline: KFold(n_splits=10) -- reproduces script 25
2. State-blocked: LeaveOneGroupOut with state_fips (40 folds)
3. County-blocked: GroupKFold(n_splits=10) with county_fips

With Moran's I of 0.64-0.87 for HRSN and disease variables, random splits
allow spatially adjacent tracts to appear in both training and test folds,
inflating OOS R-squared through spatial information leakage.

Addresses audit finding C5-2 (spatial leakage in cross-validation).

Output:
  - data/final/spatial_cv_results.csv  (fold-level results, ~1680 rows)
  - data/final/spatial_cv_summary.csv  (summary by outcome/model/strategy)
  - outputs/figures/spatial_cv_comparison.png
"""
import sys
from pathlib import Path

import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.model_selection import KFold, LeaveOneGroupOut, GroupKFold
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


def _run_cv_strategy(df, splitter, groups, model_specs, outcome_cols, cv_strategy):
    """Run all model specs x outcomes across all folds of a given CV splitter.

    Returns list of per-fold result dicts.
    """
    results = []
    X_dummy = np.zeros(len(df))  # dummy for splitter API

    # Pre-compute splits (so we iterate folds once)
    splits = list(splitter.split(X_dummy, groups=groups))
    n_folds = len(splits)
    logger.info(f"  Strategy: {cv_strategy} ({n_folds} folds)")

    for outcome in outcome_cols:
        if outcome not in df.columns:
            continue

        for model_name, features in model_specs.items():
            available = [f for f in features if f in df.columns]
            X_all = df[available].values
            y_all = df[outcome].values

            for fold_idx, (train_idx, test_idx) in enumerate(splits):
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

                    results.append({
                        "outcome": outcome,
                        "model": model_name,
                        "fold": fold_idx,
                        "cv_strategy": cv_strategy,
                        "n_train": len(train_idx),
                        "n_test": len(test_idx),
                        "rmse": rmse,
                        "mae": mae,
                        "r2_oos": r2_oos,
                        "r2_insample": model.rsquared,
                    })
                except Exception as e:
                    logger.warning(f"    {cv_strategy} {model_name} {outcome} "
                                   f"fold {fold_idx}: {e}")

    return results


def _plot_spatial_cv(summary):
    """Create comparison figure for spatial CV results."""
    strategies = ["Random (10-fold)", "County-Blocked (10-fold)", "State-Blocked (LOSO)"]
    strategy_colors = {
        "Random (10-fold)": "#3498db",
        "County-Blocked (10-fold)": "#e74c3c",
        "State-Blocked (LOSO)": "#2ecc71",
    }
    models = ["HRSN + Covariates", "SVI + Covariates"]
    outcomes = sorted(summary["outcome"].unique())

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    for ax_idx, model_name in enumerate(models):
        ax = axes[ax_idx]
        x = np.arange(len(outcomes))
        width = 0.25

        for i, strategy in enumerate(strategies):
            means = []
            stds = []
            for outcome in outcomes:
                sub = summary[
                    (summary["outcome"] == outcome) &
                    (summary["model"] == model_name) &
                    (summary["cv_strategy"] == strategy)
                ]
                if len(sub) > 0:
                    means.append(sub.iloc[0]["r2_oos_mean"])
                    stds.append(sub.iloc[0]["r2_oos_std"])
                else:
                    means.append(0)
                    stds.append(0)

            ax.bar(x + i * width, means, width, yerr=stds,
                   label=strategy, color=strategy_colors[strategy],
                   alpha=0.8, capsize=2)

        ax.set_xlabel("Disease Outcome", fontsize=11)
        ax.set_ylabel("Out-of-Sample R²", fontsize=11)
        ax.set_title(f"{model_name}\nSpatial vs Random CV", fontsize=12,
                     fontweight="bold")
        ax.set_xticks(x + width)
        ax.set_xticklabels(outcomes, rotation=30, ha="right", fontsize=9)
        ax.legend(fontsize=8)
        ax.set_ylim(0, 1)

    plt.tight_layout()
    dest = PATHS["figures"] / "spatial_cv_comparison.png"
    fig.savefig(dest, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    logger.info(f"Saved spatial CV plot to {dest}")


def run_spatial_cv():
    """Run spatially blocked cross-validation and compare with random CV."""
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
    svi_df = svi_df.replace(-999, np.nan)

    # Merge
    merged = tract_df.merge(svi_df, on="tract_fips", how="inner")

    # Drop rows with any missing
    all_features = hrsn_cols + svi_cols + covariate_cols
    available_features = [c for c in all_features if c in merged.columns]
    complete = merged.dropna(subset=available_features + outcome_cols)
    logger.info(f"Complete cases: {len(complete):,} tracts, "
                f"{complete['state_fips'].nunique()} states, "
                f"{complete['county_fips'].nunique()} counties")

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

    # Prepare group variables
    state_groups = df["state_fips"].values
    county_groups = df["county_fips"].values

    # Define CV strategies
    cv_strategies = [
        ("Random (10-fold)",
         KFold(n_splits=10, shuffle=True, random_state=RANDOM_STATE),
         None),
        ("County-Blocked (10-fold)",
         GroupKFold(n_splits=10),
         county_groups),
        ("State-Blocked (LOSO)",
         LeaveOneGroupOut(),
         state_groups),
    ]

    # Run all strategies
    logger.info("\n=== Spatially Blocked Cross-Validation ===")
    all_results = []
    for strategy_name, splitter, groups in cv_strategies:
        results = _run_cv_strategy(
            df, splitter, groups, model_specs, outcome_cols, strategy_name
        )
        all_results.extend(results)

    results_df = pd.DataFrame(all_results)
    save_csv(results_df, PATHS["final"] / "spatial_cv_results.csv")
    logger.info(f"\nTotal fold-level results: {len(results_df):,}")

    # Summary table
    summary = results_df.groupby(["outcome", "model", "cv_strategy"]).agg(
        rmse_mean=("rmse", "mean"),
        rmse_std=("rmse", "std"),
        mae_mean=("mae", "mean"),
        r2_oos_mean=("r2_oos", "mean"),
        r2_oos_std=("r2_oos", "std"),
        r2_insample_mean=("r2_insample", "mean"),
        n_folds=("fold", "nunique"),
    ).round(4).reset_index()
    save_csv(summary, PATHS["final"] / "spatial_cv_summary.csv")

    # Overall comparison: HRSN + Covariates model across strategies
    logger.info("\n=== Spatial Leakage Analysis ===")
    logger.info("Mean OOS R² for HRSN + Covariates model:")
    for strategy_name, _, _ in cv_strategies:
        sub = summary[
            (summary["model"] == "HRSN + Covariates") &
            (summary["cv_strategy"] == strategy_name)
        ]
        if len(sub) > 0:
            mean_r2 = sub["r2_oos_mean"].mean()
            logger.info(f"  {strategy_name:30s}: {mean_r2:.4f}")

    # Compute spatial leakage percentage
    random_hrsn = summary[
        (summary["model"] == "HRSN + Covariates") &
        (summary["cv_strategy"] == "Random (10-fold)")
    ]["r2_oos_mean"].mean()

    for strategy_name in ["County-Blocked (10-fold)", "State-Blocked (LOSO)"]:
        spatial_hrsn = summary[
            (summary["model"] == "HRSN + Covariates") &
            (summary["cv_strategy"] == strategy_name)
        ]["r2_oos_mean"].mean()
        if random_hrsn > 0:
            leakage_pct = (random_hrsn - spatial_hrsn) / random_hrsn * 100
            logger.info(f"  Spatial leakage ({strategy_name}): {leakage_pct:.1f}%")

    # Does HRSN still beat SVI under spatial blocking?
    logger.info("\n=== HRSN vs SVI Under Spatial Blocking ===")
    for strategy_name, _, _ in cv_strategies:
        hrsn_r2 = summary[
            (summary["model"] == "HRSN + Covariates") &
            (summary["cv_strategy"] == strategy_name)
        ]["r2_oos_mean"].mean()
        svi_r2 = summary[
            (summary["model"] == "SVI + Covariates") &
            (summary["cv_strategy"] == strategy_name)
        ]["r2_oos_mean"].mean()
        logger.info(f"  {strategy_name:30s}: HRSN={hrsn_r2:.4f}, "
                    f"SVI={svi_r2:.4f}, diff={hrsn_r2 - svi_r2:+.4f}")

    # Plot
    _plot_spatial_cv(summary)

    return results_df


if __name__ == "__main__":
    print("=" * 70)
    print("Step 28: Spatially Blocked Cross-Validation")
    print("=" * 70)
    results = run_spatial_cv()
    print(f"\nDone. {len(results):,} fold-level results.")
