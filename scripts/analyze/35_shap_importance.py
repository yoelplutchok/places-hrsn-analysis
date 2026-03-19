"""
Step 35: Random Forest Variable Importance (Permutation + SHAP).

Trains random forest models for each of the 7 disease outcomes and computes
both permutation importance and SHAP values to obtain model-agnostic,
collinearity-robust variable importance rankings.

This addresses the concern that OLS betas are distorted by multicollinearity
(VIF up to 97.4) among HRSN measures. Random forests handle collinearity
naturally, and permutation/SHAP importance provide theoretically grounded
importance metrics.

Compares RF-based importance ranking to OLS beta ranking to assess
whether the relative importance of HRSN measures is robust to model choice.

Output:
  - data/final/shap_importance.csv (7 outcomes × 13 features = 91 rows)
  - data/final/shap_importance_summary.csv (rank comparison with OLS)
  - outputs/figures/shap_importance_comparison.png
"""
import sys
from pathlib import Path

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split
import shap
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
N_TREES = 300
MAX_DEPTH = 15
SHAP_BACKGROUND = 100   # Background dataset for KernelExplainer
SHAP_NSAMPLES = 500     # Number of samples to explain
PERM_REPEATS = 10       # Permutation importance repeats


def run_shap_analysis():
    """Train random forests and compute importance for each disease."""
    ensure_dirs()
    params = load_params()

    hrsn_cols = [m["id"].lower() for m in params["hrsn_measures"]]
    outcome_cols = [m["id"].lower() for m in params["outcome_measures"]]
    hrsn_labels = {m["id"].lower(): m["name"] for m in params["hrsn_measures"]}
    outcome_labels = {m["id"].lower(): m["name"] for m in params["outcome_measures"]}
    covariate_cols = ["pct_black", "pct_hispanic", "pct_poverty",
                      "pct_college", "pct_65plus", "median_age"]
    feature_cols = hrsn_cols + covariate_cols

    # Load data
    df = load_parquet(PATHS["processed"] / "merged_tracts.parquet")
    df = df.dropna(subset=feature_cols + outcome_cols)
    logger.info(f"Complete cases: {len(df):,} tracts")

    # Load OLS results for comparison
    ols_results = pd.read_csv(PATHS["final"] / "results_matrix.csv")

    # Train/test split for honest permutation importance
    X_all = df[feature_cols].values
    rng = np.random.RandomState(RANDOM_STATE)

    all_importance = []

    for outcome in outcome_cols:
        out_label = outcome_labels.get(outcome, outcome)
        logger.info(f"\n--- {out_label} ---")

        y_all = df[outcome].values

        # Split for honest evaluation
        X_train, X_test, y_train, y_test = train_test_split(
            X_all, y_all, test_size=0.2, random_state=RANDOM_STATE
        )

        # Train random forest
        rf = RandomForestRegressor(
            n_estimators=N_TREES,
            max_depth=MAX_DEPTH,
            min_samples_leaf=5,
            max_features="sqrt",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )
        rf.fit(X_train, y_train)
        r2_train = rf.score(X_train, y_train)
        r2_test = rf.score(X_test, y_test)
        logger.info(f"  RF R² train: {r2_train:.4f}, test: {r2_test:.4f}")

        # 1. Gini importance (built-in)
        gini_imp = rf.feature_importances_

        # 2. Permutation importance (on test set — honest)
        logger.info(f"  Computing permutation importance...")
        perm_result = permutation_importance(
            rf, X_test, y_test,
            n_repeats=PERM_REPEATS,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )
        perm_imp = perm_result.importances_mean

        # 3. SHAP values using TreeExplainer with small background
        logger.info(f"  Computing SHAP values...")
        bg_idx = rng.choice(len(X_train), size=SHAP_BACKGROUND, replace=False)
        explain_idx = rng.choice(len(X_test), size=min(SHAP_NSAMPLES, len(X_test)), replace=False)

        explainer = shap.TreeExplainer(rf, X_train[bg_idx])
        sv = explainer.shap_values(X_test[explain_idx])
        mean_abs_shap = np.mean(np.abs(sv), axis=0)

        for i, feat in enumerate(feature_cols):
            feat_type = "hrsn" if feat in hrsn_cols else "covariate"
            feat_label = hrsn_labels.get(feat, feat)

            # Get corresponding OLS beta
            ols_row = ols_results[
                (ols_results["hrsn_measure"] == feat) &
                (ols_results["outcome"] == outcome)
            ]
            ols_beta = float(ols_row["beta_std"].iloc[0]) if len(ols_row) > 0 else np.nan

            all_importance.append({
                "outcome": outcome,
                "outcome_label": out_label,
                "feature": feat,
                "feature_label": feat_label,
                "feature_type": feat_type,
                "mean_abs_shap": round(mean_abs_shap[i], 6),
                "permutation_importance": round(perm_imp[i], 6),
                "rf_feature_importance": round(gini_imp[i], 6),
                "ols_beta_std": round(ols_beta, 4) if not np.isnan(ols_beta) else np.nan,
                "ols_abs_beta": round(abs(ols_beta), 4) if not np.isnan(ols_beta) else np.nan,
                "rf_r2_train": round(r2_train, 4),
                "rf_r2_test": round(r2_test, 4),
            })

        # Log top 5 by permutation importance
        sorted_idx = np.argsort(perm_imp)[::-1]
        logger.info(f"  Top 5 features (permutation):")
        for rank, idx_feat in enumerate(sorted_idx[:5]):
            feat = feature_cols[idx_feat]
            logger.info(f"    {rank + 1}. {feat:25s}: perm={perm_imp[idx_feat]:.4f}, "
                        f"SHAP={mean_abs_shap[idx_feat]:.4f}")

    importance_df = pd.DataFrame(all_importance)
    save_csv(importance_df, PATHS["final"] / "shap_importance.csv")

    # --- Rank comparison: RF vs OLS ---
    logger.info("\n=== RF vs OLS Rank Comparison (HRSN features only) ===")
    rank_rows = []
    for outcome in outcome_cols:
        out_label = outcome_labels.get(outcome, outcome)
        sub = importance_df[
            (importance_df["outcome"] == outcome) &
            (importance_df["feature_type"] == "hrsn")
        ].copy()

        if len(sub) == 0:
            continue

        sub["perm_rank"] = sub["permutation_importance"].rank(ascending=False).astype(int)
        sub["shap_rank"] = sub["mean_abs_shap"].rank(ascending=False).astype(int)
        sub["ols_rank"] = sub["ols_abs_beta"].rank(ascending=False).astype(int)

        # Spearman correlations
        spearman_perm = sub["permutation_importance"].corr(sub["ols_abs_beta"], method="spearman")
        spearman_shap = sub["mean_abs_shap"].corr(sub["ols_abs_beta"], method="spearman")

        for _, r in sub.iterrows():
            rank_rows.append({
                "outcome": outcome,
                "outcome_label": out_label,
                "feature": r["feature"],
                "feature_label": r["feature_label"],
                "perm_rank": r["perm_rank"],
                "shap_rank": r["shap_rank"],
                "ols_rank": r["ols_rank"],
                "spearman_perm_vs_ols": round(spearman_perm, 4),
                "spearman_shap_vs_ols": round(spearman_shap, 4),
            })

        logger.info(f"  {out_label:15s}: rho(perm,OLS)={spearman_perm:.3f}, "
                    f"rho(SHAP,OLS)={spearman_shap:.3f}")

    rank_df = pd.DataFrame(rank_rows)
    save_csv(rank_df, PATHS["final"] / "shap_importance_summary.csv")

    # Mean Spearman across outcomes
    by_outcome = rank_df.groupby("outcome")[["spearman_perm_vs_ols", "spearman_shap_vs_ols"]].first()
    logger.info(f"\n  Mean Spearman (perm vs OLS): {by_outcome['spearman_perm_vs_ols'].mean():.3f}")
    logger.info(f"  Mean Spearman (SHAP vs OLS): {by_outcome['spearman_shap_vs_ols'].mean():.3f}")

    # --- Plot ---
    _plot_shap_comparison(importance_df, outcome_cols, outcome_labels, hrsn_cols)

    return importance_df


def _plot_shap_comparison(importance_df, outcome_cols, outcome_labels, hrsn_cols):
    """Create importance comparison figure."""
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    axes = axes.flatten()

    for i, outcome in enumerate(outcome_cols):
        if i >= 7:
            break
        ax = axes[i]
        out_label = outcome_labels.get(outcome, outcome)
        sub = importance_df[
            (importance_df["outcome"] == outcome) &
            (importance_df["feature_type"] == "hrsn")
        ].sort_values("permutation_importance", ascending=True)

        y_pos = np.arange(len(sub))
        width = 0.35
        ax.barh(y_pos - width/2, sub["permutation_importance"], width,
                label="Permutation", color="#3498db", alpha=0.8)
        ax.barh(y_pos + width/2, sub["mean_abs_shap"], width,
                label="SHAP", color="#e74c3c", alpha=0.8)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(sub["feature_label"], fontsize=7)
        ax.set_xlabel("Importance", fontsize=9)
        ax.set_title(out_label, fontsize=10, fontweight="bold")
        ax.legend(fontsize=7)

    # Remove the 8th subplot
    axes[7].set_visible(False)

    plt.suptitle("Variable Importance: HRSN Measures by Disease Outcome\n"
                 "(Permutation + SHAP from Random Forest)",
                 fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    dest = PATHS["figures"] / "shap_importance_comparison.png"
    fig.savefig(dest, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    logger.info(f"Saved importance plot to {dest}")


if __name__ == "__main__":
    print("=" * 70)
    print("Step 35: Variable Importance (Permutation + SHAP)")
    print("=" * 70)
    results = run_shap_analysis()
    print(f"\nDone. {len(results)} importance scores computed.")
