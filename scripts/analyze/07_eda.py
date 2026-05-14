"""
Phase 3: Exploratory Data Analysis

Steps 3.1–3.5:
  3.1 Descriptive statistics table (Table 1)
  3.2 HRSN correlation matrix (multicollinearity check)
  3.4 Outcome distribution plots
  3.5 Bivariate scatter matrix (HRSN × outcomes)

Note: Step 3.3 (geographic maps) handled separately in 07b_maps.py

Output:
  - outputs/tables/table1_descriptive_stats.csv
  - outputs/figures/hrsn_correlation_matrix.png
  - outputs/figures/outcome_correlation_matrix.png
  - outputs/figures/outcome_distributions.png
  - outputs/figures/hrsn_distributions.png
  - outputs/figures/hrsn_outcome_scatter_matrix.png
"""
import sys
from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from hrsn_analysis.paths import PATHS, ensure_dirs
from hrsn_analysis.io_utils import load_parquet, load_params, save_csv
from hrsn_analysis.logging_utils import setup_logging, get_logger

setup_logging()
logger = get_logger(__name__)

# Publication-quality defaults
plt.rcParams.update({
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.labelsize": 10,
    "figure.facecolor": "white",
})


def step_3_1_descriptive_stats(df, params):
    """Generate Table 1: descriptive statistics for all key variables."""
    logger.info("=== Step 3.1: Descriptive Statistics ===")

    hrsn_cols = [m["id"].lower() for m in params["hrsn_measures"]]
    outcome_cols = [m["id"].lower() for m in params["outcome_measures"]]
    secondary_cols = [m["id"].lower() for m in params["secondary_outcomes"]]
    covariate_cols = ["pct_black", "pct_hispanic", "pct_poverty",
                      "median_income", "pct_college", "pct_no_hs",
                      "pct_65plus", "median_age", "total_population"]

    # Build nice labels
    hrsn_labels = {m["id"].lower(): m["name"] for m in params["hrsn_measures"]}
    outcome_labels = {m["id"].lower(): m["name"] for m in params["outcome_measures"]}
    secondary_labels = {m["id"].lower(): m["name"] for m in params["secondary_outcomes"]}
    cov_labels = {
        "pct_black": "% Black", "pct_hispanic": "% Hispanic",
        "pct_poverty": "% Poverty", "median_income": "Median income ($)",
        "pct_college": "% College degree", "pct_no_hs": "% No HS diploma",
        "pct_65plus": "% Age 65+", "median_age": "Median age",
        "total_population": "Total population",
    }

    all_cols = hrsn_cols + outcome_cols + secondary_cols + covariate_cols
    all_labels = {**hrsn_labels, **outcome_labels, **secondary_labels, **cov_labels}

    rows = []
    for col in all_cols:
        if col not in df.columns:
            continue
        v = df[col].dropna()
        q25, q75 = v.quantile(0.25), v.quantile(0.75)

        # Determine category
        if col in hrsn_cols:
            category = "HRSN Measure"
        elif col in outcome_cols:
            category = "Primary Outcome"
        elif col in secondary_cols:
            category = "Secondary Outcome"
        else:
            category = "Demographic Covariate"

        rows.append({
            "Category": category,
            "Variable": all_labels.get(col, col),
            "Column": col,
            "N": len(v),
            "Mean": round(v.mean(), 2),
            "SD": round(v.std(), 2),
            "Median": round(v.median(), 2),
            "Q25": round(q25, 2),
            "Q75": round(q75, 2),
            "Min": round(v.min(), 2),
            "Max": round(v.max(), 2),
        })

    table1 = pd.DataFrame(rows)
    dest = PATHS["tables"] / "table1_descriptive_stats.csv"
    table1.to_csv(dest, index=False)
    logger.info(f"Table 1 saved to {dest}")

    # Print summary
    for _, row in table1.iterrows():
        logger.info(f"  [{row['Category']}] {row['Variable']}: "
                     f"mean={row['Mean']}, sd={row['SD']}, range=[{row['Min']}, {row['Max']}]")

    return table1


def step_3_2_hrsn_correlation(df, params):
    """Compute and visualize HRSN correlation matrix."""
    logger.info("=== Step 3.2: HRSN Correlation Matrix ===")

    hrsn_cols = [m["id"].lower() for m in params["hrsn_measures"]]
    hrsn_labels = {m["id"].lower(): m["name"] for m in params["hrsn_measures"]}

    corr = df[hrsn_cols].corr()

    # Flag high correlations
    for i in range(len(hrsn_cols)):
        for j in range(i + 1, len(hrsn_cols)):
            r = corr.iloc[i, j]
            if abs(r) > 0.8:
                logger.warning(f"HIGH CORRELATION: {hrsn_cols[i]} × {hrsn_cols[j]} = {r:.3f}")
            elif abs(r) > 0.6:
                logger.info(f"  Moderate: {hrsn_cols[i]} × {hrsn_cols[j]} = {r:.3f}")

    # Plot
    fig, ax = plt.subplots(figsize=(8, 7))
    labels = [hrsn_labels.get(c, c) for c in hrsn_cols]
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="RdBu_r",
                vmin=-1, vmax=1, center=0, square=True,
                xticklabels=labels, yticklabels=labels,
                linewidths=0.5, ax=ax)
    ax.set_title("Correlation Matrix: 7 HRSN Measures\n(Pearson r, N={:,})".format(len(df)))
    plt.tight_layout()

    dest = PATHS["figures"] / "hrsn_correlation_matrix.png"
    fig.savefig(dest, dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"HRSN correlation matrix saved to {dest}")

    # Also do outcome correlations
    outcome_cols = [m["id"].lower() for m in params["outcome_measures"]]
    outcome_labels = {m["id"].lower(): m["name"] for m in params["outcome_measures"]}
    corr_out = df[outcome_cols].corr()

    fig2, ax2 = plt.subplots(figsize=(8, 7))
    labels_out = [outcome_labels.get(c, c) for c in outcome_cols]
    mask_out = np.triu(np.ones_like(corr_out, dtype=bool), k=1)
    sns.heatmap(corr_out, mask=mask_out, annot=True, fmt=".2f", cmap="RdBu_r",
                vmin=-1, vmax=1, center=0, square=True,
                xticklabels=labels_out, yticklabels=labels_out,
                linewidths=0.5, ax=ax2)
    ax2.set_title("Correlation Matrix: 7 Health Outcomes\n(Pearson r, N={:,})".format(len(df)))
    plt.tight_layout()

    dest2 = PATHS["figures"] / "outcome_correlation_matrix.png"
    fig2.savefig(dest2, dpi=300, bbox_inches="tight")
    plt.close(fig2)
    logger.info(f"Outcome correlation matrix saved to {dest2}")

    return corr


def step_3_4_distributions(df, params):
    """Plot distributions of outcomes and HRSN measures."""
    logger.info("=== Step 3.4: Distribution Plots ===")

    outcome_cols = [m["id"].lower() for m in params["outcome_measures"]]
    outcome_labels = {m["id"].lower(): m["name"] for m in params["outcome_measures"]}
    hrsn_cols = [m["id"].lower() for m in params["hrsn_measures"]]
    hrsn_labels = {m["id"].lower(): m["name"] for m in params["hrsn_measures"]}

    # Outcome distributions
    n_out = len(outcome_cols)
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()
    for i, col in enumerate(outcome_cols):
        ax = axes[i]
        vals = df[col].dropna()
        ax.hist(vals, bins=50, color="#2980b9", edgecolor="white", alpha=0.8)
        ax.set_title(outcome_labels.get(col, col))
        ax.set_xlabel("Crude Prevalence (%)")
        ax.set_ylabel("Census Tracts")
        skew = vals.skew()
        ax.text(0.95, 0.95, f"skew={skew:.2f}", transform=ax.transAxes,
                ha="right", va="top", fontsize=8,
                bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))
    # Hide unused subplot
    for j in range(n_out, len(axes)):
        axes[j].set_visible(False)
    fig.suptitle(f"Distribution of Health Outcome Prevalence (N={len(df):,} tracts)", fontsize=14)
    plt.tight_layout()

    dest = PATHS["figures"] / "outcome_distributions.png"
    fig.savefig(dest, dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Outcome distributions saved to {dest}")

    # HRSN distributions
    n_hrsn = len(hrsn_cols)
    fig2, axes2 = plt.subplots(2, 4, figsize=(16, 8))
    axes2 = axes2.flatten()
    for i, col in enumerate(hrsn_cols):
        ax = axes2[i]
        vals = df[col].dropna()
        ax.hist(vals, bins=50, color="#c0392b", edgecolor="white", alpha=0.8)
        ax.set_title(hrsn_labels.get(col, col))
        ax.set_xlabel("Crude Prevalence (%)")
        ax.set_ylabel("Census Tracts")
        skew = vals.skew()
        ax.text(0.95, 0.95, f"skew={skew:.2f}", transform=ax.transAxes,
                ha="right", va="top", fontsize=8,
                bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))
    for j in range(n_hrsn, len(axes2)):
        axes2[j].set_visible(False)
    fig2.suptitle(f"Distribution of HRSN Measure Prevalence (N={len(df):,} tracts)", fontsize=14)
    plt.tight_layout()

    dest2 = PATHS["figures"] / "hrsn_distributions.png"
    fig2.savefig(dest2, dpi=300, bbox_inches="tight")
    plt.close(fig2)
    logger.info(f"HRSN distributions saved to {dest2}")


def step_3_5_scatter_matrix(df, params):
    """Bivariate scatter matrix: 7 HRSN × 7 outcomes with regression lines."""
    logger.info("=== Step 3.5: HRSN × Outcome Scatter Matrix ===")

    hrsn_cols = [m["id"].lower() for m in params["hrsn_measures"]]
    outcome_cols = [m["id"].lower() for m in params["outcome_measures"]]
    hrsn_labels = {m["id"].lower(): m["name"] for m in params["hrsn_measures"]}
    outcome_labels = {m["id"].lower(): m["name"] for m in params["outcome_measures"]}

    n_hrsn = len(hrsn_cols)
    n_out = len(outcome_cols)

    # Use a random sample to keep scatter plots readable
    sample_n = min(5000, len(df))
    df_sample = df.sample(n=sample_n, random_state=42)

    fig, axes = plt.subplots(n_hrsn, n_out, figsize=(n_out * 2.5, n_hrsn * 2.5))

    for i, hrsn in enumerate(hrsn_cols):
        for j, outcome in enumerate(outcome_cols):
            ax = axes[i, j]
            ax.scatter(df_sample[hrsn], df_sample[outcome],
                       alpha=0.1, s=2, color="#34495e", rasterized=True)

            # Add linear fit line
            mask = df_sample[[hrsn, outcome]].notna().all(axis=1)
            if mask.sum() > 10:
                x = df_sample.loc[mask, hrsn]
                y = df_sample.loc[mask, outcome]
                z = np.polyfit(x, y, 1)
                p = np.poly1d(z)
                x_range = np.linspace(x.min(), x.max(), 100)
                ax.plot(x_range, p(x_range), color="#e74c3c", linewidth=1.5)

                # Pearson r
                r = x.corr(y)
                ax.text(0.05, 0.95, f"r={r:.2f}", transform=ax.transAxes,
                        ha="left", va="top", fontsize=7, color="#e74c3c",
                        fontweight="bold")

            # Labels
            if i == 0:
                ax.set_title(outcome_labels.get(outcome, outcome), fontsize=8)
            if j == 0:
                ax.set_ylabel(hrsn_labels.get(hrsn, hrsn), fontsize=8)
            if i == n_hrsn - 1:
                ax.set_xlabel("")
            ax.tick_params(labelsize=6)

    fig.suptitle(f"HRSN × Health Outcome Bivariate Associations\n"
                 f"(sample of {sample_n:,} tracts, Pearson r shown)",
                 fontsize=14, y=1.01)
    plt.tight_layout()

    dest = PATHS["figures"] / "hrsn_outcome_scatter_matrix.png"
    fig.savefig(dest, dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Scatter matrix saved to {dest}")


if __name__ == "__main__":
    print("=" * 70)
    print("PHASE 3: Exploratory Data Analysis")
    print("=" * 70)

    ensure_dirs()
    params = load_params()
    df = load_parquet(PATHS["processed"] / "merged_tracts.parquet")

    step_3_1_descriptive_stats(df, params)
    step_3_2_hrsn_correlation(df, params)
    step_3_4_distributions(df, params)
    step_3_5_scatter_matrix(df, params)

    print("\nPhase 3 Steps 3.1, 3.2, 3.4, 3.5 complete.")
    print("Step 3.3 (geographic maps) requires separate script: 07b_maps.py")
