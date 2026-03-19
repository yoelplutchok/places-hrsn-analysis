"""
Phase A, Step 10: Factor Analysis of HRSN Measures.

Addresses extreme multicollinearity among 7 HRSN measures (VIF up to 97).
Tests whether measures load onto latent constructs.

Steps:
  1. Bartlett's test + KMO adequacy check
  2. PCA → scree plot with parallel analysis
  3. EFA: test 1-5 factor solutions (ML extraction, promax rotation)
  4. Create composite scores: "Material Hardship" + "Social Isolation"
  5. Discriminant validity: HTMT ratios

Output:
  - data/final/pca_results.csv
  - data/final/efa_loadings.csv
  - data/final/efa_model_comparison.csv
  - data/final/factor_scores.parquet
  - data/final/discriminant_validity.csv
  - outputs/figures/scree_plot.png
  - outputs/figures/factor_loading_heatmap.png
"""
import sys
from pathlib import Path

import pandas as pd
import numpy as np
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA, FactorAnalysis as SklearnFA
from scipy.stats import chi2

# factor_analyzer may be incompatible with newer scikit-learn; use fallback
try:
    from factor_analyzer import FactorAnalyzer
    from factor_analyzer.factor_analyzer import calculate_bartlett_sphericity, calculate_kmo
    FA_AVAILABLE = True
except ImportError:
    FA_AVAILABLE = False

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from hrsn_analysis.paths import PATHS, ensure_dirs
from hrsn_analysis.io_utils import load_parquet, load_params, save_csv, save_parquet
from hrsn_analysis.logging_utils import setup_logging, get_logger

setup_logging()
logger = get_logger(__name__)


def _varimax_rotation(loadings, max_iter=500, tol=1e-6):
    """Apply varimax (orthogonal) rotation to factor loadings matrix."""
    n_vars, n_factors = loadings.shape
    rotation_matrix = np.eye(n_factors)
    d = 0

    for _ in range(max_iter):
        old_d = d
        comp = loadings @ rotation_matrix
        u, s, vt = np.linalg.svd(
            loadings.T @ (comp ** 3 - (1.0 / n_vars) * comp @ np.diag(np.sum(comp ** 2, axis=0)))
        )
        rotation_matrix = u @ vt
        d = np.sum(s)
        if d - old_d < tol:
            break

    return loadings @ rotation_matrix


def _promax_rotation(loadings, power=3, max_iter=500, tol=1e-6):
    """
    Apply promax (oblique) rotation to factor loadings matrix.

    Promax is a two-step process:
    1. Apply varimax rotation to get an initial orthogonal solution
    2. Raise loadings to a power to create a target matrix, then rotate
       obliquely toward that target

    Parameters
    ----------
    loadings : np.ndarray
        Unrotated factor loadings (n_vars x n_factors).
    power : int
        Promax power parameter (default 3, matching SPSS/SAS/R defaults).
        Higher values produce more oblique solutions.

    Returns
    -------
    pattern_loadings : np.ndarray
        Pattern matrix (direct loadings after oblique rotation).
    phi : np.ndarray
        Factor correlation matrix.
    structure_loadings : np.ndarray
        Structure matrix (pattern @ phi), reflecting total variable-factor
        correlations including shared variance through correlated factors.
    """
    # Step 1: Varimax rotation
    varimax_loadings = _varimax_rotation(loadings, max_iter=max_iter, tol=tol)

    # Step 2: Create target matrix by raising absolute loadings to power
    # while preserving signs
    target = np.sign(varimax_loadings) * np.abs(varimax_loadings) ** power

    # Step 3: Find the least-squares transformation from varimax to target
    # Solve: varimax_loadings @ T ≈ target
    T, _, _, _ = np.linalg.lstsq(varimax_loadings, target, rcond=None)

    # Step 4: Apply transformation to get pattern loadings
    pattern_loadings = varimax_loadings @ T

    # Normalize pattern loadings so column norms match varimax
    for j in range(pattern_loadings.shape[1]):
        norm_ratio = np.linalg.norm(varimax_loadings[:, j]) / np.linalg.norm(pattern_loadings[:, j])
        pattern_loadings[:, j] *= norm_ratio

    # Compute factor correlation matrix (Phi) using standard formula:
    # Phi = inv(T' @ T), then rescale to correlation matrix
    TT = T.T @ T
    phi = np.linalg.inv(TT)
    # Rescale to correlation: Phi_ij / sqrt(Phi_ii * Phi_jj)
    d = np.sqrt(np.diag(phi))
    phi = phi / np.outer(d, d)

    # Compute structure matrix: S = P @ Phi
    structure_loadings = pattern_loadings @ phi

    return pattern_loadings, phi, structure_loadings


def parallel_analysis(data, n_iter=1000):
    """
    Horn's parallel analysis: compare observed eigenvalues against
    eigenvalues from random data of the same dimensions.

    Returns array of 95th percentile random eigenvalues.
    """
    n_obs, n_vars = data.shape
    random_eigenvalues = np.zeros((n_iter, n_vars))
    rng = np.random.RandomState(42)  # Fixed seed for reproducibility

    for i in range(n_iter):
        random_data = rng.normal(size=(n_obs, n_vars))
        corr_matrix = np.corrcoef(random_data, rowvar=False)
        eigenvalues = np.linalg.eigvalsh(corr_matrix)
        random_eigenvalues[i] = np.sort(eigenvalues)[::-1]

    return np.percentile(random_eigenvalues, 95, axis=0)


def run_factor_analysis():
    """Run full factor analysis pipeline on HRSN measures."""
    ensure_dirs()
    params = load_params()
    fa_params = params.get("factor_analysis", {})

    # ---- Load data ----
    df = load_parquet(PATHS["processed"] / "merged_tracts.parquet")
    hrsn_cols = [m["id"].lower() for m in params["hrsn_measures"]]
    hrsn_labels = {m["id"].lower(): m["name"] for m in params["hrsn_measures"]}

    X = df[hrsn_cols].dropna()
    logger.info(f"Factor analysis on {len(X):,} tracts, {len(hrsn_cols)} HRSN measures")

    # ================================================================
    # Step 1: Adequacy checks
    # ================================================================
    logger.info("\n=== Step 1: Factorability Checks ===")

    if FA_AVAILABLE:
        chi_sq, p_value = calculate_bartlett_sphericity(X)
        kmo_per_var, kmo_overall = calculate_kmo(X)
    else:
        # Manual Bartlett's test
        n = len(X)
        p = len(hrsn_cols)
        corr = X.corr().values
        det = np.linalg.det(corr)
        chi_sq = -((n - 1) - (2 * p + 5) / 6) * np.log(det)
        dof = p * (p - 1) / 2
        p_value = chi2.sf(chi_sq, dof)
        # Manual KMO
        corr_inv = np.linalg.inv(corr)
        partial = -corr_inv / np.sqrt(np.outer(np.diag(corr_inv), np.diag(corr_inv)))
        np.fill_diagonal(partial, 0)
        corr_sq = corr ** 2
        np.fill_diagonal(corr_sq, 0)
        partial_sq = partial ** 2
        kmo_per_var = corr_sq.sum(axis=0) / (corr_sq.sum(axis=0) + partial_sq.sum(axis=0))
        kmo_overall = corr_sq.sum() / (corr_sq.sum() + partial_sq.sum())

    logger.info(f"Bartlett's test: chi²={chi_sq:,.1f}, p={p_value:.2e}")
    if p_value < 0.05:
        logger.info("  → Significant: correlation matrix differs from identity (good)")
    else:
        logger.warning("  → NOT significant: data may not be suitable for factor analysis")

    logger.info(f"KMO overall: {kmo_overall:.3f}")
    for col, kmo_val in zip(hrsn_cols, kmo_per_var):
        logger.info(f"  {hrsn_labels[col]:30s}: KMO = {kmo_val:.3f}")
    if kmo_overall >= 0.8:
        logger.info("  → Meritorious (KMO >= 0.8)")
    elif kmo_overall >= 0.6:
        logger.info("  → Mediocre but acceptable (KMO >= 0.6)")
    else:
        logger.warning("  → Poor (KMO < 0.6): reconsider factor analysis")

    # ================================================================
    # Step 2: PCA + scree plot with parallel analysis
    # ================================================================
    logger.info("\n=== Step 2: PCA + Scree Plot ===")

    pca = PCA(n_components=len(hrsn_cols))
    pca.fit(X)

    eigenvalues = pca.explained_variance_
    # Use correlation matrix eigenvalues for comparison with parallel analysis
    corr_matrix = X.corr().values
    corr_eigenvalues = np.sort(np.linalg.eigvalsh(corr_matrix))[::-1]

    variance_explained = pca.explained_variance_ratio_
    cumulative_var = np.cumsum(variance_explained)

    pca_df = pd.DataFrame({
        "component": range(1, len(hrsn_cols) + 1),
        "eigenvalue": corr_eigenvalues,
        "variance_explained": variance_explained,
        "cumulative_variance": cumulative_var,
    })
    save_csv(pca_df, PATHS["final"] / "pca_results.csv")

    for i, row in pca_df.iterrows():
        logger.info(f"  PC{row['component']:.0f}: eigenvalue={row['eigenvalue']:.3f}, "
                     f"var={row['variance_explained']:.1%}, cumul={row['cumulative_variance']:.1%}")

    # Parallel analysis
    logger.info("Running parallel analysis (1000 iterations)...")
    n_iter = fa_params.get("parallel_analysis_iterations", 1000)
    random_eigs = parallel_analysis(X.values, n_iter=n_iter)

    # Determine number of factors to retain (eigenvalue > parallel threshold)
    n_factors_parallel = sum(corr_eigenvalues > random_eigs)
    n_factors_kaiser = sum(corr_eigenvalues > 1.0)
    logger.info(f"Factors to retain: Kaiser criterion = {n_factors_kaiser}, "
                f"parallel analysis = {n_factors_parallel}")

    # Scree plot
    fig, ax = plt.subplots(figsize=(8, 5))
    components = range(1, len(hrsn_cols) + 1)
    ax.plot(components, corr_eigenvalues, "bo-", label="Observed eigenvalues", linewidth=2)
    ax.plot(components, random_eigs, "r--", label="Parallel analysis (95th %ile)", linewidth=1.5)
    ax.axhline(y=1.0, color="gray", linestyle=":", alpha=0.6, label="Kaiser criterion (λ=1)")
    ax.set_xlabel("Component")
    ax.set_ylabel("Eigenvalue")
    ax.set_title("Scree Plot with Parallel Analysis\n(7 HRSN Measures)")
    ax.set_xticks(list(components))
    ax.legend()
    ax.grid(True, alpha=0.3)

    dest = PATHS["figures"] / "scree_plot.png"
    fig.savefig(dest, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    logger.info(f"Scree plot saved to {dest}")

    # ================================================================
    # Step 3: EFA — test 1-5 factor solutions
    # ================================================================
    logger.info("\n=== Step 3: Exploratory Factor Analysis ===")

    max_factors = fa_params.get("max_factors", 5)
    rotation = fa_params.get("rotation", "promax")
    extraction = fa_params.get("extraction_method", "ml")

    model_comparison = []
    best_loadings = None
    # Data-driven: parallel analysis suggests n_factors_parallel factor(s).
    # Also test 2-factor solution for theoretical comparison (Material Hardship vs Social Isolation).
    best_n_factors = max(n_factors_parallel, 2)

    # Standardize for EFA (critical: sklearn FA on raw data gives loadings >> 1)
    X_efa = (X - X.mean()) / X.std()

    for n_f in range(1, max_factors + 1):
        logger.info(f"\n  Testing {n_f}-factor solution...")

        try:
            # Try factor_analyzer first, fall back to sklearn
            fa_success = False
            if FA_AVAILABLE:
                try:
                    fa = FactorAnalyzer(
                        n_factors=n_f,
                        method=extraction,
                        rotation=rotation if n_f > 1 else None,
                    )
                    fa.fit(X)  # factor_analyzer standardizes internally
                    loadings = fa.loadings_
                    communalities = fa.get_communalities()
                    factor_variance = fa.get_factor_variance()
                    total_var_explained = factor_variance[2][-1] if len(factor_variance) > 2 else np.nan
                    fa_success = True
                except Exception:
                    pass

            if not fa_success:
                # Sklearn FactorAnalysis fallback (ML extraction)
                # Must use standardized data so loadings are on correlation scale
                fa_sk = SklearnFA(n_components=n_f, max_iter=2000)
                fa_sk.fit(X_efa)
                loadings = fa_sk.components_.T  # shape: (n_vars, n_factors)
                communalities = np.sum(loadings ** 2, axis=1)
                total_var_explained = communalities.sum() / len(hrsn_cols)

                # Apply promax (oblique) rotation for n_f > 1
                # (promax is appropriate when factors are expected to correlate)
                if n_f > 1:
                    loadings, phi, structure = _promax_rotation(loadings)
                    # For oblique rotation, communalities = diag(P @ Phi @ P')
                    communalities = np.diag(loadings @ phi @ loadings.T)
                    logger.info(f"    Factor correlation (Phi): {phi[0, 1]:.3f}")
                else:
                    structure = None

            row = {
                "n_factors": n_f,
                "total_variance_explained": round(total_var_explained, 4),
                "min_communality": round(communalities.min(), 4),
                "max_communality": round(communalities.max(), 4),
            }
            model_comparison.append(row)

            logger.info(f"    Variance explained: {total_var_explained:.1%}")
            logger.info(f"    Communalities: min={communalities.min():.3f}, max={communalities.max():.3f}")

            for j, col in enumerate(hrsn_cols):
                loading_str = "  ".join([f"F{k+1}={loadings[j, k]:+.3f}" for k in range(n_f)])
                logger.info(f"    {hrsn_labels[col]:30s}: {loading_str} (pattern)")

            if n_f == best_n_factors:
                best_loadings = pd.DataFrame(
                    loadings,
                    index=[hrsn_labels[c] for c in hrsn_cols],
                    columns=[f"Factor_{k+1}" for k in range(n_f)],
                )
                # Save structure matrix for oblique rotations
                if structure is not None:
                    structure_df = pd.DataFrame(
                        structure,
                        index=[hrsn_labels[c] for c in hrsn_cols],
                        columns=[f"Factor_{k+1}" for k in range(n_f)],
                    )
                    save_csv(structure_df, PATHS["final"] / "efa_structure_matrix.csv")
                    logger.info("  Structure matrix saved")

        except Exception as e:
            logger.warning(f"    {n_f}-factor solution failed: {e}")
            model_comparison.append({"n_factors": n_f, "total_variance_explained": np.nan})

    # Save model comparison
    comparison_df = pd.DataFrame(model_comparison)
    save_csv(comparison_df, PATHS["final"] / "efa_model_comparison.csv")
    logger.info(f"\nModel comparison saved ({len(comparison_df)} models)")

    # Honest reporting of factor retention decision
    if n_factors_parallel == 1 and best_n_factors == 2:
        logger.info("\n--- Factor Retention Note ---")
        logger.info(f"Data-driven criteria (Kaiser={n_factors_kaiser}, parallel={n_factors_parallel}) "
                     "both suggest 1 factor.")
        logger.info("PC1 accounts for {:.1f}% of total variance.".format(
            corr_eigenvalues[0] / sum(corr_eigenvalues) * 100))
        logger.info("The 2-factor solution is tested for theoretical comparison only")
        logger.info("(Material Hardship vs Social Isolation), not because the data demand it.")

    # Save best loadings
    if best_loadings is not None:
        best_loadings.to_csv(PATHS["final"] / "efa_loadings.csv")
        logger.info(f"Best loadings ({best_n_factors}-factor) saved")

    # ================================================================
    # Step 3b: Factor loading heatmap
    # ================================================================
    if best_loadings is not None:
        fig, ax = plt.subplots(figsize=(6, 7))
        sns.heatmap(
            best_loadings,
            annot=True,
            fmt=".2f",
            cmap="RdBu_r",
            center=0,
            vmin=-1,
            vmax=1,
            linewidths=0.5,
            ax=ax,
        )
        ax.set_title(f"EFA Factor Loadings ({best_n_factors}-Factor Solution, Promax Rotation)")
        ax.set_ylabel("")

        dest = PATHS["figures"] / "factor_loading_heatmap.png"
        fig.savefig(dest, dpi=300, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        logger.info(f"Factor loading heatmap saved to {dest}")

    # ================================================================
    # Step 4: Create composite factor scores
    # ================================================================
    logger.info("\n=== Step 4: Composite Factor Scores ===")

    # Based on expected factor structure from plan:
    # Factor 1 (Material Hardship): foodstamp, foodinsecu, housinsecu, shututility, lacktrpt
    # Factor 2 (Social Isolation): emotionspt, loneliness
    factor_defs = fa_params.get("factor_labels", [
        {"name": "Material Hardship", "expected_measures": ["foodstamp", "foodinsecu", "housinsecu", "shututility", "lacktrpt"]},
        {"name": "Social Isolation", "expected_measures": ["emotionspt", "loneliness"]},
    ])

    # Standardize HRSN measures first
    X_std = (X - X.mean()) / X.std()

    factor_scores = pd.DataFrame(index=X.index)
    factor_scores["tract_fips"] = df.loc[X.index, "tract_fips"]

    for fdef in factor_defs:
        fname = fdef["name"]
        fmeasures = fdef["expected_measures"]
        score_col = fname.lower().replace(" ", "_")

        # Simple mean of z-scored items (robust, reproducible)
        available = [m for m in fmeasures if m in X_std.columns]
        factor_scores[score_col] = X_std[available].mean(axis=1)

        logger.info(f"  {fname}: {len(available)} measures → {score_col}")
        logger.info(f"    Measures: {available}")
        logger.info(f"    Score range: [{factor_scores[score_col].min():.3f}, {factor_scores[score_col].max():.3f}]")
        logger.info(f"    Mean: {factor_scores[score_col].mean():.3f}, SD: {factor_scores[score_col].std():.3f}")

    # Inter-factor correlation
    score_cols = [fdef["name"].lower().replace(" ", "_") for fdef in factor_defs]
    r = factor_scores[score_cols].corr()
    logger.info(f"\nInter-factor correlation: r = {r.iloc[0, 1]:.3f}")

    save_parquet(factor_scores, PATHS["final"] / "factor_scores.parquet")

    # ================================================================
    # Step 5: Discriminant validity — HTMT ratios
    # ================================================================
    logger.info("\n=== Step 5: Discriminant Validity ===")

    # HTMT: Heterotrait-Monotrait ratio of correlations
    # HTMT < 0.85 indicates good discriminant validity
    corr = X.corr()

    # Define factor membership
    f1_measures = factor_defs[0]["expected_measures"]
    f2_measures = factor_defs[1]["expected_measures"]

    # Within-factor correlations (monotrait-heteromethod)
    f1_within = []
    for i, m1 in enumerate(f1_measures):
        for m2 in f1_measures[i+1:]:
            if m1 in corr.index and m2 in corr.index:
                f1_within.append(abs(corr.loc[m1, m2]))

    f2_within = []
    for i, m1 in enumerate(f2_measures):
        for m2 in f2_measures[i+1:]:
            if m1 in corr.index and m2 in corr.index:
                f2_within.append(abs(corr.loc[m1, m2]))

    # Between-factor correlations (heterotrait-heteromethod)
    between = []
    for m1 in f1_measures:
        for m2 in f2_measures:
            if m1 in corr.index and m2 in corr.index:
                between.append(abs(corr.loc[m1, m2]))

    mean_within_f1 = np.mean(f1_within) if f1_within else np.nan
    mean_within_f2 = np.mean(f2_within) if f2_within else np.nan
    mean_between = np.mean(between) if between else np.nan

    # HTMT = mean(between) / geometric_mean(mean(within_f1), mean(within_f2))
    denom = np.sqrt(mean_within_f1 * mean_within_f2) if (mean_within_f1 > 0 and mean_within_f2 > 0) else np.nan
    htmt = mean_between / denom if denom > 0 else np.nan

    logger.info(f"  Within-factor mean |r| (Material Hardship): {mean_within_f1:.3f}")
    logger.info(f"  Within-factor mean |r| (Social Isolation): {mean_within_f2:.3f}")
    logger.info(f"  Between-factor mean |r|: {mean_between:.3f}")
    logger.info(f"  HTMT ratio: {htmt:.3f}")
    if htmt < 0.85:
        logger.info("  → Good discriminant validity (HTMT < 0.85)")
    elif htmt < 0.90:
        logger.info("  → Acceptable discriminant validity (HTMT < 0.90)")
    else:
        logger.warning("  → Poor discriminant validity (HTMT >= 0.90)")

    validity_df = pd.DataFrame([{
        "metric": "HTMT",
        "factor_1": "Material Hardship",
        "factor_2": "Social Isolation",
        "within_f1_mean_r": round(mean_within_f1, 4),
        "within_f2_mean_r": round(mean_within_f2, 4),
        "between_mean_r": round(mean_between, 4),
        "htmt_ratio": round(htmt, 4),
        "discriminant_validity": "good" if htmt < 0.85 else "acceptable" if htmt < 0.90 else "poor",
    }])
    save_csv(validity_df, PATHS["final"] / "discriminant_validity.csv")

    return factor_scores


if __name__ == "__main__":
    print("=" * 70)
    print("PHASE A: Factor Analysis of HRSN Measures")
    print("=" * 70)
    scores = run_factor_analysis()
    print(f"\nDone. Factor scores computed for {len(scores):,} tracts.")
