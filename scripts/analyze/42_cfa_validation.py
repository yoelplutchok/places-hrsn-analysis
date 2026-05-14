"""
Phase A, Step 42: Confirmatory Factor Analysis and Internal Consistency.

Validates the 2-factor HRSN structure using a split-half CFA approach:
  1. Random 50/50 split of tracts
  2. EFA on first half (confirms original 2-factor structure)
  3. CFA on second half using semopy: fit 2-factor model
  4. Report fit indices: CFI, RMSEA, SRMR
  5. Internal consistency: Cronbach's alpha and McDonald's omega

Output:
  - data/final/cfa_fit_indices.csv
  - data/final/cfa_loadings.csv
  - data/final/internal_consistency.csv
  - data/final/efa_split_half_loadings.csv
"""
import sys
from pathlib import Path

import pandas as pd
import numpy as np
from scipy import stats
from sklearn.decomposition import FactorAnalysis as SklearnFA

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from hrsn_analysis.paths import PATHS, ensure_dirs
from hrsn_analysis.io_utils import load_parquet, load_params, save_csv
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
    """Apply promax (oblique) rotation to factor loadings matrix."""
    varimax_loadings = _varimax_rotation(loadings, max_iter=max_iter, tol=tol)

    target = np.sign(varimax_loadings) * np.abs(varimax_loadings) ** power
    T, _, _, _ = np.linalg.lstsq(varimax_loadings, target, rcond=None)
    pattern_loadings = varimax_loadings @ T

    for j in range(pattern_loadings.shape[1]):
        norm_ratio = np.linalg.norm(varimax_loadings[:, j]) / np.linalg.norm(pattern_loadings[:, j])
        pattern_loadings[:, j] *= norm_ratio

    TT = T.T @ T
    phi = np.linalg.inv(TT)
    d = np.sqrt(np.diag(phi))
    phi = phi / np.outer(d, d)

    structure_loadings = pattern_loadings @ phi
    return pattern_loadings, phi, structure_loadings


def cronbach_alpha(X):
    """Compute Cronbach's alpha for a set of items (columns of X)."""
    k = X.shape[1]
    if k < 2:
        return np.nan
    item_vars = X.var(axis=0, ddof=1)
    total_var = X.sum(axis=1).var(ddof=1)
    alpha = (k / (k - 1)) * (1 - item_vars.sum() / total_var)
    return alpha


def mcdonalds_omega(X):
    """
    Compute McDonald's omega (total) for a set of items.

    omega = (sum of loadings)^2 / ((sum of loadings)^2 + sum of uniquenesses)
    Uses a single-factor ML model.
    """
    k = X.shape[1]
    if k < 2:
        return np.nan

    X_std = (X - X.mean()) / X.std()

    try:
        fa = SklearnFA(n_components=1, max_iter=2000)
        fa.fit(X_std)
        loadings = fa.components_.T.flatten()
        uniquenesses = fa.noise_variance_
        sum_loadings = loadings.sum()
        omega = sum_loadings ** 2 / (sum_loadings ** 2 + uniquenesses.sum())
        return omega
    except Exception as e:
        logger.warning(f"McDonald's omega computation failed: {e}")
        return np.nan


def run_cfa_validation():
    """Run CFA validation and internal consistency analysis."""
    ensure_dirs()
    params = load_params()
    fa_params = params.get("factor_analysis", {})

    # ---- Load data ----
    df = load_parquet(PATHS["processed"] / "merged_tracts.parquet")
    hrsn_cols = [m["id"].lower() for m in params["hrsn_measures"]]
    hrsn_labels = {m["id"].lower(): m["name"] for m in params["hrsn_measures"]}

    X = df[hrsn_cols].dropna()
    logger.info(f"CFA validation on {len(X):,} tracts, {len(hrsn_cols)} HRSN measures")

    # Factor definitions
    factor_defs = fa_params.get("factor_labels", [
        {"name": "Material Hardship",
         "expected_measures": ["foodstamp", "foodinsecu", "housinsecu", "shututility", "lacktrpt"]},
        {"name": "Social Isolation",
         "expected_measures": ["emotionspt", "loneliness"]},
    ])

    # ================================================================
    # Step 1: Random 50/50 split
    # ================================================================
    logger.info("\n=== Step 1: Split-Half Design ===")
    rng = np.random.RandomState(42)
    indices = rng.permutation(len(X))
    half = len(X) // 2

    X_efa = X.iloc[indices[:half]].copy()
    X_cfa = X.iloc[indices[half:]].copy()
    logger.info(f"  EFA half: {len(X_efa):,} tracts")
    logger.info(f"  CFA half: {len(X_cfa):,} tracts")

    # ================================================================
    # Step 2: EFA on first half (confirm structure)
    # ================================================================
    logger.info("\n=== Step 2: EFA on Development Half ===")

    X_efa_std = (X_efa - X_efa.mean()) / X_efa.std()

    fa_sk = SklearnFA(n_components=2, max_iter=2000)
    fa_sk.fit(X_efa_std)
    loadings_raw = fa_sk.components_.T
    loadings_efa, phi_efa, structure_efa = _promax_rotation(loadings_raw)

    # Compute communalities for oblique rotation
    communalities_efa = np.diag(loadings_efa @ phi_efa @ loadings_efa.T)

    efa_df = pd.DataFrame(
        loadings_efa,
        index=[hrsn_labels[c] for c in hrsn_cols],
        columns=["Factor_1_Material_Hardship", "Factor_2_Social_Isolation"],
    )
    efa_df["communality"] = communalities_efa
    save_csv(efa_df.reset_index().rename(columns={"index": "measure"}),
             PATHS["final"] / "efa_split_half_loadings.csv")

    logger.info("  EFA loadings (split-half development sample):")
    for j, col in enumerate(hrsn_cols):
        logger.info(f"    {hrsn_labels[col]:30s}: F1={loadings_efa[j, 0]:+.3f}  "
                     f"F2={loadings_efa[j, 1]:+.3f}")
    logger.info(f"  Interfactor correlation: {phi_efa[0, 1]:.3f}")

    # ================================================================
    # Step 3: CFA on second half using semopy
    # ================================================================
    logger.info("\n=== Step 3: CFA on Validation Half ===")

    try:
        import semopy

        # Build CFA model specification
        f1_items = factor_defs[0]["expected_measures"]
        f2_items = factor_defs[1]["expected_measures"]

        # semopy model syntax
        model_desc = (
            f"MaterialHardship =~ {' + '.join(f1_items)}\n"
            f"SocialIsolation =~ {' + '.join(f2_items)}\n"
            f"MaterialHardship ~~ SocialIsolation"
        )

        logger.info(f"  CFA model specification:\n{model_desc}")

        # Prepare CFA data (use raw values, semopy handles standardization)
        cfa_data = X_cfa[hrsn_cols].copy()
        cfa_data.columns = hrsn_cols  # ensure lowercase names

        model = semopy.Model(model_desc)
        model.fit(cfa_data)

        # Extract fit indices — semopy returns columns as stat names, single row "Value"
        fit_stats = semopy.calc_stats(model)
        logger.info(f"\n  CFA Fit Statistics:")

        fit_dict = {}
        for stat_name in fit_stats.columns:
            val = fit_stats.loc["Value", stat_name]
            fit_dict[stat_name] = val
            logger.info(f"    {stat_name}: {val:.4f}")

        # Key fit indices
        cfi = fit_dict.get("CFI", np.nan)
        rmsea = fit_dict.get("RMSEA", np.nan)
        srmr = fit_dict.get("SRMR", np.nan)
        chi2 = fit_dict.get("chi2", np.nan)
        chi2_df = fit_dict.get("DoF", np.nan)
        chi2_p = fit_dict.get("chi2 p-value", np.nan)

        logger.info(f"\n  Key fit indices:")
        logger.info(f"    CFI  = {cfi:.4f} (good > 0.95, acceptable > 0.90)")
        logger.info(f"    RMSEA = {rmsea:.4f} (good < 0.06, acceptable < 0.08)")
        if not np.isnan(srmr):
            logger.info(f"    SRMR = {srmr:.4f} (good < 0.08)")
        else:
            logger.info(f"    SRMR = not available from semopy")
        logger.info(f"    Chi2 = {chi2:.1f}, df = {chi2_df:.0f}, p = {chi2_p:.4e}")

        # Evaluate fit
        fit_eval = []
        if cfi > 0.95:
            fit_eval.append("CFI: GOOD")
        elif cfi > 0.90:
            fit_eval.append("CFI: ACCEPTABLE")
        else:
            fit_eval.append("CFI: POOR")

        if rmsea < 0.06:
            fit_eval.append("RMSEA: GOOD")
        elif rmsea < 0.08:
            fit_eval.append("RMSEA: ACCEPTABLE")
        else:
            fit_eval.append("RMSEA: POOR")

        if not np.isnan(srmr):
            if srmr < 0.08:
                fit_eval.append("SRMR: GOOD")
            else:
                fit_eval.append("SRMR: POOR")

        logger.info(f"    Evaluation: {', '.join(fit_eval)}")

        # Save fit indices
        fit_results = pd.DataFrame([{
            "model": "2-factor CFA (validation half)",
            "n_tracts": len(X_cfa),
            "chi2": round(chi2, 2),
            "df": int(chi2_df) if not np.isnan(chi2_df) else np.nan,
            "chi2_pvalue": chi2_p,
            "CFI": round(cfi, 4),
            "RMSEA": round(rmsea, 4),
            "SRMR": round(srmr, 4),
            "evaluation": "; ".join(fit_eval),
        }])
        save_csv(fit_results, PATHS["final"] / "cfa_fit_indices.csv")

        # Extract CFA factor loadings (semopy uses item ~ factor notation)
        estimates = model.inspect()
        logger.info(f"\n  CFA Parameter Estimates:")
        cfa_loadings = []
        for _, row in estimates.iterrows():
            if row["op"] == "~":
                est = row["Estimate"]
                # semopy uses '-' for fixed parameters
                se_raw = row.get("Std. Err", "-")
                z_raw = row.get("z-value", "-")
                p_raw = row.get("p-value", "-")
                se = float(se_raw) if se_raw != "-" else np.nan
                z = float(z_raw) if z_raw != "-" else np.nan
                p = float(p_raw) if p_raw != "-" else np.nan

                # In semopy, lval is the observed variable, rval is the factor
                item_name = row["lval"]
                factor_name = row["rval"]
                logger.info(f"    {factor_name:20s} -> {item_name:15s}: "
                             f"est={est:.4f}"
                             f"{f', SE={se:.4f}' if not np.isnan(se) else ', (fixed)'}")
                cfa_loadings.append({
                    "factor": factor_name,
                    "item": item_name,
                    "item_label": hrsn_labels.get(item_name, item_name),
                    "loading": round(est, 4),
                    "std_err": round(se, 4) if not np.isnan(se) else np.nan,
                    "z_value": round(z, 4) if not np.isnan(z) else np.nan,
                    "p_value": p,
                })

        if cfa_loadings:
            cfa_loadings_df = pd.DataFrame(cfa_loadings)
            save_csv(cfa_loadings_df, PATHS["final"] / "cfa_loadings.csv")

        cfa_success = True

    except ImportError:
        logger.warning("semopy not installed — attempting alternative CFA approach")
        cfa_success = False

    except Exception as e:
        logger.error(f"CFA failed: {e}")
        cfa_success = False

    if not cfa_success:
        # Fallback: use sklearn FA on validation half and report structure confirmation
        logger.info("\n  Fallback: sklearn FA on validation half (no formal CFA fit indices)")

        X_cfa_std = (X_cfa - X_cfa.mean()) / X_cfa.std()
        fa_cfa = SklearnFA(n_components=2, max_iter=2000)
        fa_cfa.fit(X_cfa_std)
        loadings_cfa_raw = fa_cfa.components_.T
        loadings_cfa, phi_cfa, _ = _promax_rotation(loadings_cfa_raw)

        logger.info("  Validation-half EFA loadings:")
        for j, col in enumerate(hrsn_cols):
            logger.info(f"    {hrsn_labels[col]:30s}: F1={loadings_cfa[j, 0]:+.3f}  "
                         f"F2={loadings_cfa[j, 1]:+.3f}")

        # Compare loading patterns
        loading_congruence = np.corrcoef(loadings_efa.flatten(), loadings_cfa.flatten())[0, 1]
        logger.info(f"  Tucker's congruence (loading pattern correlation): {loading_congruence:.4f}")

        fit_results = pd.DataFrame([{
            "model": "2-factor EFA validation (fallback, no CFA fit indices)",
            "n_tracts": len(X_cfa),
            "tucker_congruence": round(loading_congruence, 4),
            "interfactor_r_dev": round(phi_efa[0, 1], 4),
            "interfactor_r_val": round(phi_cfa[0, 1], 4),
            "note": "semopy not available; formal CFA requires: pip install semopy",
        }])
        save_csv(fit_results, PATHS["final"] / "cfa_fit_indices.csv")

    # ================================================================
    # Step 4: Internal Consistency
    # ================================================================
    logger.info("\n=== Step 4: Internal Consistency ===")

    consistency_results = []

    for fdef in factor_defs:
        fname = fdef["name"]
        fmeasures = [m for m in fdef["expected_measures"] if m in X.columns]

        if len(fmeasures) < 2:
            logger.warning(f"  {fname}: fewer than 2 items, skipping")
            continue

        factor_data = X[fmeasures].dropna()

        alpha = cronbach_alpha(factor_data.values)
        omega = mcdonalds_omega(factor_data.values)

        # Item-total correlations
        total = factor_data.sum(axis=1)
        item_total_corrs = {}
        for col in fmeasures:
            corrected_total = total - factor_data[col]
            r = factor_data[col].corr(corrected_total)
            item_total_corrs[col] = r

        logger.info(f"\n  {fname} ({len(fmeasures)} items):")
        logger.info(f"    Cronbach's alpha: {alpha:.4f}")
        logger.info(f"    McDonald's omega: {omega:.4f}")
        logger.info(f"    Item-total correlations:")
        for col, r in item_total_corrs.items():
            logger.info(f"      {hrsn_labels.get(col, col):30s}: r = {r:.4f}")

        # Evaluate
        if alpha >= 0.90:
            alpha_eval = "Excellent"
        elif alpha >= 0.80:
            alpha_eval = "Good"
        elif alpha >= 0.70:
            alpha_eval = "Acceptable"
        else:
            alpha_eval = "Questionable"
        logger.info(f"    Alpha evaluation: {alpha_eval}")

        consistency_results.append({
            "factor": fname,
            "n_items": len(fmeasures),
            "items": ", ".join(fmeasures),
            "cronbach_alpha": round(alpha, 4),
            "alpha_evaluation": alpha_eval,
            "mcdonalds_omega": round(omega, 4),
            "mean_item_total_r": round(np.mean(list(item_total_corrs.values())), 4),
            "min_item_total_r": round(min(item_total_corrs.values()), 4),
            "n_tracts": len(factor_data),
        })

    consistency_df = pd.DataFrame(consistency_results)
    save_csv(consistency_df, PATHS["final"] / "internal_consistency.csv")

    logger.info("\n=== Summary ===")
    if cfa_success:
        logger.info(f"  CFA fit: CFI={cfi:.4f}, RMSEA={rmsea:.4f}, SRMR={srmr:.4f}")
    for _, row in consistency_df.iterrows():
        logger.info(f"  {row['factor']}: alpha={row['cronbach_alpha']:.3f}, "
                     f"omega={row['mcdonalds_omega']:.3f}")

    return consistency_df


if __name__ == "__main__":
    print("=" * 70)
    print("PHASE A: CFA Validation and Internal Consistency")
    print("=" * 70)
    results = run_cfa_validation()
    print(f"\nDone. {len(results)} factor consistency metrics computed.")
