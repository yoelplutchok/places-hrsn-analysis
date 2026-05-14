"""
Step 29: E-Values for Sensitivity to Unmeasured Confounding.

Computes E-values (VanderWeele & Ding, 2017) for all 49 HRSN-disease
associations. The E-value quantifies the minimum strength of association
that an unmeasured confounder would need to have with both the exposure
(HRSN) and the outcome (disease) to fully explain away the observed
association, conditional on measured covariates.

For continuous outcomes with standardized betas, we use the Ding &
VanderWeele (2016) approximation: RR ~= exp(0.91 * |beta_std|).
Then: E-value = RR + sqrt(RR * (RR - 1)).

Addresses reviewer concern about unmeasured confounding (healthcare access,
health behaviors, environmental exposures, etc.).

Output:
  - data/final/evalue_analysis.csv (49 rows, one per HRSN-disease pair)
"""
import sys
from pathlib import Path

import pandas as pd
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from hrsn_analysis.paths import PATHS, ensure_dirs
from hrsn_analysis.io_utils import save_csv
from hrsn_analysis.logging_utils import setup_logging, get_logger

setup_logging()
logger = get_logger(__name__)


def beta_to_rr(beta_std):
    """Convert standardized beta to approximate risk ratio.

    Uses Ding & VanderWeele (2016) approximation for continuous outcomes:
    RR ~= exp(0.91 * |beta_std|)

    Always returns RR >= 1 (uses absolute value).
    """
    return np.exp(0.91 * abs(beta_std))


def compute_evalue(rr):
    """Compute E-value for a given risk ratio.

    Formula (VanderWeele & Ding, 2017):
      E = RR + sqrt(RR * (RR - 1))

    The E-value is the minimum strength of association on the RR scale
    that an unmeasured confounder would need with both exposure and outcome
    to explain away the observed association.

    Parameters
    ----------
    rr : float
        Risk ratio >= 1.

    Returns
    -------
    float
        E-value >= 1.0.
    """
    if rr < 1.0:
        return 1.0
    return rr + np.sqrt(rr * (rr - 1))


def compute_evalue_for_ci(beta_std, ci_lower, ci_upper):
    """Compute E-value for the confidence interval bound closest to null.

    If the CI crosses zero, the E-value for the CI is 1.0 (the association
    is not statistically significant, so no confounder strength is needed).

    Parameters
    ----------
    beta_std : float
        Standardized coefficient.
    ci_lower : float
        Lower bound of 95% CI (standardized).
    ci_upper : float
        Upper bound of 95% CI (standardized).

    Returns
    -------
    float
        E-value for CI bound.
    """
    # Check if CI crosses zero
    if ci_lower <= 0 <= ci_upper:
        return 1.0

    # Use the CI bound closest to null (smallest absolute effect)
    if beta_std > 0:
        ci_bound = ci_lower  # smallest positive effect
    else:
        ci_bound = ci_upper  # closest to zero for negative effect

    rr_ci = beta_to_rr(ci_bound)
    return compute_evalue(rr_ci)


def run_evalue_analysis():
    """Compute E-values for all 49 HRSN-disease associations."""
    ensure_dirs()

    # Load primary regression results
    results_path = PATHS["final"] / "results_matrix.csv"
    results = pd.read_csv(results_path)
    logger.info(f"Loaded {len(results)} regression results from {results_path.name}")

    rows = []
    for _, r in results.iterrows():
        beta = r["beta_std"]
        ci_lo = r["ci_lower"]
        ci_hi = r["ci_upper"]

        # Skip if beta is missing
        if pd.isna(beta):
            continue

        rr = beta_to_rr(beta)
        ev_point = compute_evalue(rr)
        ev_ci = compute_evalue_for_ci(beta, ci_lo, ci_hi)

        rows.append({
            "hrsn_measure": r["hrsn_measure"],
            "hrsn_label": r["hrsn_label"],
            "outcome": r["outcome"],
            "outcome_label": r["outcome_label"],
            "beta_std": round(beta, 4),
            "se_std": round(r["se_std"], 4),
            "ci_lower": round(ci_lo, 4),
            "ci_upper": round(ci_hi, 4),
            "pvalue": r["pvalue"],
            "pvalue_fdr": r.get("pvalue_fdr", np.nan),
            "significant_fdr_05": r.get("significant_fdr_05", False),
            "direction": "harmful" if beta > 0 else "protective",
            "rr_approx": round(rr, 4),
            "evalue_point": round(ev_point, 2),
            "evalue_ci": round(ev_ci, 2),
        })

    evalue_df = pd.DataFrame(rows)
    save_csv(evalue_df, PATHS["final"] / "evalue_analysis.csv")

    # Summary statistics
    sig = evalue_df[evalue_df["significant_fdr_05"] == True]
    logger.info(f"\n=== E-Value Summary ===")
    logger.info(f"Total associations: {len(evalue_df)}")
    logger.info(f"FDR-significant: {len(sig)}")

    if len(sig) > 0:
        logger.info(f"\nAmong {len(sig)} FDR-significant associations:")
        logger.info(f"  E-value (point estimate):")
        logger.info(f"    Median: {sig['evalue_point'].median():.2f}")
        logger.info(f"    Range:  {sig['evalue_point'].min():.2f} - "
                     f"{sig['evalue_point'].max():.2f}")
        logger.info(f"    E-value > 2.0: {(sig['evalue_point'] > 2.0).sum()}/{len(sig)}")
        logger.info(f"    E-value > 3.0: {(sig['evalue_point'] > 3.0).sum()}/{len(sig)}")

        logger.info(f"\n  E-value (CI bound):")
        logger.info(f"    Median: {sig['evalue_ci'].median():.2f}")
        logger.info(f"    Range:  {sig['evalue_ci'].min():.2f} - "
                     f"{sig['evalue_ci'].max():.2f}")

        # Top 5 most robust associations (highest E-value)
        top5 = sig.nlargest(5, "evalue_point")
        logger.info(f"\n  Top 5 most robust (highest E-value):")
        for _, r in top5.iterrows():
            logger.info(f"    {r['hrsn_label']:30s} x {r['outcome_label']:15s}: "
                        f"E={r['evalue_point']:.2f} (CI: {r['evalue_ci']:.2f}), "
                        f"beta={r['beta_std']:+.4f}")

        # Bottom 5 (most vulnerable to confounding)
        bot5 = sig.nsmallest(5, "evalue_point")
        logger.info(f"\n  Bottom 5 (most vulnerable to confounding):")
        for _, r in bot5.iterrows():
            logger.info(f"    {r['hrsn_label']:30s} x {r['outcome_label']:15s}: "
                        f"E={r['evalue_point']:.2f} (CI: {r['evalue_ci']:.2f}), "
                        f"beta={r['beta_std']:+.4f}")

    # Verification checks
    assert (evalue_df["evalue_point"] >= 1.0).all(), "E-values must be >= 1.0"
    assert (evalue_df["evalue_ci"] >= 1.0).all(), "CI E-values must be >= 1.0"
    assert (evalue_df["evalue_ci"] <= evalue_df["evalue_point"]).all(), \
        "CI E-value must be <= point E-value"
    logger.info("\nAll verification checks passed.")

    return evalue_df


if __name__ == "__main__":
    print("=" * 70)
    print("Step 29: E-Values for Unmeasured Confounding")
    print("=" * 70)
    ev = run_evalue_analysis()
    print(f"\nDone. {len(ev)} E-values computed.")
