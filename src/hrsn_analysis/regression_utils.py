"""Regression helper functions for the HRSN × Disease analysis."""
import numpy as np
import statsmodels.api as sm


def standardize(series):
    """Z-score standardize a pandas Series."""
    return (series - series.mean()) / series.std()


def run_ols_clustered(y, X, cluster_var, add_constant=True):
    """
    Run OLS regression with clustered standard errors.

    Parameters
    ----------
    y : pd.Series
        Dependent variable
    X : pd.DataFrame
        Independent variables (already standardized)
    cluster_var : pd.Series
        Cluster identifier (e.g., county FIPS)
    add_constant : bool
        Whether to add intercept

    Returns
    -------
    statsmodels RegressionResults
        Model with clustered SEs
    """
    if add_constant:
        X = sm.add_constant(X)
    # Drop NaN rows explicitly to keep cluster_var aligned
    mask = X.notna().all(axis=1) & y.notna()
    X_clean = X[mask]
    y_clean = y[mask]
    cluster_clean = cluster_var[mask]
    model = sm.OLS(y_clean, X_clean)
    results = model.fit(cov_type="cluster", cov_kwds={"groups": cluster_clean})
    return results


def extract_results(results, hrsn_var, outcome_var):
    """Extract key statistics from a regression result into a dict."""
    return {
        "hrsn_measure": hrsn_var,
        "outcome": outcome_var,
        "beta": results.params.get(hrsn_var, np.nan),
        "se": results.bse.get(hrsn_var, np.nan),
        "pvalue": results.pvalues.get(hrsn_var, np.nan),
        "ci_lower": (
            results.conf_int().loc[hrsn_var, 0]
            if hrsn_var in results.conf_int().index
            else np.nan
        ),
        "ci_upper": (
            results.conf_int().loc[hrsn_var, 1]
            if hrsn_var in results.conf_int().index
            else np.nan
        ),
        "r_squared": results.rsquared,
        "n_obs": int(results.nobs),
    }
