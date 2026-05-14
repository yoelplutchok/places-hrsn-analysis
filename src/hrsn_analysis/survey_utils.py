"""Survey-weighted analysis utilities for BRFSS individual-level data."""
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.genmod.families import Binomial

from .logging_utils import get_logger

logger = get_logger(__name__)


def recode_brfss_binary(series, recode_type):
    """
    Recode a BRFSS variable to binary (0/1) based on recode type.

    Parameters
    ----------
    series : pd.Series
        Raw BRFSS variable values.
    recode_type : str
        One of: 'always_usually', 'yes_no', 'lack_support', 'diabetes', 'obesity'.

    Returns
    -------
    pd.Series
        Binary (0/1) with NaN for refused/missing.
    """
    result = pd.Series(np.nan, index=series.index)

    if recode_type == "always_usually":
        # 1=Always, 2=Usually → 1; 3=Sometimes, 4=Rarely, 5=Never → 0; 7=Don't know, 9=Refused → NaN
        result[series.isin([1, 2])] = 1
        result[series.isin([3, 4, 5])] = 0

    elif recode_type == "yes_no":
        # 1=Yes → 1; 2=No → 0; 7=Don't know, 9=Refused → NaN
        result[series == 1] = 1
        result[series == 2] = 0

    elif recode_type == "lack_support":
        # Reverse-coded: 1=Always, 2=Usually, 3=Sometimes → 0 (has support)
        # 4=Rarely, 5=Never → 1 (lacks support); 7,9 → NaN
        result[series.isin([1, 2, 3])] = 0
        result[series.isin([4, 5])] = 1

    elif recode_type == "diabetes":
        # 1=Yes → 1; 2=Yes but only during pregnancy → 0; 3=No → 0; 4=Pre-diabetes → 0
        result[series == 1] = 1
        result[series.isin([2, 3, 4])] = 0

    elif recode_type == "obesity":
        # _BMI5CAT: 1=Underweight, 2=Normal, 3=Overweight, 4=Obese → 1
        result[series == 4] = 1
        result[series.isin([1, 2, 3])] = 0

    else:
        raise ValueError(f"Unknown recode type: {recode_type}")

    return result


def recode_demographics(df):
    """
    Recode BRFSS demographic variables for use as covariates.

    Returns DataFrame with recoded columns:
    - age_group: ordinal 1-13 (from _AGEG5YR)
    - female: binary (from SEXVAR or SEX1)
    - race_cat: categorical string (from _RACE)
    - income_cat: ordinal (from _INCOMG1)
    - education_cat: ordinal (from _EDUCAG)
    """
    result = pd.DataFrame(index=df.index)

    # Age: _AGEG5YR is 1-13 (5-year groups), 14=Don't know/refused → NaN
    if "_AGEG5YR" in df.columns:
        age = df["_AGEG5YR"].copy()
        age[age == 14] = np.nan
        result["age_group"] = age

    # Sex: SEXVAR (or SEX1) 1=Male, 2=Female
    sex_col = next((c for c in ["SEXVAR", "SEX1", "_SEX"] if c in df.columns), None)
    if sex_col is not None:
        sex = df[sex_col].copy()
        result["female"] = np.where(sex == 2, 1, np.where(sex == 1, 0, np.nan))

    # Race: _RACE 1=White, 2=Black, 3=AI/AN, 4=Asian, 5=NH/PI, 6=Other, 7=Multi, 8=Hispanic
    if "_RACE" in df.columns:
        race_map = {
            1: "white", 2: "black", 3: "aian", 4: "asian",
            5: "nhpi", 6: "other", 7: "multiracial", 8: "hispanic",
        }
        result["race_cat"] = df["_RACE"].map(race_map)
        # 9 = Don't know/refused → NaN (unmapped stays NaN)

    # Income: _INCOMG1 is ordinal 1-7; 9 → NaN
    if "_INCOMG1" in df.columns:
        inc = df["_INCOMG1"].copy()
        inc[inc == 9] = np.nan
        result["income_cat"] = inc

    # Education: _EDUCAG 1-4 ordinal; 9 → NaN
    if "_EDUCAG" in df.columns:
        edu = df["_EDUCAG"].copy()
        edu[edu == 9] = np.nan
        result["education_cat"] = edu

    return result


def weighted_logistic(y, X, weights, psu=None, strata=None):
    """
    Run a survey-weighted logistic regression using GLM with Binomial family.

    Weights are normalized to sum to the actual sample size to prevent
    artificial inflation of effective N (which occurs when raw BRFSS
    probability weights are passed as freq_weights). Cluster-robust
    standard errors are computed using PSU (primary sampling unit) to
    approximate the complex survey design.

    Parameters
    ----------
    y : pd.Series
        Binary outcome (0/1).
    X : pd.DataFrame
        Predictor matrix (with constant already added).
    weights : pd.Series
        Survey probability weights (e.g., _LLCPWT). Will be normalized
        to sum to len(y) before fitting.
    psu : pd.Series, optional
        Primary sampling unit variable (_PSU) for cluster-robust SEs.
    strata : pd.Series, optional
        Stratification variable (_STSTR). Used as clustering fallback
        if psu is not provided.

    Returns
    -------
    statsmodels GLMResults
    """
    # Normalize weights to sum to actual sample size
    # This preserves relative weighting while keeping effective N correct
    n = len(y)
    w_sum = weights.sum()
    if w_sum > 0:
        scaled_weights = weights * n / w_sum
    else:
        scaled_weights = weights

    model = sm.GLM(y, X, family=Binomial(), freq_weights=scaled_weights)

    if psu is not None:
        results = model.fit(cov_type="cluster", cov_kwds={"groups": psu})
    elif strata is not None:
        results = model.fit(cov_type="cluster", cov_kwds={"groups": strata})
    else:
        results = model.fit(cov_type="HC1")

    return results


def extract_or(results, var_name):
    """
    Extract odds ratio with 95% CI from logistic regression results.

    Parameters
    ----------
    results : GLMResults
        Fitted logistic regression.
    var_name : str
        Name of the variable to extract OR for.

    Returns
    -------
    dict
        Keys: or, or_lower, or_upper, pvalue, coef, se.
    """
    coef = results.params[var_name]
    se = results.bse[var_name]
    pval = results.pvalues[var_name]
    ci = results.conf_int().loc[var_name]

    return {
        "coef": coef,
        "se": se,
        "or": np.exp(coef),
        "or_lower": np.exp(ci[0]),
        "or_upper": np.exp(ci[1]),
        "pvalue": pval,
    }
