# CDC PLACES HRSN Analysis: Research Summary  

## Overview

This project is a comprehensive analysis of the CDC's newly released (August 2024) census-tract-level health-related social needs (HRSN) measures from the PLACES platform. We characterized their measurement properties, validated them against independent data, quantified a critical shared-modeling bias, and applied them to decompose racial health disparities. The work has been split into two papers targeting public health journals.

---

## Data

- **Primary dataset**: 60,156 US census tracts across 39 states and DC (from CDC PLACES 2024, ACS 2022, CDC/ATSDR SVI 2022)
- **7 HRSN exposures** (tract-level age-adjusted prevalence from BRFSS/MRP): food insecurity, SNAP receipt, housing insecurity, utility shutoff threat, transportation barriers, loneliness/social isolation, lack of social/emotional support
- **7 chronic disease outcomes** (also from PLACES/BRFSS/MRP): diabetes, depression, asthma, obesity, CHD, COPD, stroke
- **Independent validation data**: ACS direct proxies, CMS Medicare claims (ICD codes from 100% FFS claims), Map the Meal Gap (CPS-modeled food insecurity), USDA Food Access Atlas, Eviction Lab (court records), Chetty et al. Social Capital Atlas, CDC WONDER cause-specific mortality
- **Individual-level data**: BRFSS 2023 (up to 145,248 respondents for cross-level comparison)

---

## Paper 1: Measurement Structure and Predictive Validity

### Research Questions

1. What is the factor structure and internal consistency of the 7 PLACES HRSN measures?
2. Do they predict chronic disease beyond the Social Vulnerability Index (SVI)?
3. Are the associations genuine, or artifacts of shared BRFSS modeling?
4. How much does shared modeling inflate effect sizes?

### Methods and Key Techniques

**Factor analysis (psychometric validation)**
- Split-half design: development (n=30,078) and validation (n=30,078)
- KMO (0.88) + Bartlett's test for sampling adequacy
- Parallel analysis for factor number determination
- EFA with ML extraction and promax (oblique) rotation
- CFA on hold-out half using semopy (CFI=0.93, Tucker's congruence=0.9999)
- Cronbach's alpha + McDonald's omega for internal consistency
- HTMT ratio for discriminant validity

**49 ecological regressions (7 HRSN x 7 diseases)**
- OLS with county-clustered standard errors (2,294 clusters)
- Benjamini-Hochberg FDR correction across 49 tests
- All variables standardized (z-scored)

**Incremental validity over SVI**
- Nested model comparison (SVI-only, HRSN-only, combined)
- Cluster-robust Wald tests (not naive F-tests)

**Cross-validation (3 strategies)**
- Random 10-fold CV
- County-blocked 10-fold CV (GroupKFold) to detect spatial leakage
- State-blocked leave-one-state-out CV (40 folds)

**Independent validation (breaking the shared-modeling chain)**
- V3: ACS direct proxies -> CMS Medicare claims (fully independent on both sides, county-level)
- V5: Five non-BRFSS domain-specific proxies (Map the Meal Gap, Eviction Lab, USDA Food Access, Social Capital Atlas) -> CMS claims

**Attenuation analysis**
- Compared primary beta magnitudes to V3 betas for matched pairs to quantify shared-modeling inflation

**Sensitivity analyses**
- Population-weighted least squares (WLS)
- Reduced-covariate models (dropping MRP-overlapping covariates)
- Spatial error models (GM_Error_Het) for residual autocorrelation
- E-values for unmeasured confounding robustness
- Random forest variable importance (permutation + SHAP via TreeExplainer)
- Variance decomposition (within-county vs. between-county R^2, ICCs)

### Key Results

| Finding | Detail |
|---|---|
| Factor structure | 2 factors: Material Hardship (5 items, alpha=0.97, omega=0.99) and Social Isolation (2 items, alpha=0.90, omega=0.91); 93% variance explained |
| CFA replication | Tucker's congruence = 0.9999; CFI = 0.93 |
| Multicollinearity | Material Hardship pairwise r = 0.88-0.99; max VIF = 97.4 (individual items) -> 8.8 (composites) |
| Significance | 44/49 associations significant after FDR (q<0.05) |
| Strongest associations | Food insecurity -> diabetes (beta=0.73), utility shutoff -> COPD (0.72) |
| HRSN vs SVI | Mean R^2: HRSN = 0.81, SVI = 0.70; incremental R^2 = 9-17 pp (all Wald P<.001) |
| Cross-validation | OOS R^2 = 0.81 (random), 0.79 (county-blocked, only 2% leakage), 0.46 (state-blocked) |
| Independent validation V3 | 76% directional concordance (ACS -> CMS claims) |
| Independent validation V5 | 93% concordance for MMG food insecurity, 86% for eviction filings |
| Shared-modeling inflation | Median 2.0x; Pearson r = 0.55 between primary and independent betas |
| E-values | Median = 2.45; 30/44 exceed 2.0; food insecurity -> diabetes = 3.31 |
| RF importance | Mean Spearman rho = 0.67 vs. OLS rankings; RF test R^2 = 0.86 |
| Variance decomposition | Mean within-county R^2 (0.73) > between-county R^2 (0.53) |

---

## Paper 2: Racial Disparities and Ecological Amplification


### Research Questions

1. What share of Black-White chronic disease gaps is explained by community-level HRSN differences?
2. How do ecological estimates compare with individual-level estimates (ecological amplification)?
3. Does Material Hardship mediate SVI-disease associations?

### Methods and Key Techniques

**Oaxaca-Blinder decomposition at two levels**
- *Ecological*: Majority-Black (>=50%, n=5,436) vs. Majority-White (>=60%, n=38,240) tracts
- *Individual*: Black (n~13,010) vs. White (n~132,238) BRFSS 2023 respondents
- Neumark/pooled specification with group indicator
- 500 stratified bootstrap resamples for 95% CIs
- Factor-level decomposition (2 composite scores) to address multicollinearity instability

**Cross-level comparison**
- Ecological amplification ratio = ecological % explained / individual % explained
- Restricted to outcomes with consistent positive B-W gaps at both levels

**Threshold sensitivity**
- Repeated ecological OB at 40%, 50%, 60% majority-Black thresholds

**Mediation analysis**
- 28 models (4 SVI themes x 7 diseases)
- Baron and Kenny framework with 1,000 bootstrap resamples
- Testing Material Hardship as mediator of SVI -> disease pathway

### Key Results

| Finding | Detail |
|---|---|
| Ecological HRSN contribution | 23% (obesity) to 99% (COPD) of Black-White disease gaps explained |
| Key decompositions | Diabetes: 66%, stroke: 65%, CHD: 65%, obesity: 23% |
| Individual-level HRSN | Diabetes: 13.5%, obesity: 14.4%, stroke: 41.4% |
| **Ecological amplification** | **Diabetes: 4.9x, obesity: 1.6x, stroke: 1.6x** |
| Threshold stability | Diabetes: 65-68% across 40-60% thresholds; obesity: 23-24% |
| Mediation | All 28 indirect effects significant; Material Hardship mediates all SVI -> disease paths |
| Strongest mediation | COPD (path b = 0.65-0.69), diabetes (0.63-0.65), stroke (0.63-0.65) |
| Unstable outcomes | Depression, CHD, COPD have reversed individual-level B-W gaps (excluded from amplification) |

The central finding is "ecological amplification": community-level HRSN differences explain approximately 5x more of the Black-White diabetes gap than individual-level HRSN differences. This quantifies how spatial concentration of social disadvantage in majority-Black neighborhoods compounds individual-level risk.

---

## Pipeline and Infrastructure

The analysis is fully reproducible via a 44-script pipeline:

| Stage | Scripts | Purpose |
|---|---|---|
| `scripts/collect/` | 01-09 | Data acquisition (PLACES API, ACS, SVI, CMS, BRFSS, CDC WONDER, external sources) |
| `scripts/process/` | 10-18 | Cleaning, merging, tract filtering, variable construction |
| `scripts/analyze/` | 19-44 | Regressions, factor analysis, cross-validation, OB decomposition, validation, mediation, SHAP, spatial models, E-values, CFA, policy simulations |
| `scripts/visualize/` | -- | Figures and tables |

**Core library** (`src/hrsn_analysis/`): shared utilities for paths, I/O, logging, and survey weights.

**Configuration**: `configs/params.yml` defines all measures, outcomes, covariates, and data source paths.

**Key output files**: all results are in `data/final/` as CSV/parquet (49 regression results, cross-validation results, OB decomposition, validation concordance, SHAP importance, mediation results, E-values, CFA loadings, etc.).

---

## Technical Stack

- **Language**: Python 3.11
- **Core packages**: statsmodels, scikit-learn, scipy, numpy, pandas, semopy (CFA), pysal/spreg (spatial models), shap (TreeExplainer)
- **Statistical methods used**: OLS with clustered SEs, WLS, EFA/CFA, Oaxaca-Blinder decomposition, 10-fold/GroupKFold/LOSO cross-validation, random forests, SHAP, spatial error models (GM_Error_Het), E-values, mediation analysis, Benjamini-Hochberg FDR, bootstrap inference, variance decomposition (ICC/multilevel), parallel analysis, quantile regression, policy simulation

---

## What Makes This Work slightly interesting

1. **First systematic psychometric evaluation** of the PLACES HRSN measures (factor structure, internal consistency, discriminant validity)
2. **First head-to-head comparison** of HRSN vs. SVI for chronic disease prediction with proper cross-validation
3. **First quantification of shared-modeling inflation** in PLACES-derived ecological associations (the 2x calibration factor)
4. **Multi-source independent validation** using 5+ external data systems (CMS claims, ACS, Map the Meal Gap, Eviction Lab, USDA, Chetty social capital, CDC WONDER mortality)
5. **First quantification of "ecological amplification"** of HRSN contributions to racial health disparities across multiple diseases
6. **Cross-level comparison** (ecological vs. individual Oaxaca-Blinder) for HRSN and chronic disease --- rarely done with this many outcomes and independent data sources

---

## Potential Collaboration Directions

- **Individual-level linkage using restricted-access geocoded BRFSS**: Access to geocoded BRFSS microdata (with FIPS codes) would enable true multilevel models (individuals nested in tracts) rather than ecological-only analysis, directly addressing the study's primary methodological limitation. This requires an approved data use agreement through an institutional affiliation.
- **Clinical EHR validation**: Deidentified electronic health record data from a health system (academic medical center, VA, integrated network) with clinical diagnoses, lab values, and social needs screening results would provide individual-level clinical validation far stronger than self-reported BRFSS — and independent of the PLACES shared-modeling framework entirely.
