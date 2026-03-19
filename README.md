# CDC PLACES HRSN Analysis

Census-tract-level analysis of the CDC's newly released (August 2024) health-related social needs (HRSN) measures from the PLACES platform. We characterized their measurement properties, validated them against independent data, quantified a critical shared-modeling bias, and applied them to decompose racial health disparities.

## Data

- **Primary dataset**: 60,156 US census tracts across 39 states and DC (CDC PLACES 2024, ACS 2022, CDC/ATSDR SVI 2022)
- **7 HRSN exposures** (tract-level, BRFSS/MRP): food insecurity, SNAP receipt, housing insecurity, utility shutoff threat, transportation barriers, loneliness/social isolation, lack of social/emotional support
- **7 chronic disease outcomes** (PLACES/BRFSS/MRP): diabetes, depression, asthma, obesity, CHD, COPD, stroke
- **Independent validation data**: ACS direct proxies, CMS Medicare claims, Map the Meal Gap, USDA Food Access Atlas, Eviction Lab, Chetty et al. Social Capital Atlas, CDC WONDER mortality
- **Individual-level data**: BRFSS 2023 (~145,000 respondents for cross-level comparison)

## Key Findings

**Measurement (Paper 1)**
- The 7 HRSN measures resolve into 2 factors: Material Hardship (5 items, alpha=0.97) and Social Isolation (2 items, alpha=0.90), confirmed by split-half CFA (Tucker's congruence=0.9999)
- 44/49 HRSN-disease associations significant after FDR correction; strongest: food insecurity -> diabetes (beta=0.73)
- HRSN outperforms SVI for chronic disease prediction (mean R^2: 0.81 vs 0.70; incremental R^2 = 9-17 pp)
- Cross-validation: OOS R^2 = 0.81, county-blocked = 0.79 (only 2% spatial leakage), zero overfitting
- Independent validation (ACS -> CMS claims): 76% directional concordance; domain-specific proxies: 86-93% for strong matches
- Shared BRFSS modeling inflates effect sizes by median 2.0x (quantifiable calibration factor)

**Racial Disparities (Paper 2)**
- Community-level HRSN differences explain 23-99% of Black-White chronic disease gaps (diabetes: 66%, stroke: 65%)
- Individual-level HRSN explains far less (diabetes: 13.5%, stroke: 41.4%)
- Ecological amplification ratio: 4.9x for diabetes, 1.6x for obesity, 1.6x for stroke
- All 28 mediation models significant: Material Hardship mediates SVI -> disease pathways
- Threshold sensitivity confirms stability (diabetes: 65-68% across 40-60% majority-Black thresholds)

## Methods

- EFA/CFA with split-half validation, parallel analysis, promax rotation
- OLS with county-clustered standard errors + Benjamini-Hochberg FDR
- Incremental validity via cluster-robust Wald tests
- 10-fold, county-blocked, and state-blocked (leave-one-state-out) cross-validation
- Independent validation using 5+ external data systems
- Oaxaca-Blinder decomposition at ecological and individual levels (Neumark/pooled, 500 bootstrap)
- Mediation analysis (Baron-Kenny, 1,000 bootstrap)
- Spatial error models (GM_Error_Het), E-values, random forest/SHAP importance, variance decomposition

## Setup

```bash
conda env create -f environment.yml
conda activate hrsn-analysis
pip install -e .
```

## Pipeline

The analysis runs through a 44-script pipeline in `scripts/`:

| Stage | Scripts | Purpose |
|---|---|---|
| `scripts/collect/` | 01-40 | Data acquisition (PLACES API, ACS, SVI, CMS, BRFSS, CDC WONDER, external sources) |
| `scripts/process/` | 04-13 | Cleaning, merging, tract filtering, variable construction |
| `scripts/analyze/` | 07-44 | Regressions, factor analysis, cross-validation, OB decomposition, validation, mediation, SHAP, spatial models, E-values, CFA |
| `scripts/visualize/` | -- | Figures and tables |

```bash
make collect     # Download raw data
make process     # Reshape, merge, create master dataset
make analyze     # Run regressions
make factor      # EFA/CFA factor analysis
make brfss       # Individual-level BRFSS analysis
make spatial     # Spatial autocorrelation and error models
make svi         # SVI comparison and incremental validity
make decomposition  # Oaxaca-Blinder decomposition
make validation  # Independent validation framework
make enhancements   # Run all enhancement analyses
```

## Project Structure

```
configs/params.yml          # Analysis configuration (measures, outcomes, covariates)
src/hrsn_analysis/          # Core library (paths, I/O, logging, survey weights)
scripts/collect/            # Data acquisition scripts
scripts/process/            # Data processing scripts
scripts/analyze/            # Analysis scripts
scripts/visualize/          # Visualization scripts
tests/                      # Tests
```

## Technical Stack

Python 3.11 with statsmodels, scikit-learn, scipy, pandas, semopy, pysal/spreg, shap, geopandas, factor-analyzer

## License

MIT
