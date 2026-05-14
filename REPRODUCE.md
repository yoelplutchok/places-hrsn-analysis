# Reproducing the PLACES HRSN Analysis

This guide walks through three reproduction levels, in increasing order of effort:

1. **Verify the published numbers** — no download, ~1 minute
2. **Re-run the analyses from the merged dataset** — no raw-data download, ~30 minutes
3. **Re-run the full pipeline from raw data** — requires ~1.8 GB of downloads, ~2-3 hours

## Prerequisites

- Python 3.11+
- Conda or mamba (recommended) or a virtualenv
- ~5 GB free disk (for raw + processed + final data)
- Internet connection (for raw-data download in Level 3)

## Environment setup (all levels)

```bash
git clone https://github.com/<your-username>/places-hrsn-analysis.git
cd places-hrsn-analysis
conda env create -f environment.yml
conda activate hrsn-analysis
pip install -e .
```

This installs:
- pandas, numpy, scipy, statsmodels, scikit-learn (core analysis)
- geopandas, libpysal, esda, mgwr, spreg (spatial)
- factor-analyzer, semopy (factor analysis + CFA)
- censusdis (Census API)

## Level 1 — Verify the published numbers (~1 minute)

All result tables from both papers are in `data/final/`. To spot-check the main claims:

```bash
python -c "
import pandas as pd

# Paper 1 Table 1 — primary 49 regression results
r = pd.read_csv('data/final/results_matrix.csv')
assert r.shape == (49, 21)
assert r['significant_fdr_05'].sum() == 44   # 44/49 FDR-significant
print('Top 3 effect sizes:')
print(r.nlargest(3, 'beta_std')[['hrsn_measure','outcome','beta_std']])

# Paper 1 Table 3 — SVI incremental validity
print('\nIncremental R² (HRSN beyond SVI):')
svi = pd.read_csv('data/final/svi_incremental_validity.csv')
print(svi[['outcome','r2_svi_only','r2_hrsn_only','delta_r2_hrsn_over_svi']].round(3))

# Paper 1 Table 5 — attenuation analysis
att = pd.read_csv('data/final/attenuation_analysis.csv')
print(f'\nV3 same-direction: {att.same_direction.sum()}/{len(att)} (76% concordance)')
sd = att[att.same_direction & att.inflation_ratio.notna()]
print(f'Median inflation: {sd.inflation_ratio.median():.2f}x across 21 same-direction pairs')

# Paper 2 Table 1 — Oaxaca-Blinder, Black vs White
ob = pd.read_csv('data/final/ob_decomposition_ecological.csv')
bw = ob[ob.comparison == 'Black_vs_White']
print('\nBlack-White HRSN endowment share (HRSN-only rows):')
for o in ['diabetes','obesity','stroke','copd','depression','chd','casthma']:
    pct = bw[(bw.outcome==o) & (bw.variable_type=='hrsn')]['pct_of_gap'].sum()
    print(f'  {o:12s}: {pct:.1f}%')

# Paper 2 Table 2 — Individual-level OB
iob = pd.read_csv('data/final/individual_ob_all_outcomes.csv')
print('\nIndividual-level HRSN endowment share (BRFSS 2023):')
print(iob[['outcome','prevalence_white','prevalence_black','pct_explained_hrsn']].round(2))
"
```

Every number cited in the abstracts and main tables is reproducible from `data/final/*.csv`.

## Level 2 — Re-run analyses from the merged dataset (~30 minutes)

The single most important intermediate is `data/processed/merged_tracts.parquet` (60,156 × 38). All analysis scripts read from this file and write to `data/final/`.

```bash
# Verify the merged dataset is intact
python -c "
import pandas as pd
df = pd.read_parquet('data/processed/merged_tracts.parquet')
assert df.shape == (60156, 38), f'Unexpected shape: {df.shape}'
print('merged_tracts.parquet OK:', df.shape)
"

# Re-run the analysis stages
make analyze       # primary regressions, mutually adjusted, factor regressions
python scripts/analyze/10_factor_analysis.py
python scripts/analyze/15_spatial_autocorrelation.py
python scripts/analyze/18_svi_comparison.py
python scripts/analyze/19_disparity_decomposition.py
python scripts/analyze/23_validation_framework.py
python scripts/analyze/24_attenuation_analysis.py
python scripts/analyze/25_cross_validation.py
python scripts/analyze/28_spatial_cross_validation.py
python scripts/analyze/29_evalue_analysis.py
python scripts/analyze/30_factor_ob_decomposition.py
python scripts/analyze/31_variance_decomposition.py
python scripts/analyze/35_shap_importance.py
python scripts/analyze/36_mediation_analysis.py
python scripts/analyze/41_enhanced_validation.py   # requires V5 raw data (see Level 3)
python scripts/analyze/42_cfa_validation.py
python scripts/analyze/43_individual_ob_all_outcomes.py   # requires BRFSS raw data
python scripts/analyze/44_ob_threshold_sensitivity.py
```

Output: every CSV in `data/final/` should be regenerated. Compare against the committed versions with `git diff data/final/` — differences should be limited to floating-point precision in the trailing decimals.

**Note:** Scripts 14 (BRFSS individual regressions), 41 (enhanced V5 validation), and 43 (individual OB) require external raw data (BRFSS 2023 SAS XPT files, Map the Meal Gap, Eviction Lab, etc.) and will skip with a clear error if those files are missing. See Level 3.

## Level 3 — Re-run the full pipeline from raw data (~2-3 hours)

### Step 3a. Download raw data

```bash
# Programmatic downloads (most data sources)
make collect      # PLACES, ACS, shapefiles

# Additional collect scripts (run individually)
python scripts/collect/12_download_brfss.py            # BRFSS 2023 SAS XPT (1.1 GB)
python scripts/collect/17_download_svi.py              # CDC/ATSDR SVI 2022
python scripts/collect/22_download_cms_medicare.py     # CMS MMD (manual download URL provided)
python scripts/collect/33_download_food_access.py      # USDA Food Access Atlas
python scripts/collect/34_process_cdc_wonder.py        # CDC WONDER mortality (manual download via web form)
python scripts/collect/38_download_social_capital.py   # Chetty Social Capital Atlas
python scripts/collect/39_download_eviction_lab.py     # Eviction Lab V1 (S3)
python scripts/collect/40_process_food_access_v5.py    # V5 supplementary
```

Several sources require manual interactions:

| Source | Why manual | Where to get it |
|---|---|---|
| CMS Mapping Medicare Disparities | Disabled API; web download only | data.cms.gov/mapping-medicare-disparities |
| CDC WONDER (mortality) | Web form only; API is national | wonder.cdc.gov (database D158) |
| Map the Meal Gap | Requires free Feeding America registration | feedingamerica.org/research/map-the-meal-gap/by-county |
| EJScreen | gaftp.epa.gov URLs are 404 | Try zenodo.org/records/14767363 |

`data/raw/ACQUISITION.md` documents every data source with URL, download date, file size, row count, and licensing.

### Step 3b. Process raw data into the merged dataset

```bash
make process      # 04_reshape_places, 05_process_census, 06_merge_datasets
python scripts/process/13_process_brfss.py
```

Output: `data/processed/merged_tracts.parquet`, `data/processed/brfss_analytic.parquet`, etc.

### Step 3c. Run the analysis pipeline

Same as Level 2 above.

## Pipeline overview

```
scripts/collect/    -> data/raw/         (download from external sources)
scripts/process/    -> data/processed/   (reshape, merge, build analytic dataset)
scripts/analyze/    -> data/final/       (regressions, factor analysis, OB, validation, etc.)
scripts/visualize/  -> outputs/          (figures and supplementary tables)
```

The 44 analysis scripts are numbered roughly in dependency order. Numbers up to 30 are core analyses; 31+ are extensions added during revision (mediation, mortality, V5 enhanced validation, CFA, threshold sensitivity).

## Common reproduction issues

- **`semopy` import error** — CFA requires `pip install semopy` (in environment.yml's pip section)
- **`mgwr` slow** — GWR (script 16) takes ~20 minutes; safe to skip for headline numbers
- **SHAP slow** — `35_shap_importance.py` requires `max_depth=15` and `SHAP_BACKGROUND=100`; with defaults it can hang
- **Spatial weights NaN** — All spatial code uses `w_subset()` to handle missing data; do not zero-fill

## Configuration

`configs/params.yml` defines all analysis parameters:

- 7 HRSN measure names + labels
- 7 chronic disease outcome names + labels
- Demographic covariates
- Race/ethnicity thresholds (majority-Black ≥ 50%, majority-White ≥ 60%, majority-Hispanic ≥ 50%)
- Bootstrap iterations (500 for OB, 1000 for mediation)
- FDR threshold (0.05)

To run sensitivity analyses with different thresholds (e.g., 40% or 60% majority-Black), edit `configs/params.yml` and re-run `scripts/analyze/44_ob_threshold_sensitivity.py`.

## Library structure

`src/hrsn_analysis/` is an installed Python package (`pip install -e .`) providing:

- `paths.py` — canonical PATHS dict for `raw/`, `processed/`, `final/`, `outputs/`
- `io_utils.py` — `load_parquet`, `save_csv`, file-state validators
- `logging_utils.py` — uniform logging across scripts
- `regression_utils.py` — cluster-robust OLS wrapper with FDR
- `survey_utils.py` — BRFSS survey-weighted estimators (PSU clustering, weight normalization)

## Reporting reproduction issues

If you can't reproduce a number cited in the manuscripts:

1. Check that `data/processed/merged_tracts.parquet` has shape (60,156, 38)
2. Check that `data/final/results_matrix.csv` has 49 rows and 44 FDR-significant
3. Open an issue on this repository with the script, the expected value, and the value you got
