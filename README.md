# CDC PLACES HRSN Analysis — Reproducibility Repository

This repository contains the analysis code, intermediate datasets, and final results matrices used in our two-paper analysis of the CDC PLACES 2024 Health-Related Social Needs (HRSN) measures across 60,156 US census tracts.

It accompanies:

1. **Paper 1 (measurement)** — Measurement structure and predictive validity of the 7 PLACES HRSN measures; validation against independent ACS and CMS Medicare claims data; quantification of shared-modeling inflation (the "V1/V3 attenuation framework").
2. **Paper 2 (disparities)** — Cross-level Oaxaca-Blinder decomposition of Black-White chronic disease disparities at both the ecological (PLACES) and individual (BRFSS) levels.

## What's in this repo

```
scripts/        44 analysis scripts (collect/, process/, analyze/, visualize/)
src/            hrsn_analysis Python package (paths, I/O, regression, survey utilities)
configs/        params.yml — analysis parameters (HRSN measures, outcomes, covariates, thresholds)
data/processed/ Merged tract-level analytic dataset and processed inputs (parquet)
data/final/     49+ result CSVs/parquet — every number cited in the manuscripts
data/raw/       ACQUISITION.md — full provenance for every external data source
Makefile        Pipeline orchestration (make collect / process / analyze)
environment.yml Conda environment specification
pyproject.toml  Python package metadata
REPRODUCE.md    Step-by-step reproduction guide
```

## What's NOT in this repo

- **Raw data files** (~1.8 GB total: BRFSS, SMART BRFSS, ACS shapefiles, CDC PLACES, SVI, CMS MMD, etc.). See `data/raw/ACQUISITION.md` and `scripts/collect/*.py` for download URLs and instructions. Most sources are public-domain US Government datasets; a few (Map the Meal Gap, Eviction Lab) require free registration.
- **Manuscripts and supplementary docs** — Available upon publication or by request to the corresponding author.

## Quick start

```bash
# 1. Create environment
conda env create -f environment.yml
conda activate hrsn-analysis
pip install -e .

# 2. To verify the published numbers from the processed dataset (no download required)
python -c "
import pandas as pd
print(pd.read_csv('data/final/results_matrix.csv').shape)            # (49, 21)
print(pd.read_csv('data/final/svi_incremental_validity.csv').head()) # Table 3 in Paper 1
print(pd.read_csv('data/final/ob_decomposition_ecological.csv').head())  # Table 1 in Paper 2
"

# 3. To re-run analyses from the merged dataset (data/processed/merged_tracts.parquet)
make analyze       # primary regressions, mutually adjusted, etc.

# 4. To re-run everything end-to-end (requires raw data download; see REPRODUCE.md)
make collect       # downloads ~1.8 GB of raw data
make process       # builds data/processed/merged_tracts.parquet
make analyze
```

See **REPRODUCE.md** for the full reproduction workflow.

## Key results (one-line summary)

- 7 HRSN measures resolve into 2 factors: **Material Hardship** (α = 0.97) and **Social Isolation** (α = 0.90), split-half CFA Tucker's congruence = 0.9999.
- HRSN adds 9–17 pp of R² beyond CDC SVI for 7 chronic disease outcomes (OOS R² = 0.81 vs 0.70; all Wald *P* < .001).
- 44 of 49 HRSN × disease associations FDR-significant; strongest = food insecurity → diabetes (β = 0.73).
- V1 (PLACES → ACS-proxy) concordance with primary = 50% (chance); V3 (both sides independent: ACS → CMS Medicare claims) = 76%; median primary-to-V3 inflation ratio = 2.0×.
- Tract-level HRSN differences statistically account for 23–99% of Black-White chronic disease gaps at primary scale, 12–50% after V3 calibration; cross-level amplification survives calibration only for diabetes (2.4×).

## Data sources (high-level)

| Source | Coverage | Used for |
|---|---|---|
| CDC PLACES 2024 | 60,156 tracts, 39 states + DC | HRSN exposures + disease outcomes (primary) |
| CDC/ATSDR SVI 2022 | All US tracts | SVI incremental validity comparator |
| ACS 5-year 2022 | All US tracts | Demographic covariates + V3 direct proxies (SNAP, rent burden, no-vehicle, living alone) |
| CMS Mapping Medicare Disparities | 3,241 counties | V3 independent disease outcomes (Medicare claims) |
| BRFSS 2023 | 145,248 Black/White respondents | Individual-level Oaxaca-Blinder + BRFSS validation |
| Map the Meal Gap (Feeding America) | 3,153 counties | V5 food insecurity proxy |
| USDA Food Access Atlas | 72,531 tracts | V5 low-food-access proxy |
| Eviction Lab V1 | 1,422 counties | V5 eviction filing rate |
| Chetty et al. Social Capital Atlas | 3,089 counties | V5 social capital |
| CDC WONDER (D158) | 3,018 counties | Mortality validation (V4) |

Full provenance, URLs, file sizes, and access notes: see `data/raw/ACQUISITION.md`.

## Citation

If you use this code or data, please cite:

> *[Paper 1 citation — to be added at publication]*
> *[Paper 2 citation — to be added at publication]*

## License

MIT License — see [LICENSE](LICENSE). All US Government data referenced (CDC, Census, CMS) is public domain; third-party datasets retain their own licenses (see `data/raw/ACQUISITION.md`).

## Contact

For questions about the analysis, raw data acquisition, or reproduction, please open an issue on this repository.
