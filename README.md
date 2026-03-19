# Which Social Needs Matter Most for Which Diseases?

Census tract-level analysis linking CDC PLACES 2024 health-related social needs (HRSN) measures to chronic disease prevalence.

## Project Status

**Phase 0: Project Setup** — Complete ✅

## Setup

```bash
make setup
conda activate hrsn-analysis
```

## Data Sources

- **CDC PLACES 2024** — HRSN and chronic disease prevalence at census tract level
- **Census ACS 5-Year (2022)** — Demographic covariates
- **TIGER/Line Shapefiles** — Census tract boundaries for mapping

See `data/raw/ACQUISITION.md` for detailed data source documentation.

## Pipeline

```bash
make collect    # Download raw data
make process    # Reshape, merge, create master dataset
make eda        # Exploratory data analysis
make analyze    # Run regressions
make visualize  # Generate figures
make sensitivity # Sensitivity analyses
```

## License

MIT
