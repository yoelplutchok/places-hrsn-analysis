# Data Acquisition Log

This document tracks all data sources used in the PLACES HRSN analysis.

## Data Sources

### 1. CDC PLACES 2024 — Census Tract Data (GIS-Friendly Format)

| Field | Value |
|-------|-------|
| **File** | `places/places_tract_2024.csv` |
| **Source** | CDC PLACES 2024 Release, GIS-friendly format |
| **URL** | `https://data.cdc.gov/api/views/yjkw-uj5s/rows.csv?accessType=DOWNLOAD` |
| **Download Date** | 2026-02-11 |
| **File Size** | 64 MB |
| **Rows** | 83,522 census tracts |
| **Columns** | 88 (39 measures × 2 columns each + metadata) |
| **Format** | Wide — one row per tract, one column per measure |
| **Prevalence Type** | Crude prevalence only (`_CrudePrev`); age-adjusted not available at tract level |
| **HRSN Measures** | 7 (LONELINESS, FOODSTAMP, FOODINSECU, HOUSINSECU, SHUTUTILITY, LACKTRPT, EMOTIONSPT) |
| **License** | Public domain (US Government work) |
| **Notes** | HRSN measures available in ~39 states + DC only; other measures available nationally |

### 2. CDC PLACES — Data Dictionary

| Field | Value |
|-------|-------|
| **File** | `places/places_data_dictionary.json` |
| **Source** | CDC PLACES Socrata API |
| **URL** | `https://data.cdc.gov/resource/m35w-spkz.json` |
| **Download Date** | 2026-02-11 |
| **Records** | 44 measure definitions |
| **HRSN (SOCLNEED) measures** | 7 confirmed |
| **License** | Public domain (US Government work) |

### 3. Census ACS 5-Year Estimates (2022) — Tract Demographics

| Field | Value |
|-------|-------|
| **File** | `census/acs_tract_demographics_2022.csv` |
| **Source** | US Census Bureau, American Community Survey 5-Year Estimates (2018-2022) |
| **Access Method** | `censusdis` Python library (Census API wrapper) |
| **Download Date** | 2026-02-11 |
| **File Size** | 6.2 MB |
| **Rows** | 84,415 census tracts (50 states + DC) |
| **Columns** | 17 (STATE, COUNTY, TRACT + 14 ACS variables) |
| **License** | Public domain (US Government work) |

**ACS Variables Downloaded:**

| Variable | Description |
|----------|-------------|
| B01003_001E | Total population |
| B01002_001E | Median age |
| B02001_002E | White alone |
| B02001_003E | Black/African American alone |
| B03003_003E | Hispanic/Latino |
| B17001_002E | Below poverty level |
| B19013_001E | Median household income |
| B15003_001E | Population 25+ (education denominator) |
| B15003_017E | High school diploma |
| B15003_018E | GED or equivalent |
| B15003_022E | Bachelor's degree |
| B15003_023E | Master's degree |
| B15003_024E | Professional school degree |
| B15003_025E | Doctorate degree |

### 4. Census TIGER/Line — Tract Boundary Geometries (2022)

| Field | Value |
|-------|-------|
| **File** | `geo/us_tracts_2022.gpkg` |
| **Source** | US Census Bureau TIGER/Line via censusdis |
| **Access Method** | `censusdis` with `with_geometry=True` |
| **Download Date** | 2026-02-11 |
| **File Size** | 103.8 MB |
| **Tracts** | 84,415 (50 states + DC) |
| **CRS** | EPSG:4269 (NAD83) |
| **Format** | GeoPackage (.gpkg) |
| **GEOID Format** | 11-digit (2-digit state + 3-digit county + 6-digit tract) |
| **License** | Public domain (US Government work) |

## Download Scripts

| Script | Datasets |
|--------|----------|
| `scripts/collect/01_download_places.py` | PLACES tract data + data dictionary |
| `scripts/collect/02_download_census.py` | Census ACS demographics |
| `scripts/collect/03_download_shapefiles.py` | TIGER/Line tract geometries |

## Data Quality Notes

- **Total population range**: 0 to 38,907 (median: 3,754)
- **Median household income range**: $2,499 to $250,001 (median: $71,944)
- **GEOID consistency**: All files use 11-digit census tract FIPS codes
- **Tract count alignment**: Census (84,415) > PLACES (83,522) — some tracts in Census but not PLACES
- **HRSN availability**: Only in states participating in BRFSS HRSN module (~39 states + DC)
