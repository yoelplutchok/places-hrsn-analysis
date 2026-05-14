.PHONY: setup collect process analyze visualize sensitivity validate clean all factor brfss spatial enhancements validation

setup:
	conda env create -f environment.yml || conda env update -f environment.yml
	pip install -e .
	@echo "Activate with: conda activate hrsn-analysis"

collect:
	python scripts/collect/01_download_places.py
	python scripts/collect/02_download_census.py
	python scripts/collect/03_download_shapefiles.py

process:
	python scripts/process/04_reshape_places.py
	python scripts/process/05_process_census.py
	python scripts/process/06_merge_datasets.py

eda:
	python scripts/analyze/07_eda.py

analyze:
	python scripts/analyze/08_run_regressions.py
	python scripts/analyze/09_mutually_adjusted.py

visualize:
	python scripts/visualize/11_heatmap.py
	python scripts/visualize/12_forest_plots.py
	python scripts/visualize/13_maps.py
	python scripts/visualize/14_supplementary.py

sensitivity:
	python scripts/analyze/10_sensitivity.py

# --- Enhancement targets ---

factor:
	python scripts/analyze/10_factor_analysis.py
	python scripts/analyze/11_factor_regressions.py

brfss:
	python scripts/collect/12_download_brfss.py
	python scripts/process/13_process_brfss.py
	python scripts/analyze/14_brfss_regressions.py

spatial:
	python scripts/analyze/15_spatial_autocorrelation.py
	python scripts/analyze/15b_spatial_error_sensitivity.py
	python scripts/analyze/16_gwr_analysis.py

svi:
	python scripts/collect/17_download_svi.py
	python scripts/analyze/18_svi_comparison.py

decomposition:
	python scripts/analyze/19_disparity_decomposition.py

heterogeneity:
	python scripts/analyze/20_heterogeneity_explained.py

contextual:
	python scripts/analyze/21_contextual_effects.py

validation:
	python scripts/collect/22_download_cms_medicare.py
	python scripts/analyze/23_validation_framework.py

novel: svi decomposition heterogeneity contextual

enhancements: factor brfss spatial novel validation

dashboard:
	cd dashboard && streamlit run app.py

clean:
	rm -rf data/processed/*
	rm -rf data/final/*
	rm -rf outputs/*

all: collect process eda analyze visualize sensitivity enhancements
