"""
Phase B, Step 12: Download BRFSS 2023 data.

Downloads the BRFSS 2023 SAS Transport file from CDC (~200MB unzipped).
Verifies key variables exist.

Output:
  - data/raw/brfss/LLCP2023.XPT
"""
import sys
import zipfile
import tempfile
from pathlib import Path

import requests
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from hrsn_analysis.paths import PATHS, ensure_dirs
from hrsn_analysis.io_utils import load_params
from hrsn_analysis.logging_utils import setup_logging, get_logger

setup_logging()
logger = get_logger(__name__)


def download_brfss():
    """Download BRFSS 2023 SAS Transport file from CDC."""
    ensure_dirs()
    params = load_params()
    brfss_params = params["brfss"]

    brfss_dir = PATHS["brfss_raw"]
    brfss_dir.mkdir(parents=True, exist_ok=True)

    xpt_path = brfss_dir / brfss_params["xpt_filename"]
    url = brfss_params["download_url"]

    # Check if already downloaded
    if xpt_path.exists():
        size_mb = xpt_path.stat().st_size / (1024 * 1024)
        logger.info(f"BRFSS file already exists: {xpt_path} ({size_mb:.0f} MB)")
        logger.info("Skipping download. Delete file to re-download.")
        return xpt_path

    logger.info(f"Downloading BRFSS 2023 from: {url}")
    logger.info("This file is ~80MB compressed, ~200MB unzipped. May take a few minutes.")

    # Download with progress bar
    zip_path = brfss_dir / "LLCP2023XPT.zip"
    response = requests.get(url, stream=True, timeout=300)
    response.raise_for_status()

    total_size = int(response.headers.get("content-length", 0))
    with open(zip_path, "wb") as f:
        with tqdm(total=total_size, unit="B", unit_scale=True, desc="Downloading BRFSS") as pbar:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
                pbar.update(len(chunk))

    logger.info(f"Downloaded: {zip_path} ({zip_path.stat().st_size / (1024*1024):.0f} MB)")

    # Extract
    logger.info("Extracting ZIP archive...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(brfss_dir)
        logger.info(f"Extracted files: {zf.namelist()}")

    # Verify the XPT file exists (CDC zips sometimes have trailing spaces)
    if not xpt_path.exists():
        # Try finding the file with glob (handles trailing spaces, case differences)
        candidates = list(brfss_dir.glob("*.XPT*")) + list(brfss_dir.glob("*.xpt*"))
        candidates = [f for f in candidates if f.suffix.strip().upper() == ".XPT" or
                       f.name.strip().upper().endswith(".XPT")]
        if candidates:
            actual = candidates[0]
            actual.rename(xpt_path)
            logger.info(f"Renamed '{actual.name}' → {xpt_path.name}")
        else:
            raise FileNotFoundError(f"Expected {xpt_path} not found after extraction")

    size_mb = xpt_path.stat().st_size / (1024 * 1024)
    logger.info(f"BRFSS XPT file ready: {xpt_path} ({size_mb:.0f} MB)")

    # Quick verification: check that key variables exist
    logger.info("Verifying key variables...")
    import pandas as pd
    # Read just the first few rows to check columns
    df_check = pd.read_sas(xpt_path, format="xport", chunksize=100)
    chunk = next(df_check)
    columns = set(chunk.columns)

    expected_vars = [
        brfss_params["weight_var"],
        brfss_params["strata_var"],
        brfss_params["psu_var"],
    ]
    # Add HRSN variables
    for hrsn_info in brfss_params["hrsn_variables"].values():
        expected_vars.append(hrsn_info["brfss_var"])
    # Add outcome variables
    for out_info in brfss_params["outcome_variables"].values():
        expected_vars.append(out_info["brfss_var"])

    found = [v for v in expected_vars if v in columns]
    missing = [v for v in expected_vars if v not in columns]

    logger.info(f"  Found {len(found)}/{len(expected_vars)} expected variables")
    if missing:
        logger.warning(f"  Missing variables: {missing}")
        logger.warning("  These may be in a different module/year. Check BRFSS codebook.")
    else:
        logger.info("  All expected variables present")

    logger.info(f"  Total columns in file: {len(columns)}")

    # Clean up zip
    zip_path.unlink()
    logger.info(f"Removed ZIP file: {zip_path}")

    return xpt_path


if __name__ == "__main__":
    print("=" * 70)
    print("PHASE B: Download BRFSS 2023")
    print("=" * 70)
    path = download_brfss()
    print(f"\nDone. BRFSS data at: {path}")
