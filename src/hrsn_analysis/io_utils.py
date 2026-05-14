"""Data I/O utilities with logging."""
from pathlib import Path

import pandas as pd
import yaml

from .paths import PATHS
from .logging_utils import get_logger

logger = get_logger(__name__)


def load_csv(filepath, **kwargs):
    df = pd.read_csv(filepath, **kwargs)
    logger.info(f"Loaded {filepath.name}: {len(df):,} rows × {len(df.columns)} cols")
    return df


def save_csv(df, filepath, **kwargs):
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(filepath, index=False, **kwargs)
    logger.info(f"Saved {filepath.name}: {len(df):,} rows")


def load_parquet(filepath, **kwargs):
    df = pd.read_parquet(filepath, **kwargs)
    logger.info(f"Loaded {filepath.name}: {len(df):,} rows × {len(df.columns)} cols")
    return df


def save_parquet(df, filepath, **kwargs):
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(filepath, index=False, **kwargs)
    logger.info(f"Saved {filepath.name}: {len(df):,} rows")


def load_params():
    params_file = PATHS["configs"] / "params.yml"
    with open(params_file) as f:
        return yaml.safe_load(f)
