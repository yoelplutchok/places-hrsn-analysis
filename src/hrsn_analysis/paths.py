"""Centralized path management for the project."""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

PATHS = {
    # Data directories
    "raw": PROJECT_ROOT / "data" / "raw",
    "processed": PROJECT_ROOT / "data" / "processed",
    "final": PROJECT_ROOT / "data" / "final",
    "places_raw": PROJECT_ROOT / "data" / "raw" / "places",
    "census_raw": PROJECT_ROOT / "data" / "raw" / "census",
    "geo_raw": PROJECT_ROOT / "data" / "raw" / "geo",
    "brfss_raw": PROJECT_ROOT / "data" / "raw" / "brfss",
    "svi_raw": PROJECT_ROOT / "data" / "raw" / "svi",
    # Output directories
    "figures": PROJECT_ROOT / "outputs" / "figures",
    "maps": PROJECT_ROOT / "outputs" / "figures" / "maps",
    "tables": PROJECT_ROOT / "outputs" / "tables",
    "interactive": PROJECT_ROOT / "outputs" / "interactive",
    # Config and logs
    "configs": PROJECT_ROOT / "configs",
    "logs": PROJECT_ROOT / "logs",
}


def ensure_dirs():
    """Create all directories if they don't exist."""
    for path in PATHS.values():
        path.mkdir(parents=True, exist_ok=True)
