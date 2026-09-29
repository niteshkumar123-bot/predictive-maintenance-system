import os
import json
import logging
from pathlib import Path
import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("PredictiveMaintenance")

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_RAW_DIR = ROOT_DIR / "data" / "raw"
DATA_PROC_DIR = ROOT_DIR / "data" / "processed"
MODELS_DIR = ROOT_DIR / "models"
REPORTS_DIR = ROOT_DIR / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

def ensure_directories():
    """Ensure all required project directories exist."""
    for d in [DATA_RAW_DIR, DATA_PROC_DIR, MODELS_DIR, REPORTS_DIR, FIGURES_DIR]:
        d.mkdir(parents=True, exist_ok=True)
    logger.info("All project directories verified/created.")

def save_json(data: dict, filepath: Path):
    """Save dictionary to JSON file with formatting."""
    with open(filepath, "w") as f:
        json.dump(data, f, indent=4)
    logger.info(f"Saved JSON metrics to {filepath}")

def load_json(filepath: Path) -> dict:
    """Load dictionary from JSON file."""
    if not filepath.exists():
        return {}
    with open(filepath, "r") as f:
        return json.load(f)
