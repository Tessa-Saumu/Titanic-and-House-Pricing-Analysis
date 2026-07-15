from pathlib import Path

# This dynamically finds the absolute path of the repository root
ROOT_DIR = Path(__file__).resolve().parent.parent

# Define standard data directories
RAW_DATA_DIR = ROOT_DIR / "datasets" / "raw"
PROCESSED_DATA_DIR = ROOT_DIR / "datasets" / "processed"