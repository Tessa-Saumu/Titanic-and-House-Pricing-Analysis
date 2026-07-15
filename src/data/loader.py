"""
Data loading utilities.
Handles the safe ingestion of raw and processed datasets, ensuring
paths are resolved correctly regardless of where the script is executed.
"""

import pandas as pd
from src.config import RAW_DATA_DIR
from src.config import PROCESSED_DATA_DIR

def load_raw_data(filename: str) -> pd.DataFrame:
    """
    Loads a raw CSV dataset from the datasets/raw/ directory.
    """
    file_path = RAW_DATA_DIR / filename
    if not file_path.exists():
        raise FileNotFoundError(f"Dataset not found at {file_path}")
    
    return pd.read_csv(file_path)


def save_processed_data(df: pd.DataFrame, filename: str) -> None:
    """
    Saves a cleaned DataFrame to the datasets/processed/ directory.
    """
    file_path = PROCESSED_DATA_DIR / filename
    df.to_csv(file_path, index=False)
    print(f"Successfully saved {filename} to processed directory.")