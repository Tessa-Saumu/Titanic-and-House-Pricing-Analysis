import pandas as pd
from src.config import RAW_DATA_DIR

def load_raw_data(filename: str) -> pd.DataFrame:
    """
    Loads a raw CSV dataset from the datasets/raw/ directory.
    """
    file_path = RAW_DATA_DIR / filename
    if not file_path.exists():
        raise FileNotFoundError(f"Dataset not found at {file_path}")
    
    return pd.read_csv(file_path)