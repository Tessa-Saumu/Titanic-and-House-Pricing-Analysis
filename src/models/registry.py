"""
Model Registry module.
Handles the serialization (saving) and deserialization (loading) of trained ML pipelines.
"""

import joblib
import datetime
from pathlib import Path
from src.config import ROOT_DIR

# Define standard model directory
MODELS_DIR = ROOT_DIR / "models"

def save_model(pipeline, domain: str, model_name: str) -> None:
    """
    Saves a trained pipeline to disk with both a timestamp and a 'latest' alias.
    
    Args:
        pipeline: The trained Scikit-Learn pipeline.
        domain: The project domain (e.g., 'titanic', 'housing').
        model_name: The algorithm used (e.g., 'logistic_regression').
    """
    # Ensure directory exists
    domain_dir = MODELS_DIR / domain
    domain_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate timestamp (e.g., 20260729_2135)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M")
    
    # Define file paths
    timestamped_path = domain_dir / f"{model_name}_{timestamp}.pkl"
    latest_path = domain_dir / "latest_model.pkl"
    
    # Save the files
    joblib.dump(pipeline, timestamped_path)
    joblib.dump(pipeline, latest_path)
    
    print(f"Model saved successfully to {timestamped_path}")
    print(f"Alias updated: {latest_path}")

def load_latest_model(domain: str):
    """Loads the latest model for a given domain."""
    latest_path = MODELS_DIR / domain / "latest_model.pkl"
    if not latest_path.exists():
        raise FileNotFoundError(f"No model found at {latest_path}")
    return joblib.load(latest_path)