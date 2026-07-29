"""
Model evaluation utilities.
Calculates and formats core Machine Learning metrics, generates dynamic business 
insights, and saves timestamped training reports.
"""

import numpy as np
import logging
import datetime
from typing import Dict, Optional, Tuple
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    mean_squared_error, r2_score
)
from src.config import ROOT_DIR

logger = logging.getLogger(__name__)
REPORTS_DIR = ROOT_DIR / "reports" / "runs"


def evaluate_classification(y_true: np.ndarray, y_pred: np.ndarray, return_dict: bool = False) -> Optional[Tuple[Dict[str, float], str]]:
    """
    Evaluates a classification model and prints business-readable metrics.
    Includes dynamic insights comparing precision and recall.
    """
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred)
    }
    
    print("--- Classification Evaluation ---")
    print(f"Accuracy:  {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall:    {metrics['recall']:.4f}")
    print(f"F1-Score:  {metrics['f1']:.4f}")
    print("-" * 33)
    
    # Custom insight logic
    if metrics['recall'] > metrics['precision']:
        insight = "The model leans toward Recall (better at catching positives, higher risk of false alarms)."
    elif metrics['precision'] > metrics['recall']:
        insight = "The model leans toward Precision (conservative predictions, higher risk of missing positives)."
    else:
        insight = "Precision and Recall are perfectly balanced."
        
    print(f"Insight: {insight}")
        
    if return_dict:
        return metrics, insight


def evaluate_regression(y_true: np.ndarray, y_pred: np.ndarray, return_dict: bool = False) -> Optional[Tuple[Dict[str, float], str]]:
    """
    Evaluates a regression model, prints metrics, and returns dynamic insights.
    """
    metrics = {
        "rmse": np.sqrt(mean_squared_error(y_true, y_pred)),
        "r2": r2_score(y_true, y_pred)
    }
    
    print("--- Regression Evaluation ---")
    print(f"RMSE: {metrics['rmse']:,.2f}")
    print(f"R²:   {metrics['r2']:.4f}")
    print("-" * 29)
    
    insight = f"The model explains {metrics['r2'] * 100:.1f}% of the variance in property prices. Predictions deviate by an average of ${metrics['rmse']:,.2f} on the actual price scale."
    
    print(f"Insight: {insight}")
    
    if return_dict:
        return metrics, insight


def save_training_report(domain: str, model_name: str, metrics: Dict[str, float], insight: str, data_shape: tuple):
    """
    Generates and saves a timestamped training report to simulate an MLOps audit trail.
    """
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    safe_time = datetime.datetime.now().strftime("%Y%m%d_%H%M")
    
    report_path = REPORTS_DIR / f"{domain}_{model_name}_{safe_time}.txt"
    
    with open(report_path, "w") as f:
        f.write(f"--- Model Training Report ---\n")
        f.write(f"Date: {timestamp}\n")
        f.write(f"Domain: {domain.capitalize()}\n")
        f.write(f"Model: {model_name}\n")
        f.write(f"Training Data Shape: {data_shape}\n")
        f.write("-" * 29 + "\n\n")
        
        f.write("Metrics:\n")
        for key, value in metrics.items():
            # Format large numbers for RMSE vs standard floats for Accuracy
            if value > 1000:
                f.write(f"- {key.upper()}: {value:,.2f}\n")
            else:
                f.write(f"- {key.capitalize()}: {value:.4f}\n")
                
        f.write(f"\nInsight:\n{insight}\n")
        
    print(f"Training report saved to {report_path.relative_to(ROOT_DIR)}")