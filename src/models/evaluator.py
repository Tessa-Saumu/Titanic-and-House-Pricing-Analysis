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
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    mean_squared_error, mean_absolute_error, r2_score
)
from src.config import ROOT_DIR

logger = logging.getLogger(__name__)
REPORTS_DIR = ROOT_DIR / "reports" / "runs"


def evaluate_classification(y_true: np.ndarray, y_pred: np.ndarray, y_prob: Optional[np.ndarray] = None, return_dict: bool = False) -> Optional[Tuple[Dict[str, float], str]]:
    """
    Evaluates a classification model and prints business-readable metrics.
    
    Args:
        y_true (np.ndarray): The ground truth labels.
        y_pred (np.ndarray): The predicted labels.
        y_prob (np.ndarray, optional): The predicted probabilities for the positive class (used for ROC-AUC). Defaults to None.
        return_dict (bool, optional): Whether to return the metrics and insights. Defaults to False.
        
    Returns:
        Optional[Tuple[Dict[str, float], str]]: A dictionary of metrics and the generated insight string, if return_dict is True.
    """
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred)
    }
    
    if y_prob is not None:
        metrics["roc_auc"] = roc_auc_score(y_true, y_prob)
    
    print("--- Classification Evaluation ---")
    print(f"Accuracy:  {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall:    {metrics['recall']:.4f}")
    print(f"F1-Score:  {metrics['f1']:.4f}")
    if y_prob is not None:
        print(f"ROC-AUC:   {metrics['roc_auc']:.4f}")
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
    Evaluates a regression model, prints metrics (MAE, MSE, RMSE, R2), and returns dynamic insights.
    
    Args:
        y_true (np.ndarray): The ground truth target values.
        y_pred (np.ndarray): The predicted target values.
        return_dict (bool, optional): Whether to return the metrics and insights. Defaults to False.
        
    Returns:
        Optional[Tuple[Dict[str, float], str]]: A dictionary of metrics and the generated insight string, if return_dict is True.
    """
    mse = mean_squared_error(y_true, y_pred)
    metrics = {
        "mae": mean_absolute_error(y_true, y_pred),
        "mse": mse,
        "rmse": np.sqrt(mse),
        "r2": r2_score(y_true, y_pred)
    }
    
    print("--- Regression Evaluation ---")
    print(f"MAE:  {metrics['mae']:,.2f}")
    print(f"MSE:  {metrics['mse']:,.2f}")
    print(f"RMSE: {metrics['rmse']:,.2f}")
    print(f"R²:   {metrics['r2']:.4f}")
    print("-" * 29)
    
    insight = f"The model explains {metrics['r2'] * 100:.1f}% of the variance in property prices. Predictions deviate by an average of ${metrics['mae']:,.2f} absolute error."
    
    print(f"Insight: {insight}")
    
    if return_dict:
        return metrics, insight


def save_training_report(domain: str, model_name: str, metrics: Dict[str, float], insight: str, data_shape: tuple) -> None:
    """
    Generates and saves a timestamped training report to simulate an MLOps audit trail.
    
    Args:
        domain (str): The business domain (e.g., 'housing', 'titanic').
        model_name (str): The name of the trained algorithm.
        metrics (Dict[str, float]): The evaluation metrics dictionary.
        insight (str): The generated dynamic insight.
        data_shape (tuple): The shape of the training dataset.
        
    Returns:
        None
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
            if value > 1000:
                f.write(f"- {key.upper()}: {value:,.2f}\n")
            else:
                f.write(f"- {key.upper()}: {value:.4f}\n")
                
        f.write(f"\nInsight:\n{insight}\n")
        
    print(f"Training report saved to {report_path.relative_to(ROOT_DIR)}")