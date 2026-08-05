"""
Visualization module for the project.
Enforces a consistent 'Data Journalism' aesthetic and autosaves plots.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from matplotlib.ticker import StrMethodFormatter
from pathlib import Path
import numpy as np
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import learning_curve
import shap

JOURNALISM_PALETTE = [
    "#008fd5",  # Blue
    "#fc4f30",  # Red
    "#e5ae38",  # Gold
    "#6d904f",  # Green
    "#8b8b8b",  # Grey
    "#810f7c",  # Purple
]


def set_journalism_style():
    sns.set_theme(
        style="whitegrid",
        rc={
            "axes.facecolor": "#F0F0F0",
            "figure.facecolor": "#F0F0F0",
            "axes.edgecolor": "#F0F0F0",
            "grid.color": "white",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.titlesize": 16,
            "axes.titleweight": "bold",
            "axes.titlelocation": "left",
            "axes.labelsize": 12,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
            "lines.linewidth": 2.5,
        },
    )


def _format_axes(ax):
    ax.yaxis.set_major_formatter(StrMethodFormatter("{x:,.0f}"))

    if ax.get_xlim()[1] > 1000:
        ax.xaxis.set_major_formatter(StrMethodFormatter("{x:,.0f}"))


def _get_order(df, column):
    """Automatically determine sensible ordering."""
    s = df[column].dropna()

    if pd.api.types.is_numeric_dtype(s):
        return sorted(s.unique())

    return list(s.value_counts().index)


def _save_plot(title):
    """Automatically saves the plot to the figures directory."""
    # Navigate from src/visualization/plots.py to the root directory
    root_dir = Path(__file__).resolve().parent.parent.parent
    fig_dir = root_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    
    # Create a safe filename from the title
    safe_title = "".join([c if c.isalnum() or c.isspace() else "" for c in title])
    filename = safe_title.replace(" ", "_").lower() + ".png"
    
    plt.savefig(fig_dir / filename, dpi=300, bbox_inches="tight")


def plot_histogram(df, column, title, bins=30):

    plt.figure(figsize=(8, 5))

    ax = sns.histplot(
        data=df,
        x=column,
        bins=bins,
        kde=True,
        color=JOURNALISM_PALETTE[0],
        edgecolor="white",
    )

    plt.title(title)
    plt.xlabel(column.replace("_", " ").title())
    plt.ylabel("Frequency")

    _format_axes(ax)

    plt.tight_layout()
    _save_plot(title)
    plt.show()


def plot_bar_chart(df, column, title, order=None):

    if order is None:
        order = _get_order(df, column)

    plt.figure(figsize=(8, 5))

    sns.countplot(
        data=df,
        x=column,
        order=order,
        hue=column,
        palette=JOURNALISM_PALETTE[: len(order)],
        legend=False,
    )

    plt.title(title)
    plt.xlabel(column.replace("_", " ").title())
    plt.ylabel("Count")

    plt.tight_layout()
    _save_plot(title)
    plt.show()


def plot_box(df, cat_col, num_col, title, order=None):

    if order is None:
        order = _get_order(df, cat_col)

    plt.figure(figsize=(8, 5))

    ax = sns.boxplot(
        data=df,
        x=cat_col,
        y=num_col,
        order=order,
        hue=cat_col,
        palette=JOURNALISM_PALETTE[: len(order)],
        legend=False,
    )

    plt.title(title)
    plt.xlabel(cat_col.replace("_", " ").title())
    plt.ylabel(num_col.replace("_", " ").title())

    _format_axes(ax)

    plt.tight_layout()
    _save_plot(title)
    plt.show()


def plot_categorical_target_rate(df, cat_col, target_col, title, order=None):

    if order is None:
        order = _get_order(df, cat_col)

    plt.figure(figsize=(8, 5))

    palette = JOURNALISM_PALETTE[: len(order)]

    ax = sns.barplot(
        data=df,
        x=cat_col,
        y=target_col,
        order=order,
        hue=cat_col,
        palette=palette,
        estimator="mean",
        errorbar=None,
        legend=False,
    )

    plt.title(title)
    plt.xlabel(cat_col.replace("_", " ").title())
    plt.ylabel(f"{target_col.replace('_', ' ').title()} Rate")

    ax.yaxis.set_major_formatter(StrMethodFormatter("{x:.0%}"))

    counts = df[cat_col].value_counts().reindex(order)

    for patch, count in zip(ax.patches, counts):

        ax.annotate(
            f"n={int(count)}",
            (
                patch.get_x() + patch.get_width() / 2,
                patch.get_height() * 0.05,
            ),
            ha="center",
            va="bottom",
            color="white",
            fontsize=9,
            fontweight="bold",
        )

    plt.tight_layout()
    _save_plot(title)
    plt.show()


def plot_correlation_heatmap(df, title):

    plt.figure(figsize=(10, 6))

    corr = df.select_dtypes(include="number").corr()

    sns.heatmap(
        corr,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        center=0,
        linewidths=0.5,
        linecolor="white",
    )

    plt.title(title)

    plt.tight_layout()
    _save_plot(title)
    plt.show()
    
from sklearn.metrics import confusion_matrix

def plot_confusion_matrix(y_true, y_pred, title="Confusion Matrix"):
    """Plots a stylized confusion matrix heatmap."""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    
    # Using 'Blues' to match JOURNALISM_PALETTE[0] aesthetic
    ax = sns.heatmap(
        cm, 
        annot=True, 
        fmt='d',
        cmap="Blues", 
        cbar=False,
        linewidths=1,
        linecolor="white",
        square=True,
        annot_kws={"size": 14, "weight": "bold"}
    )
    
    plt.title(title)
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    
    plt.tight_layout()
    _save_plot(title)
    plt.show()

def plot_actual_vs_predicted(y_true, y_pred, title="Actual vs. Predicted"):
    """Plots regression actuals vs predictions with a baseline of perfect prediction."""
    plt.figure(figsize=(8, 5))
    
    plt.scatter(
        y_true, 
        y_pred, 
        alpha=0.6, 
        color=JOURNALISM_PALETTE[0],
        edgecolor="white",
        s=80
    )
    
    # Perfect prediction diagonal line
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    plt.plot(
        [min_val, max_val], 
        [min_val, max_val], 
        color=JOURNALISM_PALETTE[1], 
        linestyle="--", 
        linewidth=2,
        label="Perfect Prediction"
    )
    
    plt.title(title)
    plt.xlabel("Actual Values")
    plt.ylabel("Predicted Values")
    
    # Apply existing axis formatter for large numbers
    ax = plt.gca()
    _format_axes(ax)
    ax.xaxis.set_major_formatter(StrMethodFormatter("{x:,.0f}"))
    
    plt.legend()
    plt.tight_layout()
    _save_plot(title)
    plt.show()

def plot_learning_curve(estimator, X, y, title="Learning Curve"):
    """
    Plots the learning curve to diagnose overfitting or underfitting.
    
    Args:
        estimator: The scikit-learn estimator/pipeline.
        X (np.ndarray or pd.DataFrame): Training features.
        y (np.ndarray or pd.Series): Target variable.
        title (str): Title for the plot.
    """
    plt.figure(figsize=(8, 5))
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=5, n_jobs=-1, 
        train_sizes=np.linspace(0.1, 1.0, 10), scoring="accuracy" if len(np.unique(y)) == 2 else "r2"
    )
    
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    
    plt.fill_between(train_sizes, train_scores_mean - train_scores_std, train_scores_mean + train_scores_std, alpha=0.1, color=JOURNALISM_PALETTE[0])
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std, test_scores_mean + test_scores_std, alpha=0.1, color=JOURNALISM_PALETTE[1])
    
    plt.plot(train_sizes, train_scores_mean, 'o-', color=JOURNALISM_PALETTE[0], label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color=JOURNALISM_PALETTE[1], label="Cross-validation score")
    
    plt.title(title)
    plt.xlabel("Training Examples")
    plt.ylabel("Score")
    plt.legend(loc="best")
    plt.tight_layout()
    _save_plot(title)
    plt.show()

def plot_roc_curve(y_true, y_probs, title="ROC Curve"):
    """
    Plots the Receiver Operating Characteristic (ROC) curve.
    
    Args:
        y_true (np.ndarray): Ground truth binary labels.
        y_probs (np.ndarray): Predicted probabilities for the positive class.
        title (str): Title for the plot.
    """
    fpr, tpr, _ = roc_curve(y_true, y_probs)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(7, 6))
    plt.plot(fpr, tpr, color=JOURNALISM_PALETTE[0], lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], color=JOURNALISM_PALETTE[4], lw=2, linestyle='--')
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    plt.legend(loc="lower right")
    plt.tight_layout()
    _save_plot(title)
    plt.show()

def plot_shap_summary(model, X_transformed, feature_names, title="SHAP Feature Importance"):
    """
    Plots a SHAP summary plot for model interpretability.
    
    Args:
        model: The trained inner estimator (not the full pipeline).
        X_transformed (np.ndarray): The preprocessed feature matrix.
        feature_names (list): The list of feature names.
        title (str): The plot title for saving.
    """
    # Create explainer based on model type
    if type(model).__name__ in ['RandomForestRegressor', 'RandomForestClassifier', 'XGBRegressor', 'XGBClassifier']:
        explainer = shap.TreeExplainer(model)
    else:
        # Fallback to KernelExplainer for SVM/KNN or LinearExplainer for others.
        # Using a summary sample to speed up KernelExplainer if needed
        background = shap.sample(X_transformed, 100)
        explainer = shap.Explainer(model.predict, background)
        
    shap_values = explainer(X_transformed)
    
    plt.figure(figsize=(10, 6))
    plt.title(title, pad=20, fontsize=16, fontweight='bold')
    shap.summary_plot(shap_values, X_transformed, feature_names=feature_names, show=False)
    
    plt.tight_layout()
    _save_plot(title)
    plt.show()