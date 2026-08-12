"""
Visualization module for the project.

Enforces a consistent "Data Journalism" aesthetic and autosaves plots
to the figures directory, optionally organized into subfolders.
"""

import logging
from pathlib import Path
from typing import Any, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import shap
from matplotlib.ticker import StrMethodFormatter
from sklearn.metrics import auc, confusion_matrix, roc_curve
from sklearn.model_selection import learning_curve


logger = logging.getLogger(__name__)


JOURNALISM_PALETTE = [
    "#008fd5",  # Blue
    "#fc4f30",  # Red
    "#e5ae38",  # Gold
    "#6d904f",  # Green
    "#8b8b8b",  # Grey
    "#810f7c",  # Purple
]


def set_journalism_style() -> None:
    """Set the global Seaborn theme to a Data Journalism aesthetic."""
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


def _format_axes(ax: plt.Axes) -> None:
    """Format axes with commas for large numeric values."""
    ax.yaxis.set_major_formatter(StrMethodFormatter("{x:,.0f}"))

    if ax.get_xlim()[1] > 1000:
        ax.xaxis.set_major_formatter(StrMethodFormatter("{x:,.0f}"))


def _get_order(df: pd.DataFrame, column: str) -> List[Any]:
    """Automatically determine a sensible ordering for a categorical column."""
    series = df[column].dropna()

    if pd.api.types.is_numeric_dtype(series):
        return sorted(series.unique())

    return list(series.value_counts().index)


def _save_plot(title: str, subfolder: str = "") -> None:
    """
    Save the current figure to the figures directory.

    Args:
        title: Plot title used to generate the filename.
        subfolder: Optional subdirectory inside the figures directory.
    """
    # Navigate from src/visualization/plots.py to the project root.
    root_dir = Path(__file__).resolve().parent.parent.parent

    fig_dir = root_dir / "figures"

    if subfolder:
        fig_dir = fig_dir / Path(subfolder)

    fig_dir.mkdir(parents=True, exist_ok=True)

    # Create a filesystem-safe filename from the title.
    safe_title = "".join(
        character
        for character in title
        if character.isalnum() or character.isspace()
    )

    filename = safe_title.replace(" ", "_").lower() + ".png"
    file_path = fig_dir / filename

    figure = plt.gcf()
    figure.savefig(file_path, dpi=300, bbox_inches="tight")

    logger.info("Plot saved to %s", file_path)


def plot_histogram(
    df: pd.DataFrame,
    column: str,
    title: str,
    bins: int = 30,
    subfolder: str = "",
) -> None:
    """Plot a histogram with an optional KDE curve."""
    fig, ax = plt.subplots(figsize=(8, 5))

    sns.histplot(
        data=df,
        x=column,
        bins=bins,
        kde=True,
        color=JOURNALISM_PALETTE[0],
        edgecolor="white",
        ax=ax,
    )

    ax.set_title(title)
    ax.set_xlabel(column.replace("_", " ").title())
    ax.set_ylabel("Frequency")

    _format_axes(ax)

    fig.tight_layout()
    _save_plot(title, subfolder)
    plt.show()
    plt.close(fig)


def plot_bar_chart(
    df: pd.DataFrame,
    column: str,
    title: str,
    order: Optional[List[Any]] = None,
    subfolder: str = "",
) -> None:
    """Plot a categorical count bar chart."""
    if order is None:
        order = _get_order(df, column)

    fig, ax = plt.subplots(figsize=(8, 5))

    sns.countplot(
        data=df,
        x=column,
        order=order,
        hue=column,
        palette=JOURNALISM_PALETTE[: len(order)],
        legend=False,
        ax=ax,
    )

    ax.set_title(title)
    ax.set_xlabel(column.replace("_", " ").title())
    ax.set_ylabel("Count")

    fig.tight_layout()
    _save_plot(title, subfolder)
    plt.show()
    plt.close(fig)


def plot_box(
    df: pd.DataFrame,
    cat_col: str,
    num_col: str,
    title: str,
    order: Optional[List[Any]] = None,
    subfolder: str = "",
) -> None:
    """Plot a box plot of a numeric variable across categories."""
    if order is None:
        order = _get_order(df, cat_col)

    fig, ax = plt.subplots(figsize=(8, 5))

    sns.boxplot(
        data=df,
        x=cat_col,
        y=num_col,
        order=order,
        hue=cat_col,
        palette=JOURNALISM_PALETTE[: len(order)],
        legend=False,
        ax=ax,
    )

    ax.set_title(title)
    ax.set_xlabel(cat_col.replace("_", " ").title())
    ax.set_ylabel(num_col.replace("_", " ").title())

    _format_axes(ax)

    fig.tight_layout()
    _save_plot(title, subfolder)
    plt.show()
    plt.close(fig)


def plot_categorical_target_rate(
    df: pd.DataFrame,
    cat_col: str,
    target_col: str,
    title: str,
    order: Optional[List[Any]] = None,
    subfolder: str = "",
) -> None:
    """Plot the mean target rate across categorical groups."""
    if order is None:
        order = _get_order(df, cat_col)

    fig, ax = plt.subplots(figsize=(8, 5))

    palette = JOURNALISM_PALETTE[: len(order)]

    sns.barplot(
        data=df,
        x=cat_col,
        y=target_col,
        order=order,
        hue=cat_col,
        palette=palette,
        estimator="mean",
        errorbar=None,
        legend=False,
        ax=ax,
    )

    ax.set_title(title)
    ax.set_xlabel(cat_col.replace("_", " ").title())
    ax.set_ylabel(f"{target_col.replace('_', ' ').title()} Rate")

    ax.yaxis.set_major_formatter(StrMethodFormatter("{x:.0%}"))

    counts = df[cat_col].value_counts().reindex(order)

    for patch, count in zip(ax.patches, counts):
        height = patch.get_height()

        ax.annotate(
            f"n={int(count)}",
            (
                patch.get_x() + patch.get_width() / 2,
                height * 0.05,
            ),
            ha="center",
            va="bottom",
            color="white",
            fontsize=9,
            fontweight="bold",
        )

    fig.tight_layout()
    _save_plot(title, subfolder)
    plt.show()
    plt.close(fig)


def plot_correlation_heatmap(
    df: pd.DataFrame,
    title: str,
    subfolder: str = "",
) -> None:
    """Plot a correlation heatmap for numeric columns."""
    fig, ax = plt.subplots(figsize=(10, 6))

    corr = df.select_dtypes(include="number").corr()

    sns.heatmap(
        corr,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        center=0,
        linewidths=0.5,
        linecolor="white",
        ax=ax,
    )

    ax.set_title(title)

    fig.tight_layout()
    _save_plot(title, subfolder)
    plt.show()
    plt.close(fig)


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    title: str = "Confusion Matrix",
    subfolder: str = "",
) -> None:
    """Plot a stylized confusion matrix heatmap."""
    cm = confusion_matrix(y_true, y_pred)

    fig, ax = plt.subplots(figsize=(6, 5))

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar=False,
        linewidths=1,
        linecolor="white",
        square=True,
        annot_kws={"size": 14, "weight": "bold"},
        ax=ax,
    )

    ax.set_title(title)
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")

    fig.tight_layout()
    _save_plot(title, subfolder)
    plt.show()
    plt.close(fig)


def plot_actual_vs_predicted(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    title: str = "Actual vs. Predicted",
    subfolder: str = "",
) -> None:
    """Plot regression actual values against predicted values."""
    fig, ax = plt.subplots(figsize=(8, 5))

    ax.scatter(
        y_true,
        y_pred,
        alpha=0.6,
        color=JOURNALISM_PALETTE[0],
        edgecolor="white",
        s=80,
    )

    # Perfect prediction diagonal line.
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())

    ax.plot(
        [min_val, max_val],
        [min_val, max_val],
        color=JOURNALISM_PALETTE[1],
        linestyle="--",
        linewidth=2,
        label="Perfect Prediction",
    )

    ax.set_title(title)
    ax.set_xlabel("Actual Values")
    ax.set_ylabel("Predicted Values")

    _format_axes(ax)
    ax.xaxis.set_major_formatter(StrMethodFormatter("{x:,.0f}"))

    ax.legend()

    fig.tight_layout()
    _save_plot(title, subfolder)
    plt.show()
    plt.close(fig)


def plot_learning_curve(
    estimator: Any,
    X: np.ndarray | pd.DataFrame,
    y: np.ndarray | pd.Series,
    title: str = "Learning Curve",
    subfolder: str = "",
) -> None:
    """
    Plot a learning curve to diagnose overfitting or underfitting.

    Args:
        estimator: Scikit-learn estimator or pipeline.
        X: Training features.
        y: Target variable.
        title: Plot title.
        subfolder: Optional subdirectory inside the figures directory.
    """
    fig, ax = plt.subplots(figsize=(8, 5))

    scoring = "accuracy" if len(np.unique(y)) == 2 else "r2"

    train_sizes, train_scores, test_scores = learning_curve(
        estimator,
        X,
        y,
        cv=5,
        n_jobs=-1,
        train_sizes=np.linspace(0.1, 1.0, 10),
        scoring=scoring,
    )

    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)

    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    ax.fill_between(
        train_sizes,
        train_scores_mean - train_scores_std,
        train_scores_mean + train_scores_std,
        alpha=0.1,
        color=JOURNALISM_PALETTE[0],
    )

    ax.fill_between(
        train_sizes,
        test_scores_mean - test_scores_std,
        test_scores_mean + test_scores_std,
        alpha=0.1,
        color=JOURNALISM_PALETTE[1],
    )

    ax.plot(
        train_sizes,
        train_scores_mean,
        "o-",
        color=JOURNALISM_PALETTE[0],
        label="Training score",
    )

    ax.plot(
        train_sizes,
        test_scores_mean,
        "o-",
        color=JOURNALISM_PALETTE[1],
        label="Cross-validation score",
    )

    ax.set_title(title)
    ax.set_xlabel("Training Examples")
    ax.set_ylabel("Score")
    ax.legend(loc="best")

    fig.tight_layout()
    _save_plot(title, subfolder)
    plt.show()
    plt.close(fig)


def plot_roc_curve(
    y_true: np.ndarray,
    y_probs: np.ndarray,
    title: str = "ROC Curve",
    subfolder: str = "",
) -> None:
    """
    Plot the Receiver Operating Characteristic curve.

    Args:
        y_true: Ground-truth binary labels.
        y_probs: Predicted probabilities for the positive class.
        title: Plot title.
        subfolder: Optional subdirectory inside the figures directory.
    """
    fpr, tpr, _ = roc_curve(y_true, y_probs)
    roc_auc = auc(fpr, tpr)

    fig, ax = plt.subplots(figsize=(7, 6))

    ax.plot(
        fpr,
        tpr,
        color=JOURNALISM_PALETTE[0],
        lw=2,
        label=f"ROC curve (AUC = {roc_auc:.3f})",
    )

    ax.plot(
        [0, 1],
        [0, 1],
        color=JOURNALISM_PALETTE[4],
        lw=2,
        linestyle="--",
        label="Random Classifier",
    )

    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(title)
    ax.legend(loc="lower right")

    fig.tight_layout()
    _save_plot(title, subfolder)
    plt.show()
    plt.close(fig)


def plot_shap_summary(
    model: Any,
    X_transformed: np.ndarray,
    feature_names: List[str],
    title: str = "SHAP Feature Importance",
    subfolder: str = "",
) -> None:
    """
    Plot a SHAP summary plot for model interpretability.

    Args:
        model: Trained inner estimator, not the full preprocessing pipeline.
        X_transformed: Preprocessed feature matrix.
        feature_names: Names of the transformed features.
        title: Plot title.
        subfolder: Optional subdirectory inside the figures directory.
    """
    # Use TreeExplainer for tree-based estimators.
    if type(model).__name__ in [
        "RandomForestRegressor",
        "RandomForestClassifier",
        "XGBRegressor",
        "XGBClassifier",
    ]:
        explainer = shap.TreeExplainer(model)
    else:
        # Use a representative background sample for non-tree models.
        background = shap.sample(X_transformed, min(100, len(X_transformed)))
        explainer = shap.Explainer(model.predict, background)

    shap_values = explainer(X_transformed)

    fig = plt.figure(figsize=(10, 6))

    plt.title(
        title,
        pad=20,
        fontsize=16,
        fontweight="bold",
    )

    shap.summary_plot(
        shap_values,
        X_transformed,
        feature_names=feature_names,
        show=False,
    )

    plt.tight_layout()
    _save_plot(title, subfolder)
    plt.show()
    plt.close(fig)