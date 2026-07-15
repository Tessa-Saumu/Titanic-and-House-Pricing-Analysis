"""
Visualization module for the project.
Enforces a consistent 'Data Journalism' aesthetic and autosaves plots.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from matplotlib.ticker import StrMethodFormatter
from pathlib import Path

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