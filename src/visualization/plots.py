"""
Visualization module for the project.
Enforces a consistent 'Data Journalism' aesthetic and provides 
reusable plotting functions.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from matplotlib.ticker import StrMethodFormatter

def set_journalism_style():
    """Applies a Data Journalism aesthetic globally."""
    journalism_palette = ['#008fd5', '#fc4f30', '#e5ae38', '#6d904f', '#8b8b8b', '#810f7c']
    sns.set_theme(
        style="whitegrid",
        palette=journalism_palette,
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
            "lines.linewidth": 2.5
        }
    )

def _format_axes(ax):
    """Helper function to remove scientific notation and add commas."""
    ax.yaxis.set_major_formatter(StrMethodFormatter('{x:,.0f}'))
    if ax.get_xlim()[1] > 1000: # Format x-axis if values are large
        ax.xaxis.set_major_formatter(StrMethodFormatter('{x:,.0f}'))

def plot_histogram(df: pd.DataFrame, column: str, title: str, bins: int = 30) -> None:
    plt.figure(figsize=(8, 5))
    ax = sns.histplot(data=df, x=column, bins=bins, kde=True, color='#008fd5', edgecolor="white")
    plt.title(title)
    plt.xlabel(column.capitalize())
    plt.ylabel("Frequency")
    _format_axes(ax)
    plt.tight_layout()
    plt.show()

def plot_bar_chart(df: pd.DataFrame, column: str, title: str) -> None:
    plt.figure(figsize=(8, 5))
    ax = sns.countplot(data=df, x=column, color='#008fd5', order=df[column].value_counts().index)
    plt.title(title)
    plt.xlabel(column.capitalize())
    plt.ylabel("Count")
    plt.tight_layout()
    plt.show()

def plot_box(df: pd.DataFrame, cat_col: str, num_col: str, title: str) -> None:
    plt.figure(figsize=(8, 5))
    # Fixed Warning: Added hue and legend=False. Removed hardcoded palette.
    ax = sns.boxplot(data=df, x=cat_col, y=num_col, hue=cat_col, legend=False)
    plt.title(title)
    plt.xlabel(cat_col.capitalize())
    plt.ylabel(num_col.capitalize())
    _format_axes(ax)
    plt.tight_layout()
    plt.show()

def plot_categorical_target_rate(df: pd.DataFrame, cat_col: str, target_col: str, title: str) -> None:
    """
    Plots the mean rate of a binary target variable across categories.
    E.g., Survival Rate by Passenger Class.
    """
    plt.figure(figsize=(8, 5))
    # sns.barplot automatically calculates the mean (which equals the % rate for 0/1 targets)
    ax = sns.barplot(data=df, x=cat_col, y=target_col, hue=cat_col, legend=False)
    plt.title(title)
    plt.xlabel(cat_col.capitalize())
    plt.ylabel(f"{target_col.capitalize()} Rate (%)")
    
    # Format Y axis as percentages
    ax.yaxis.set_major_formatter(StrMethodFormatter('{x:.0%}'))
    
    # Add count annotations to address your sample size question!
    counts = df[cat_col].value_counts().to_dict()
    for i, p in enumerate(ax.patches):
        category = ax.get_xticklabels()[i].get_text()
        # Fallback to integer conversion if possible, otherwise use string key
        try:
            count = counts.get(int(category), counts.get(category, 0))
        except ValueError:
            count = counts.get(category, 0)
        
        ax.annotate(f'n={count}', 
                    (p.get_x() + p.get_width() / 2., 0.05), # Place near bottom
                    ha='center', va='bottom', color='white', weight='bold', fontsize=10)

    plt.tight_layout()
    plt.show()

def plot_correlation_heatmap(df: pd.DataFrame, title: str) -> None:
    plt.figure(figsize=(10, 6))
    num_df = df.select_dtypes(include=['int64', 'float64'])
    corr = num_df.corr()
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0, 
                vmin=-1, vmax=1, linewidths=0.5, linecolor='white')
    plt.title(title)
    plt.tight_layout()
    plt.show()