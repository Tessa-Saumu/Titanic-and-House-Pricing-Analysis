"""
Visualization module for the project.
Enforces a consistent 'Data Journalism' aesthetic (light grey backgrounds, 
high contrast colors, clear typography) across all notebooks and provides 
reusable plotting functions to keep notebook code sparse and readable.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def set_journalism_style():
    """
    Applies a Data Journalism aesthetic globally to Matplotlib and Seaborn.
    Features: Light grey background, white gridlines, no top/right borders, 
    and a high-contrast color palette.
    """
    # The FiveThirtyEight color palette (Blue, Red, Yellow, Green, Grey, Purple)
    journalism_palette = ['#008fd5', '#fc4f30', '#e5ae38', '#6d904f', '#8b8b8b', '#810f7c']
    
    sns.set_theme(
        style="whitegrid",
        palette=journalism_palette,
        rc={
            "axes.facecolor": "#F0F0F0",      # Light grey background
            "figure.facecolor": "#F0F0F0",    # Light grey figure edge
            "axes.edgecolor": "#F0F0F0",      # Hide axis border
            "grid.color": "white",            # Stark white gridlines
            "axes.spines.top": False,         # Remove top border
            "axes.spines.right": False,       # Remove right border
            "axes.titlesize": 16,             # Larger titles
            "axes.titleweight": "bold",       # Bold titles
            "axes.titlelocation": "left",     # Left-aligned titles (Journalism style)
            "axes.labelsize": 12,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
            "lines.linewidth": 2.5            # Thicker lines for readability
        }
    )

def plot_histogram(df: pd.DataFrame, column: str, title: str, bins: int = 30) -> None:
    """Plots a univariate numerical distribution."""
    plt.figure(figsize=(8, 5))
    sns.histplot(data=df, x=column, bins=bins, kde=True, color='#008fd5', edgecolor="white")
    plt.title(title)
    plt.xlabel(column.capitalize())
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.show()

def plot_bar_chart(df: pd.DataFrame, column: str, title: str) -> None:
    """Plots a univariate categorical distribution."""
    plt.figure(figsize=(8, 5))
    sns.countplot(data=df, x=column, color='#008fd5', order=df[column].value_counts().index)
    plt.title(title)
    plt.xlabel(column.capitalize())
    plt.ylabel("Count")
    plt.tight_layout()
    plt.show()

def plot_box(df: pd.DataFrame, cat_col: str, num_col: str, title: str) -> None:
    """Plots a bivariate relationship (Categorical vs Numerical)."""
    plt.figure(figsize=(8, 5))
    sns.boxplot(data=df, x=cat_col, y=num_col, palette=['#008fd5', '#fc4f30', '#e5ae38'])
    plt.title(title)
    plt.xlabel(cat_col.capitalize())
    plt.ylabel(num_col.capitalize())
    plt.tight_layout()
    plt.show()

def plot_correlation_heatmap(df: pd.DataFrame, title: str) -> None:
    """Plots a correlation heatmap for numerical features."""
    plt.figure(figsize=(10, 6))
    # Select only numerical columns for correlation to avoid errors
    num_df = df.select_dtypes(include=['int64', 'float64'])
    corr = num_df.corr()
    
    # Use a diverging palette (blue to red) for correlations
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0, 
                vmin=-1, vmax=1, linewidths=0.5, linecolor='white')
    plt.title(title)
    plt.tight_layout()
    plt.show()