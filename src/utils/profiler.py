"""
Data profiling utilities.
Provides functions to generate metadata summaries of pandas DataFrames,
including data types, null counts, and unique value metrics.
"""

import pandas as pd

def generate_data_profile(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generates a comprehensive profile of a pandas DataFrame.
    """
    if df.empty:
        raise ValueError("The provided DataFrame is empty.")

    profile = pd.DataFrame({
        'Data Type': df.dtypes,
        'Non-Null Count': df.notnull().sum(),
        'Missing Values': df.isnull().sum(),
        'Missing Percentage (%)': (df.isnull().sum() / len(df) * 100).round(2),
        'Unique Values': df.nunique()
    })
    
    return profile