"""
Statistical testing utilities.
Abstracts scipy.stats functions to return rich, business-friendly dictionaries
containing raw test statistics, p-values, and plain-English conclusions.
"""

import pandas as pd
import numpy as np
from scipy import stats

def _generate_conclusion(p_value: float, alpha: float, test_name: str, h1_desc: str) -> str:
    """
    Internal helper to generate a plain-English conclusion based on the p-value.
    """
    if p_value < alpha:
        return f"Reject Null Hypothesis (p={p_value:.4f} < {alpha}): {h1_desc} is statistically significant."
    else:
        return f"Fail to Reject Null (p={p_value:.4f} >= {alpha}): Not enough evidence to prove {h1_desc}."


def check_normality(data: pd.Series, alpha: float = 0.05) -> dict:
    """
    Performs the Shapiro-Wilk test for normality on a continuous variable.
    
    Args:
        data (pd.Series): The numerical data to test.
        alpha (float): The significance level. Defaults to 0.05.
        
    Returns:
        dict: Test results including statistic, p-value, and plain-English conclusion.
    """
    # Drop NaNs purely for the scope of the calculation
    clean_data = data.dropna()
    
    # Run Shapiro-Wilk test
    stat, p_value = stats.shapiro(clean_data)
    
    conclusion = _generate_conclusion(
        p_value=p_value, 
        alpha=alpha, 
        test_name="Shapiro-Wilk", 
        h1_desc="Deviation from normality"
    )
    
    return {
        "test": "Shapiro-Wilk Normality Test",
        "statistic": stat,
        "p_value": p_value,
        "is_normal": p_value >= alpha, # If p >= alpha, we assume it is normal
        "conclusion": conclusion
    }


def run_mann_whitney(group1: pd.Series, group2: pd.Series, alpha: float = 0.05) -> dict:
    """
    Performs the non-parametric Mann-Whitney U test for two independent groups.
    Ideal for skewed data (e.g., comparing House Prices between AC / No AC).
    
    Args:
        group1 (pd.Series): Continuous data for group 1.
        group2 (pd.Series): Continuous data for group 2.
        alpha (float): Significance level. Defaults to 0.05.
        
    Returns:
        dict: Test results including statistic, p-value, and conclusion.
    """
    clean_g1 = group1.dropna()
    clean_g2 = group2.dropna()
    
    stat, p_value = stats.mannwhitneyu(clean_g1, clean_g2, alternative='two-sided')
    
    conclusion = _generate_conclusion(
        p_value=p_value, 
        alpha=alpha, 
        test_name="Mann-Whitney U", 
        h1_desc="Difference in distributions between the two groups"
    )
    
    return {
        "test": "Mann-Whitney U Test",
        "statistic": stat,
        "p_value": p_value,
        "is_significant": p_value < alpha,
        "conclusion": conclusion
    }


def run_chi_square(feature1: pd.Series, feature2: pd.Series, alpha: float = 0.05) -> dict:
    """
    Performs a Chi-Square Test of Independence between two categorical variables.
    (e.g., Pclass vs. Survived on the Titanic).
    
    Args:
        feature1 (pd.Series): First categorical variable.
        feature2 (pd.Series): Second categorical variable.
        alpha (float): Significance level. Defaults to 0.05.
        
    Returns:
        dict: Test results including statistic, p-value, degrees of freedom, and conclusion.
    """
    # Create a contingency table (crosstab)
    contingency_table = pd.crosstab(feature1.dropna(), feature2.dropna())
    
    # Run Chi-Square test
    chi2_stat, p_value, dof, expected = stats.chi2_contingency(contingency_table)
    
    conclusion = _generate_conclusion(
        p_value=p_value, 
        alpha=alpha, 
        test_name="Chi-Square", 
        h1_desc="Dependence/Association between the categorical variables"
    )
    
    return {
        "test": "Chi-Square Test of Independence",
        "statistic": chi2_stat,
        "p_value": p_value,
        "dof": dof,
        "is_significant": p_value < alpha,
        "conclusion": conclusion
    }