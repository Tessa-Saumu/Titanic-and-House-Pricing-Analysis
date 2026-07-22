# Week 3: Statistical Analysis & Hypothesis Testing

## Objective
Transition from visual Exploratory Data Analysis (EDA) to rigorous mathematical proofs. The goal of Week 3 was to systematically validate the "signals" discovered in Week 1-2 using probability theory, advanced descriptive statistics, and formal hypothesis testing, ensuring that only statistically significant features are passed to the machine learning pipeline.

## Architectural Updates
- **Dependency Management:** Updated `requirements.txt` to include `scipy` and `statsmodels`.
- **Custom Utility Module (`src/utils/stats.py`):** Rather than cluttering the Jupyter Notebooks with Scipy boilerplate and `dropna()` operations, a centralized statistics module was developed. This module returns rich dictionaries containing test statistics, p-values, and dynamically generated plain-English conclusions to cleanly bridge complex math with business logic.

## Key Findings: Housing Dataset
- **Failure of Normality:** Advanced descriptive statistics revealed extreme right-skewness (>1.0) and kurtosis in `Price` and `Area`. The Shapiro-Wilk test officially rejected the normality assumption ($p=0.0000$), necessitating the use of non-parametric testing.
- **Systematic Feature Validation:** A programmatic Mann-Whitney U test loop was executed against all binary categorical features. All amenities proved to be statistically significant drivers of price.
- **Weak Signal Identification:** `Hotwaterheating` barely passed the threshold ($p=0.0461$), making it a potential candidate for feature pruning.
- **Confounding Variables:** Established that `Area` acts as a heavy confounding variable for features like `Bathrooms`, creating collinearity that models must account for.

## Key Findings: Titanic Dataset
- **Exhaustive Probability & Bayes' Theorem:** Mapped the complete probability space of the ship. Discovered the "Inversion of Proportions" phenomenon: while Females and 1st Class passengers were massive minorities on the ship (35% and 24% respectively), their extraordinarily high survival rates caused them to become the dominating demographics inside the lifeboats.
- **Hypothesis Testing:** Executed Chi-Square Tests of Independence on `Sex` (Stat: 258.43) and `Pclass` (Stat: 100.98). Both vehemently rejected the null hypothesis, mathematically proving that gender and socio-economic status dictated survival.
- **Baseline Accuracy Established:** Established a hard baseline of 61.8% accuracy (the unconditional probability of mortality) that future predictive models must beat.

## Next Steps (Week 4: Feature Engineering & ML Preprocessing)
The statistical foundations set in Week 3 dictate our Week 4 pipeline requirements:
1. **Robust Scaling:** Housing numerical features (`Price`, `Area`) require `RobustScaler` or Log Transformations due to extreme skewness.
2. **Encoding:** Strong categorical signals (`Sex`, `Pclass`) require strict One-Hot Encoding to prevent algorithmic misinterpretation.
3. **Imputation:** `Age` will be safely median-imputed within a Scikit-Learn pipeline to prevent data leakage.