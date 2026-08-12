# Week 6: Feature Engineering & Synergistic Stacking

## Objective
To break through the algorithmic performance ceiling established in Week 5 by shifting focus to the data space. This week utilized custom Scikit-Learn transformers to safely construct, mathematically filter, and synergistically stack new features without introducing data leakage.

## Methodology: Isolation vs. Synergy
We split the week's experiments into two distinct methodologies:
1. **Isolated Testing & Regularization (Notebook 01):** We tested hypotheses individually and used L1 Regularization (Lasso) and Recursive Feature Elimination (RFE) to observe how mathematical selectors punish redundant features.
2. **Incremental Forward Stacking (Notebook 02):** Recognizing that features interact non-linearly in the real world, we built a custom stacking algorithm. It tested hypotheses sequentially via Cross-Validation, locking in features only if they improved the global metric, before passing the optimized dataset to a Hyperparameter tournament.

## Key Findings & Results

### Housing Prices (Regression)
- **The Multicollinearity Trap:** VIF (Variance Inflation Factor) analysis correctly triggered `inf` warnings when `total_rooms` was combined with `bedrooms` and `bathrooms`, forcing us to dynamically drop source columns to protect the linear matrix.
- **Synergistic Success:** Stacking density, amenity, estate, and room features together lowered the cross-validated training RMSE to ~$995k. 
- **Champion:** `Tuned_Engineered_Linear`. The final Test RMSE dropped to **$1,289,091** (a $25,000+ improvement over the Week 5 baseline).

### Titanic Survival (Classification)
- **RFE Feature Replacement:** RFE objectively proved the value of our feature engineering by ranking `title`, `family_size`, and `age_x_pclass` as priority 1, while actively dropping the raw `Age`, `SibSp`, and `Parch` columns as inferior noise.
- **The Algorithmic Upset:** In Week 5, the Support Vector Classifier (SVC) defeated all tree models on raw data. However, once we injected `Title` extraction into the dataset, **XGBoost** was able to utilize the new categorical splits to dethrone the SVC. 
- **Champion:** `Tuned_Engineered_XGBoost`. The final F1-Score improved from 0.7500 to **0.7704**, with an impressive ROC-AUC of 0.8371.

## Challenges Encountered
- **Automated Selectors vs. Non-Linear Boundaries:** We discovered that RFE (which relies on linear coefficients) heavily penalized continuous variables like `Age`. Blindly trusting RFE would have destroyed the non-linear boundaries required by our SVC, highlighting the danger of trusting automated feature selection without domain knowledge. 

## Next Steps (Week 7: Deployment)
Both `Tuned_Engineered_Linear` and `Tuned_Engineered_XGBoost` are fully encapsulated in Scikit-Learn `Pipelines` alongside their custom `TransformerMixin` classes. They have been serialized via `joblib` and are ready to be served via a FastAPI/Streamlit application.