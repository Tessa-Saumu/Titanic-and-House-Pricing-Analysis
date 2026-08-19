# Week 6: Feature Engineering & Model Optimization

## Objective
To break through the algorithmic performance ceiling established in Week 5 by shifting focus to the data space. This week utilized custom Scikit-Learn transformers to safely construct, mathematically filter, transform, and synergistically stack new features, followed by hyperparameter tuning to finalize production models.

---

## 1. Feature Engineering
We engineered new features to capture domain knowledge that the raw data failed to express.
*   **Titanic (Text Extraction & Combinations):** Extracted passenger `Title` (Mr, Mrs, Master) from the raw `Name` string to serve as a powerful socio-economic and age proxy. Combined `SibSp` and `Parch` to create a new `family_size` feature.
*   **Titanic (Interaction Features):** Generated an `age_x_pclass` interaction feature to mathematically force the model to recognize that age impacts survival differently depending on socio-economic class.
*   **Housing (Aggregations):** Created an `is_estate` binary flag (area > 8000 and stories > 2) to help linear models capture luxury outliers. 

## 2. Feature Transformation
We prepared the data for modeling using strict Scikit-Learn `ColumnTransformers` to prevent target leakage.
*   **Encoding & Scaling:** Applied `OneHotEncoder(drop='first')` to all categorical variables to prevent multicollinearity, and `RobustScaler` to standard numericals.
*   **Handling Skewed Variables:** EDA proved `Area` was massively right-skewed. To fix this, we applied a `QuantileTransformer(output_distribution='normal')` to forcefully map the skewed tail into a Gaussian distribution, satisfying the core mathematical assumption of Linear Regression. We also applied `TransformedTargetRegressor(func=np.log1p)` to normalize the skewed `Price` target.

## 3. Feature Selection
Instead of guessing which features to drop, we used strict mathematical and statistical techniques.
*   **Statistical Selection (Housing):** Variance Inflation Factor (VIF) analysis proved that attempting to use `Total_Rooms`, `bedrooms`, and `bathrooms` simultaneously resulted in a singular matrix (`VIF = inf`). We dropped the redundant source columns. We then applied L1 Regularization (Lasso), which mathematically shrank the noisy coefficients to absolute zero.
*   **Recursive Feature Elimination (Titanic):** We used RFE to rank our feature space. RFE objectively proved the value of our engineering by ranking `title`, `family_size`, and `age_x_pclass` as Priority 1 (Keep), while actively dropping the raw `Age`, `SibSp`, and `Parch` columns as inferior noise.

## 4. Model Optimization
After defining the optimal feature space via Incremental Forward Stacking, we executed hyperparameter tuning.
*   **Techniques Used:** We passed our Top 3 algorithms through a tournament using both `GridSearchCV` and `RandomizedSearchCV` with 5-fold Cross-Validation (`cv=5`) to guarantee robustness and prevent overfitting.
*   **Optimization Results:**
    *   *Housing:* `RandomSearch` identified that the `LinearRegression` baseline (with `fit_intercept=True`) generalized best on the newly engineered data.
    *   *Titanic:* `GridSearch` identified that `XGBoost` (`max_depth=10`, `min_samples_split=5`) outperformed our previous SVC champion when utilizing the newly extracted categorical features.

## 5. Performance Evaluation
We evaluated the optimized models against the frozen Week 5 baseline.

### Housing (Regression)
*   **Metrics:** MAE, MSE, RMSE, R² Score.
*   **Performance Change:** The final Test RMSE dropped to **$1,289,091** (a $25,000+ improvement over the Week 5 baseline). By explicitly transforming the skewed `Area` and flagging luxury estates, the model's ability to predict expensive outliers improved dramatically.

### Titanic (Classification)
*   **Metrics:** Accuracy, Precision, Recall, F1-Score, ROC-AUC.
*   **Performance Change:** In Week 5, the SVC defeated all tree models on raw data. However, once we injected `Title` extraction, **XGBoost** utilized the new categorical splits to dethrone the SVC. The final F1-Score improved from 0.7500 to **0.7704**, with an impressive ROC-AUC of 0.8371.