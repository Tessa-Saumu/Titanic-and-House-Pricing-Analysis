# Week 4: Supervised Learning, Pipelines, & MLOps

## Objective
Transition from inferential statistics to predictive machine learning. This week focused on establishing rigorous, leak-proof Scikit-Learn pipelines to train baseline supervised learning models for classification (Titanic) and regression (Housing), while implementing a production-grade model registry to prepare for deployment.

---

## 1. Datasets Used
- **Titanic Dataset (Classification):** 889 passengers (after dropping missing `Embarked`). Target: `Survived` (Binary).
- **House Prices Dataset (Regression):** 545 properties. Target: `Price` (Continuous).

---

## 2. Leak-Proof ML Pipelines
To ensure reproducibility and prevent data leakage, manual pandas manipulations were replaced with strict Machine Learning pipelines.
1. **Stratified & Random Splitting:** Isolated 20% of the data into strict holdout test sets. Used `stratify=y` for the Titanic dataset to maintain the 62/38 baseline mortality ratio.
2. **Pipeline Construction (`ColumnTransformer`):**
   - Implemented `SimpleImputer(strategy='median')` for missing numericals (`Age`), fit *only* on the training data.
   - Applied `RobustScaler` to handle extreme high-end outliers without distorting the interquartile range.
   - Applied `OneHotEncoder(drop='first')` to categorical variables to prevent perfect multicollinearity (the Dummy Variable Trap).
3. **Target Normalization:** Wrapped the Housing Linear Regression estimator in a `TransformedTargetRegressor(func=np.log1p)` to dynamically normalize the highly skewed `Price` target during training, automatically inverting the predictions for evaluation.

---

## 3. MLOps & Artifact Management (Enhancement)
To simulate an enterprise environment and prevent technical debt ahead of deployment (Week 7), a custom `registry.py` and `evaluator.py` module were constructed.
- **Timestamped Serialization:** Trained pipelines are serialized via `joblib` into a `models/` directory, saving both a timestamped version for auditing and a `latest_model.pkl` alias for API consumption.
- **Automated Experiment Tracking:** Model metrics and dynamic business insights are automatically written to a timestamped `.txt` report inside `reports/runs/`, ensuring a historical record of all training executions without relying on Jupyter Notebook states.

---

## 4. Model Performance

### Titanic (Logistic Regression)
- **Accuracy:** 79.78% (Successfully beats the 61.8% baseline)
- **Precision:** 73.24%
- **Recall:** 75.36%
- **F1-Score:** 0.7429
- **Insight:** The model leans slightly toward Recall (identifying 52 out of 69 actual survivors). The Confusion Matrix reveals 17 False Negatives and 19 False Positives, showing balanced but slightly conservative survivor identification.

### Housing (Linear Regression)
- **RMSE:** $1,314,648.20
- **R² Score:** 0.6581
- **Insight:** The model explains roughly 65.8% of the variance in property prices. The RMSE indicates our predictions deviate by an average of ~$1.3M on the actual price scale. 

---

## 5. Challenges Encountered & Key Insights

**Heteroscedasticity in Housing Prices:** 
An analysis of the "Actual vs. Predicted" scatter plot reveals a distinct funnel shape. The model predicts lower-to-middle market homes quite accurately, but the residuals widen significantly at the higher end of the market. The linear model systematically underpredicts luxury outliers.

**The Bivariate vs. Multivariate Trap (Housing):** 
In Week 3, a standard Mann-Whitney U test suggested `hotwaterheating` was the weakest feature signal in the dataset. However, our trained Linear Regression model assigned it the 4th highest positive weight (0.130). This highlighted a crucial Machine Learning principle: when controlling for other variables (like Area and Bathrooms) in a multivariate space, the presence of hot water heating yields a strong price premium that bivariate tests cannot capture. 

**Validating the Chi-Square (Titanic):**
The Logistic Regression weights perfectly validated the inferential statistics from Week 3. `Sex_male` (-2.589) and `Pclass_3` (-1.955) were the most devastating negative drivers of survival odds, mathematically confirming that gender and socio-economic status overrode all other predictive features on the ship.

---

## Next Steps (Week 5: Advanced Machine Learning)
- Address the Housing heteroscedasticity by migrating from simple Linear Regression to non-linear, tree-based models (Decision Trees, Random Forest, Gradient Boosting) to better capture luxury outliers.
- Introduce Hyperparameter tuning to optimize performance beyond the baseline constraints.