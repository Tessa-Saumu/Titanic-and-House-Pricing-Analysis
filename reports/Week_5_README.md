\# Week 5: Advanced Machine Learning \& Model Optimization



\## Objective

The objective of Phase 8 was to transition from baseline modeling to advanced, non-linear machine learning. We aimed to establish idiosyncratic preprocessing pipelines, train a suite of advanced algorithms (KNN, Huber, SVC, Decision Trees, Random Forest, XGBoost), dynamically isolate the top performers, and apply Hyperparameter Tuning (`GridSearchCV`) to finalize our production models.



\## Architectural Enhancements

\- \*\*Idiosyncratic Preprocessing:\*\* Built specialized `ColumnTransformers`. Linear/Distance models received `RobustScaler` and `OneHotEncoder(drop='first')` to prevent multicollinearity. Tree-based models received unscaled passthroughs and kept all categorical classes to prevent forced, inefficient splits.

\- \*\*Model-Agnostic Explainability:\*\* Integrated \*\*SHAP (SHapley Additive exPlanations)\*\* to generate global feature importance charts based on game theory, ensuring we could interpret non-tree models (like SVC and Linear Regression) fairly.



\---



\## Part 1: Housing Prices (Regression)



\### Experiment Setup

We compared distance-based (KNN), robust parametric (Huber), log-transformed parametric (Linear Regression), and ensemble trees (Random Forest, XGBoost) to predict property values. 



\### Performance Comparison Table (Top 3)

| Model | MAE | MSE | RMSE | R² Score | Tuning Time (s) |

| :--- | :--- | :--- | :--- | :--- | :--- |

| \*\*Baseline Linear (Log Transformed)\*\* | \*\*$960,123\*\* | \*\*1.728e+12\*\* | \*\*$1,314,648\*\* | \*\*0.6580\*\* | \*\*0.49\*\* |

| Tuned Huber Regressor | $985,273 | 1.784e+12 | $1,335,955 | 0.6468 | 11.19 |

| Tuned Random Forest | $1,030,289 | 1.991e+12 | $1,411,317 | 0.6059 | 19.46 |

| \*Default XGBoost (Untuned)\* | \*$1,057,455\* | \*2.068e+12\* | \*$1,438,064\* | \*0.5908\* | \*0.23\* |



\### Key Insight: The Complexity Trap

The most valuable business insight generated was the failure of advanced tree ensembles. The simple Linear Regression (augmented with a `TransformedTargetRegressor` to handle target kurtosis) outperformed both Random Forest and XGBoost. Because the dataset is small (545 rows) and largely linear, the tree models overfit the noise. This proves that algorithm selection must be dictated by data size and shape, not just industry hype.



\---



\## Part 2: Titanic Survival (Classification)



\### Experiment Setup

We compared Logistic Regression, KNN, SVC (RBF Kernel), Decision Trees, Random Forest, and XGBoost to predict binary survival probabilities. We optimized for the \*\*F1-Score\*\* due to the baseline 62/38 class imbalance.



\### Performance Comparison Table (Top 3)

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC | Tuning Time (s) |

| :--- | :--- | :--- | :--- | :--- | :--- | :--- |

| \*\*SVC (RBF Kernel)\*\* | \*\*0.8202\*\* | \*\*0.8000\*\* | \*\*0.7059\*\* | \*\*0.7500\*\* | \*\*0.8499\*\* | \*\*0.11\*\* |

| Tuned XGBoost | 0.7978 | 0.7759 | 0.6618 | 0.7143 | 0.8295 | 4.46 |

| Tuned Random Forest | 0.7978 | 0.7759 | 0.6618 | 0.7143 | 0.8222 | 20.55 |

| \*Baseline Logistic (Untuned)\* | \*0.8090\* | \*0.7931\* | \*0.6765\* | \*0.7302\* | \*0.8583\* | \*0.07\* |



\### Key Insight: Non-Linear Boundaries

The \*\*Support Vector Classifier (SVC)\*\* using a Radial Basis Function (RBF) kernel emerged as the champion. The Titanic dataset contains highly complex survival intersections (e.g., 3rd class men died, 1st class women survived, but 3rd class children had mixed odds). The RBF kernel was able to perfectly warp the feature space to draw these margins without succumbing to the overfitting that plagued the tree models.



\---



\## Conclusion \& Next Steps

Both the `Linear\_Regression` (Housing) and `SVC` (Titanic) models have been serialized via `joblib` into our `/models` registry. We have successfully proven that robust data preprocessing (scaling, target transformation) combined with the \*correct\* algorithmic architecture beats blindly applying XGBoost. 



\*\*Next Phase (Week 7):\*\* Model Deployment \& API Construction.

