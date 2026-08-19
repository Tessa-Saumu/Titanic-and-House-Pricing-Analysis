# AnalystLab: End-to-End Machine Learning Microservice

## Project Description
This repository represents the culmination of an intensive 7-week Machine Learning internship. It elevates a standard data science analysis into a **production-ready, decoupled microservice architecture**. 

The project features a highly optimized Machine Learning backend that serves predictions via RESTful API endpoints, and a custom-designed, insight-driven frontend workspace for users to simulate prediction scenarios.

---

## Problem Statement
The pipeline concurrently solves two classic ML problems, addressing specific data complexities for each:
1. **Housing Price Prediction (Regression):** Estimating continuous property market values based on physical and location characteristics. *Challenge:* Overcoming severe target skewness and heteroscedasticity on luxury outliers.
2. **Titanic Survival (Classification):** Estimating binary survival probabilities based on historical passenger demographics. *Challenge:* Navigating severe class imbalances and discovering complex non-linear survival boundaries.

---

## Models Used
Following rigorous statistical selection, feature engineering, synergistic stacking, and hyperparameter tuning, the following champion models were deployed:
- **Housing Model:** `Tuned Engineered Linear Regression`. (Augmented with a `TransformedTargetRegressor` for log-scaling and a `QuantileTransformer` to normalize spatial features).
- **Titanic Model:** `Tuned Engineered XGBoost`. (Leveraging extracted titles and family-size aggregations to capture non-linear demographic interactions).

---

## Technologies Used
- **Machine Learning & Data:** `scikit-learn`, `xgboost`, `pandas`, `numpy`, `scipy`
- **Model Serialization:** `joblib`
- **Backend API:** `FastAPI`, `uvicorn`, `pydantic`
- **Frontend UI:** `streamlit` (with heavy custom CSS injection)
- **Experiment Tracking:** Custom MLOps File Registry

---

## API Endpoints

The backend is a strictly typed FastAPI application serving the serialized Scikit-Learn pipelines.

### `GET /`
- **Description:** System health check.
- **Output:** `{"status": "API is live and routing traffic."}`

### `POST /predict/housing`
- **Description:** Accepts property characteristics and returns the predicted USD value.

### `POST /predict/titanic`
- **Description:** Accepts passenger demographics and returns survival classification and model confidence.

---

## Input and Output Format

The API enforces strict data contracts using Pydantic. 

### Housing Request & Response Example
**POST payload (`application/json`):**
```json
{
  "area": 5000,
  "bedrooms": 3,
  "bathrooms": 2,
  "stories": 2,
  "mainroad": "yes",
  "guestroom": "no",
  "basement": "no",
  "hotwaterheating": "no",
  "airconditioning": "yes",
  "parking": 0,
  "prefarea": "yes",
  "furnishingstatus": "furnished"
}
Response:
code
JSON
{
  "prediction": "$6,161,400.68",
  "confidence": null,
  "message": "Housing prediction generated successfully."
}
Titanic Request & Response Example
POST payload (application/json):
code
JSON
{
  "Pclass": 3,
  "Name": "Smith, Mr. John",
  "Sex": "male",
  "Age": 30.0,
  "SibSp": 0,
  "Parch": 0,
  "Fare": 15.50,
  "Embarked": "S"
}
Response:
code
JSON
{
  "prediction": "Did Not Survive",
  "confidence": 0.868,
  "message": "Titanic prediction generated successfully."
}
```

## Setup Instructions
### Clone the repository:
```code
Bash
git clone https://github.com/yourusername/analystlab-internship.git
cd analystlab-internship
```
### Create a virtual environment (Recommended):
```code
Bash
python -m venv .venv
source .venv/bin/activate  # On Windows use: .venv\Scripts\activate
```
### Install dependencies:
```code
Bash
pip install -r requirements.txt
```
### How to Run the Project
This project uses a decoupled architecture. We have provided an orchestration script to boot both the backend and frontend simultaneously.
**Run the Orchestrator:**
```code
Bash
python run_app.py
```
**Frontend Workspace/Streamlit Dashboard**: Open your browser to http://localhost:8501

**Backend Swagger Docs**: Open your browser to http://localhost:8000/docs to test the API directly without the UI.

To stop the servers, simply press CTRL+C in the terminal.

---