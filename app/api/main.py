"""
FastAPI application for serving ML predictions.
Loads serialized Scikit-Learn pipelines into memory and exposes HTTP POST endpoints.
"""

import logging
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from contextlib import asynccontextmanager

from src.config import ROOT_DIR
# VERY IMPORTANT: Transformers must be imported so joblib can unpickle the pipelines
from src.utils.transformers import HousingFeatureEngineer, TitanicFeatureEngineer
from app.api.schemas import HousingRequest, TitanicRequest, PredictionResponse

# Configure Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Global dictionary to hold ML models in memory
ml_models = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    FastAPI lifecycle manager. Loads models before accepting requests.
    """
    logger.info("Starting up FastAPI Server. Loading ML Models...")
    try:
        ml_models["housing"] = joblib.load(ROOT_DIR / "models" / "housing" / "latest_model.pkl")
        ml_models["titanic"] = joblib.load(ROOT_DIR / "models" / "titanic" / "latest_model.pkl")
        logger.info("Successfully loaded Housing and Titanic models.")
    except Exception as e:
        logger.error(f"Failed to load models: {str(e)}")
        ml_models["housing"] = None
        ml_models["titanic"] = None
    
    yield  # Server runs here
    
    logger.info("Shutting down API. Clearing models from memory.")
    ml_models.clear()

# Initialize FastAPI App
app = FastAPI(title="AnalystLab ML Deployment API", version="1.0", lifespan=lifespan)


@app.get("/")
def health_check() -> dict:
    """Returns a simple health check status."""
    return {"status": "API is live and routing traffic."}


@app.post("/predict/housing", response_model=PredictionResponse)
def predict_housing(request: HousingRequest) -> PredictionResponse:
    """
    Predicts the price of a house based on property features.
    
    Args:
        request (HousingRequest): The Pydantic validated JSON payload.
        
    Returns:
        PredictionResponse: Formatted USD price string.
    """
    if ml_models["housing"] is None:
        raise HTTPException(status_code=503, detail="Housing model is currently unavailable.")
    
    try:
        # Convert Pydantic object to DataFrame
        df_input = pd.DataFrame([request.model_dump()])
        
        # Pipeline handles engineering, scaling, log-inversion, and prediction
        prediction_val = ml_models["housing"].predict(df_input)[0]
        
        # Format as currency
        formatted_price = f"${prediction_val:,.2f}"
        
        logger.info(f"Housing Prediction Generated: {formatted_price}")
        return PredictionResponse(
            prediction=formatted_price,
            confidence=None,
            message="Housing prediction generated successfully."
        )
    except Exception as e:
        logger.error(f"Error during housing prediction: {str(e)}")
        raise HTTPException(status_code=400, detail=f"Prediction failed: {str(e)}")


@app.post("/predict/titanic", response_model=PredictionResponse)
def predict_titanic(request: TitanicRequest) -> PredictionResponse:
    """
    Predicts whether a passenger survived the Titanic disaster.
    
    Args:
        request (TitanicRequest): The Pydantic validated JSON payload.
        
    Returns:
        PredictionResponse: Survival string and probability confidence.
    """
    if ml_models["titanic"] is None:
        raise HTTPException(status_code=503, detail="Titanic model is currently unavailable.")
    
    try:
        df_input = pd.DataFrame([request.model_dump()])
        
        prediction_class = ml_models["titanic"].predict(df_input)[0]
        prediction_proba = ml_models["titanic"].predict_proba(df_input)[0][1] # Probability of Class 1 (Survived)
        
        result_text = "Survived" if prediction_class == 1 else "Did Not Survive"
        
        # Calculate confidence percentage based on the class chosen
        confidence = float(prediction_proba if prediction_class == 1 else (1 - prediction_proba))
        
        logger.info(f"Titanic Prediction: {result_text} (Confidence: {confidence:.2%})")
        return PredictionResponse(
            prediction=result_text,
            confidence=confidence,
            message="Titanic prediction generated successfully."
        )
    except Exception as e:
        logger.error(f"Error during titanic prediction: {str(e)}")
        raise HTTPException(status_code=400, detail=f"Prediction failed: {str(e)}")