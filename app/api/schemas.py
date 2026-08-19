"""
Pydantic Schemas for API Validation.
Defines the expected input and output data structures for the ML endpoints.
"""

from pydantic import BaseModel, Field
from typing import Optional

class HousingRequest(BaseModel):
    """Schema for incoming House Pricing prediction requests."""
    area: int = Field(..., description="Total area of the property in square feet")
    bedrooms: int = Field(..., description="Number of bedrooms")
    bathrooms: int = Field(..., description="Number of bathrooms")
    stories: int = Field(..., description="Number of stories")
    mainroad: str = Field(..., description="Is it on the main road? (yes/no)")
    guestroom: str = Field(..., description="Does it have a guestroom? (yes/no)")
    basement: str = Field(..., description="Does it have a basement? (yes/no)")
    hotwaterheating: str = Field(..., description="Does it have hot water heating? (yes/no)")
    airconditioning: str = Field(..., description="Does it have AC? (yes/no)")
    parking: int = Field(..., description="Number of parking spaces (0-3)")
    prefarea: str = Field(..., description="Is it in a preferred area? (yes/no)")
    furnishingstatus: str = Field(..., description="Status (furnished/semi-furnished/unfurnished)")

    model_config = {
        "json_schema_extra": {
            "example": {
                "area": 7420, "bedrooms": 4, "bathrooms": 2, "stories": 3,
                "mainroad": "yes", "guestroom": "no", "basement": "no",
                "hotwaterheating": "no", "airconditioning": "yes",
                "parking": 2, "prefarea": "yes", "furnishingstatus": "furnished"
            }
        }
    }

class TitanicRequest(BaseModel):
    """Schema for incoming Titanic Survival prediction requests."""
    Pclass: int = Field(..., description="Passenger Class (1, 2, or 3)")
    Name: str = Field(..., description="Passenger full name with title")
    Sex: str = Field(..., description="Gender (male/female)")
    Age: float = Field(..., description="Passenger age in years")
    SibSp: int = Field(..., description="Number of siblings/spouses aboard")
    Parch: int = Field(..., description="Number of parents/children aboard")
    Fare: float = Field(..., description="Ticket fare paid")
    Embarked: str = Field(..., description="Port of Embarkation (C, Q, S)")

    model_config = {
        "json_schema_extra": {
            "example": {
                "Pclass": 1, "Name": "Cumings, Mrs. John Bradley",
                "Sex": "female", "Age": 38.0, "SibSp": 1, "Parch": 0,
                "Fare": 71.28, "Embarked": "C"
            }
        }
    }

class PredictionResponse(BaseModel):
    """Standardized response schema for all model predictions."""
    prediction: str = Field(..., description="The predicted output (e.g., '$1,200,000' or 'Survived')")
    confidence: Optional[float] = Field(None, description="Prediction confidence/probability if applicable")
    message: str = Field(..., description="Status message")