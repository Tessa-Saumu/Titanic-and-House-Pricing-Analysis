"""
Custom Scikit-Learn Transformers for Feature Engineering.
Ensures feature creation happens inside the pipeline to prevent data leakage.
"""

import pandas as pd
import numpy as np
import logging
from typing import Optional
from sklearn.base import BaseEstimator, TransformerMixin

logger = logging.getLogger(__name__)

class HousingFeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Custom transformer to execute Week 6 hypotheses for the Housing dataset.
    Flags allow testing features in isolation.
    """
    def __init__(self, h1_density: bool = False, h2_amenity: bool = False, 
                 h3_poly: bool = False, h4_estate: bool = False, h5_rooms: bool = False):
        self.h1_density = h1_density
        self.h2_amenity = h2_amenity
        self.h3_poly = h3_poly
        self.h4_estate = h4_estate
        self.h5_rooms = h5_rooms

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X_eng = X.copy()
        
        # Helper calculation for rooms
        total_rooms = X_eng['bedrooms'] + X_eng['bathrooms']
        
        if self.h1_density:
            X_eng['area_per_room'] = X_eng['area'] / (total_rooms + 1) # +1 prevents div zero
            
        if self.h2_amenity:
            # Convert yes/no to 1/0 and sum
            amenities = ['airconditioning', 'hotwaterheating', 'prefarea']
            X_eng['luxury_score'] = sum((X_eng[col] == 'yes').astype(int) for col in amenities)
            
        if self.h3_poly:
            X_eng['area_squared'] = X_eng['area'] ** 2
            
        if self.h4_estate:
            X_eng['is_estate'] = ((X_eng['area'] > 8000) & (X_eng['stories'] > 2)).astype(int).astype(str)
            
        if self.h5_rooms:
            X_eng['total_rooms'] = total_rooms
            # Note: We don't drop bedrooms/bathrooms here; we handle dropping in the ColumnTransformer
            
        return X_eng


class TitanicFeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Custom transformer to execute Week 6 hypotheses for the Titanic dataset.
    """
    def __init__(self, h1_title: bool = False, h2_family: bool = False, 
                 h3_alone: bool = False, h4_wealth: bool = False, h5_interact: bool = False):
        self.h1_title = h1_title
        self.h2_family = h2_family
        self.h3_alone = h3_alone
        self.h4_wealth = h4_wealth
        self.h5_interact = h5_interact

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X_eng = X.copy()
        
        if self.h1_title:
            # Extract title using regex and group rare ones
            X_eng['title'] = X_eng['Name'].str.extract(r' ([A-Za-z]+)\.', expand=False)
            rare_titles = ['Lady', 'Countess','Capt', 'Col', 'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona']
            X_eng['title'] = X_eng['title'].replace(rare_titles, 'Rare')
            X_eng['title'] = X_eng['title'].replace({'Mlle': 'Miss', 'Ms': 'Miss', 'Mme': 'Mrs'})
            
        if self.h2_family or self.h3_alone:
            family_size = X_eng['SibSp'] + X_eng['Parch'] + 1
            if self.h2_family:
                X_eng['family_size'] = family_size
            if self.h3_alone:
                X_eng['is_alone'] = (family_size == 1).astype(int).astype(str)
                
        if self.h4_wealth:
            # Bin Fare into 4 quantiles. Use qcut on training data logic.
            # For simplicity in a stateless transformer, we use hardcoded bins or rank.
            X_eng['fare_bin'] = pd.qcut(X_eng['Fare'], 4, labels=['Budget', 'Economy', 'Premium', 'Luxury'], duplicates='drop')
            X_eng['fare_bin'] = X_eng['fare_bin'].astype(str)
            
        if self.h5_interact:
            # Impute age temporarily just for the interaction term math
            temp_age = X_eng['Age'].fillna(X_eng['Age'].median())
            X_eng['age_x_pclass'] = temp_age * X_eng['Pclass']
            
        return X_eng