import pandas as pd
import pytest
from src.utils.profiler import generate_data_profile

def test_generate_data_profile_valid():
    # Arrange: Create dummy data
    df = pd.DataFrame({
        'A': [1, 2, None],
        'B': ['cat', 'dog', 'cat']
    })
    
    # Act: Run profiler
    profile = generate_data_profile(df)
    
    # Assert: Check expected outputs
    assert profile.loc['A', 'Missing Values'] == 1
    assert profile.loc['A', 'Missing Percentage (%)'] == 33.33
    assert profile.loc['B', 'Unique Values'] == 2

def test_generate_data_profile_empty():
    df = pd.DataFrame()
    with pytest.raises(ValueError):
        generate_data_profile(df)