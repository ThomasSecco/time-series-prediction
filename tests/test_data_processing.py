import pytest
import pandas as pd
from src.data_processing import preprocess_data

def test_preprocess_data():
    data = {
        "date": ["2024-01-01", "2024-01-02", "2024-01-03"],
        "sales": [100, None, 200],
    }
    df = pd.DataFrame(data)
    df["date"] = pd.to_datetime(df["date"])
    
    processed_df, scaler = preprocess_data(df)
    assert not processed_df["sales"].isnull().any()
    assert "year" in processed_df.columns
