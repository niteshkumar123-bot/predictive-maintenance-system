import pytest
import pandas as pd
import numpy as np
from src.data_preprocessing import generate_synthetic_data, preprocess_pipeline
from src.feature_engineering import engineer_features

def test_synthetic_data_generation():
    df = generate_synthetic_data(num_samples=1000)
    assert len(df) == 1000
    assert "failure" in df.columns
    assert "temperature" in df.columns

def test_feature_engineering():
    df = generate_synthetic_data(num_samples=100)
    df_eng = engineer_features(df)
    assert "temperature_pressure_ratio" in df_eng.columns
    assert "power_consumption" in df_eng.columns

def test_preprocessing_pipeline():
    df = generate_synthetic_data(num_samples=500)
    X_train, X_test, y_train, y_test, preprocessor = preprocess_pipeline(df, save_artifacts=False)
    assert X_train.shape[0] == 400
    assert X_test.shape[0] == 100
    assert len(y_train) == 400
    assert len(y_test) == 100
