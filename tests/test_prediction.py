import pytest
from src.predict import FailurePredictor
from src.data_preprocessing import load_or_generate_data, preprocess_pipeline
from src.train import train_and_tune_models
from src.evaluate import evaluate_models

def test_end_to_end_training_and_prediction():
    df = load_or_generate_data()
    X_train, X_test, y_train, y_test, preprocessor = preprocess_pipeline(df, save_artifacts=True)
    models = train_and_tune_models(X_train, y_train)
    evaluate_models(models, X_test, y_test)

    predictor = FailurePredictor()
    sample_input = {
        "temperature": 75.0,
        "vibration": 2.1,
        "pressure": 100.0,
        "rotational_speed": 1500.0,
        "torque": 40.0,
        "operating_hours": 3000.0,
        "voltage": 220.0,
        "current": 15.0,
        "humidity": 50.0,
        "maintenance_count": 1,
        "machine_age": 3.0,
        "failure_history": 0,
        "tool_wear": 30.0,
        "load_percentage": 70.0
    }

    result = predictor.predict(sample_input)
    assert "prediction" in result
    assert "failure_probability" in result
    assert "risk_level" in result
    assert result["risk_level"] in ["LOW", "MEDIUM", "HIGH"]
