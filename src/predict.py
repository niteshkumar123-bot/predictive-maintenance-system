import joblib
import pandas as pd

from src.feature_engineering import engineer_features
from src.utils import MODELS_DIR, logger

PREDICTION_THRESHOLD = 0.50
LOW_RISK_THRESHOLD = 0.30
HIGH_RISK_THRESHOLD = 0.70


class FailurePredictor:
    """Load trained artifacts and return a failure prediction for one machine."""

    def __init__(self):
        self.model_path = MODELS_DIR / "best_model.pkl"
        self.scaler_path = MODELS_DIR / "scaler.pkl"
        self.model = None
        self.scaler = None
        self.load_artifacts()

    def load_artifacts(self):
        if self.model_path.exists():
            self.model = joblib.load(self.model_path)
        else:
            logger.warning(f"Model file not found at {self.model_path}")

        if self.scaler_path.exists():
            self.scaler = joblib.load(self.scaler_path)
        else:
            logger.warning(f"Preprocessor file not found at {self.scaler_path}")

    def predict(self, input_data: dict) -> dict:
        if self.model is None or self.scaler is None:
            self.load_artifacts()
        if self.model is None or self.scaler is None:
            raise RuntimeError(
                "Model artifacts are missing. Run 'python run_pipeline.py' before prediction."
            )

        df = engineer_features(pd.DataFrame([input_data]))
        df = df.drop(columns=["machine_id"], errors="ignore")

        if hasattr(self.scaler, "feature_names_in_"):
            expected = list(self.scaler.feature_names_in_)
            df = df.reindex(columns=expected)

        X = self.scaler.transform(df)
        probability = float(self.model.predict_proba(X)[0][1])
        prediction = int(probability >= PREDICTION_THRESHOLD)

        if probability < LOW_RISK_THRESHOLD:
            risk_level = "LOW"
        elif probability < HIGH_RISK_THRESHOLD:
            risk_level = "MEDIUM"
        else:
            risk_level = "HIGH"

        return {
            "prediction": prediction,
            "failure_probability": round(probability, 4),
            "risk_level": risk_level,
            "prediction_threshold": PREDICTION_THRESHOLD,
        }
