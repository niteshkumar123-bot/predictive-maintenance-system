from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, ConfigDict

from src.predict import FailurePredictor
from src.utils import logger

app = FastAPI(
    title="Predictive Maintenance API",
    description="API for industrial machine failure prediction.",
    version="1.1.0",
)

predictor = FailurePredictor()


class SensorInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    temperature: float = Field(..., ge=0, le=200, examples=[78.5])
    vibration: float = Field(..., ge=0, le=20, examples=[3.2])
    pressure: float = Field(..., ge=0, le=200, examples=[101.2])
    rotational_speed: float = Field(..., ge=0, le=5000, examples=[1490.0])
    torque: float = Field(..., ge=0, le=150, examples=[42.1])
    operating_hours: float = Field(..., ge=0, le=20000, examples=[4500.0])
    voltage: float = Field(..., ge=0, le=500, examples=[221.0])
    current: float = Field(..., ge=0, le=100, examples=[15.2])
    humidity: float = Field(..., ge=0, le=100, examples=[55.0])
    maintenance_count: int = Field(..., ge=0, le=100, examples=[2])
    machine_age: float = Field(..., ge=0, le=50, examples=[4.5])
    failure_history: int = Field(..., ge=0, le=1, examples=[0])
    tool_wear: float = Field(..., ge=0, le=100, examples=[45.0])
    load_percentage: float = Field(..., ge=0, le=100, examples=[75.0])


@app.get("/health")
def health_check():
    artifacts_ready = predictor.model is not None and predictor.scaler is not None
    return {
        "status": "healthy" if artifacts_ready else "degraded",
        "service": "Predictive Maintenance API",
        "model_ready": artifacts_ready,
    }


@app.post("/predict")
def predict_failure(data: SensorInput):
    try:
        return predictor.predict(data.model_dump())
    except Exception as exc:
        logger.error(f"Prediction error: {exc}")
        raise HTTPException(status_code=503, detail=str(exc)) from exc
