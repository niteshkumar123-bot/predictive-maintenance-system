import numpy as np
import pandas as pd
from src.utils import logger

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create domain-specific derived features for predictive maintenance."""
    logger.info("Applying feature engineering...")
    df = df.copy()

    # Derived features
    df["temperature_pressure_ratio"] = df["temperature"] / (df["pressure"] + 1e-5)
    df["vibration_temperature_interaction"] = df["vibration"] * df["temperature"]
    df["power_consumption"] = df["voltage"] * df["current"]
    
    # Age ratio: Operating hours relative to estimated maximum lifespan (approx 10 years * 8760 hours/year)
    df["operating_age_ratio"] = df["operating_hours"] / (df["machine_age"] * 8760.0 + 1e-5)
    
    # Maintenance frequency relative to runtime
    df["maintenance_frequency"] = df["maintenance_count"] / (df["operating_hours"] + 1e-5)
    
    # Wear rate
    df["wear_rate"] = df["tool_wear"] / (df["operating_hours"] + 1e-5)

    logger.info(f"Feature engineering completed. Total columns: {df.shape[1]}")
    return df
