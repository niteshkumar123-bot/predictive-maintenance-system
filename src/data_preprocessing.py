import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
import joblib

from src.utils import DATA_RAW_DIR, DATA_PROC_DIR, MODELS_DIR, logger

def generate_synthetic_data(num_samples: int = 50000) -> pd.DataFrame:
    """Generate a realistic synthetic industrial predictive maintenance dataset."""
    logger.info(f"Generating synthetic industrial dataset with {num_samples} rows...")
    np.random.seed(42)

    machine_ids = [f"MACH_{i:04d}" for i in np.random.randint(1, 200, size=num_samples)]
    temperature = np.random.normal(75.0, 15.0, num_samples)
    vibration = np.random.exponential(2.5, num_samples)
    pressure = np.random.normal(100.0, 10.0, num_samples)
    rotational_speed = np.random.normal(1500.0, 150.0, num_samples)
    torque = np.random.normal(40.0, 8.0, num_samples)
    operating_hours = np.random.uniform(100, 9000, num_samples)
    voltage = np.random.normal(220.0, 5.0, num_samples)
    current = np.random.normal(15.0, 2.0, num_samples)
    humidity = np.random.uniform(20.0, 90.0, num_samples)
    maintenance_count = np.random.poisson(2, num_samples)
    machine_age = np.random.uniform(1.0, 10.0, num_samples)
    failure_history = np.random.binomial(1, 0.2, num_samples)
    tool_wear = np.random.uniform(0.0, 100.0, num_samples)
    load_percentage = np.random.uniform(30.0, 100.0, num_samples)

    # Compute failure probability based on physical wear indicators
    failure_score = (
        0.03 * (temperature - 75) +
        0.15 * (vibration - 2.5) +
        0.02 * (tool_wear) +
        0.01 * (operating_hours / 1000) +
        0.05 * (failure_history * 5) +
        0.04 * (torque - 40) +
        np.random.normal(0, 1.0, num_samples)
    )
    # Convert log-odds to probability
    prob = 1 / (1 + np.exp(-failure_score + 3.0))
    failure = np.random.binomial(1, np.clip(prob, 0.01, 0.95))

    df = pd.DataFrame({
        "machine_id": machine_ids,
        "temperature": temperature,
        "vibration": vibration,
        "pressure": pressure,
        "rotational_speed": rotational_speed,
        "torque": torque,
        "operating_hours": operating_hours,
        "voltage": voltage,
        "current": current,
        "humidity": humidity,
        "maintenance_count": maintenance_count,
        "machine_age": machine_age,
        "failure_history": failure_history,
        "tool_wear": tool_wear,
        "load_percentage": load_percentage,
        "failure": failure
    })

    DATA_RAW_DIR.mkdir(parents=True, exist_ok=True)
    file_path = DATA_RAW_DIR / "machine_data.csv"
    df.to_csv(file_path, index=False)
    logger.info(f"Synthetic dataset saved to {file_path}")
    return df

def load_or_generate_data() -> pd.DataFrame:
    """Load raw dataset if exists, otherwise generate it."""
    file_path = DATA_RAW_DIR / "machine_data.csv"
    if file_path.exists():
        logger.info(f"Loading existing raw dataset from {file_path}")
        return pd.read_csv(file_path)
    else:
        return generate_synthetic_data()

def preprocess_pipeline(df: pd.DataFrame, save_artifacts: bool = True):
    """Clean data, handle missing values, duplicates, split and scale."""
    logger.info("Starting data preprocessing pipeline...")
    
    # Clean duplicates and missing values
    initial_len = len(df)
    df = df.drop_duplicates()
    logger.info(f"Dropped {initial_len - len(df)} duplicate rows.")
    
    # Separate features and target
    if "machine_id" in df.columns:
        df = df.drop(columns=["machine_id"])
        
    X = df.drop(columns=["failure"])
    y = df["failure"]

    numerical_cols = X.select_dtypes(include=[np.number]).columns.tolist()

    # Preprocessing transformer
    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numerical_cols)
        ]
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    X_train_transformed = preprocessor.fit_transform(X_train)
    X_test_transformed = preprocessor.transform(X_test)

    if save_artifacts:
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        joblib.dump(preprocessor, MODELS_DIR / "scaler.pkl")
        logger.info("Saved preprocessing scaler pipeline to models/scaler.pkl")

    DATA_PROC_DIR.mkdir(parents=True, exist_ok=True)
    np.save(DATA_PROC_DIR / "X_train.npy", X_train_transformed)
    np.save(DATA_PROC_DIR / "X_test.npy", X_test_transformed)
    np.save(DATA_PROC_DIR / "y_train.npy", y_train.to_numpy())
    np.save(DATA_PROC_DIR / "y_test.npy", y_test.to_numpy())

    logger.info("Preprocessing complete. Data saved to processed directory.")
    return X_train_transformed, X_test_transformed, y_train.to_numpy(), y_test.to_numpy(), preprocessor
