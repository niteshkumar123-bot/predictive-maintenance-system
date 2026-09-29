import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import RandomizedSearchCV

from src.utils import logger

try:
    import mlflow
    MLFLOW_AVAILABLE = True
except ImportError:
    mlflow = None
    MLFLOW_AVAILABLE = False


def train_and_tune_models(X_train: np.ndarray, y_train: np.ndarray):
    """Train Logistic Regression, tuned Random Forest and tuned XGBoost."""
    logger.info("Initializing model training and tuning...")

    if MLFLOW_AVAILABLE:
        mlflow.set_experiment("predictive_maintenance_experiment")

    models_results = {}

    # 1. Logistic Regression
    logger.info("Training Logistic Regression...")
    lr = LogisticRegression(max_iter=1000, random_state=42, class_weight="balanced")
    lr.fit(X_train, y_train)
    models_results["Logistic Regression"] = lr
    if MLFLOW_AVAILABLE:
        with mlflow.start_run(run_name="Logistic_Regression"):
            mlflow.log_param("model_type", "LogisticRegression")

    # 2. Random Forest with RandomizedSearchCV
    logger.info("Tuning Random Forest...")
    rf = RandomForestClassifier(random_state=42, class_weight="balanced")
    rf_params = {
        "n_estimators": [50, 100],
        "max_depth": [None, 10, 20],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
    }
    rf_search = RandomizedSearchCV(
        rf, rf_params, n_iter=2, cv=2, scoring="f1", random_state=42, n_jobs=-1
    )
    rf_search.fit(X_train, y_train)
    best_rf = rf_search.best_estimator_
    models_results["Random Forest"] = best_rf
    if MLFLOW_AVAILABLE:
        with mlflow.start_run(run_name="Random_Forest"):
            mlflow.log_params(rf_search.best_params_)
            mlflow.log_param("model_type", "RandomForest")

    # 3. XGBoost with RandomizedSearchCV
    logger.info("Tuning XGBoost...")
    positives = max(int(np.sum(y_train == 1)), 1)
    negatives = int(np.sum(y_train == 0))
    scale_pos_weight = float(negatives) / positives
    xgb = XGBClassifier(
        random_state=42,
        scale_pos_weight=scale_pos_weight,
        eval_metric="logloss",
        tree_method="hist",
    )
    xgb_params = {
        "n_estimators": [50, 100],
        "max_depth": [3, 5],
        "learning_rate": [0.05, 0.1],
        "subsample": [0.8, 1.0],
        "colsample_bytree": [0.8, 1.0],
    }
    xgb_search = RandomizedSearchCV(
        xgb, xgb_params, n_iter=2, cv=2, scoring="f1", random_state=42, n_jobs=-1
    )
    xgb_search.fit(X_train, y_train)
    best_xgb = xgb_search.best_estimator_
    models_results["XGBoost"] = best_xgb
    if MLFLOW_AVAILABLE:
        with mlflow.start_run(run_name="XGBoost"):
            mlflow.log_params(xgb_search.best_params_)
            mlflow.log_param("model_type", "XGBoost")

    logger.info("All models trained successfully.")
    return models_results
