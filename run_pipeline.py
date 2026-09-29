from src.data_preprocessing import load_or_generate_data, preprocess_pipeline
from src.feature_engineering import engineer_features
from src.train import train_and_tune_models
from src.evaluate import evaluate_models
from notebooks.exploratory_analysis import run_eda
from src.utils import ensure_directories, logger

def main():
    logger.info("Starting Predictive Maintenance Pipeline...")
    ensure_directories()

    # 1. Load or Generate Dataset
    df = load_or_generate_data()

    # 2. Feature Engineering
    df = engineer_features(df)

    # 3. Preprocessing and Splitting
    X_train, X_test, y_train, y_test, preprocessor = preprocess_pipeline(df, save_artifacts=True)

    # 4. Train Models & Hyperparameter Tuning
    models = train_and_tune_models(X_train, y_train)

    # 5. Evaluate and Save Best Model & Metrics
    evaluate_models(models, X_test, y_test)

    # 6. Generate EDA Reports & Figures
    run_eda()

    logger.info("Pipeline execution successfully completed!")

if __name__ == "__main__":
    main()
