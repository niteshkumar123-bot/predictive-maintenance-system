import json
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)

from src.utils import REPORTS_DIR, FIGURES_DIR, MODELS_DIR, save_json, logger

try:
    import mlflow
    import mlflow.sklearn
    MLFLOW_AVAILABLE = True
except ImportError:
    mlflow = None
    MLFLOW_AVAILABLE = False


def evaluate_models(models, X_test, y_test):
    """Evaluate models, select the highest-F1 model, and save artifacts."""
    logger.info("Evaluating trained models...")
    metrics_summary = {}
    best_model_name = None
    best_f1 = -1.0
    best_model = None

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    for name, model in models.items():
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else y_pred

        metrics_summary[name] = {
            "accuracy": float(accuracy_score(y_test, y_pred)),
            "precision": float(precision_score(y_test, y_pred, zero_division=0)),
            "recall": float(recall_score(y_test, y_pred, zero_division=0)),
            "f1_score": float(f1_score(y_test, y_pred, zero_division=0)),
            "roc_auc": float(roc_auc_score(y_test, y_prob)),
        }

        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(5, 4))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
        plt.title(f"Confusion Matrix - {name}")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        safe_name = name.lower().replace(" ", "_")
        plt.savefig(FIGURES_DIR / f"confusion_matrix_{safe_name}.png", bbox_inches="tight")
        plt.close()

        if metrics_summary[name]["f1_score"] > best_f1:
            best_f1 = metrics_summary[name]["f1_score"]
            best_model_name = name
            best_model = model

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    save_json(metrics_summary, REPORTS_DIR / "model_metrics.json")

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_model, MODELS_DIR / "best_model.pkl")
    logger.info(f"Best model selected by F1: {best_model_name} ({best_f1:.4f})")

    if MLFLOW_AVAILABLE:
        try:
            with mlflow.start_run(run_name="Best_Model_Registration"):
                mlflow.log_metric("best_f1_score", best_f1)
                mlflow.sklearn.log_model(best_model, "best_model")
        except Exception as exc:
            logger.warning(f"Could not log best model to MLflow: {exc}")

    return metrics_summary, best_model_name
