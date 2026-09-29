# Predictive Maintenance & Failure Detection System

An end-to-end machine-learning project that estimates the likelihood of industrial machine failure from sensor and operational data.

> **Important:** the current dataset is synthetic and is intended for demonstration/learning. The model should not be treated as an industrial safety system.

## What the project does

The system accepts machine telemetry such as temperature, vibration, pressure, rotational speed, torque, operating hours, voltage, current, humidity, machine age, failure history, tool wear and load percentage.

It then:

1. Generates/loads the dataset.
2. Creates domain-inspired derived features.
3. Splits and preprocesses the data using a fitted imputation/scaling pipeline.
4. Trains Logistic Regression, tuned Random Forest and tuned XGBoost models.
5. Compares accuracy, precision, recall, F1-score and ROC-AUC.
6. Selects the model with the highest F1-score.
7. Saves the trained model and preprocessing artifact.
8. Exposes predictions through a Streamlit dashboard and FastAPI endpoint.

## Prediction logic

- **Failure prediction threshold:** 50% probability.
- **LOW risk:** probability < 30%.
- **MEDIUM risk:** 30%–<70%.
- **HIGH risk:** >= 70%.

The risk level is a presentation rule; it is not a medical, safety, or industrial shutdown recommendation.

## Project structure

```text
predictive-maintenance-system/
├── api/                  # FastAPI service
├── dashboard/            # Streamlit UI
├── data/                 # Raw/processed data
├── models/               # Trained model + preprocessing artifact
├── notebooks/            # EDA script
├── reports/              # Metrics and plots
├── src/                  # ML pipeline modules
├── tests/                # Automated tests
├── run_pipeline.py       # End-to-end training pipeline
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── DEPLOYMENT.md
```

## Run locally

```bash
python -m venv venv
# Windows PowerShell
venv\Scripts\Activate.ps1

pip install -r requirements.txt
python run_pipeline.py
streamlit run dashboard/app.py
```

FastAPI:

```bash
uvicorn api.main:app --reload
```

API documentation is available at `/docs` while the API is running.

## Example API payload

```json
{
  "temperature": 75,
  "vibration": 2.5,
  "pressure": 100,
  "rotational_speed": 1500,
  "torque": 40,
  "operating_hours": 4500,
  "voltage": 220,
  "current": 15,
  "humidity": 50,
  "maintenance_count": 2,
  "machine_age": 4,
  "failure_history": 0,
  "tool_wear": 45,
  "load_percentage": 75
}
```

## Limitations

- The training data is synthetic.
- Synthetic labels are generated from predefined relationships, so performance does not establish real-world industrial accuracy.
- The project is a demonstration and should not be used for autonomous safety-critical decisions.
- A production version should use real time-series maintenance data, temporal validation, calibration, monitoring and domain validation.

## Future improvements

- Train on a public/real predictive-maintenance dataset.
- Add temporal/time-to-failure modeling.
- Calibrate predicted probabilities.
- Add drift and model monitoring.
- Add authentication and rate limiting to the API.
- Add CI tests and automated deployment.
