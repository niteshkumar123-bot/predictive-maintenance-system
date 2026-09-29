# Deployment

## Local

1. Create/activate a virtual environment.
2. Install dependencies:
   `pip install -r requirements.txt`
3. Generate the dataset, preprocessing artifact, trained models, metrics and EDA reports:
   `python run_pipeline.py`
4. Run Streamlit:
   `streamlit run dashboard/app.py`
5. Run FastAPI:
   `uvicorn api.main:app --host 0.0.0.0 --port 8000`

## Streamlit Community Cloud

The repository should contain these generated artifacts before deployment:

- `models/best_model.pkl`
- `models/scaler.pkl`
- `reports/model_metrics.json`
- `data/raw/machine_data.csv`

Set the Streamlit entrypoint to `dashboard/app.py`.

The dashboard adds the project root to `sys.path`, so the `src` package can be imported reliably when Streamlit is launched from the dashboard directory.

## Docker

The Docker image runs `python run_pipeline.py` during the image build, so a clean image contains the model artifacts required by the API and dashboard.

For local Docker Compose:

`docker compose up --build`
