import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st
import pandas as pd
import plotly.express as px
import json
from src.utils import REPORTS_DIR, FIGURES_DIR, DATA_RAW_DIR
from src.predict import FailurePredictor
from src.data_preprocessing import load_or_generate_data
from src.feature_engineering import engineer_features

st.set_page_config(
    page_title="Predictive Maintenance Dashboard",
    page_icon="⚙️",
    layout="wide"
)

@st.cache_resource
def get_predictor():
    return FailurePredictor()

predictor = get_predictor()

# Sidebar Navigation
st.sidebar.title("Navigation")
section = st.sidebar.radio("Go to", ["Overview", "Prediction", "Analytics", "Model Performance"])

# Load dataset and metrics for dashboard
df_raw = load_or_generate_data()
df = engineer_features(df_raw)
metrics_path = REPORTS_DIR / "model_metrics.json"
metrics_data = {}
if metrics_path.exists():
    with open(metrics_path, "r") as f:
        metrics_data = json.load(f)

if section == "Overview":
    st.title("⚙️ Industrial Predictive Maintenance Dashboard")
    st.markdown("Monitor machine telemetry, analyze degradation indicators, and predict asset failures proactively.")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Machines", f"{df_raw['machine_id'].nunique():,}")
    with col2:
        failure_rate = (df['failure'].mean()) * 100
        st.metric("Failure Rate", f"{failure_rate:.2f}%")
    with col3:
        best_roc = max([m.get("roc_auc", 0) for m in metrics_data.values()]) if metrics_data else 0.85
        st.metric("Top Model ROC-AUC", f"{best_roc:.3f}" if metrics_data else "N/A")
    with col4:
        best_f1 = max([m.get("f1_score", 0) for m in metrics_data.values()]) if metrics_data else 0.75
        st.metric("Top Model F1-Score", f"{best_f1:.3f}" if metrics_data else "N/A")

    st.markdown("---")
    st.subheader("Sample Sensor Telemetry Data")
    st.dataframe(df_raw.head(10), use_container_width=True)

elif section == "Prediction":
    st.title("🔮 Real-Time Machine Failure Prediction")
    st.markdown("Input operational sensor values below to evaluate failure risk.")

    with st.form("prediction_form"):
        col1, col2, col3 = st.columns(3)
        with col1:
            temperature = st.number_input("Temperature (°C)", min_value=0.0, max_value=200.0, value=75.0)
            vibration = st.number_input("Vibration (mm/s)", min_value=0.0, max_value=20.0, value=2.5)
            pressure = st.number_input("Pressure (bar)", min_value=0.0, max_value=200.0, value=100.0)
            rotational_speed = st.number_input("Rotational Speed (RPM)", min_value=0.0, max_value=5000.0, value=1500.0)
            torque = st.number_input("Torque (Nm)", min_value=0.0, max_value=150.0, value=40.0)
        with col2:
            operating_hours = st.number_input("Operating Hours", min_value=0.0, max_value=20000.0, value=4500.0)
            voltage = st.number_input("Voltage (V)", min_value=0.0, max_value=500.0, value=220.0)
            current = st.number_input("Current (A)", min_value=0.0, max_value=100.0, value=15.0)
            humidity = st.number_input("Humidity (%)", min_value=0.0, max_value=100.0, value=50.0)
            maintenance_count = st.number_input("Maintenance Count", min_value=0, max_value=100, value=2, step=1)
        with col3:
            machine_age = st.number_input("Machine Age (Years)", min_value=0.0, max_value=50.0, value=4.0)
            failure_history = st.selectbox("Failure History", [0, 1])
            tool_wear = st.number_input("Tool Wear (min)", min_value=0.0, max_value=100.0, value=45.0)
            load_percentage = st.number_input("Load Percentage (%)", min_value=0.0, max_value=100.0, value=75.0)

        submitted = st.form_submit_button("Predict Machine Failure")

    if submitted:
        input_data = {
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
            "load_percentage": load_percentage
        }

        try:
            res = predictor.predict(input_data)
            st.markdown("---")
            st.subheader("Prediction Results")
            
            c1, c2, c3 = st.columns(3)
            with c1:
                st.metric("Prediction", "Failure Likely" if res["prediction"] == 1 else "Normal")
            with c2:
                st.metric("Failure Probability", f"{res['failure_probability']*100:.2f}%")
            with c3:
                risk = res["risk_level"]
                color = "red" if risk == "HIGH" else ("orange" if risk == "MEDIUM" else "green")
                st.markdown(f"**Risk Level:** :{color}[{risk}]")

            st.caption("Prediction threshold: 50% failure probability. Risk bands: Low <30%, Medium 30–69.99%, High ≥70%.")

            if res["risk_level"] == "HIGH":
                st.error("⚠️ Recommendation: High predicted failure risk. Arrange maintenance inspection promptly.")
            elif res["risk_level"] == "MEDIUM":
                st.warning("⚡ Recommendation: Schedule inspection during the next routine window.")
            else:
                st.success("✅ Recommendation: Machine operating under normal parameters.")

        except Exception as e:
            st.error(f"Prediction failed: {e}")

elif section == "Analytics":
    st.title("📊 Exploratory Data Analytics")

    col1, col2 = st.columns(2)
    with col1:
        fig_dist = px.histogram(df_raw, x="temperature", color="failure", barmode="overlay", title="Temperature Distribution by Failure")
        st.plotly_chart(fig_dist, use_container_width=True)
    with col2:
        fig_vib = px.histogram(df_raw, x="vibration", color="failure", barmode="overlay", title="Vibration Distribution by Failure")
        st.plotly_chart(fig_vib, use_container_width=True)

    st.subheader("Correlation Heatmap")
    num_df = df_raw.select_dtypes(include=["number"])
    corr = num_df.corr()
    fig_corr = px.imshow(corr, text_auto=False, color_continuous_scale="RdBu_r", title="Feature Correlation Matrix")
    st.plotly_chart(fig_corr, use_container_width=True)

elif section == "Model Performance":
    st.title("🏆 Model Performance & Comparison")

    if metrics_data:
        metrics_df = pd.DataFrame(metrics_data).T.reset_index().rename(columns={"index": "Model"})
        st.dataframe(metrics_df, use_container_width=True)

        fig_bar = px.bar(metrics_df, x="Model", y=["f1_score", "roc_auc", "precision", "recall"], barmode="group", title="Model Metric Comparison")
        st.plotly_chart(fig_bar, use_container_width=True)
    else:
        st.info("No saved metrics found. Please run the training pipeline first (`python run_pipeline.py`).")
