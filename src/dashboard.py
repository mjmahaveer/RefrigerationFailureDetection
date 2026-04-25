import streamlit as st
import pandas as pd
import plotly.express as px
import os

st.set_page_config(layout="wide")

st.title(" Industrial Refrigeration Monitoring Dashboard")

# -------------------------
# LOAD DATA
# -------------------------
try:
    df = pd.read_csv("data/processed_data.csv")

    if df.empty:
        st.error(" Data file is empty. Run pipeline first.")
        st.stop()

except Exception as e:
    st.error(f" Error loading data: {e}")
    st.stop()

# -------------------------
# FIX TIMESTAMP (CLEAN)
# -------------------------
if "timestamp" in df.columns:
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["timestamp"])
    df = df.sort_values("timestamp")

# -------------------------
# FAILURE TYPE MAPPING
# -------------------------
failure_map = {
    0: "Normal",
    1: "Compressor Failure",
    2: "Cooling Failure",
    3: "Defrost Failure",
    4: "Electrical Failure"
}

failure_colors = {
    "Normal": "green",
    "Compressor Failure": "orange",
    "Cooling Failure": "red",
    "Defrost Failure": "purple",
    "Electrical Failure": "black"
}

# -------------------------
# SIDEBAR
# -------------------------
st.sidebar.header("⚙️ Controls")

if "unit_id" in df.columns:
    units = df["unit_id"].unique()
    selected_unit = st.sidebar.selectbox("Select Unit", units)
    df = df[df["unit_id"] == selected_unit]

# -------------------------
# DATE FILTER
# -------------------------
if "timestamp" in df.columns:
    df["date"] = df["timestamp"].dt.date

    min_date = df["date"].min()
    max_date = df["date"].max()

    if min_date != max_date:
        date_range = st.sidebar.slider(
            "Select Date Range",
            min_value=min_date,
            max_value=max_date,
            value=(min_date, max_date)
        )

        df = df[
            (df["date"] >= date_range[0]) &
            (df["date"] <= date_range[1])
        ]

# -------------------------
# KPI METRICS
# -------------------------
st.subheader(" Key Metrics")

cols = st.columns(4)

try:
    cols[0].metric("Temperature", round(df["temperature_sensor"].iloc[-1], 2))
    cols[1].metric("Pressure", round(df["pressure_sensor"].iloc[-1], 2))
    cols[2].metric("Current", round(df["current_sensor"].iloc[-1], 2))
    cols[3].metric("Humidity", round(df["humidity_sensor"].iloc[-1], 2))
except:
    st.warning("KPI unavailable")

# -------------------------
# MAIN TREND
# -------------------------
st.subheader(" Sensor Trends")

params = [
    "temperature_sensor",
    "pressure_sensor",
    "vibration_sensor",
    "current_sensor"
]

params = [p for p in params if p in df.columns]

fig = px.line(
    df,
    x="timestamp",
    y=params,
    title="Sensor Trends Over Time"
)
st.plotly_chart(fig, use_container_width=True)

# -------------------------
# LSTM FAILURE PREDICTION
# -------------------------
st.subheader(" Failure Prediction")

try:
    pred_df = pd.read_csv("data/failure_predictions.csv")

    if "timestamp" in pred_df.columns:
        pred_df["timestamp"] = pd.to_datetime(pred_df["timestamp"], errors="coerce")

    fig = px.line(
        pred_df,
        x="timestamp" if "timestamp" in pred_df.columns else pred_df.index,
        y="failure_probability",
        title="Failure Probability"
    )

    st.plotly_chart(fig, use_container_width=True)

    # ALERT COLORS
    def color_alert(val):
        if val == "🔴 Critical":
            return "background-color:red;color:white"
        elif val == "🟡 Warning":
            return "background-color:orange"
        return "background-color:green;color:white"

    if "alert" in pred_df.columns:
        st.dataframe(
            pred_df.tail(20).style.applymap(color_alert, subset=["alert"])
        )
    else:
        st.dataframe(pred_df.tail(20))

except:
    st.warning("Run failure_prediction.py first")

# -------------------------
# FAILURE TYPE CLASSIFICATION
# -------------------------
st.subheader(" Failure Type Classification")

try:
    cls_df = pd.read_csv("data/failure_classification.csv")

    cls_df["actual_label"] = cls_df["actual"].map(failure_map)
    cls_df["predicted_label"] = cls_df["predicted"].map(failure_map)

    # st.dataframe(cls_df.tail(20))

    if "timestamp" in cls_df.columns:
        cls_df = cls_df.sort_values("timestamp")

    st.dataframe(cls_df.tail(20))

    fig = px.histogram(
        cls_df,
        x="predicted_label",
        color="predicted_label",
        title="Failure Type Distribution",
        color_discrete_map=failure_colors
    )

    st.plotly_chart(fig, use_container_width=True)

except:
    st.warning("Run classification.py first")

# -------------------------
#  ANOMALY DETECTION
# -------------------------
st.subheader(" Anomaly Detection")

if "anomaly" in df.columns:

    fig = px.scatter(
        df,
        x="timestamp",
        y="temperature_sensor",
        color=df["anomaly"].astype(str),
        title="Anomaly Detection (Isolation Forest)",
    )

    st.plotly_chart(fig, use_container_width=True)

    anomalies = df[df["anomaly"] == -1]

    if not anomalies.empty:
        st.error(f" {len(anomalies)} anomalies detected")
        st.dataframe(anomalies.tail(10))
    else:
        st.success(" No anomalies")

else:
    st.warning("Run anomaly_detection.py first")

# -------------------------
#  ALERT LOGS
# -------------------------
st.subheader(" Alerts")

try:
    alerts = pd.read_csv("data/alert_logs.csv")

    if not alerts.empty:
        st.dataframe(alerts.tail(20))
    else:
        st.info("No alerts")

except:
    st.warning("Alert log not found")

# -------------------------
# DATA PREVIEW
# -------------------------
st.subheader(" Data Preview")
st.dataframe(df.tail(50))

# -------------------------
# SHAP
# -------------------------
st.subheader(" Explainability (SHAP)")

shap_folder = "data/shap"

if os.path.exists(shap_folder):

    col1, col2 = st.columns(2)

    try:
        col1.image(f"{shap_folder}/summary_plot.png")
        col2.image(f"{shap_folder}/bar_plot.png")
        st.image(f"{shap_folder}/force_plot.png")

    except:
        st.warning("Run SHAP module")

else:
    st.warning("SHAP not found")