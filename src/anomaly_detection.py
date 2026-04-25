import pandas as pd
from sklearn.ensemble import IsolationForest
import joblib
import os

def run_anomaly_detection():

    print(" Running Anomaly Detection...")

    df = pd.read_csv("data/processed_data.csv")

    # -------------------------
    # TIMESTAMP FIX
    # -------------------------
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")

    # -------------------------
    # FEATURE SELECTION
    # -------------------------
    features = df.select_dtypes(include=['float64', 'int64']).drop(
        columns=["failure"], errors='ignore'
    ).fillna(0)

    # -------------------------
    # MODEL
    # -------------------------
    model = IsolationForest(
        contamination=0.05,
        random_state=42
    )

    preds = model.fit_predict(features)

    # -------------------------
    # CLEAN OUTPUT
    # -------------------------
    df["anomaly"] = preds
    df["anomaly_flag"] = df["anomaly"].apply(lambda x: 1 if x == -1 else 0)

    # -------------------------
    # SAVE MODEL
    # -------------------------
    os.makedirs("models", exist_ok=True)
    joblib.dump(model, "models/isolation_forest.pkl")

    # -------------------------
    # SAVE FULL RESULTS
    # -------------------------
    os.makedirs("data", exist_ok=True)

    df.to_csv("data/anomaly_results.csv", index=False)

    # -------------------------
    # SAVE ONLY ANOMALIES (LOG)
    # -------------------------
    anomaly_log = df[df["anomaly_flag"] == 1]

    anomaly_log.to_csv("data/anomaly_logs.csv", index=False)

    print(f" Total anomalies detected: {len(anomaly_log)}")
    print(" Saved → data/anomaly_results.csv")
    print(" Saved → data/anomaly_logs.csv")

    print(" Anomaly detection completed")