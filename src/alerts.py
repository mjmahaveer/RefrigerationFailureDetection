import pandas as pd
import os

def run_alerts():

    print(" Running Alert System...")

    df = pd.read_csv("data/processed_data.csv")

    alerts = []

    for _, row in df.iterrows():

        alert_type = None

        #  Critical (Failure prediction)
        if "predicted_probability" in row and row["predicted_probability"] > 0.8:
            alert_type = "CRITICAL FAILURE RISK"

        #  Warning (Anomaly)
        elif "anomaly" in row and row["anomaly"] == 1:
            alert_type = "ANOMALY DETECTED"

        #  Threshold alerts
        elif row.get("temp_deviation", 0) > 0.7:
            alert_type = "HIGH TEMPERATURE DEVIATION"

        if alert_type:
            alerts.append({
                "timestamp": row.get("timestamp"),
                "unit_id": row.get("unit_id"),
                "alert": alert_type
            })

    alert_df = pd.DataFrame(alerts)

    os.makedirs("data", exist_ok=True)
    alert_df.to_csv("data/alert_logs.csv", index=False)

    print(f" Alerts generated: {len(alert_df)}")