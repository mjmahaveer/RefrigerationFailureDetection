import pandas as pd
import numpy as np
import os
import joblib

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report


def run_classification():

    print(" Running Failure Type Classification...")

    df = pd.read_csv("data/processed_data.csv")

    # -------------------------
    # FAILURE LABEL MAPPING
    # -------------------------
    failure_map = {
        0: "Normal",
        1: "Compressor Failure",
        2: "Cooling Failure",
        3: "Defrost Failure",
        4: "Electrical Failure"
    }

    # -------------------------
    # CREATE FAILURE TYPE LABEL
    # -------------------------
    df["failure_type"] = 0  # default = Normal

    # Priority-based assignment (important!)
    df.loc[df["current_overload"] > 0.7, "failure_type"] = 4   # Electrical
    df.loc[df["defrost_impact"] > 0.5, "failure_type"] = 3     # Defrost
    df.loc[df["temp_deviation"] > 0.7, "failure_type"] = 2     # Cooling
    df.loc[df["vibration_ratio"] > 0.7, "failure_type"] = 1    # Compressor

    # -------------------------
    # FEATURE SELECTION
    # -------------------------
    X = df.select_dtypes(include=['float64', 'int64']).copy()

    # Drop unwanted
    X = X.drop(columns=["failure_type"], errors="ignore")
    X = X.drop(columns=["timestamp"], errors="ignore")

    y = df["failure_type"]

    # -------------------------
    # TRAIN TEST SPLIT
    # -------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # -------------------------
    # MODEL
    
    # -------------------------
    model = RandomForestClassifier(n_estimators=100, random_state=42)

    model.fit(X_train, y_train)

    # -------------------------
    # EVALUATION
    # -------------------------
    y_pred = model.predict(X_test)

    print("\n Classification Report:")
    print(classification_report(y_test, y_pred))

    # -------------------------
    # SAVE MODEL
    # -------------------------
    os.makedirs("models", exist_ok=True)
    joblib.dump(model, "models/random_forest.pkl")

    # -------------------------
    # SAVE RESULTS ( UPDATED)
    # -------------------------
    df_results = X_test.copy()

    # df_results["actual"] = y_test.values
    # df_results["predicted"] = y_pred
    # df_results = X_test.copy()

    # Add timestamp back (IMPORTANT)
    if "timestamp" in df.columns:
        df_results["timestamp"] = df.loc[X_test.index, "timestamp"]

    df_results["actual"] = y_test.values
    df_results["predicted"] = y_pred

    # Convert timestamp properly
    if "timestamp" in df_results.columns:
        df_results["timestamp"] = pd.to_datetime(df_results["timestamp"], errors="coerce")



    #  ADD HUMAN-READABLE LABELS
    df_results["actual_label"] = df_results["actual"].map(failure_map)
    df_results["predicted_label"] = df_results["predicted"].map(failure_map)

    #  ADD SEVERITY LEVEL
    def get_severity(label):
        if label in ["Cooling Failure", "Electrical Failure"]:
            return "HIGH"
        elif label in ["Compressor Failure", "Defrost Failure"]:
            return "MEDIUM"
        else:
            return "LOW"

    df_results["severity"] = df_results["predicted_label"].apply(get_severity)

    # -------------------------
    # SAVE FILE
    # -------------------------
    os.makedirs("data", exist_ok=True)
    df_results.to_csv("data/failure_classification.csv", index=False)

    print(" Saved → data/failure_classification.csv")
    print(" Classification completed successfully")