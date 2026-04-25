import pandas as pd
import numpy as np
import os

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score,
    recall_score, f1_score, roc_auc_score
)

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

import matplotlib.pyplot as plt
import seaborn as sns

# Optional XGBoost
try:
    import xgboost as xgb
    XGB_AVAILABLE = True
except:
    print(" XGBoost not installed. Skipping...")
    XGB_AVAILABLE = False


# -------------------------
# LOAD DATA
# -------------------------
def load_data():
    df = pd.read_csv("data/processed_data.csv")

    # Create failure label
    df["failure"] = (
        (df["temp_deviation"] > 0.7) |
        (df["pressure_drift"] > 0.6) |
        (df["vibration_ratio"] > 0.7)
    ).astype(int)

    X = df.drop(columns=["timestamp", "failure"], errors="ignore")
    y = df["failure"]

    return train_test_split(X, y, test_size=0.2, random_state=42)


# -------------------------
# TRAIN & EVALUATE MODELS
# -------------------------
def train_and_compare(X_train, X_test, y_train, y_test):

    models = {
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "SVM": SVC(probability=True)
    }

    if XGB_AVAILABLE:
        models["XGBoost"] = xgb.XGBClassifier(
            use_label_encoder=False,
            eval_metric='logloss'
        )

    results = {}

    for name, model in models.items():

        print(f"\n🔹 Training {name}...")

        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]

        results[name] = {
            "Accuracy": accuracy_score(y_test, y_pred),
            "Precision": precision_score(y_test, y_pred),
            "Recall": recall_score(y_test, y_pred),
            "F1-score": f1_score(y_test, y_pred),
            "ROC-AUC": roc_auc_score(y_test, y_prob)
        }

        print(results[name])

    return pd.DataFrame(results).T


# -------------------------
# SAVE + PLOT RESULTS
# -------------------------
def save_and_plot(results_df):

    os.makedirs("data", exist_ok=True)
    os.makedirs("data/plots", exist_ok=True)

    # Save CSV
    results_df.to_csv("data/model_comparison.csv")

    print("\n Model Comparison Results:")
    print(results_df)

    # Plot comparison
    plt.figure(figsize=(10, 6))
    results_df.plot(kind='bar')
    plt.title("Model Comparison")
    plt.ylabel("Score")
    plt.xticks(rotation=45)

    plt.tight_layout()
    plt.savefig("data/plots/model_comparison.png")
    plt.close()

    print(" Saved → data/model_comparison.csv")
    print(" Plot → data/plots/model_comparison.png")


# -------------------------
# MAIN FUNCTION
# -------------------------
def run_all_models():

    print(" Running Model Comparison...")

    X_train, X_test, y_train, y_test = load_data()

    results_df = train_and_compare(X_train, X_test, y_train, y_test)

    save_and_plot(results_df)

    print(" Model comparison completed")