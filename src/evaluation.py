import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

from sklearn.metrics import (
    confusion_matrix, roc_curve, auc,
    accuracy_score, precision_score,
    recall_score, f1_score, classification_report
)
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

# -------------------------
# OPTIONAL XGBOOST
# -------------------------
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

    df["failure"] = (
        (df["temp_deviation"] > 0.7) |
        (df["pressure_drift"] > 0.6) |
        (df["vibration_ratio"] > 0.7)
    ).astype(int)

    #  Class distribution check
    print("\n Class distribution:")
    print(df["failure"].value_counts())

    X = df.drop(columns=["timestamp", "failure"], errors="ignore")
    y = df["failure"]

    return train_test_split(X, y, test_size=0.2, random_state=42)

# -------------------------
# TRAIN MODELS
# -------------------------
def train_models(X_train, y_train):

    models = {
        "Random Forest": RandomForestClassifier(n_estimators=100),
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "SVM": SVC(probability=True, kernel='rbf')
    }

    if XGB_AVAILABLE:
        models["XGBoost"] = xgb.XGBClassifier(
            use_label_encoder=False,
            eval_metric='logloss'
        )

    trained = {}

    for name, model in models.items():
        print(f" Training {name}...")
        model.fit(X_train, y_train)
        trained[name] = model

    return trained

# -------------------------
# CONFUSION MATRIX
# -------------------------
def plot_confusion_matrices(models, X_test, y_test):
    os.makedirs("data/plots", exist_ok=True)

    for name, model in models.items():
        y_pred = model.predict(X_test)

        cm = confusion_matrix(y_test, y_pred)

        plt.figure(figsize=(5,4))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(f"{name} - Confusion Matrix")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")

        plt.savefig(f"data/plots/{name}_confusion_matrix.png")
        plt.close()

# -------------------------
# ROC CURVES
# -------------------------
def plot_roc_curves(models, X_test, y_test):
    plt.figure(figsize=(8,6))

    for name, model in models.items():

        #  Safe probability handling
        if hasattr(model, "predict_proba"):
            y_prob = model.predict_proba(X_test)[:, 1]
        else:
            y_prob = model.decision_function(X_test)

        fpr, tpr, _ = roc_curve(y_test, y_prob)
        roc_auc = auc(fpr, tpr)

        plt.plot(fpr, tpr, label=f"{name} (AUC={roc_auc:.2f})")

    plt.plot([0, 1], [0, 1], linestyle='--')

    plt.title("ROC Curve Comparison")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend()

    os.makedirs("data/plots", exist_ok=True)
    plt.savefig("data/plots/roc_comparison.png")
    plt.close()

# -------------------------
# MODEL METRICS
# -------------------------
def calculate_metrics(models, X_test, y_test):

    results = {}

    for name, model in models.items():
        y_pred = model.predict(X_test)

        #  Safe metrics
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)

        results[name] = {
            "Accuracy": acc,
            "Precision": prec,
            "Recall": rec,
            "F1-score": f1
        }

        print(f"\n {name}")
        print(classification_report(y_test, y_pred))

    df = pd.DataFrame(results).T

    #  Ensure folder exists
    os.makedirs("data", exist_ok=True)
    df.to_csv("data/model_metrics.csv")

    return df

# -------------------------
# MODEL COMPARISON PLOT
# -------------------------
def plot_model_comparison(df):

    plt.figure(figsize=(10,6))
    df.plot(kind='bar')

    plt.title("Model Performance Comparison")
    plt.ylabel("Score")
    plt.xticks(rotation=45)
    plt.grid(axis='y')

    os.makedirs("data/plots", exist_ok=True)
    plt.savefig("data/plots/model_comparison.png")
    plt.close()

# -------------------------
# MAIN FUNCTION
# -------------------------
def run_evaluation():

    X_train, X_test, y_train, y_test = load_data()

    models = train_models(X_train, y_train)

    plot_confusion_matrices(models, X_test, y_test)
    plot_roc_curves(models, X_test, y_test)

    metrics_df = calculate_metrics(models, X_test, y_test)
    plot_model_comparison(metrics_df)

    print("\n Final Metrics:")
    print(metrics_df)

    print("\n Evaluation complete → check data/plots/")