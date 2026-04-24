
```markdown
# Failure Detection Project
#  Predictive Failure Detection in Industrial Refrigeration Systems

##  Overview
This project implements an **end-to-end IIoT + Machine Learning pipeline** for detecting, predicting, and classifying failures in industrial coolers and refrigeration systems.

It simulates sensor data, applies multiple ML models, performs anomaly detection, and visualizes everything in an interactive **Streamlit dashboard**.

---

##  Features

### 🔹 Data Pipeline
- Synthetic sensor data generation (5-min intervals)
- Preprocessing & feature engineering
- Time-series structured dataset

### 🔹 Machine Learning Models
- Random Forest
- Logistic Regression
- SVM
- XGBoost (optional)
- LSTM (time-series prediction)

### 🔹 Advanced Analytics
- Failure prediction (probability-based)
- Failure type classification:
  - 0 → Normal
  - 1 → Compressor Failure
  - 2 → Cooling Failure
  - 3 → Defrost Failure
  - 4 → Electrical Failure

### 🔹 Anomaly Detection
- Isolation Forest for detecting abnormal patterns

### 🔹 Explainability
- SHAP for model interpretability

### 🔹 Dashboard
- Real-time visualization (Streamlit)
- Sensor trends
- Failure prediction graph
- Classification insights
- Alerts & anomaly visualization

---

##  Project Structure

```

project/
│
├── main.py
├── dashboard.py
│
├── data/
│   ├── processed_data.csv
│   ├── failure_predictions.csv
│   ├── failure_classification.csv
│   ├── alert_logs.csv
│   └── model_results.csv
│
├── models/
│
└── src/
├── data_generator.py
├── preprocessing.py
├── model_runner.py
├── failure_prediction.py
├── classification.py
├── anomaly_detection.py
├── alerts.py
└── shap_explainer.py

````

---

##  Installation

```bash
pip install pandas numpy matplotlib seaborn scikit-learn streamlit plotly joblib xgboost tensorflow shap
````

---

##  How to Run

### Step 1: Run Full Pipeline

```bash
python main.py
```

This will:

* Generate data
* Train models
* Run predictions
* Detect anomalies
* Generate alerts
* Save all outputs

---

### Step 2: Launch Dashboard

```bash
streamlit run dashboard.py
```

Open in browser:

```
http://localhost:8501
```

---

##  Dashboard Modules

### Sensor Monitoring

* Temperature, pressure, vibration, etc.

### Failure Prediction

* LSTM-based probability over time
* Alerts:

  * Normal
  * Warning
  * Critical

### Failure Classification

* Type-wise failure detection
* Visual distribution

### 🚨 Anomaly Detection

* Isolation Forest visualization
* Highlighted abnormal points

### Explainability

* SHAP plots for feature importance

---

##  Key Outputs

| File                       | Description      |
| -------------------------- | ---------------- |
| processed_data.csv         | Clean dataset    |
| failure_predictions.csv    | LSTM predictions |
| failure_classification.csv | Failure types    |
| alert_logs.csv             | Generated alerts |
| model_results.csv          | Model accuracy   |

---

##  Tech Stack

* Python
* Scikit-learn
* TensorFlow / Keras
* XGBoost
* Streamlit
* Plotly
* SHAP

---

##  Use Cases

* Predictive maintenance
* Industrial IoT monitoring
* Smart refrigeration systems
* Fault diagnosis systems

---

##  Future Improvements

* Real-time streaming (Kafka / MQTT)
* Edge deployment (Raspberry Pi)
* Auto model retraining
* Live alert notifications
* API integration

---

##  Author

Developed as part of:
**IIoT + Machine Learning Project for Industrial Failure Detection**

---

```
