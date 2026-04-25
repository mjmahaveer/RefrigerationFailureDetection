import pandas as pd
import numpy as np
import os

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split


def run_lstm():

    print(" Running LSTM Failure Prediction...")

    df = pd.read_csv("data/processed_data.csv")

    # -------------------------
    # TIMESTAMP
    # -------------------------
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"])

    # -------------------------
    # CREATE FAILURE LABEL
    # -------------------------
    df["failure"] = (
        (df["temp_deviation"] > 0.7) |
        (df["pressure_drift"] > 0.6) |
        (df["vibration_ratio"] > 0.7)
    ).astype(int)

    # -------------------------
    # FEATURES
    # -------------------------
    features = df.select_dtypes(include=['float64', 'int64']).fillna(0)

    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(features)

    # -------------------------
    # CREATE SEQUENCES
    # -------------------------
    X, y, timestamps = [], [], []

    window = 72  #  6 HOURS

    for i in range(len(scaled) - window):
        X.append(scaled[i:i+window])
        y.append(df["failure"].iloc[i+window])

        # save timestamp
        if "timestamp" in df.columns:
            timestamps.append(df["timestamp"].iloc[i+window])
        else:
            timestamps.append(i)

    X, y = np.array(X), np.array(y)

    # -------------------------
    # TRAIN / TEST SPLIT
    # -------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # -------------------------
    # MODEL
    # -------------------------
    model = Sequential([
        LSTM(64, return_sequences=False, input_shape=(X.shape[1], X.shape[2])),
        Dense(1, activation='sigmoid')
    ])

    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    # -------------------------
    # TRAIN
    # -------------------------
    model.fit(X_train, y_train, epochs=5, batch_size=32, verbose=1)

    loss, acc = model.evaluate(X_test, y_test)
    print(f" LSTM Accuracy: {acc:.4f}")

    # -------------------------
    # FULL DATA PREDICTIONS (IMPORTANT)
    # -------------------------
    all_probs = model.predict(X).flatten()
    all_preds = (all_probs > 0.5).astype(int)

    df_results = pd.DataFrame({
        "timestamp": timestamps,
        "failure_probability": all_probs,
        "predicted_failure": all_preds
    })

    # -------------------------
    #  ALERT LOGIC
    # -------------------------
    df_results["alert"] = df_results["failure_probability"].apply(
        lambda x: "🔴 Critical" if x > 0.8 else
                  "🟡 Warning" if x > 0.5 else
                  "🟢 Normal"
    )

    # -------------------------
    #  FUTURE PREDICTION (6 HOURS)
    # -------------------------
    future_steps = 72

    last_seq = X[-1].copy()
    future_probs = []

    for _ in range(future_steps):
        pred = model.predict(last_seq.reshape(1, window, X.shape[2]))[0][0]
        future_probs.append(pred)

        # shift window
        new_row = last_seq[-1]
        last_seq = np.vstack([last_seq[1:], new_row])

    df_future = pd.DataFrame({
        "step": range(1, future_steps + 1),
        "future_failure_probability": future_probs
    })

    # -------------------------
    # SAVE
    # -------------------------
    os.makedirs("data", exist_ok=True)
    os.makedirs("models", exist_ok=True)

    df_results.to_csv("data/failure_predictions.csv", index=False)
    df_future.to_csv("data/future_predictions.csv", index=False)

    model.save("models/lstm_model.h5")

    print(" Saved → data/failure_predictions.csv")
    print(" Saved → data/future_predictions.csv")
    print(" LSTM model trained successfully")