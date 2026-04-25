import pandas as pd
from sklearn.preprocessing import MinMaxScaler

def preprocess_data():
    print(" Loading raw data...")
    df = pd.read_csv("data/raw_sensor_data.csv")

    if df.empty:
        print(" Raw data is empty")
        return

    # -------------------------
    # BASIC CLEANING
    # -------------------------
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df.sort_values(["unit_id", "timestamp"], inplace=True)

    df.fillna(method="ffill", inplace=True)
    df.fillna(method="bfill", inplace=True)

    # -------------------------
    # RECOMPUTE EXTRA INTELLIGENCE (CORRECT WAY)
    # -------------------------
    print(" Recomputing engineered features...")

    # 1. Temp deviation (correct per row)
    df["temp_deviation"] = df["temperature_sensor"] - df["active_setpoint"]

    # 2. Pressure drift (per unit)
    df["pressure_drift"] = df.groupby("unit_id")["pressure_sensor"].diff().fillna(0)

    # 3. Vibration ratio
    df["vibration_ratio"] = df["vibration_sensor"] / df["current_sensor"].replace(0, 1)

    # 4. Current overload (per unit normalization)
    df["current_overload"] = df.groupby("unit_id")["current_sensor"].transform(
        lambda x: x / x.max()
    )

    # 5. Alarm flags (recompute cleanly)
    df["high_temp_alarm"] = (df["case_temperature"] > df["alarm_high_setpoint"]).astype(int)
    df["low_temp_alarm"] = (df["case_temperature"] < df["alarm_low_setpoint"]).astype(int)

    # 6. Defrost impact
    df["defrost_impact"] = df["defrost_cycle_status"] * df["temperature_sensor"]

    # -------------------------
    # VERIFY FEATURES
    # -------------------------
    print(" Final columns:")
    print(df.columns.tolist())

    # -------------------------
    # NORMALIZATION (SAFE)
    # -------------------------
    scaler = MinMaxScaler()

    numeric_cols = df.select_dtypes(include=["float64", "int64"]).columns

    # Keep binary flags unscaled
    exclude_cols = [
        "high_temp_alarm",
        "low_temp_alarm",
        "defrost_cycle_status"
    ]

    numeric_cols = [c for c in numeric_cols if c not in exclude_cols]

    df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

    # -------------------------
    # SAVE OUTPUT
    # -------------------------
    df.to_csv("data/processed_data.csv", index=False)

    print(" Processed data saved successfully!")