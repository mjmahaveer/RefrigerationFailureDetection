import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os


def generate_sensor_data():

    os.makedirs("data", exist_ok=True)

    rows = []
    start = datetime.now() - timedelta(days=10)

    for unit in range(1, 11):  # 10 refrigeration units

        prev_pressure = 60
        compressor_on = 1
        runtime = 0

        for i in range(2000):  # more data = better ML
            ts = start + timedelta(minutes=5*i)

            # -------------------------
            # ENVIRONMENT
            # -------------------------
            ambient_temp = np.random.normal(30, 3)
            door_open = np.random.choice([0, 1], p=[0.9, 0.1])

            # -------------------------
            # COMPRESSOR LOGIC
            # -------------------------
            if np.random.rand() < 0.05:
                compressor_on = 1 - compressor_on  # toggle

            runtime = runtime + 5 if compressor_on else 0

            # -------------------------
            # CORE SENSORS (REALISTIC)
            # -------------------------
            if compressor_on:
                temperature = np.random.normal(4, 0.5)
            else:
                temperature = np.random.normal(8, 1.5)

            temperature += door_open * np.random.uniform(1, 3)

            pressure = 60 + temperature * 0.7 + np.random.normal(0, 1.5)

            current = np.random.uniform(6, 12) if compressor_on else np.random.uniform(1, 3)

            vibration = current * 0.25 + np.random.normal(0, 0.1)

            humidity = np.random.uniform(50, 80)

            # -------------------------
            # REFRIGERATION PARAMETERS
            # -------------------------
            active_sp = np.random.uniform(2, 6)
            control_temp = active_sp + np.random.normal(0, 0.3)
            case_temp = temperature + np.random.normal(0, 0.4)

            defrost = np.random.choice([0, 1], p=[0.9, 0.1])

            if defrost:
                case_temp += np.random.uniform(2, 5)

            alarm_high = active_sp + 4
            alarm_low = active_sp - 3

            # -------------------------
            # ADDITIONAL INDUSTRIAL PARAMETERS
            # -------------------------
            suction_pressure = pressure - np.random.uniform(5, 10)
            discharge_pressure = pressure + np.random.uniform(10, 20)

            power_consumption = current * np.random.uniform(220, 240)

            energy_efficiency = temperature / (power_consumption + 1)

            # -------------------------
            # FAILURE INJECTION (SMART)
            # -------------------------
            failure_type = 0

            rand = np.random.rand()

            # Compressor failure
            if rand < 0.02:
                vibration += np.random.uniform(1, 2)
                current += np.random.uniform(3, 5)
                failure_type = 1

            # Cooling failure
            elif rand < 0.04:
                temperature += np.random.uniform(4, 7)
                pressure += np.random.uniform(5, 10)
                failure_type = 2

            # Defrost failure
            elif rand < 0.06:
                defrost = 1
                case_temp += np.random.uniform(5, 8)
                failure_type = 3

            # Electrical failure
            elif rand < 0.08:
                current += np.random.uniform(5, 8)
                failure_type = 4

            # -------------------------
            # DERIVED FEATURES
            # -------------------------
            temp_deviation = temperature - active_sp

            pressure_drift = pressure - prev_pressure
            prev_pressure = pressure

            vibration_ratio = vibration / current if current != 0 else 0
            current_overload = current / 12

            high_temp_alarm = 1 if case_temp > alarm_high else 0
            low_temp_alarm = 1 if case_temp < alarm_low else 0

            defrost_impact = defrost * temperature

            # -------------------------
            # APPEND ROW
            # -------------------------
            rows.append([
                ts, unit,

                # core
                temperature, pressure, vibration, current, humidity,

                # refrigeration
                case_temp, defrost, active_sp, control_temp,
                alarm_high, alarm_low,

                # industrial extras
                suction_pressure, discharge_pressure,
                power_consumption, ambient_temp,
                door_open, runtime, energy_efficiency,

                # intelligence
                temp_deviation, pressure_drift, vibration_ratio,
                current_overload, high_temp_alarm,
                low_temp_alarm, defrost_impact,

                # label
                failure_type
            ])

    columns = [
        "timestamp", "unit_id",

        "temperature_sensor",
        "pressure_sensor",
        "vibration_sensor",
        "current_sensor",
        "humidity_sensor",

        "case_temperature",
        "defrost_cycle_status",
        "active_setpoint",
        "control_temperature",
        "alarm_high_setpoint",
        "alarm_low_setpoint",

        "suction_pressure",
        "discharge_pressure",
        "power_consumption",
        "ambient_temperature",
        "door_open",
        "compressor_runtime",
        "energy_efficiency",

        "temp_deviation",
        "pressure_drift",
        "vibration_ratio",
        "current_overload",
        "high_temp_alarm",
        "low_temp_alarm",
        "defrost_impact",

        "failure_type"
    ]

    df = pd.DataFrame(rows, columns=columns)

    print(" Generated rows:", len(df))
    print(" Realistic industrial dataset created")

    df.to_csv("data/raw_sensor_data.csv", index=False)