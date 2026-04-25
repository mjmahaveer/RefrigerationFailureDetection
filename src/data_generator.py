import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os

def generate_sensor_data():
    os.makedirs("data", exist_ok=True)

    rows = []
    start = datetime.now() - timedelta(days=5)

    for unit in range(1, 11):  # 10 units
        prev_pressure = None

        for i in range(1000):
            ts = start + timedelta(minutes=5*i)

            # -------------------------
            # CORE SENSORS
            # -------------------------
            temperature = np.random.normal(5, 1)
            pressure = 60 + temperature * 0.6 + np.random.normal(0, 2)
            current = np.random.uniform(4, 10)
            vibration = current * 0.3 + np.random.normal(0, 0.2)
            humidity = np.random.uniform(40, 70)

            # -------------------------
            # REFRIGERATION PARAMETERS
            # -------------------------
            active_sp = np.random.uniform(2, 6)
            control_temp = active_sp + np.random.normal(0, 0.3)
            case_temp = control_temp + np.random.normal(0, 0.5)

            defrost = np.random.choice([0, 1], p=[0.85, 0.15])

            alarm_high = active_sp + 4
            alarm_low = active_sp - 3

            # -------------------------
            # FAILURE INJECTION
            # -------------------------
            if np.random.rand() < 0.05:
                temperature += np.random.uniform(2, 5)
                pressure += np.random.uniform(5, 10)
                vibration += np.random.uniform(0.5, 1.5)
                current += np.random.uniform(2, 4)

            # -------------------------
            # EXTRA INTELLIGENCE (INITIAL VERSION)
            # -------------------------

            # Temp deviation
            temp_deviation = temperature - active_sp

            # Pressure drift
            if prev_pressure is None:
                pressure_drift = 0
            else:
                pressure_drift = pressure - prev_pressure

            prev_pressure = pressure

            # Vibration ratio
            vibration_ratio = vibration / current if current != 0 else 0

            # Current overload (approx)
            current_overload = current / 10  # normalized approx

            # Alarm flags
            high_temp_alarm = 1 if case_temp > alarm_high else 0
            low_temp_alarm = 1 if case_temp < alarm_low else 0

            # Defrost impact
            defrost_impact = defrost * temperature

            rows.append([
                ts, unit,
                temperature, pressure, vibration, current, humidity,
                case_temp, defrost,
                active_sp, control_temp,
                alarm_high, alarm_low,

                # EXTRA INTELLIGENCE
                temp_deviation,
                pressure_drift,
                vibration_ratio,
                current_overload,
                high_temp_alarm,
                low_temp_alarm,
                defrost_impact
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

        # EXTRA INTELLIGENCE
        "temp_deviation",
        "pressure_drift",
        "vibration_ratio",
        "current_overload",
        "high_temp_alarm",
        "low_temp_alarm",
        "defrost_impact"
    ]

    df = pd.DataFrame(rows, columns=columns)

    print(" Generated rows:", len(df))
    print(" Columns:", df.columns.tolist())

    df.to_csv("data/raw_sensor_data.csv", index=False)