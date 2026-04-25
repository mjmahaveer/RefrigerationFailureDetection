import time
import pandas as pd
import json

def simulate_stream():

    df = pd.read_csv("data/processed_data.csv")

    print(" Simulating MQTT stream...")

    for _, row in df.iterrows():
        message = row.to_dict()
        print(json.dumps(message))  # simulate publish
        time.sleep(1)