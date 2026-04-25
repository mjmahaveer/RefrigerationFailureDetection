from flask import Flask, jsonify
import pandas as pd

app = Flask(__name__)

@app.route("/data")
def get_data():
    df = pd.read_csv("data/processed_data.csv")
    return df.tail(100).to_json(orient="records")

@app.route("/alerts")
def get_alerts():
    df = pd.read_csv("data/alert_logs.csv")
    return df.tail(50).to_json(orient="records")

if __name__ == "__main__":
    app.run(debug=True)