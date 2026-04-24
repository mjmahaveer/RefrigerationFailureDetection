import os
import argparse

from src.data_generator_v1 import generate_sensor_data
from src.preprocessing import preprocess_data
from src.model_runner import run_all_models
from src.evaluation import run_evaluation
from src.failure_prediction import run_lstm
from src.classification import run_classification

from src.anomaly_detection import run_anomaly_detection


from src.correlation_monitor import run_correlation_analysis
from src.alerts import run_alerts

# Optional SHAP
try:
    from src.shap_explainer import run_shap
    SHAP_AVAILABLE = True
except:
    SHAP_AVAILABLE = False


def main(run_shap_flag=False):
    os.makedirs("data", exist_ok=True)

    print("\n STARTING PIPELINE\n")

    print(" Generating data...")
    generate_sensor_data()

    print(" Preprocessing...")
    preprocess_data()
    print(" Running anomaly detection...")
    run_anomaly_detection()   # 


    print(" Running ML models...")
    run_all_models()

    print(" Running failure prediction (LSTM)...")
    run_lstm()
    print(" Running classification...")
    run_classification()


    print(" Running evaluation...")
    run_evaluation()

    print(" Running Correlation Analysis...")
    run_correlation_analysis()

    print(" Running Alerts...")
    run_alerts()



    # Optional SHAP (SAFE)
    if run_shap_flag and SHAP_AVAILABLE:
        print(" Generating SHAP explainability...")
        run_shap()
    else:
        print(" SHAP skipped")

    print("\n ALL DONE\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--shap", action="store_true", help="Run SHAP")

    args = parser.parse_args()

    main(run_shap_flag=args.shap)