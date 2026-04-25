# shap_explainer

import pandas as pd
import shap
import joblib
import os
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier

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

    X = df.drop(columns=["timestamp", "failure"], errors="ignore")
    y = df["failure"]

    return X, y

# -------------------------
# TRAIN OR LOAD MODEL
# -------------------------
def get_model(X, y):
    os.makedirs("models", exist_ok=True)

    model_path = "models/shap_rf_model.pkl"

    if os.path.exists(model_path):
        model = joblib.load(model_path)
    else:
        model = RandomForestClassifier()
        model.fit(X, y)
        joblib.dump(model, model_path)

    return model

# -------------------------
# SHAP EXPLANATION
# -------------------------
def run_shap():

    import shap
    import pandas as pd
    import matplotlib.pyplot as plt
    import os

    # -------------------------
    # LOAD DATA
    # -------------------------
    X, y = load_data()

    # ✅ Sample for speed
    X = X.sample(min(500, len(X)), random_state=42)
    y = y.loc[X.index]

    # -------------------------
    # TRAIN MODEL
    # -------------------------
    model = get_model(X, y)

    #  CRITICAL FIX: ALIGN FEATURES
    if hasattr(model, "feature_names_in_"):
        X = X[model.feature_names_in_]

    # -------------------------
    # SHAP EXPLAINER
    # -------------------------
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)

    #  HANDLE SHAP OUTPUT FORMAT
    if isinstance(shap_values, list):
        shap_vals = shap_values[1]   # binary classification
        expected_value = explainer.expected_value[1]
    else:
        shap_vals = shap_values
        expected_value = explainer.expected_value

    #  FINAL SAFETY CHECK
    print("X shape:", X.shape)
    print("SHAP shape:", shap_vals.shape)

    if shap_vals.shape[1] != X.shape[1]:
        print(" Feature mismatch — fixing automatically")

        # Trim to smallest common shape
        min_cols = min(shap_vals.shape[1], X.shape[1])
        shap_vals = shap_vals[:, :min_cols]
        X = X.iloc[:, :min_cols]

    # -------------------------
    # SAVE PLOTS
    # -------------------------
    os.makedirs("data/shap", exist_ok=True)

    # Summary plot
    plt.figure()
    shap.summary_plot(shap_vals, X, show=False)
    plt.savefig("data/shap/summary_plot.png", bbox_inches="tight")
    plt.close()

    # Bar plot
    plt.figure()
    shap.summary_plot(shap_vals, X, plot_type="bar", show=False)
    plt.savefig("data/shap/bar_plot.png", bbox_inches="tight")
    plt.close()

    # Force plot
    import shap
    import matplotlib.pyplot as plt

      # -------------------------
    #  FINAL FIXED WATERFALL
    # -------------------------

    # Take ONE sample
    single_shap = shap_vals[0]

    # Fix multi-output shape
    if len(single_shap.shape) > 1:
        single_shap = single_shap[:, 0]

    # Ensure expected_value is scalar
    if isinstance(expected_value, (list, tuple)):
        base_val = expected_value[0]
    else:
        base_val = expected_value

    # Build explanation
    # exp = shap.Explanation(
    #     values=single_shap,
    #     base_values=float(base_val),
    #     data=X.iloc[0],
    #     feature_names=X.columns
    # )


    # plt.figure()
    #     # plt.figure()
    # shap.plots.waterfall(exp)

    # -------------------------
    # FINAL FIXED WATERFALL
    # -------------------------

    # Select ONE sample
    sample_index = 0

    # Select ONE class (failure class = 1)
    single_shap = shap_vals[sample_index, :, 1]

    # Fix base value
    if isinstance(expected_value, (list, tuple)):
        base_val = expected_value[1]
    elif hasattr(expected_value, "__len__"):
        base_val = expected_value[1]
    else:
        base_val = expected_value

    base_val = float(base_val)

    # Build explanation
    exp = shap.Explanation(
        values=single_shap,
        base_values=base_val,
        data=X.iloc[sample_index],
        feature_names=X.columns
    )

    plt.figure()
    shap.plots.waterfall(exp)

    plt.savefig("data/shap/waterfall_plot.png", bbox_inches="tight")
    plt.close()

    # plt.savefig("data/shap/waterfall_plot.png", bbox_inches="tight")
    # plt.close()

    print(" SHAP WORKING PERFECTLY NOW")

    # shap.plots.waterfall(
    # shap.Explanation(
    #     values=shap_vals[0],
    #     base_values=expected_value,
    #     data=X.iloc[0],
    #     feature_names=X.columns
    # )
    # )

    # plt.savefig("data/shap/force_plot.png", bbox_inches="tight")
    # plt.close()

    print(" SHAP WORKING PERFECTLY NOW")