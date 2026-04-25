import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

def run_correlation_analysis():

    print("📊 Running Correlation Analysis...")

    df = pd.read_csv("data/processed_data.csv")

    numeric_df = df.select_dtypes(include=['float64', 'int64'])

    corr = numeric_df.corr()

    os.makedirs("data/plots", exist_ok=True)

    plt.figure(figsize=(12, 8))
    sns.heatmap(corr, cmap="coolwarm", annot=False)

    plt.title("Sensor vs Refrigeration Parameter Correlation")
    plt.savefig("data/plots/correlation_heatmap.png")
    plt.close()

    print("📁 Saved → data/plots/correlation_heatmap.png")