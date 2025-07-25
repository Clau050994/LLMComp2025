import pandas as pd
import matplotlib.pyplot as plt

def plot_robustness_vs_accuracy(df):
    plt.figure(figsize=(8,6))
    # ✅ Use correct latency column
    plt.scatter(df["Robustness Score"], df["Accuracy"], 
                c=df["Inference Time (ms)"], s=100, cmap="plasma")
    
    plt.xlabel("Robustness Score")
    plt.ylabel("Accuracy (%)")
    plt.colorbar(label="Inference Time (ms)")
    plt.title("Robustness vs Accuracy")

    # ✅ Add model name labels
    for _, row in df.iterrows():
        plt.text(row["Robustness Score"], row["Accuracy"], row["Model"], fontsize=7, ha='right')

    plt.grid(True)
    plt.tight_layout()
    plt.savefig("/aul/homes/cvaro009/Desktop/LLMComp2025/src/evaluation/usecase3/plots_BASELINE/plot_robustness_vs_accuracy.png")

if __name__ == "__main__":
    df = pd.read_csv("/aul/homes/cvaro009/Desktop/LLMComp2025/results_baseline_comparison/baseline_model_comparison_summary.csv")
    # Remove % and convert to float
    df["Accuracy"] = df["Accuracy"].str.replace('%', '').astype(float)
    df["F1 Score"] = df["F1 Score"].str.replace('%', '').astype(float)
    plot_robustness_vs_accuracy(df)