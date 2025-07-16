import pandas as pd
import matplotlib.pyplot as plt

def plot_robustness_vs_accuracy(df):
    plt.figure(figsize=(8,6))
    # ✅ Use correct latency column
    plt.scatter(df["Robust Accuracy"], df["Clean Accuracy"], 
                c=df["Inference Time Clean (ms)"], s=100, cmap="plasma")
    
    plt.xlabel("Robust Accuracy (%)")
    plt.ylabel("Clean Accuracy (%)")
    plt.colorbar(label="Inference Time (ms)")
    plt.title("Robustness vs Accuracy")

    # ✅ Add model name labels
    for _, row in df.iterrows():
        plt.text(row["Robust Accuracy"], row["Clean Accuracy"], row["Model"], fontsize=7, ha='right')

    plt.grid(True)
    plt.savefig("/aul/homes/cvaro009/Desktop/LLMComp2025/src/evaluation/usecase3/plots/plot_robustness_vs_accuracy.png")

if __name__ == "__main__":
    df = pd.read_csv("/aul/homes/cvaro009/Desktop/LLMComp2025/onnx_results_usecase3/model_comparison_summary_with_onnx.csv")
    df["Clean Accuracy"] = df["Clean Accuracy"].str.replace('%', '').astype(float)
    df["Robust Accuracy"] = df["Robust Accuracy"].str.replace('%', '').astype(float)
    plot_robustness_vs_accuracy(df)

