import pandas as pd
import matplotlib.pyplot as plt

def plot_disk_vs_memory(df):
    plt.figure(figsize=(8,6))
    plt.scatter(df["Model Size (MB)"], df["Peak Memory (MB)"], s=100)
    for _, row in df.iterrows():
        plt.text(row["Model Size (MB)"], row["Peak Memory (MB)"], row["Model"], fontsize=8)
    plt.xlabel("Model Size (MB)")
    plt.ylabel("Peak Memory Usage (MB)")
    plt.title("Disk Size vs RAM Footprint")
    plt.grid(True)
    plt.savefig("/aul/homes/cvaro009/Desktop/LLMComp2025/src/evaluation/usecase3/plots/plot_disk_vs_memory.png")

if __name__ == "__main__":
    df = pd.read_csv("/aul/homes/cvaro009/Desktop/LLMComp2025/onnx_results_usecase3/model_comparison_summary_with_onnx.csv")
    plot_disk_vs_memory(df)
