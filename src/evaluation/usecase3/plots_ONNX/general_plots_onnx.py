import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# === CONFIG ===
ONNX_CSV = os.path.expanduser("/aul/homes/cvaro009/Desktop/LLMComp2025/onnx_results_usecase3/model_comparison_summary_with_onnx.csv")
PLOT_DIR = os.path.expanduser("/aul/homes/cvaro009/Desktop/LLMComp2025/src/evaluation/usecase3/plots_ONNX")
os.makedirs(PLOT_DIR, exist_ok=True)

# === LOAD & PREPROCESS ===
def load_data():
    df = pd.read_csv(ONNX_CSV)

    # Remove % and convert to float
    for col in ["F1", "Accuracy", "Robust F1", "Robust Accuracy"]:
        df[col] = df[col].str.replace('%', '').astype(float)

    # Derived metrics
    df["Robustness Score"] = (df["Robust F1"] + df["Robust Accuracy"]) / 2
    df["Speed Score"] = 100 * df["Inference Time (ms)"].min() / df["Inference Time (ms)"]

    return df

# === PLOTS ===

def plot_memory_vs_accuracy(df):
    plt.figure(figsize=(10, 6))
    scatter = plt.scatter(
        df["Peak Memory (MB)"], df["Accuracy"],
        c=df["Speed Score"], cmap="plasma", s=100
    )
    plt.axvline(2048, color='red', linestyle='--', label='2GB Limit')
    for _, row in df.iterrows():
        plt.text(row["Peak Memory (MB)"], row["Accuracy"], row["Model"], fontsize=7, ha='right')
    plt.xlabel("Peak Memory Usage (MB)")
    plt.ylabel("Accuracy (%)")
    plt.colorbar(label="Speed Score")
    plt.legend()
    plt.title("Memory vs Accuracy (Color = Speed Score)")
    plt.grid(True)
    plt.savefig(os.path.join(PLOT_DIR, "memory_vs_accuracy.png"))
    plt.close()

def plot_disk_vs_memory(df):
    plt.figure(figsize=(10, 6))
    colors = sns.color_palette("hls", len(df))
    for i, (_, row) in enumerate(df.iterrows()):
        plt.scatter(row["Model Size (MB)"], row["Peak Memory (MB)"], color=colors[i], s=100, label=row["Model"])
    plt.xlabel("Model Size (MB)")
    plt.ylabel("Peak Memory (MB)")
    plt.title("Disk Size vs RAM Footprint")
    plt.grid(True)
    plt.legend(title="Model", bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "disk_vs_memory.png"))
    plt.close()

def plot_grouped_bar_chart(df):
    metrics = ["Accuracy", "F1", "Robustness Score", "Speed Score"]
    for m in metrics:
        df[m + "_norm"] = 100 * (df[m] - df[m].min()) / (df[m].max() - df[m].min())
    melted = df.melt(id_vars="Model", value_vars=[m + "_norm" for m in metrics],
                     var_name="Metric", value_name="Score")
    melted["Metric"] = melted["Metric"].str.replace("_norm", "")
    plt.figure(figsize=(12, 6))
    sns.barplot(data=melted, x="Metric", y="Score", hue="Model")
    plt.title("Grouped Bar Chart: Normalized Metrics")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "grouped_bar_chart.png"))
    plt.close()

def plot_accuracy_vs_latency(df):
    plt.figure(figsize=(10, 6))
    scatter = plt.scatter(
        df["Inference Time (ms)"], df["Accuracy"],
        c=df["Robustness Score"], cmap="viridis", s=df["Model Size (MB)"] / 2
    )
    for _, row in df.iterrows():
        plt.text(row["Inference Time (ms)"], row["Accuracy"], row["Model"], fontsize=7, ha='right')
    plt.xlabel("Inference Time (ms)")
    plt.ylabel("Accuracy (%)")
    plt.title("Accuracy vs Inference Time (Color = Robustness, Size = Model Size)")
    plt.colorbar(label="Robustness Score")
    plt.grid(True)
    plt.savefig(os.path.join(PLOT_DIR, "accuracy_vs_latency.png"))
    plt.close()
def plot_robustness_vs_accuracy(df):
    plt.figure(figsize=(8,6))
    plt.scatter(
        df["Robustness Score"], df["Accuracy"],
        c=df["Speed Score"], s=100, cmap="plasma"
    )
    for _, row in df.iterrows():
        plt.text(row["Robustness Score"], row["Accuracy"], row["Model"], fontsize=7, ha='right')
    plt.xlabel("Robustness Score")
    plt.ylabel("Accuracy (%)")
    plt.colorbar(label="Speed Score")
    plt.title("Robustness vs Accuracy (Color = Speed Score)")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "robustness_vs_accuracy.png"))
    plt.close()
# === MAIN ===
if __name__ == "__main__":
    df = load_data()
    plot_memory_vs_accuracy(df)
    plot_disk_vs_memory(df)
    plot_grouped_bar_chart(df)
    plot_accuracy_vs_latency(df)
    plot_robustness_vs_accuracy(df)  # New plot for robustness vs accuracy
    print("✅ All ONNX plots saved to:", PLOT_DIR)