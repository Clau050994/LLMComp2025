import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# === CONFIG ===
BASELINE_CSV = os.path.expanduser("/aul/homes/cvaro009/Desktop/LLMComp2025/results_baseline_comparison/baseline_model_comparison_summary.csv")
PLOT_DIR = os.path.expanduser("/aul/homes/cvaro009/Desktop/LLMComp2025/src/evaluation/usecase3/plots_BASELINE")
os.makedirs(PLOT_DIR, exist_ok=True)

# === LOAD & PREPROCESS ===
def load_data():
    df = pd.read_csv(BASELINE_CSV)
    df["F1 Score"] = df["F1 Score"].str.replace('%', '').astype(float)
    df["Accuracy"] = df["Accuracy"].str.replace('%', '').astype(float)
    df["Speed Score"] = 100 * df["Inference Time (ms)"].min() / df["Inference Time (ms)"]
    return df

# === PLOTS ===

def plot_memory_vs_accuracy(df):
    plt.figure(figsize=(10, 6))
    scatter = plt.scatter(
        df["Memory (MB)"], df["Accuracy"],
        c=df["Speed Score"], cmap="plasma", s=100
    )
    plt.axvline(2048, color='red', linestyle='--', label='2GB Limit')
    for _, row in df.iterrows():
        plt.text(row["Memory (MB)"], row["Accuracy"], row["Model"], fontsize=7, ha='right')
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

    # Assign a unique color to each model
    colors = sns.color_palette("hls", len(df))  # hls gives vibrant colors
    for i, (_, row) in enumerate(df.iterrows()):
        plt.scatter(row["Model Size (MB)"], row["Memory (MB)"], color=colors[i], s=100, label=row["Model"])

    plt.xlabel("Model Size (MB)")
    plt.ylabel("Memory (MB)")
    plt.title("Disk Size vs RAM Footprint")
    plt.grid(True)

    # Add legend with model names
    plt.legend(title="Model", bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    plt.tight_layout()

    plt.savefig(os.path.join(PLOT_DIR, "disk_vs_memory.png"))
    plt.close()


def plot_grouped_bar_chart(df):
    metrics = ["Accuracy", "F1 Score", "Robustness Score", "Speed Score"]
    for m in metrics:
        df[m + "_norm"] = 100 * (df[m] - df[m].min()) / (df[m].max() - df[m].min())
    melted = df.melt(id_vars="Model", value_vars=[m + "_norm" for m in metrics], var_name="Metric", value_name="Score")
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
        c=df["Robustness Score"], cmap="viridis", s=df["Model Size (MB)"] / 20
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
from math import pi

def plot_radar_chart(df):
    metrics = ["Accuracy", "F1 Score", "Robustness Score", "Speed Score"]

    # Normalize each metric
    df_norm = df.copy()
    for m in metrics:
        df_norm[m + "_norm"] = 100 * (df[m] - df[m].min()) / (df[m].max() - df[m].min())

    categories = [m + "_norm" for m in metrics]
    angles = [n / float(len(categories)) * 2 * pi for n in range(len(categories))]
    angles += [angles[0]]  # close the loop

    plt.figure(figsize=(10, 10))
    ax = plt.subplot(111, polar=True)

    for _, row in df_norm.iterrows():
        values = row[categories].tolist()
        values += [values[0]]
        ax.plot(angles, values, label=row["Model"])
        ax.fill(angles, values, alpha=0.1)

    plt.xticks(angles[:-1], metrics)
    plt.title("Radar Chart: Normalized Baseline Metrics")
    plt.legend(loc='upper right', bbox_to_anchor=(1.2, 1.1))
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "radar_chart.png"))
    plt.close()


# === MAIN ===
if __name__ == "__main__":
    df = load_data()
    plot_memory_vs_accuracy(df)
    plot_disk_vs_memory(df)
    plot_grouped_bar_chart(df)
    plot_accuracy_vs_latency(df)
    plot_radar_chart(df)
    print("✅ All baseline plots saved to:", PLOT_DIR)
