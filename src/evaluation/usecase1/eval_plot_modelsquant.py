import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from math import pi

# === CONFIG ===
ROOT = "/aul/homes/melsh008/First_Case_Scenario"
PLOT_DIR = "/aul/homes/melsh008/temp-storage-space-on-diamond/eval_plots"
os.makedirs(PLOT_DIR, exist_ok=True)

# === UTILITY ===
def estimate_model_size(model_path):
    size = 0
    for root, _, files in os.walk(model_path):
        for f in files:
            size += os.path.getsize(os.path.join(root, f))
    return round(size / (1024 ** 2), 2)  # MB

def load_metrics_json(model_path):
    for name in ["final_eval_metrics.json", "final_metrics.json"]:
        path = os.path.join(model_path, name)
        if os.path.isfile(path):
            with open(path) as f:
                return json.load(f)
    return None

def load_data():
    summary_csv = os.path.join(ROOT, "evaluation_results", "verification_model_comparison_summary.csv")
    if not os.path.isfile(summary_csv):
        print(f"❌ CSV file not found at: {summary_csv}")
        return pd.DataFrame()

    df = pd.read_csv(summary_csv)

    # Clean and convert percentage-based metrics
    percent_cols = ["Accuracy", "F1 Score", "Robustness Score"]
    for col in percent_cols:
        df[col] = (
            df[col]
            .astype(str)
            .str.replace('%', '', regex=False)
            .str.strip()
            .astype(float)
        )

    # Ensure other numeric columns are parsed correctly
    numeric_cols = ["Inference Time (ms)", "Memory (MB)", "Model Size (MB)"]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # Normalize model names (e.g., remove trailing spaces, unify case)
    df["Model"] = df["Model"].astype(str).str.strip().str.upper()

    # Compute Speed Score (lower inference time = better)
    df["Speed Score"] = 100 * df["Inference Time (ms)"].min() / df["Inference Time (ms)"]

    print("✅ Loaded summary data with models:", df["Model"].tolist())
    return df




# === PLOTTING ===
def plot_memory_vs_accuracy(df):
    plt.figure(figsize=(10, 6))
    plt.scatter(df["Memory (MB)"], df["Accuracy"], c=df["Speed Score"], cmap="plasma", s=100)
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
    colors = sns.color_palette("hls", len(df))
    for i, (_, row) in enumerate(df.iterrows()):
        plt.scatter(row["Model Size (MB)"], row["Memory (MB)"], color=colors[i], s=100, label=row["Model"])
    plt.xlabel("Model Size (MB)")
    plt.ylabel("Memory (MB)")
    plt.title("Disk Size vs RAM Footprint")
    plt.grid(True)
    plt.legend(title="Model", bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "disk_vs_memory.png"))
    plt.close()

def plot_robustness_vs_accuracy(df):
    plt.figure(figsize=(10, 6))
    
    # Plot the scatter
    scatter = plt.scatter(
        df["Robustness Score"],
        df["Accuracy"],
        c=df["Speed Score"],
        cmap="coolwarm",
        s=100,
        edgecolors="k"
    )
    
    # Sort to avoid overlap chaos
    df_sorted = df.sort_values(by=["Accuracy", "Robustness Score"], ascending=False)
    
    # Keep track of used label positions to avoid overlaps
    used_coords = set()
    for _, row in df_sorted.iterrows():
        x, y = row["Robustness Score"], row["Accuracy"]
        label = row["Model"]

        # Check if nearby label already exists
        offset_x, offset_y = 0, 0
        while (round(x + offset_x, 2), round(y + offset_y, 2)) in used_coords:
            offset_x += 0.05
            offset_y += 0.05

        used_coords.add((round(x + offset_x, 2), round(y + offset_y, 2)))
        plt.text(x + offset_x, y + offset_y, label, fontsize=8)

    plt.xlabel("Robustness Score")
    plt.ylabel("Accuracy (%)")
    plt.title("Robustness vs Accuracy (Color = Speed Score)")
    plt.colorbar(label="Speed Score")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "robustness_vs_accuracy.png"))
    plt.close()




def plot_grouped_bar_chart(df):
    metrics = ["Accuracy", "F1 Score", "Robustness Score", "Speed Score"]
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
    import numpy as np

    plt.figure(figsize=(12, 7))

    # Normalize sizes
    max_size = df["Model Size (MB)"].max()
    min_size = df["Model Size (MB)"].min()
    df["Size"] = df["Model Size (MB)"].apply(
        lambda x: 100 + 400 * (x - min_size) / (max_size - min_size + 1e-5)
    )

    # Log transform inference time to spread out values
    df["Log Inference Time"] = np.log10(df["Inference Time (ms)"].clip(lower=1))

    # Color mapping
    color_map = dict(zip(df["Model"], sns.color_palette("tab10", len(df))))
    df["Color"] = df["Model"].map(color_map)

    # Track used label positions
    used_label_coords = set()

    # Plot
    for i, (_, row) in enumerate(df.iterrows()):
        x = row["Log Inference Time"]
        y = row["Accuracy"]
        size = row["Size"]
        color = row["Color"]
        model = row["Model"]

        # Draw circle
        plt.scatter(x, y, s=size, color=color, edgecolors='black', alpha=0.85)

        # Try to find non-overlapping label position
        offset_step = 0.5
        max_attempts = 10
        dx, dy = 0.01, 0  # initial offset
        for attempt in range(max_attempts):
            label_pos = (round(x + dx, 3), round(y + dy, 3))
            if label_pos not in used_label_coords:
                used_label_coords.add(label_pos)
                break
            dy += offset_step * ((-1) ** attempt)  # alternate up/down

        # Draw label with background
        plt.text(
            x + dx, y + dy, model,
            fontsize=9, ha='left', va='center',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.75, edgecolor='gray')
        )

    # Axes and titles
    plt.xlabel("Log Inference Time (ms)")
    plt.ylabel("Accuracy (%)")
    plt.title("Accuracy vs Inference Time (Log Scale)\n(Circle Size = Model Size, Color = Model)")
    plt.grid(True, linestyle='--', alpha=0.6)

    # Legend
    handles = [plt.Line2D([0], [0], marker='o', color='w', label=model,
                          markerfacecolor=color_map[model],
                          markeredgecolor='black', markersize=10)
               for model in df["Model"]]
    plt.legend(title="Model", handles=handles, bbox_to_anchor=(1.05, 1), loc='upper left')

    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "accuracy_vs_latency.png"))
    plt.close()


def plot_radar_chart(df):
    metrics = ["Accuracy", "F1 Score", "Robustness Score", "Speed Score"]
    df_norm = df.copy()
    for m in metrics:
        df_norm[m + "_norm"] = 100 * (df[m] - df[m].min()) / (df[m].max() - df[m].min())
    categories = [m + "_norm" for m in metrics]
    angles = [n / float(len(categories)) * 2 * pi for n in range(len(categories))]
    angles += [angles[0]]
    plt.figure(figsize=(10, 10))
    ax = plt.subplot(111, polar=True)
    for _, row in df_norm.iterrows():
        values = row[categories].tolist()
        values += [values[0]]
        ax.plot(angles, values, label=row["Model"])
        ax.fill(angles, values, alpha=0.1)
    plt.xticks(angles[:-1], metrics)
    plt.title("Radar Chart: Normalized Verification Metrics")
    plt.legend(loc='upper right', bbox_to_anchor=(1.2, 1.1))
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "radar_chart.png"))
    plt.close()

# === MAIN ===
if __name__ == "__main__":
    df = load_data()
    if df.empty:
        print("❌ No model evaluation data found.")
    else:
        plot_memory_vs_accuracy(df)
        plot_disk_vs_memory(df)
        plot_grouped_bar_chart(df)
        plot_accuracy_vs_latency(df)
        plot_radar_chart(df)
        plot_robustness_vs_accuracy(df)  # <- added here
        print("✅ All plots saved to:", PLOT_DIR)

