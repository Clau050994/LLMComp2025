import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import glob
import os

# === Define the folders containing the run results ===
run_dirs = [
    "/aul/homes/cvaro009/Desktop/LLMComp2025/onnx_results_usecase3",
    "/aul/homes/cvaro009/Desktop/LLMComp2025/onnx_results_usecase3_run2",
    "/aul/homes/cvaro009/Desktop/LLMComp2025/onnx_results_usecase3_run3",
    "/aul/homes/cvaro009/Desktop/LLMComp2025/onnx_results_usecase3_run4",
    "/aul/homes/cvaro009/Desktop/LLMComp2025/onnx_results_usecase3_run5",
]

target_filename = "model_comparison_summary_with_onnx.csv"

# === Load all valid CSVs from those folders ===
dfs = []
for i, run_dir in enumerate(run_dirs):
    matches = glob.glob(os.path.join(run_dir, "**", target_filename), recursive=True)
    if not matches:
        print(f"❌ Could not find {target_filename} in {run_dir}")
        continue
    csv_path = matches[0]
    df = pd.read_csv(csv_path)
    df["run"] = i + 1
    df["run_dir"] = run_dir
    dfs.append(df)

# === Combine all runs into one DataFrame ===
all_data = pd.concat(dfs, ignore_index=True)

# === Rename columns to remove 'Clean'
rename_cols = {
    "Clean F1": "F1",
    "Clean Accuracy": "Accuracy",
    "Inference Time Clean (ms)": "Latency (ms)"
}
all_data.rename(columns=rename_cols, inplace=True)

# === Clean percentage columns safely
for col in ["F1", "Accuracy", "Robust F1", "Robust Accuracy"]:
    if col in all_data.columns and all_data[col].dtype == "object":
        all_data[col] = all_data[col].str.replace("%", "").astype(float)

# === Sort model names for consistent plotting
all_data["Model"] = all_data["Model"].astype("category")
all_data["Model"] = all_data["Model"].cat.set_categories(
    sorted(all_data["Model"].unique()), ordered=True
)

# === Metrics to plot
metrics = [
    ("F1", "F1 (%)"),
    ("Accuracy", "Accuracy (%)"),
    ("Robust F1", "Robust F1 (%)"),
    ("Robust Accuracy", "Robust Accuracy (%)"),
    ("Latency (ms)", "Latency (ms)"),
    ("Model Size (MB)", "Model Size (MB)"),
    ("Peak Memory (MB)", "Peak Memory (MB)")
]

# === Output directory
plot_dir = "/aul/homes/cvaro009/Desktop/LLMComp2025/src/evaluation/usecase3/plots_ONNX/standard_deviation"
os.makedirs(plot_dir, exist_ok=True)

# === Plot each metric as a seaborn boxplot
for col, label in metrics:
    if col in all_data.columns:
        plt.figure(figsize=(10, 6))
        sns.boxplot(x="Model", y=col, data=all_data, palette="Set2")
        plt.title(f"{label} by Model (all runs)", fontsize=14)
        plt.xlabel("Model", fontsize=12)
        plt.ylabel(label, fontsize=12)
        plt.xticks(rotation=30, ha="right", fontsize=10)
        plt.yticks(fontsize=10)
        plt.tight_layout()
        plt.savefig(os.path.join(plot_dir, f"boxplot_{col.replace(' ', '_')}.png"))
        plt.close()

# === Generate and save standard deviation summary table
std_table = all_data.groupby("Model").std(numeric_only=True)
mean_table = all_data.groupby("Model").mean(numeric_only=True)
summary = mean_table.join(std_table, lsuffix="_mean", rsuffix="_std")
summary.to_csv(os.path.join(plot_dir, "model_stddev_summary.csv"))

print("Boxplots and stddev summary saved to:", plot_dir)
