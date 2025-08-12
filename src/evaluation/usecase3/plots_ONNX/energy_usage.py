import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# === Load the CSV (with precomputed energy columns) ===
csv_path = "/aul/homes/cvaro009/Desktop/LLMComp2025/onnx_results_usecase3/energy_usage_onnx.csv"
df = pd.read_csv(csv_path)

# === Setup ===
latency_col = "Inference Time (ms)"
model_size_col = "Model Size (MB)"
peak_mem_col = "Peak Memory (MB)"
models = df["Model"]
x = np.arange(len(models))
width = 0.18
output_path = "/aul/homes/cvaro009/Desktop/LLMComp2025/src/evaluation/usecase3/plots_ONNX/energy_usage_summary.csv"

# === 1. Grouped Bar Plot: Energy per Query by Model and Device ===
plt.figure(figsize=(12,6))
plt.bar(x - 1.5*width, df["Energy per Query (J) [GPU]"], width, label="GPU")
plt.bar(x - 0.5*width, df["Energy per Query (J) [CPU]"], width, label="CPU")
plt.bar(x + 0.5*width, df["Energy per Query (J) [Laptop]"], width, label="Laptop")
plt.bar(x + 1.5*width, df["Energy per Query (J) [Phone]"], width, label="Phone")
plt.xticks(x, models, rotation=45)
plt.ylabel("Energy per Query (J)")
plt.title("Energy per Query by Model and Device")
plt.legend()
plt.tight_layout()
plt.savefig(output_path.replace(".csv", "_energy_barplot_FROM_CSV.png"))
plt.close()

# === 2. Scatter Plot: Latency vs. Energy per Query (All Devices) ===
plt.figure(figsize=(10, 7))
devices = [
    ("Energy per Query (J) [GPU]", "GPU", "royalblue"),
    ("Energy per Query (J) [CPU]", "CPU", "orange"),
    ("Energy per Query (J) [Laptop]", "Laptop", "green"),
    ("Energy per Query (J) [Phone]", "Phone", "red")
]
for col, label, color in devices:
    plt.scatter(df[latency_col], df[col], s=120, label=label, color=color, edgecolor='black')
    for i, txt in enumerate(models):
        plt.annotate(txt, 
                     (df[latency_col][i], df[col][i]),
                     textcoords="offset points", xytext=(5,5), ha='left', fontsize=8, color=color)
plt.xlabel("Latency (ms)", fontsize=12)
plt.ylabel("Energy per Query (J)", fontsize=12)
plt.title("Latency vs. Energy per Query (All Devices)", fontsize=14)
plt.xscale("log")
plt.yscale("log")
plt.grid(True, which="both", linestyle='--', linewidth=0.5, alpha=0.7)
plt.legend()
plt.tight_layout()
plt.savefig(output_path.replace(".csv", "_latency_vs_energy_scatter_ALLDEVICES_FROM_CSV.png"))
plt.close()

# === 3. Dual-Axis Plot: Model Size + Energy by Device ===
fig, ax1 = plt.subplots(figsize=(12, 6))
ax1.bar(x - 1.5*width, df["Energy per Query (J) [GPU]"], width, label="GPU")
ax1.bar(x - 0.5*width, df["Energy per Query (J) [CPU]"], width, label="CPU")
ax1.bar(x + 0.5*width, df["Energy per Query (J) [Laptop]"], width, label="Laptop")
ax1.bar(x + 1.5*width, df["Energy per Query (J) [Phone]"], width, label="Phone")
ax1.set_ylabel("Energy per Query (J)")
ax1.set_xticks(x)
ax1.set_xticklabels(models, rotation=45)
ax1.legend(loc='upper left')
ax2 = ax1.twinx()
ax2.plot(x, df[model_size_col], color='red', marker='o', label="Model Size (MB)")
ax2.set_ylabel("Model Size (MB)")
ax2.legend(loc='upper right')
plt.title("Model Size and Energy per Query by Device")
plt.tight_layout()
plt.savefig(output_path.replace(".csv", "_modelsize_vs_energy_ALLDEVICES_FROM_CSV.png"))
plt.close()

# === 4. Dual-Axis Plot: Peak Memory + Energy by Device ===
fig, ax1 = plt.subplots(figsize=(12, 6))
ax1.bar(x - 1.5*width, df["Energy per Query (J) [GPU]"], width, label="GPU")
ax1.bar(x - 0.5*width, df["Energy per Query (J) [CPU]"], width, label="CPU")
ax1.bar(x + 0.5*width, df["Energy per Query (J) [Laptop]"], width, label="Laptop")
ax1.bar(x + 1.5*width, df["Energy per Query (J) [Phone]"], width, label="Phone")
ax1.set_ylabel("Energy per Query (J)")
ax1.set_xticks(x)
ax1.set_xticklabels(models, rotation=45)
ax1.legend(loc='upper left')
ax2 = ax1.twinx()
ax2.plot(x, df[peak_mem_col], color='green', marker='x', label="Peak Memory (MB)")
ax2.set_ylabel("Peak Memory (MB)")
ax2.legend(loc='upper right')
plt.title("Peak Memory and Energy per Query by Device")
plt.tight_layout()
plt.savefig(output_path.replace(".csv", "_peakmem_vs_energy_ALLDEVICES_FROM_CSV.png"))
plt.close()

print("✅ All plots generated from CSV columns successfully.")
