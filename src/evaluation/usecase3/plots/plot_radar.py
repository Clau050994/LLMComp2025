import pandas as pd
import matplotlib.pyplot as plt
from math import pi
import os

df = pd.read_csv("/aul/homes/cvaro009/Desktop/LLMComp2025/onnx_results_usecase3/model_comparison_summary_with_onnx.csv")
metrics = ["Clean Accuracy", "Clean F1", "Robust Accuracy", "Robust F1", "Speed Score"]

# Convert % to float and compute Speed Score
for col in metrics[:-1]:
    df[col] = df[col].str.replace('%', '').astype(float)
df["Speed Score"] = 100 * df["Inference Time Clean (ms)"].min() / df["Inference Time Clean (ms)"]

# Normalize
df_norm = df.copy()
for metric in metrics:
    df_norm[metric + "_norm"] = 100 * (df[metric] - df[metric].min()) / (df[metric].max() - df[metric].min())

# Radar plot
categories = [m + "_norm" for m in metrics]
angles = [n / float(len(categories)) * 2 * pi for n in range(len(categories))] + [0]

plt.figure(figsize=(10, 10))
ax = plt.subplot(111, polar=True)

for i, row in df_norm.iterrows():
    values = row[categories].tolist() + [row[categories[0]]]
    ax.plot(angles, values, label=row["Model"])
    ax.fill(angles, values, alpha=0.1)

plt.xticks(angles[:-1], metrics)
plt.title("Radar Chart: Normalized SLM Metrics")
plt.legend(loc='upper right', bbox_to_anchor=(1.2, 1.1))
plt.tight_layout()
plt.savefig("/aul/homes/cvaro009/Desktop/LLMComp2025/src/evaluation/usecase3/plots/radar_chart.png")
