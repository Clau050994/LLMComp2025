import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv("/aul/homes/cvaro009/Desktop/LLMComp2025/onnx_results_usecase3/model_comparison_summary_with_onnx.csv")
metrics = ["Clean Accuracy", "Clean F1", "Robust Accuracy", "Robust F1", "Speed Score"]
for col in metrics[:-1]:
    df[col] = df[col].str.replace('%', '').astype(float)
df["Speed Score"] = 100 * df["Inference Time Clean (ms)"].min() / df["Inference Time Clean (ms)"]
for m in metrics:
    df[m + "_norm"] = 100 * (df[m] - df[m].min()) / (df[m].max() - df[m].min())

melted = df.melt(id_vars="Model", value_vars=[m + "_norm" for m in metrics], var_name="Metric", value_name="Score")
melted["Metric"] = melted["Metric"].str.replace("_norm", "")

plt.figure(figsize=(12, 6))
sns.barplot(data=melted, x="Metric", y="Score", hue="Model")
plt.title("Grouped Bar Chart: Normalized Metrics")
plt.tight_layout()
plt.savefig("/aul/homes/cvaro009/Desktop/LLMComp2025/src/evaluation/usecase3/plots/grouped_bar_chart.png")  