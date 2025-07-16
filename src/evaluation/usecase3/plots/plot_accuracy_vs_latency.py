import pandas as pd
import matplotlib.pyplot as plt

# Load and preprocess data
df = pd.read_csv("/aul/homes/cvaro009/Desktop/LLMComp2025/onnx_results_usecase3/model_comparison_summary_with_onnx.csv")
df["Clean Accuracy"] = df["Clean Accuracy"].str.replace('%', '').astype(float)
df["Robust Accuracy"] = df["Robust Accuracy"].str.replace('%', '').astype(float)

# Create scatter plot
plt.figure(figsize=(10, 6))
scatter = plt.scatter(
    df["Inference Time Clean (ms)"], df["Clean Accuracy"],
    s=df["Model Size (MB)"] * 0.5,
    c=df["Robust Accuracy"], cmap="viridis", alpha=0.8
)

# Add model name labels
for _, row in df.iterrows():
    plt.text(
        row["Inference Time Clean (ms)"], row["Clean Accuracy"],
        row["Model"], fontsize=7, ha='left'
    )

# Format plot
plt.xscale("log")
plt.xlabel("Inference Time (ms, log scale)")
plt.ylabel("Clean Accuracy (%)")
plt.colorbar(label="Robust Accuracy (%)")
plt.title("Accuracy vs Latency (Bubble = Model Size)")
plt.grid(True)
plt.savefig("/aul/homes/cvaro009/Desktop/LLMComp2025/src/evaluation/usecase3/plots/accuracy_vs_latency.png")
