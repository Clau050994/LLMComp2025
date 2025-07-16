import pandas as pd
import matplotlib.pyplot as plt

# Load and prepare data
df = pd.read_csv("/aul/homes/cvaro009/Desktop/LLMComp2025/onnx_results_usecase3/model_comparison_summary_with_onnx.csv")
df["Clean Accuracy"] = df["Clean Accuracy"].str.replace('%', '').astype(float)
df["Speed Score"] = 100 * df["Inference Time Clean (ms)"].min() / df["Inference Time Clean (ms)"]

# Create plot
plt.figure(figsize=(10, 6))
scatter = plt.scatter(
    df["Peak Memory (MB)"], df["Clean Accuracy"],
    c=df["Speed Score"], cmap="plasma", s=100
)

# Add 2GB RAM line
plt.axvline(2048, color='red', linestyle='--', label='2GB Limit')

# Add model name labels
for _, row in df.iterrows():
    plt.text(row["Peak Memory (MB)"], row["Clean Accuracy"], row["Model"], fontsize=7, ha='right')

# Format plot
plt.xlabel("Peak Memory Usage (MB)")
plt.ylabel("Clean Accuracy (%)")
plt.colorbar(label="Speed Score")
plt.legend()
plt.title("Memory vs Accuracy (Color = Speed Score)")
plt.grid(True)
plt.savefig("/aul/homes/cvaro009/Desktop/LLMComp2025/src/evaluation/usecase3/plots/memory_vs_accuracy.png")
