import pandas as pd
import matplotlib.pyplot as plt

def plot_params_vs_accuracy(df):
    plt.figure(figsize=(8,6))
    scatter = plt.scatter(
        df["Parameters"], df["Clean Accuracy"],
        s=100, c=df["Robust Accuracy"], cmap="coolwarm"
    )
    plt.xscale("log")
    plt.xlabel("Number of Parameters (log scale)")
    plt.ylabel("Clean Accuracy (%)")
    plt.colorbar(label="Robust Accuracy (%)")
    plt.title("Model Parameters vs Accuracy")

    # ✅ Add model name labels to each point
    for _, row in df.iterrows():
        plt.text(
            row["Parameters"], row["Clean Accuracy"], row["Model"],
            fontsize=7, ha='right', va='center'
        )

    plt.grid(True)
    plt.savefig("/aul/homes/cvaro009/Desktop/LLMComp2025/src/evaluation/usecase3/plots/plot_params_vs_accuracy.png")

if __name__ == "__main__":
    df = pd.read_csv("/aul/homes/cvaro009/Desktop/LLMComp2025/onnx_results_usecase3/model_comparison_summary_with_onnx.csv")
    df["Clean Accuracy"] = df["Clean Accuracy"].str.replace('%', '').astype(float)
    df["Robust Accuracy"] = df["Robust Accuracy"].str.replace('%', '').astype(float)

    # PATCH: Add Parameters column manually
    df["Parameters"] = df["Model"].map({
        "PHI3_ONNX": 250_000_000,
        "ALBERT_ONNX": 12_000_000,
        "MOBILEBERT_ONNX": 25_000_000,
        "DISTILBERT_ONNX": 65_000_000,
        "TINYLLAMA_ONNX": 6_800_000,
        "MOBILELLAMA_ONNX": 20_000_000
    })

    plot_params_vs_accuracy(df)
