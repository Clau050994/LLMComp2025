import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification,
    pipeline
)
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support, 
    confusion_matrix, classification_report
)
import time
from pathlib import Path
import onnxruntime as ort
import argparse

# Set style for plots
plt.style.use('fivethirtyeight')
sns.set_palette("colorblind")

class ModelComparison:
    def __init__(self, base_dir="models/traveler_classification/", 
                 test_data_path="data/processed/unified/unified_test.csv", 
                 results_dir="results/model_comparison"):
        """Initialize model comparison with paths."""
        self.base_dir = base_dir
        self.test_data_path = test_data_path
        self.results_dir = results_dir
        self.models_to_compare = []
        self.results = {}
        self.onnx_models = {}
        
        # Create results directory
        os.makedirs(results_dir, exist_ok=True)
        
        # Load test data
        self.test_df = pd.read_csv(test_data_path)
        print(f"Loaded test data with {len(self.test_df)} examples")
        
        # Label mapping
        self.id2label = {0: "Low Risk", 1: "Medium Risk", 2: "High Risk"}
        self.label2id = {v: k for k, v in self.id2label.items()}

    def add_model(self, model_name, model_type="hf"):
        """Add a model to compare."""
        model_path = os.path.join(self.base_dir, f"{model_name}_risk_assessment")
        
        if not os.path.exists(model_path):
            print(f"Warning: Model path {model_path} not found. Skipping.")
            return
            
        print(f"Adding model: {model_name} ({model_type})")
        self.models_to_compare.append({
            "name": model_name,
            "path": model_path,
            "type": model_type,
            "display_name": self.get_display_name(model_name)
        })
    
    def get_display_name(self, model_name):
        """Get a cleaner display name for plots."""
        name_map = {
            "distilbert": "DistilBERT",
            "albert": "ALBERT",
            "mobilebert": "MobileBERT",
            "tinyllama": "TinyLLaMA",
            "mobilellama": "MobileLLaMA",
            "grok": "Grok-3 mini",
            "flan": "Flan-T5"
        }
        return name_map.get(model_name, model_name)
    
    def add_onnx_model(self, model_name):
        """Add an ONNX model to compare."""
        onnx_path = os.path.join("models", "onnx", f"{model_name}_risk_assessment", "model.onnx")
        tokenizer_path = os.path.join(self.base_dir, f"{model_name}_risk_assessment")
        
        if not os.path.exists(onnx_path):
            print(f"Warning: ONNX model {onnx_path} not found. Skipping.")
            return
            
        print(f"Adding ONNX model: {model_name}")
        self.models_to_compare.append({
            "name": f"{model_name}_onnx",
            "path": onnx_path,
            "tokenizer_path": tokenizer_path,
            "type": "onnx",
            "display_name": f"{self.get_display_name(model_name)} (ONNX)"
        })
        
    def evaluate_model(self, model_info):
        """Evaluate a single model and return performance metrics."""
        model_name = model_info["name"]
        model_path = model_info["path"]
        model_type = model_info["type"]
        
        print(f"\nEvaluating {model_name}...")
        
        # For PyTorch/HF models
        if model_type == "hf":
            try:
                # Load tokenizer and model
                tokenizer = AutoTokenizer.from_pretrained(model_path)
                model = AutoModelForSequenceClassification.from_pretrained(model_path)
                
                # Get model size
                model_size = sum(p.numel() for p in model.parameters()) / 1e6  # in millions
                
                # Create classifier pipeline
                classifier = pipeline("text-classification", model=model, tokenizer=tokenizer, device=0 if torch.cuda.is_available() else -1)
                
                # Measure inference time
                start_time = time.time()
                
                # Process test examples
                predictions = []
                for _, row in self.test_df.iterrows():
                    result = classifier(row["input_text"])
                    pred_label = int(result[0]["label"].split("_")[-1])
                    predictions.append(pred_label)
                
                # Calculate time
                total_time = time.time() - start_time
                avg_inference_time = total_time / len(self.test_df) * 1000  # in ms
                
                # Compute metrics
                true_labels = self.test_df["label"].values
                precision, recall, f1, _ = precision_recall_fscore_support(
                    true_labels, predictions, average="weighted"
                )
                accuracy = accuracy_score(true_labels, predictions)
                cm = confusion_matrix(true_labels, predictions)
                
                # Get file size
                model_file_size = self.get_directory_size(model_path) / (1024 * 1024)  # MB
                
                return {
                    "name": model_name,
                    "display_name": model_info["display_name"],
                    "accuracy": accuracy * 100,
                    "precision": precision * 100,
                    "recall": recall * 100,
                    "f1": f1 * 100,
                    "parameters": model_size,
                    "file_size": model_file_size,
                    "inference_time_ms": avg_inference_time,
                    "confusion_matrix": cm,
                    "predictions": predictions,
                    "true_labels": true_labels
                }
            
            except Exception as e:
                print(f"Error evaluating {model_name}: {e}")
                return None
                
        # For ONNX models
        elif model_type == "onnx":
            try:
                tokenizer_path = model_info["tokenizer_path"]
                tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
                session = ort.InferenceSession(model_path, providers=['CUDAExecutionProvider', 'CPUExecutionProvider'] 
                                            if 'CUDAExecutionProvider' in ort.get_available_providers() 
                                            else ['CPUExecutionProvider'])
                
                # Measure inference time
                start_time = time.time()
                
                # Process test examples
                predictions = []
                for _, row in self.test_df.iterrows():
                    # Tokenize input
                    inputs = tokenizer(row["input_text"], return_tensors="np", padding=True, truncation=True)
                    
                    # Run inference
                    ort_inputs = {
                        "input_ids": inputs["input_ids"],
                        "attention_mask": inputs["attention_mask"]
                    }
                    logits = session.run(None, ort_inputs)[0]
                    
                    # Get prediction
                    pred_label = np.argmax(logits, axis=1)[0]
                    predictions.append(pred_label)
                
                # Calculate time
                total_time = time.time() - start_time
                avg_inference_time = total_time / len(self.test_df) * 1000  # in ms
                
                # Compute metrics
                true_labels = self.test_df["label"].values
                precision, recall, f1, _ = precision_recall_fscore_support(
                    true_labels, predictions, average="weighted"
                )
                accuracy = accuracy_score(true_labels, predictions)
                cm = confusion_matrix(true_labels, predictions)
                
                # Get file size
                model_file_size = os.path.getsize(model_path) / (1024 * 1024)  # MB
                
                return {
                    "name": model_name,
                    "display_name": model_info["display_name"],
                    "accuracy": accuracy * 100,
                    "precision": precision * 100,
                    "recall": recall * 100,
                    "f1": f1 * 100,
                    "parameters": "N/A",  # Parameter count not available for ONNX
                    "file_size": model_file_size,
                    "inference_time_ms": avg_inference_time,
                    "confusion_matrix": cm,
                    "predictions": predictions,
                    "true_labels": true_labels
                }
            
            except Exception as e:
                print(f"Error evaluating ONNX model {model_name}: {e}")
                return None
    
    def get_directory_size(self, directory):
        """Calculate the total size of a directory in bytes."""
        total_size = 0
        for path in Path(directory).glob('**/*'):
            if path.is_file():
                total_size += path.stat().st_size
        return total_size
    
    def run_comparison(self):
        """Run evaluation on all models and collect results."""
        for model_info in self.models_to_compare:
            result = self.evaluate_model(model_info)
            if result:
                self.results[model_info["name"]] = result
                
        # Save results
        with open(os.path.join(self.results_dir, "model_comparison_results.json"), "w") as f:
            # Convert numpy arrays to lists for JSON serialization
            serializable_results = {}
            for model_name, result in self.results.items():
                serializable_result = {k: v if not isinstance(v, np.ndarray) else v.tolist() 
                                      for k, v in result.items()}
                serializable_results[model_name] = serializable_result
            
            json.dump(serializable_results, f, indent=2)
        
    def generate_visualizations(self):
        """Generate visualization comparisons."""
        if not self.results:
            print("No results available for visualization. Run comparison first.")
            return
            
        # Convert results to DataFrame for easier plotting
        results_df = pd.DataFrame([
            {
                "Model": result["display_name"],
                "Accuracy": result["accuracy"],
                "F1 Score": result["f1"],
                "Precision": result["precision"],
                "Recall": result["recall"],
                "Inference Time (ms)": result["inference_time_ms"],
                "File Size (MB)": result["file_size"],
                "Parameters (M)": result["parameters"] if result["parameters"] != "N/A" else None
            }
            for result in self.results.values()
        ])
        
        # Sort by F1 score
        results_df = results_df.sort_values("F1 Score", ascending=False)
        
        # Save as CSV
        results_df.to_csv(os.path.join(self.results_dir, "model_comparison_summary.csv"), index=False)
        
        print("\nModel Performance Summary:")
        print(results_df[["Model", "F1 Score", "Accuracy", "Inference Time (ms)", "File Size (MB)"]])
        
        # 1. Performance Metrics Bar Chart
        plt.figure(figsize=(14, 8))
        performance_df = results_df.melt(
            id_vars="Model", 
            value_vars=["Accuracy", "F1 Score", "Precision", "Recall"],
            var_name="Metric", value_name="Score"
        )
        
        sns.barplot(x="Model", y="Score", hue="Metric", data=performance_df)
        plt.title("Model Performance Comparison", fontsize=16)
        plt.xlabel("Model", fontsize=14)
        plt.ylabel("Score (%)", fontsize=14)
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(os.path.join(self.results_dir, "performance_comparison.png"), dpi=300)
        plt.close()
        
        # 2. Inference Time vs Accuracy Scatter Plot
        plt.figure(figsize=(12, 8))
        sns.scatterplot(
            x="Inference Time (ms)", 
            y="F1 Score", 
            size="File Size (MB)",
            sizes=(100, 1000),
            alpha=0.7,
            hue="Model",
            data=results_df
        )
        
        for i, row in results_df.iterrows():
            plt.text(
                row["Inference Time (ms)"] + 0.5, 
                row["F1 Score"] + 0.1, 
                row["Model"], 
                fontsize=10
            )
            
        plt.title("F1 Score vs Inference Time", fontsize=16)
        plt.xlabel("Inference Time (ms)", fontsize=14)
        plt.ylabel("F1 Score (%)", fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(self.results_dir, "inference_vs_f1.png"), dpi=300)
        plt.close()
        
        # 3. Model Size vs Performance
        plt.figure(figsize=(12, 8))
        
        hb = sns.scatterplot(
            x="File Size (MB)", 
            y="F1 Score",
            size="Inference Time (ms)",
            sizes=(100, 1000),
            alpha=0.7,
            hue="Model",
            data=results_df
        )
        
        for i, row in results_df.iterrows():
            plt.text(
                row["File Size (MB)"] * 1.05, 
                row["F1 Score"] - 0.1, 
                row["Model"], 
                fontsize=10
            )
        
        plt.title("F1 Score vs Model Size", fontsize=16)
        plt.xlabel("Model Size (MB)", fontsize=14)
        plt.ylabel("F1 Score (%)", fontsize=14)
        plt.xscale('log')  # Log scale for better visualization of size differences
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(self.results_dir, "size_vs_f1.png"), dpi=300)
        plt.close()
        
        # 4. Confusion Matrices
        for model_name, result in self.results.items():
            plt.figure(figsize=(8, 6))
            cm = result["confusion_matrix"]
            sns.heatmap(
                cm, 
                annot=True, 
                fmt='d', 
                cmap='Blues',
                xticklabels=[self.id2label[i] for i in range(3)],
                yticklabels=[self.id2label[i] for i in range(3)]
            )
            plt.title(f"Confusion Matrix - {result['display_name']}", fontsize=14)
            plt.xlabel("Predicted Label", fontsize=12)
            plt.ylabel("True Label", fontsize=12)
            plt.tight_layout()
            plt.savefig(os.path.join(self.results_dir, f"confusion_matrix_{model_name}.png"), dpi=300)
            plt.close()
            
        # 5. Performance across risk categories
        for model_name, result in self.results.items():
            # Get per-class metrics
            report = classification_report(
                result["true_labels"], 
                result["predictions"],
                output_dict=True
            )
            
            # Extract metrics by class
            class_metrics = {
                "Low Risk": {metric: report["0"][metric] for metric in ["precision", "recall", "f1-score"]},
                "Medium Risk": {metric: report["1"][metric] for metric in ["precision", "recall", "f1-score"]},
                "High Risk": {metric: report["2"][metric] for metric in ["precision", "recall", "f1-score"]}
            }
            
            # Convert to DataFrame
            class_df = pd.DataFrame({
                "Risk Category": list(class_metrics.keys()),
                "Precision": [v["precision"] for v in class_metrics.values()],
                "Recall": [v["recall"] for v in class_metrics.values()],
                "F1 Score": [v["f1-score"] for v in class_metrics.values()]
            })
            
            # Plot
            plt.figure(figsize=(10, 6))
            class_plot_df = class_df.melt(
                id_vars="Risk Category",
                value_vars=["Precision", "Recall", "F1 Score"],
                var_name="Metric", value_name="Score"
            )
            
            sns.barplot(x="Risk Category", y="Score", hue="Metric", data=class_plot_df)
            plt.title(f"Performance by Risk Category - {result['display_name']}", fontsize=14)
            plt.xlabel("Risk Category", fontsize=12)
            plt.ylabel("Score", fontsize=12)
            plt.ylim(0, 1.0)
            plt.tight_layout()
            plt.savefig(os.path.join(self.results_dir, f"class_performance_{model_name}.png"), dpi=300)
            plt.close()
        
        # 6. Combined performance visualization
        plt.figure(figsize=(15, 10))
        
        # Create a radar chart for each model
        categories = ['Accuracy', 'F1 Score', 'Inference Speed', 'Size Efficiency']
        colors = plt.cm.rainbow(np.linspace(0, 1, len(results_df)))
        
        # Normalize the metrics for the radar chart
        max_inference = results_df["Inference Time (ms)"].max()
        max_size = results_df["File Size (MB)"].max()
        
        # Prepare data for radar chart
        model_data = []
        
        for i, row in results_df.iterrows():
            # Normalize and invert inference time and size (lower is better)
            speed_score = 100 * (1 - (row["Inference Time (ms)"] / max_inference))
            size_score = 100 * (1 - (row["File Size (MB)"] / max_size))
            model_data.append([
                row["Accuracy"],  # Accuracy
                row["F1 Score"],  # F1
                speed_score,      # Inference Speed (normalized and inverted)
                size_score        # Size Efficiency (normalized and inverted)
            ])
        
        # Plot radar chart
        angles = np.linspace(0, 2*np.pi, len(categories), endpoint=False).tolist()
        angles += angles[:1]  # Close the loop
        
        fig, ax = plt.subplots(figsize=(12, 10), subplot_kw=dict(polar=True))
        
        for i, model in enumerate(results_df["Model"]):
            values = model_data[i]
            values += values[:1]  # Close the loop
            ax.plot(angles, values, 'o-', linewidth=2, color=colors[i], label=model)
            ax.fill(angles, values, alpha=0.1, color=colors[i])
        
        ax.set_thetagrids(np.degrees(angles[:-1]), categories)
        ax.set_ylim(0, 100)
        ax.grid(True)
        plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
        plt.title("Model Performance Trade-offs", fontsize=16)
        plt.tight_layout()
        plt.savefig(os.path.join(self.results_dir, "model_tradeoffs_radar.png"), dpi=300)
        plt.close()
        
        print(f"\nVisualizations saved to {self.results_dir}")

def main():
    parser = argparse.ArgumentParser(description="Compare different models for traveler risk classification")
    parser.add_argument("--include-onnx", action="store_true", help="Include ONNX model versions in comparison")
    args = parser.parse_args()
    
    # Initialize comparison
    comparison = ModelComparison()
    
    # Add Hugging Face models
    comparison.add_model("distilbert")
    comparison.add_model("albert")
    comparison.add_model("mobilebert")
    comparison.add_model("tinyllama")
    comparison.add_model("mobilellama")
    
    # Add other models if they exist
    comparison.add_model("grok")
    comparison.add_model("flan")
    
    # Add ONNX models if requested
    if args.include_onnx:
        comparison.add_onnx_model("distilbert")
        comparison.add_onnx_model("albert")
        comparison.add_onnx_model("mobilebert")
        comparison.add_onnx_model("tinyllama")
        comparison.add_onnx_model("mobilellama")
    
    # Run comparison
    comparison.run_comparison()
    
    # Generate visualizations
    comparison.generate_visualizations()

if __name__ == "__main__":
    main()