import os
import gc
import json
import time
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from pathlib import Path
from datasets import Dataset
from transformers import AutoTokenizer
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, ConfusionMatrixDisplay
from torch.utils.data import DataLoader
import psutil
import wandb
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
try:
    import onnxruntime
    ONNX_AVAILABLE = True
    print("Available ONNX providers:", onnxruntime.get_available_providers())
except ImportError:
    ONNX_AVAILABLE = False

def set_seed(seed=42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def load_test_data(data_path, shuffle=False, seed=42):
    print(f"Loading test data from {data_path} (shuffle={shuffle}, seed={seed})")
    test_df = pd.read_csv(data_path)
    if 'text' in test_df.columns and 'input_text' not in test_df.columns:
        test_df['input_text'] = test_df['text']
    
    # Shuffle data if requested (introduces randomness for std dev calculation)
    if shuffle:
        test_df = test_df.sample(frac=1, random_state=seed).reset_index(drop=True)
        print(f"Data shuffled with seed {seed}")
    
    print(f"Loaded test data with {len(test_df)} examples")
    return test_df, Dataset.from_pandas(test_df)

def plot_confusion_matrix(y_true, y_pred, model_name, out_dir):
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap=plt.cm.Blues)
    plt.title(f"Confusion Matrix: {model_name}")
    plt.savefig(os.path.join(out_dir, f"confusion_matrix_{model_name}.png"))
    plt.close()

def evaluate_accuracy(tokenizer, test_dataset, model_name, out_dir, session, robustness=True, batch_size=16, seed=42):
    import random
    # Set seed for reproducible randomness in robustness testing
    random.seed(seed)
    np.random.seed(seed)
    
    process = psutil.Process(os.getpid())
    peak_memory = 0

    def preprocess_function(examples):
        return tokenizer(
            examples["input_text"],
            truncation=True,
            padding="max_length",
            max_length=512
        )

    if robustness:
        def perturb(text):
            chars = list(text)
            # Use seed-based randomness for consistent perturbations across runs
            for _ in range(max(1, len(chars) // 20)):
                idx = random.randint(0, len(chars) - 2)
                chars[idx], chars[idx + 1] = chars[idx + 1], chars[idx]
            return ''.join(chars)
        test_dataset = test_dataset.map(lambda x: {"input_text": perturb(x["input_text"])})

    try:
        test_tokenized = test_dataset.map(preprocess_function, batched=True)
        if "input_text" in test_tokenized.column_names:
            test_tokenized = test_tokenized.remove_columns(["input_text"])
        test_tokenized.set_format(
            "torch",
            columns=["input_ids", "attention_mask", "label"] +
            (["token_type_ids"] if "token_type_ids" in test_tokenized.column_names else [])
        )
        test_dataloader = DataLoader(test_tokenized, batch_size=batch_size)
        y_pred, y_true = [], []
        start_time = time.time()
        onnx_input_names = set(i.name for i in session.get_inputs())
        for batch in tqdm(test_dataloader, desc=f"Evaluating {model_name} (ONNX){' [Robust]' if robustness else ''}"):
            ort_inputs = {
                "input_ids": batch["input_ids"].cpu().numpy(),
                "attention_mask": batch["attention_mask"].cpu().numpy()
            }
            if "token_type_ids" in batch and "token_type_ids" in onnx_input_names:
                ort_inputs["token_type_ids"] = batch["token_type_ids"].cpu().numpy()
            ort_outputs = session.run(None, ort_inputs)
            predictions = np.argmax(ort_outputs[0], axis=-1)
            y_pred.extend(predictions)
            y_true.extend(batch["label"].cpu().numpy())
            mem = process.memory_info().rss / (1024 * 1024)
            peak_memory = max(peak_memory, mem)

        accuracy = accuracy_score(y_true, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted')
        cm = confusion_matrix(y_true, y_pred)
        total_time = time.time() - start_time
        avg_time_per_sample = total_time / len(y_true) * 1000

        plot_confusion_matrix(y_true, y_pred, model_name + ("_robust" if robustness else ""), out_dir)

        return {
            "accuracy": accuracy * 100,
            "precision": precision * 100,
            "recall": recall * 100,
            "f1": f1 * 100,
            "confusion_matrix": cm.tolist(),
            "total_inference_time_s": total_time,
            "avg_inference_time_ms": avg_time_per_sample,
            "peak_memory_mb": peak_memory
        }
    except Exception as e:
        print(f"Error in evaluate_accuracy for {model_name}: {e}")
        import traceback
        traceback.print_exc()
        return None

def measure_model_size(model_path, model_name=None):
    """Calculate model size - special handling for Phi3 quantized model"""
    total_size = 0
    file_sizes = {}
    
    if model_name == "phi3":
        # For Phi3, only count the quantized model files
        quantized_files = ["model_int4.onnx", "model_int4.onnx.data"]
        for filename in quantized_files:
            file_path = Path(model_path) / filename
            if file_path.exists():
                size_mb = file_path.stat().st_size / (1024 * 1024)
                total_size += size_mb
                file_sizes[filename] = size_mb
        
        print(f"  Phi3 quantized files: {list(file_sizes.keys())}")
        print(f"  Total quantized size: {total_size:.2f} MB")
    else:
        # For other models, count all relevant files
        relevant_extensions = ['.onnx', '.json', '.txt', '.bin', '.data']
        for path in Path(model_path).glob('**/*'):
            if path.is_file() and any(path.suffix.lower() == ext for ext in relevant_extensions):
                size_mb = path.stat().st_size / (1024 * 1024)
                total_size += size_mb
                file_sizes[path.name] = size_mb
    
    return {
        "disk_size_mb": total_size,
        "file_sizes_mb": file_sizes,
        "parameters": None,
        "trainable_parameters": None
    }

def evaluate_onnx_model(model_name, onnx_dir, test_df, test_dataset, results_dir, seed):
    print(f"\n{'-'*60}\nEvaluating ONNX model: {model_name} (Run {seed})\n{'-'*60}")
    try:
        if not ONNX_AVAILABLE:
            raise ImportError("onnxruntime is not installed.")

        tokenizer = AutoTokenizer.from_pretrained(onnx_dir)
        onnx_path = os.path.join(onnx_dir, "model_int4.onnx") if model_name == "phi3" else os.path.join(onnx_dir, "model.onnx")

        # Optimize session options
        so = onnxruntime.SessionOptions()
        so.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
        so.enable_mem_pattern = False
        so.execution_mode = onnxruntime.ExecutionMode.ORT_SEQUENTIAL

        # Add CPU fallback for stability
        session = onnxruntime.InferenceSession(
            onnx_path,
            sess_options=so,
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
        )

        print("Using providers:", session.get_providers())

        # Use the updated size calculation with model_name parameter
        size_metrics = measure_model_size(onnx_dir, model_name)
        print(f"  ONNX Model size: {size_metrics['disk_size_mb']:.2f} MB")

        # Adjust batch size
        batch_size = 2 if model_name == "phi3" else 16

        print("Evaluating clean accuracy...")
        clean_metrics = evaluate_accuracy(
            tokenizer, test_dataset, model_name + f"_onnx_clean_run{seed}", results_dir, session,
            robustness=False, batch_size=batch_size, seed=seed
        )

        print("Evaluating robustness accuracy...")
        robust_metrics = evaluate_accuracy(
            tokenizer, test_dataset, model_name + f"_onnx_robust_run{seed}", results_dir, session,
            robustness=True, batch_size=batch_size, seed=seed
        )

        if clean_metrics and robust_metrics:
            print(f"  Clean Accuracy: {clean_metrics['accuracy']:.2f}%")
            print(f"  Clean F1 Score: {clean_metrics['f1']:.2f}%")
            print(f"  Robust Accuracy: {robust_metrics['accuracy']:.2f}%")
            print(f"  Robust F1 Score: {robust_metrics['f1']:.2f}%")

            results = {
                "name": model_name + f"_onnx_run{seed}",
                "path": onnx_dir,
                "size": size_metrics,
                "clean": clean_metrics,
                "robust": robust_metrics,
                "seed": seed
            }

            os.makedirs(results_dir, exist_ok=True)
            with open(os.path.join(results_dir, f"{model_name}_onnx_results_run{seed}.json"), "w") as f:
                json.dump(results, f, indent=2)
        else:
            print(f"  ❌ Evaluation failed for {model_name}")
            results = {"name": model_name + f"_onnx_run{seed}", "path": onnx_dir, "error": "Evaluation failed", "seed": seed}

        # Cleanup
        del session
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return results

    except Exception as e:
        print(f"Error evaluating ONNX {model_name}: {e}")
        import traceback
        traceback.print_exc()
        return {"name": model_name + f"_onnx_run{seed}", "path": onnx_dir, "error": str(e), "seed": seed}

def main():
    wandb.init(
        project="LLMComp2025",
        name="onnx_stddev_analysis",
        config={"num_runs": 5, "max_length": 512, "batch_size": 16, "evaluation_type": "onnx_stddev"}
    )
    
    test_data_path = "/aul/homes/cvaro009/Desktop/LLMComp2025/data/processed/usecase3/unified/unified_test.csv"
    results_dir = "/aul/homes/cvaro009/Desktop/LLMComp2025/onnx_stddev_results"
    os.makedirs(results_dir, exist_ok=True)

    models = [
        {"name": "phi3", "onnx_path": "/disk/diamond-scratch/cvaro009/data/usecase3/onnx_models/phi3_onnx"},
        {"name": "albert", "onnx_path": "/disk/diamond-scratch/cvaro009/data/usecase3/onnx_models/albert_onnx"},
        {"name": "mobilebert", "onnx_path": "/disk/diamond-scratch/cvaro009/data/usecase3/onnx_models/mobilebert_onnx"},
        {"name": "distilbert", "onnx_path": "/disk/diamond-scratch/cvaro009/data/usecase3/onnx_models/distilbert_onnx"},
        {"name": "tinyllama", "onnx_path": "/disk/diamond-scratch/cvaro009/data/usecase3/onnx_models/tinyllama_onnx"},
        {"name": "mobilellama", "onnx_path": "/disk/diamond-scratch/cvaro009/data/usecase3/onnx_models/mobilellama_onnx"},
    ]

    num_runs = 5  # Number of runs for standard deviation calculation
    # Change this number if you want more/fewer runs:
    # num_runs = 3   # For faster testing
    # num_runs = 10  # For more robust statistics
    all_results = {}

    for model_info in models:
        model_name = model_info["name"]
        onnx_dir = model_info["onnx_path"]
        
        if not (onnx_dir and os.path.exists(onnx_dir)):
            print(f"ONNX directory {onnx_dir} not found for {model_name}.")
            continue
            
        print(f"\n{'='*80}")
        print(f"Running {num_runs} evaluations for {model_name.upper()}")
        print(f"{'='*80}")
        
        model_results = []
        for seed in range(num_runs):
            print(f"\n--- Run {seed+1}/{num_runs} for {model_name} ---")
            set_seed(seed)
            
            # Load data with shuffling using different seed
            test_df, test_dataset = load_test_data(test_data_path, shuffle=True, seed=seed)
            
            result = evaluate_onnx_model(model_name, onnx_dir, test_df, test_dataset, results_dir, seed)
            
            if result and "error" not in result:
                model_results.append(result)
                print(f"✅ Run {seed+1} completed successfully")
            else:
                print(f"❌ Run {seed+1} failed: {result.get('error', 'Unknown error')}")
        
        all_results[model_name] = model_results
        print(f"\nCompleted {len(model_results)}/{num_runs} successful runs for {model_name}")

    # Save all results
    with open(os.path.join(results_dir, "all_runs_results.json"), "w") as f:
        json.dump(all_results, f, indent=2)

    # Calculate statistics and create summary
    summary_rows = []
    for model_name, runs in all_results.items():
        if not runs:
            continue

        # Helper function to extract metrics across runs
        def extract_metric(metric_key, robustness=False):
            return [r["robust" if robustness else "clean"][metric_key] for r in runs]

        # Calculate means and standard deviations
        clean_f1_values = extract_metric('f1', False)
        clean_acc_values = extract_metric('accuracy', False)
        robust_f1_values = extract_metric('f1', True)
        robust_acc_values = extract_metric('accuracy', True)
        inference_time_values = extract_metric('avg_inference_time_ms', False)

        summary_rows.append({
            "Model": model_name.upper(),
            "Clean F1 Mean": f"{np.mean(clean_f1_values):.2f}%",
            "Clean F1 Std": f"{np.std(clean_f1_values):.3f}%",
            "Clean Accuracy Mean": f"{np.mean(clean_acc_values):.2f}%",
            "Clean Accuracy Std": f"{np.std(clean_acc_values):.3f}%",
            "Robust F1 Mean": f"{np.mean(robust_f1_values):.2f}%",
            "Robust F1 Std": f"{np.std(robust_f1_values):.3f}%", 
            "Robust Accuracy Mean": f"{np.mean(robust_acc_values):.2f}%",
            "Robust Accuracy Std": f"{np.std(robust_acc_values):.3f}%",
            "Inference Time Mean (ms)": f"{np.mean(inference_time_values):.2f}",
            "Inference Time Std (ms)": f"{np.std(inference_time_values):.3f}",
            "Runs Completed": len(runs)
        })

    if summary_rows:
        summary_df = pd.DataFrame(summary_rows)
        summary_df.to_csv(os.path.join(results_dir, "onnx_stddev_summary.csv"), index=False)
        
        print("\n🎉 ONNX Standard Deviation Analysis Summary:")
        print("=" * 140)
        print(summary_df.to_string(index=False))
        print("=" * 140)
        print(f"\n📊 Results saved to {results_dir}")
        
        # Log to wandb
        wandb.log({"stddev_summary_table": wandb.Table(dataframe=summary_df)})
        
        # Log individual model statistics
        for model_name, runs in all_results.items():
            if runs:
                clean_f1_values = [r["clean"]["f1"] for r in runs]
                wandb.log({
                    f"{model_name}_clean_f1_mean": np.mean(clean_f1_values),
                    f"{model_name}_clean_f1_std": np.std(clean_f1_values),
                    f"{model_name}_runs_completed": len(runs)
                })
    else:
        print("⚠️  No successful evaluations to summarize")

    wandb.finish()

if __name__ == "__main__":
    main()