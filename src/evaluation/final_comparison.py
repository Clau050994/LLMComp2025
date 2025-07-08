import os
import json
import time
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
try:
    import onnxruntime
    ONNX_AVAILABLE = True
    print("Available ONNX providers:", onnxruntime.get_available_providers())
except ImportError:
    ONNX_AVAILABLE = False

def set_seed(seed=42):
    import random
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def load_test_data(data_path):
    print(f"Loading test data from {data_path}")
    test_df = pd.read_csv(data_path)
    if 'text' in test_df.columns and 'input_text' not in test_df.columns:
        test_df['input_text'] = test_df['text']
    print(f"Loaded test data with {len(test_df)} examples")
    return test_df, Dataset.from_pandas(test_df)

def plot_confusion_matrix(y_true, y_pred, model_name, out_dir):
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap=plt.cm.Blues)
    plt.title(f"Confusion Matrix: {model_name}")
    plt.savefig(os.path.join(out_dir, f"confusion_matrix_{model_name}.png"))
    plt.close()

def evaluate_accuracy(tokenizer, test_dataset, model_name, out_dir, session, robustness=True):
    """
    Evaluates accuracy, F1, inference time, and peak memory usage.
    If robustness=True, perturbs the input text before evaluation.
    """
    import psutil
    import random
    process = psutil.Process(os.getpid())
    peak_memory = 0

    def preprocess_function(examples):
        return tokenizer(
            examples["input_text"],
            truncation=True,
            padding="max_length",
            max_length=512
        )

    # Perturb input for robustness evaluation
    if robustness:
        def perturb(text):
            chars = list(text)
            for _ in range(max(1, len(chars)//20)):
                idx = random.randint(0, len(chars)-2)
                chars[idx], chars[idx+1] = chars[idx+1], chars[idx]
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
        test_dataloader = DataLoader(test_tokenized, batch_size=16)
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
            # Track peak memory
            mem = process.memory_info().rss / (1024 * 1024)  # MB
            if mem > peak_memory:
                peak_memory = mem
        accuracy = accuracy_score(y_true, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted')
        cm = confusion_matrix(y_true, y_pred)
        total_time = time.time() - start_time
        avg_time_per_sample = total_time / len(y_true) * 1000  # ms
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
def measure_model_size(model_path):
    total_size = 0
    for path in Path(model_path).glob('**/*'):
        if path.is_file():
            total_size += path.stat().st_size
    file_sizes = {}
    for path in Path(model_path).glob('*'):
        if path.is_file():
            file_sizes[path.name] = path.stat().st_size / (1024 * 1024)  # MB
    return {
        "disk_size_mb": total_size / (1024 * 1024),
        "file_sizes_mb": file_sizes,
        "parameters": None,
        "trainable_parameters": None
    }

def evaluate_onnx_model(model_name, onnx_dir, test_df, test_dataset, results_dir):
    print(f"\n{'-'*60}\nEvaluating ONNX model: {model_name}\n{'-'*60}")
    try:
        if not ONNX_AVAILABLE:
            raise ImportError("onnxruntime is not installed.")
        tokenizer = AutoTokenizer.from_pretrained(onnx_dir)
        onnx_path = os.path.join(onnx_dir, "model.onnx")
        session = onnxruntime.InferenceSession(
            onnx_path,
            providers=["CUDAExecutionProvider"]
        )
        print("Using providers:", session.get_providers())

        # Model size
        # Model size
        size_metrics = measure_model_size(onnx_dir)
        print(f"  ONNX Model size: {size_metrics['disk_size_mb']:.2f} MB")

        # Evaluate clean accuracy
        print("Evaluating clean accuracy...")
        clean_metrics = evaluate_accuracy(
            tokenizer,
            test_dataset,
            model_name + "_onnx_clean",
            results_dir,
            session,
            robustness=False
        )

        # Evaluate robustness accuracy
        print("Evaluating robustness accuracy...")
        robust_metrics = evaluate_accuracy(
            tokenizer,
            test_dataset,
            model_name + "_onnx_robust",
            results_dir,
            session,
            robustness=True
        )
        
        print(f"  Clean Accuracy: {clean_metrics['accuracy']:.2f}%")
        print(f"  Clean F1 Score: {clean_metrics['f1']:.2f}%")
        print(f"  Clean Avg inference time: {clean_metrics['avg_inference_time_ms']:.2f} ms per example")

        print(f"  Robust Accuracy: {robust_metrics['accuracy']:.2f}%")
        print(f"  Robust F1 Score: {robust_metrics['f1']:.2f}%")
        print(f"  Robust Avg inference time: {robust_metrics['avg_inference_time_ms']:.2f} ms per example")

        results = {
            "name": model_name + "_onnx",
            "path": onnx_dir,
            "size": size_metrics,
            "clean": clean_metrics,
            "robust": robust_metrics,
        }


        os.makedirs(results_dir, exist_ok=True)
        with open(os.path.join(results_dir, f"{model_name}_onnx_results.json"), "w") as f:
            json.dump(results, f, indent=2)
        return results
    except Exception as e:
        print(f"Error evaluating ONNX {model_name}: {e}")
        import traceback
        traceback.print_exc()
        return {"name": model_name + "_onnx", "path": onnx_dir, "error": str(e)}

def main():
    wandb.init(
        project="LLMComp2025",
        name="final_comparison_with_onnx",
        config={
            "seed": 42,
            "max_length": 512,
            "batch_size": 16,
            "evaluation_type": "onnx_models"
        }   
    )
    set_seed(42)
    test_data_path = "/aul/homes/cvaro009/Desktop/LLMComp2025/data/processed/unified/unified_test.csv"
    results_dir = "/aul/homes/cvaro009/Desktop/LLMComp2025/onnx_results"
    os.makedirs(results_dir, exist_ok=True)
    test_df, test_dataset = load_test_data(test_data_path)
    models = [
        {
            "name": "albert",
            "onnx_path": "/disk/diamond-scratch/cvaro009/data/onnx_models/albert_onnx"
        },
        {
            "name": "mobilebert",
            "onnx_path": "/disk/diamond-scratch/cvaro009/data/onnx_models/mobilebert_onnx"
        },
        {
            "name": "distilbert",
            "onnx_path": "/disk/diamond-scratch/cvaro009/data/onnx_models/distilbert_onnx"
        },
        {
            "name": "tinyllama",
            "onnx_path": "/disk/diamond-scratch/cvaro009/data/onnx_models/tinyllama_onnx"
        },
        {
            "name": "mobilellama",
            "onnx_path": "/disk/diamond-scratch/cvaro009/data/onnx_models/mobile_llama_onnx",
        },
        {
            "name": "phi3",
            "onnx_path": "/disk/diamond-scratch/cvaro009/data/onnx_models/mobilellama_onnx"
        }
        
    ]
    all_results = {}
    for model_info in models:
        model_name = model_info["name"]
        onnx_dir = model_info.get("onnx_path")
        if onnx_dir and os.path.exists(onnx_dir):
            onnx_result = evaluate_onnx_model(model_name, onnx_dir, test_df, test_dataset, results_dir)
            if onnx_result and "error" not in onnx_result:
                all_results[model_name + "_onnx"] = onnx_result
        else:
            print(f"ONNX directory {onnx_dir} not found for {model_name}. Skipping ONNX evaluation.")
    # Save all results
    with open(os.path.join(results_dir, "all_models_results_with_onnx.json"), "w") as f:
        json.dump(all_results, f, indent=2)
    # Summary table
    summary = []
    for model_name, result in all_results.items():
        summary.append({
            "Model": model_name.upper(),
            "Clean F1": f"{result['clean']['f1']:.2f}%",
            "Clean Accuracy": f"{result['clean']['accuracy']:.2f}%",
            "Robust F1": f"{result['robust']['f1']:.2f}%",
            "Robust Accuracy": f"{result['robust']['accuracy']:.2f}%",
            "Inference Time Clean (ms)": f"{result['clean']['avg_inference_time_ms']:.2f}",
            "Inference Time Robust (ms)": f"{result['robust']['avg_inference_time_ms']:.2f}",
            "Model Size (MB)": f"{result['size']['disk_size_mb']:.2f}",
            "Peak Memory (MB)": f"{result['clean']['peak_memory_mb']:.2f}",
    })

    summary_df = pd.DataFrame(summary)
    summary_df.to_csv(os.path.join(results_dir, "model_comparison_summary_with_onnx.csv"), index=False)
    print("\nONNX Model Performance Summary:")
    print(summary_df.to_string(index=False))
    print(f"\nResults saved to {results_dir}")

if __name__ == "__main__":
    main()