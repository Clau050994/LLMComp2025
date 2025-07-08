import os
import json
import time
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from pathlib import Path
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, ConfusionMatrixDisplay
import psutil
import random
from torch.utils.data import DataLoader
import wandb


def set_seed(seed=42):
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

def evaluate_accuracy(model, tokenizer, test_dataset, device, model_name, out_dir):
    # Check if tokenizer has pad_token, if not set it
    if tokenizer.pad_token is None:
        if tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            tokenizer.add_special_tokens({'pad_token': '[PAD]'})
            model.resize_token_embeddings(len(tokenizer))
    
    def preprocess_function(examples):
        return tokenizer(examples["input_text"], truncation=True, padding="max_length", max_length=128)
    
    test_tokenized = test_dataset.map(preprocess_function, batched=True)
    if "input_text" in test_tokenized.column_names:
        test_tokenized = test_tokenized.remove_columns(["input_text"])
    test_tokenized.set_format("torch", columns=["input_ids", "attention_mask", "label"])
    
    # Use smaller batch size for models that have trouble with batching
    batch_size = 1 if model_name in ['phi2', 'phi3'] else 16
    test_dataloader = DataLoader(test_tokenized, batch_size=batch_size)
    
    model.to(device)
    model.eval()
    
    start_time = time.time()
    y_pred, y_true = [], []
    
    with torch.no_grad():
        for batch in tqdm(test_dataloader, desc=f"Evaluating {model_name}"):
            batch = {k: v.to(device) for k, v in batch.items()}
            
            try:
                outputs = model(**{k: v for k, v in batch.items() if k != "label"})
                predictions = outputs.logits.argmax(dim=-1)
                y_pred.extend(predictions.cpu().numpy())
                y_true.extend(batch["label"].cpu().numpy())
            except Exception as e:
                print(f"Error processing batch for {model_name}: {e}")
                # Skip this batch and continue
                continue
    
    if len(y_pred) == 0:
        print(f"Warning: No predictions were made for {model_name}")
        return {
            "accuracy": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "confusion_matrix": [[0, 0], [0, 0]],
            "total_inference_time_s": 0.0,
            "avg_inference_time_ms": 0.0
        }
    
    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="weighted")
    cm = confusion_matrix(y_true, y_pred)
    total_time = time.time() - start_time
    avg_time_per_sample = total_time / len(y_true) * 1000  # ms
    
    plot_confusion_matrix(y_true, y_pred, model_name, out_dir)
    
    return {
        "accuracy": accuracy * 100,
        "precision": precision * 100,
        "recall": recall * 100,
        "f1": f1 * 100,
        "confusion_matrix": cm.tolist(),
        "total_inference_time_s": total_time,
        "avg_inference_time_ms": avg_time_per_sample
    }

def measure_latency(model, tokenizer, device, num_runs=50, batch_sizes=[1, 8]):
    sample_text = "Traveler from Canada with valid passport seeking entry for tourism for 2 weeks"
    latency_results = {}

    # If the tokenizer has no pad_token, skip batch_size >1
    if tokenizer.pad_token is None and tokenizer.eos_token is None:
        print("Warning: Tokenizer has no pad token or eos token. Only measuring batch_size=1 latency.")
        batch_sizes = [1]

    encoded = tokenizer(sample_text, return_tensors="pt").to(device)

    for _ in range(10):
        _ = model(**encoded)

    for batch_size in batch_sizes:
        batch_encoded = {k: v.repeat(batch_size, 1) for k, v in encoded.items()} if batch_size > 1 else encoded
        latencies = []
        for _ in range(num_runs):
            if device == "cuda":
                torch.cuda.synchronize()
            start_time = time.time()
            _ = model(**batch_encoded)
            if device == "cuda":
                torch.cuda.synchronize()
            latencies.append((time.time() - start_time) * 1000)  # ms
        per_item_latencies = [l / batch_size for l in latencies]
        latency_results[f"batch_size_{batch_size}"] = {
            "mean_ms": np.mean(latencies),
            "p50_ms": np.percentile(latencies, 50),
            "p90_ms": np.percentile(latencies, 90),
            "p99_ms": np.percentile(latencies, 99),
            "per_item_mean_ms": np.mean(per_item_latencies)
        }
    return latency_results


def measure_model_size(model_path):
    from transformers import AutoConfig

    total_size = 0
    for path in Path(model_path).glob('**/*'):
        if path.is_file():
            total_size += path.stat().st_size

    file_sizes = {}
    for path in Path(model_path).glob('*'):
        if path.is_file():
            file_sizes[path.name] = path.stat().st_size / (1024 * 1024)  # MB

    # Load config and create model for counting parameters
    config = AutoConfig.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_config(config)

    param_count = sum(p.numel() for p in model.parameters())
    trainable_param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)

    return {
        "disk_size_mb": total_size / (1024 * 1024),
        "file_sizes_mb": file_sizes,
        "parameters": param_count,
        "trainable_parameters": trainable_param_count
    }


def measure_memory_usage(model, tokenizer, device, batch_size=1):
    sample_text = "Traveler from Canada with valid passport seeking entry for tourism for 2 weeks"
    encoded = tokenizer(sample_text, return_tensors="pt").to(device)
    if batch_size > 1:
        encoded = {k: v.repeat(batch_size, 1) for k, v in encoded.items()}
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()
        _ = model(**encoded)
        peak_memory = torch.cuda.max_memory_allocated() / (1024 * 1024)  # MB
    else:
        process = psutil.Process(os.getpid())
        base_memory = process.memory_info().rss / (1024 * 1024)
        _ = model(**encoded)
        post_memory = process.memory_info().rss / (1024 * 1024)
        peak_memory = post_memory - base_memory
    return {"peak_memory_mb": peak_memory}

def add_noise(text, noise_type="typos", noise_level=0.1):
    if noise_type == "typos":
        chars = list(text)
        n_errors = max(1, int(len(chars) * noise_level))
        indices = random.sample(range(len(chars)), n_errors)
        for idx in indices:
            chars[idx] = random.choice("abcdefghijklmnopqrstuvwxyz ")
        return "".join(chars)
    elif noise_type == "missing_words":
        words = text.split()
        n_to_delete = max(1, int(len(words) * noise_level))
        indices = random.sample(range(len(words)), n_to_delete)
        return " ".join([w for i, w in enumerate(words) if i not in indices])
    elif noise_type == "ocr_errors":
        ocr_map = {"0": "o", "1": "l", "2": "z", "5": "s", "8": "b", "o": "0", "l": "1", "z": "2", "s": "5", "b": "8"}
        chars = list(text)
        n_errors = max(1, int(len(chars) * noise_level))
        indices = random.sample(range(len(chars)), n_errors)
        for idx in indices:
            if chars[idx] in ocr_map:
                chars[idx] = ocr_map[chars[idx]]
        return "".join(chars)
    return text

def test_robustness(model, tokenizer, test_df, device, num_samples=50):
    sample_indices = random.sample(range(len(test_df)), num_samples)
    text_column = "input_text" if "input_text" in test_df.columns else "text"
    sample_texts = test_df.iloc[sample_indices][text_column].tolist()
    sample_labels = test_df.iloc[sample_indices]["label"].tolist()
    encoded = tokenizer(sample_texts, padding=True, truncation=True, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**encoded)
    baseline_preds = outputs.logits.argmax(dim=-1).cpu().numpy()
    baseline_acc = accuracy_score(sample_labels, baseline_preds) * 100
    results = {"baseline_accuracy": baseline_acc}
    noise_types = ["typos", "missing_words", "ocr_errors"]
    noise_levels = [0.05, 0.1, 0.2]
    for noise_type in noise_types:
        noise_results = {}
        for noise_level in noise_levels:
            noisy_texts = [add_noise(text, noise_type, noise_level) for text in sample_texts]
            encoded = tokenizer(noisy_texts, padding=True, truncation=True, return_tensors="pt").to(device)
            with torch.no_grad():
                outputs = model(**encoded)
            noisy_preds = outputs.logits.argmax(dim=-1).cpu().numpy()
            noisy_acc = accuracy_score(sample_labels, noisy_preds) * 100
            robustness = noisy_acc / baseline_acc if baseline_acc > 0 else 0
            noise_results[f"level_{noise_level}"] = {
                "accuracy": noisy_acc,
                "robustness_score": robustness,
                "accuracy_drop": baseline_acc - noisy_acc
            }
        results[noise_type] = noise_results
    return results

def evaluate_model(model_name, model_path, test_df, test_dataset, results_dir):
    print(f"\n{'-'*60}\nEvaluating model: {model_name}\n{'-'*60}")
    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {device}")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if 'phi' in model_name.lower():
            model = AutoModelForSequenceClassification.from_pretrained(model_path, num_labels=3, ignore_mismatched_sizes=True)
        else:
            model = AutoModelForSequenceClassification.from_pretrained(model_path)
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model.to(device)
        print("Measuring model size...")
        size_metrics = measure_model_size(model_path)
        print(f"  Model size: {size_metrics['disk_size_mb']:.2f} MB")
        print(f"  Parameters: {size_metrics['parameters']:,}")
        print("Evaluating accuracy...")
        accuracy_metrics = evaluate_accuracy(model, tokenizer, test_dataset, device, model_name, results_dir)
        print(f"  Accuracy: {accuracy_metrics['accuracy']:.2f}%")
        print(f"  F1 Score: {accuracy_metrics['f1']:.2f}%")
        print(f"  Avg inference time: {accuracy_metrics['avg_inference_time_ms']:.2f} ms per example")
        print("Measuring latency...")
        latency_metrics = measure_latency(model, tokenizer, device)
        print(f"  Batch size 1: {latency_metrics['batch_size_1']['mean_ms']:.2f} ms")
        if 'batch_size_8' in latency_metrics:
            print(f"  Batch size 8 (per item): {latency_metrics['batch_size_8']['per_item_mean_ms']:.2f} ms")
        print("Measuring memory usage...")
        memory_metrics = measure_memory_usage(model, tokenizer, device)
        print(f"  Peak memory: {memory_metrics['peak_memory_mb']:.2f} MB")
        print("Testing robustness...")
        robustness_metrics = test_robustness(model, tokenizer, test_df, device)
        print(f"  Baseline accuracy: {robustness_metrics['baseline_accuracy']:.2f}%")
        print(f"  Robustness to 10% typos: {robustness_metrics['typos']['level_0.1']['robustness_score']:.2f}")
        results = {
            "name": model_name,
            "path": model_path,
            "size": size_metrics,
            "accuracy": accuracy_metrics,
            "latency": latency_metrics,
            "memory": memory_metrics,
            "robustness": robustness_metrics
        }
        os.makedirs(results_dir, exist_ok=True)
        with open(os.path.join(results_dir, f"{model_name}_results.json"), "w") as f:
            json.dump(results, f, default=lambda obj: obj.tolist() if isinstance(obj, np.ndarray) else float(obj) if isinstance(obj, (np.float32, np.float64)) else int(obj) if isinstance(obj, np.integer) else obj, indent=2)
        model.cpu()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()    
        return results
    except Exception as e:
        print(f"Error evaluating {model_name}: {e}")
        import traceback
        traceback.print_exc()
        return {"name": model_name, "path": model_path, "error": str(e)}

def main():
    set_seed(42)
    test_data_path = "/aul/homes/cvaro009/Desktop/LLMComp2025/data/processed/unified/unified_test.csv"
    results_dir = "/aul/homes/cvaro009/Desktop/LLMComp2025/results"
    os.makedirs(results_dir, exist_ok=True)
    test_df, test_dataset = load_test_data(test_data_path)
    models = [
    {"name": "albert", "path": "/disk/diamond-scratch/cvaro009/data/albert"},
    {"name": "mobilebert", "path": "/disk/diamond-scratch/cvaro009/data/mobilebert"},
    {"name": "distilbert", "path": "/disk/diamond-scratch/cvaro009/data/distilbert"},
    {"name": "tinyllama", "path": "/disk/diamond-scratch/cvaro009/data/tinyllama"},
    {"name": "mobilellama", "path": "/disk/diamond-scratch/cvaro009/data/mobilellama"},
    {"name": "phi3", "path": "/disk/diamond-scratch/cvaro009/data/phi3_risk_classification_qlora"},
]
    # Set up WandB
    wandb.init(
        project="risk-model-evaluation",
        name="baseline_model_comparison",
        config={
            "test_data": test_data_path,
            "num_models": len(models)
        }
    )

    all_results = {}
    for model_info in models:
        model_name = model_info["name"]
        model_path = model_info["path"]
        if not os.path.exists(model_path):
            print(f"Warning: Model path {model_path} not found. Skipping.")
            continue
        result = evaluate_model(model_name, model_path, test_df, test_dataset, results_dir)
        if result and "error" not in result:
            wandb.log({
                  f"{model_name}/accuracy": result["accuracy"]["accuracy"],
                  f"{model_name}/f1": result["accuracy"]["f1"],
                  f"{model_name}/precision": result["accuracy"]["precision"],
                  f"{model_name}/recall": result["accuracy"]["recall"],
                  f"{model_name}/avg_inference_time_ms": result["accuracy"]["avg_inference_time_ms"],
                  f"{model_name}/model_size_mb": result["size"]["disk_size_mb"],
                  f"{model_name}/peak_memory_mb": result["memory"]["peak_memory_mb"],
                  f"{model_name}/robustness_10pct_typos": result["robustness"]["typos"]["level_0.1"]["robustness_score"]
            })
            all_results[model_name] = result
    with open(os.path.join(results_dir, "all_models_results.json"), "w") as f:
        json.dump(all_results, f, default=lambda obj: obj.tolist() if isinstance(obj, np.ndarray) else float(obj) if isinstance(obj, (np.float32, np.float64)) else int(obj) if isinstance(obj, np.integer) else obj, indent=2)
    summary = []
    for model_name, result in all_results.items():
        summary.append({
            "Model": model_name.upper(),
            "F1 Score": f"{result['accuracy']['f1']:.2f}%",
            "Accuracy": f"{result['accuracy']['accuracy']:.2f}%",
            "Inference Time (ms)": f"{result['accuracy']['avg_inference_time_ms']:.2f}",
            "Model Size (MB)": f"{result['size']['disk_size_mb']:.2f}",
            "Memory (MB)": f"{result['memory']['peak_memory_mb']:.2f}",
            "Robustness Score": f"{result['robustness']['typos']['level_0.1']['robustness_score']:.2f}"
        })
    summary_df = pd.DataFrame(summary)
    wandb.log({"model_comparison_summary": wandb.Table(dataframe=summary_df)})
    print("\nSummary of Model Performance:")
    summary_df.to_csv(os.path.join(results_dir, "baseline_model_comparison_summary.csv"), index=False)
    print("\nModel Performance Summary:")
    print(summary_df.to_string(index=False))
    print(f"\nResults saved to {results_dir}")

if __name__ == "__main__":
    main()