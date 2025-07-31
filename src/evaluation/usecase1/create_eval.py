import os
import json
import time
from datasets import Dataset  # add import at top if not already
import torch
import numpy as np
from transformers import AutoConfig
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from pathlib import Path
from transformers import AutoModelForCausalLM
from peft import PeftModel
from datasets import load_dataset
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


def load_test_data_jsonl(data_path):
    print(f"Loading test data from {data_path}")
    ds = load_dataset('json', data_files=data_path)['train']
    def map_label(example):
        example['input_text'] = example['input']
        example['label'] = 1 if example['response'].lower().startswith("yes") else 0
        return example
    ds = ds.map(map_label)
    print(f"Loaded test data with {len(ds)} examples")
    return ds.to_pandas(), ds


def evaluate_phi3_generation(model, tokenizer, dataset, device, batch_size=8):
    model.eval()
    preds = []
    labels = []

    def normalize_label(text):
        text = text.lower()
        if "yes" in text:
            return "yes"
        elif "no" in text:
            return "no"
        else:
            return "unknown"

    total_time = 0.0
    total_samples = 0

    for i in tqdm(range(0, len(dataset), batch_size), desc="Evaluating phi3 generation"):
        batch = dataset.select(range(i, min(i+batch_size, len(dataset))))
        prompts = [sample["input"] for sample in batch]
        labels.extend([sample["label"] if "label" in sample else None for sample in batch])
        total_samples += len(prompts)

        inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True)
        device = next(model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        start = time.time()
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=50,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id
            )
        end = time.time()

        total_time += (end - start)

        decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)

        for prompt, gen in zip(prompts, decoded):
            pred_text = gen.replace(prompt, "").strip().lower()
            if "yes" in pred_text:
                preds.append("yes")
            elif "no" in pred_text:
                preds.append("no")
            else:
                preds.append("unknown")

    labels_norm = [normalize_label(l) if isinstance(l, str) else ("yes" if l == 1 else "no") for l in labels]

    accuracy = accuracy_score(labels_norm, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(labels_norm, preds, average="weighted")

    avg_inference_time_ms = (total_time / total_samples) * 1000 if total_samples > 0 else None

    return {
        "accuracy": accuracy * 100,
        "precision": precision * 100,
        "recall": recall * 100,
        "f1": f1 * 100,
        "avg_inference_time_ms": avg_inference_time_ms
    }





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


def plot_confusion_matrix(y_true, y_pred, model_name, out_dir):
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap=plt.cm.Blues)
    plt.title(f"Confusion Matrix: {model_name}")
    plt.savefig(os.path.join(out_dir, f"confusion_matrix_{model_name}.png"))
    plt.close()


def evaluate_accuracy(model, tokenizer, test_dataset, device, model_name, out_dir):
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

    batch_size = 1 if model_name.lower() in ['phi3'] else 16
    test_dataloader = DataLoader(test_tokenized, batch_size=batch_size)

    model.to(device)
    model.eval()

    start_time = time.time()
    y_pred, y_true = [], []

    with torch.no_grad():
        for batch in tqdm(test_dataloader, desc=f"Evaluating {model_name}"):
            batch = {k: (v.to(device) if isinstance(v, torch.Tensor) else torch.tensor(v).to(device)) for k, v in batch.items()}
            try:
                outputs = model(**{k: v for k, v in batch.items() if k != "label"})
                predictions = outputs.logits.argmax(dim=-1)
                y_pred.extend(predictions.cpu().numpy())
                y_true.extend(batch["label"].cpu().numpy())
            except Exception as e:
                print(f"Error processing batch for {model_name}: {e}")
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


def measure_latency(model, tokenizer, device, input_text="Test input", runs=100):
    model.eval()
    inputs = tokenizer(input_text, return_tensors="pt").to(device)
    # Warm up
    with torch.no_grad():
        for _ in range(10):
            _ = model(**inputs)

    times = []
    with torch.no_grad():
        for _ in range(runs):
            start = time.time()
            _ = model(**inputs)
            times.append(time.time() - start)
    mean_latency_ms = np.mean(times) * 1000
    return {"mean_ms": mean_latency_ms}


def measure_model_size(model_path):
    total_size = 0
    for path in Path(model_path).glob('*'):
        if path.is_file():
            total_size += path.stat().st_size

    file_sizes = {}
    for path in Path(model_path).glob('*'):
        if path.is_file():
            file_sizes[path.name] = path.stat().st_size / (1024 * 1024)  # MB

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


def measure_peak_gpu_memory(device):
    if device == "cuda":
        torch.cuda.synchronize()
        peak_mem_bytes = torch.cuda.max_memory_allocated()
        peak_mem_mb = peak_mem_bytes / (1024 * 1024)
        return peak_mem_mb
    else:
        return 0.0


def add_typos(text, typo_rate=0.1):
    text = list(text)
    n_typos = max(1, int(len(text) * typo_rate))
    for _ in range(n_typos):
        idx = random.randint(0, len(text) - 2)
        # Swap adjacent characters
        text[idx], text[idx + 1] = text[idx + 1], text[idx]
    return "".join(text)


def create_noisy_dataset(dataset, typo_rate=0.1):
    noisy_samples = []
    for sample in dataset:
        noisy_input = add_typos(sample['input'], typo_rate)
        new_sample = sample.copy()
        new_sample['input'] = noisy_input
        noisy_samples.append(new_sample)
    return noisy_samples


def test_robustness(model, tokenizer, test_dataset, device, model_name, out_dir, typo_rate=0.1):
    print(f"Running robustness test with {typo_rate*100}% typo noise for {model_name}")
    noisy_data = create_noisy_dataset(test_dataset, typo_rate)
    noisy_dataset = Dataset.from_list(noisy_data)  # Convert list back to HF Dataset
    noisy_accuracy_metrics = evaluate_accuracy(model, tokenizer, noisy_dataset, device, model_name + "_noisy", out_dir)
    clean_accuracy_metrics = evaluate_accuracy(model, tokenizer, test_dataset, device, model_name, out_dir)
    robustness_score = noisy_accuracy_metrics["accuracy"] / clean_accuracy_metrics["accuracy"] if clean_accuracy_metrics["accuracy"] > 0 else 0.0
    return {
        "typos": {
            f"level_{typo_rate}": {
                "robustness_score": robustness_score
            }
        },
        "baseline_accuracy": clean_accuracy_metrics["accuracy"]
    }


def evaluate_model(model_name, model_path, test_df, test_dataset, results_dir):
    print(f"\n{'-' * 60}\nEvaluating model: {model_name}\n{'-' * 60}")
    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {device}")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

        if model_name.lower() == 'phi3':
            from transformers import BitsAndBytesConfig

            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            )

            base_model = AutoModelForCausalLM.from_pretrained(
                "microsoft/phi-3-mini-4k-instruct",
                device_map={"": "cuda:0"},     # load entire model on cuda:0
                trust_remote_code=True,
                quantization_config=bnb_config,
                torch_dtype=torch.bfloat16,
                low_cpu_mem_usage=True,
            )
            model = PeftModel.from_pretrained(base_model, model_path)
            model.to("cuda:0")  # ensure model is on cuda:0
            model.eval()

            print("Evaluating phi3 generation...")
            accuracy_metrics = evaluate_phi3_generation(model, tokenizer, test_dataset, "cuda:0")

        else:
            model = AutoModelForSequenceClassification.from_pretrained(model_path, num_labels=2)
            model.to(device)
            model.eval()

            print("Evaluating accuracy...")
            accuracy_metrics = evaluate_accuracy(model, tokenizer, test_dataset, device, model_name, results_dir)

            peak_memory = measure_peak_gpu_memory(device)

            print("Measuring model size...")
            size_metrics = measure_model_size(model_path)
            print(f"  Model size: {size_metrics['disk_size_mb']:.2f} MB")
            print(f"  Parameters: {size_metrics['parameters']:,}")


            latency_metrics = measure_latency(model, tokenizer, device)

            robustness_metrics = test_robustness(model, tokenizer, test_dataset, device, model_name, results_dir)

            results = {
                "name": model_name,
                "path": model_path,
                "size": size_metrics,
                "accuracy": accuracy_metrics,
                "latency": latency_metrics,
                "memory": {"peak_memory_mb": peak_memory},
                "robustness": robustness_metrics
            }
            os.makedirs(results_dir, exist_ok=True)
            with open(os.path.join(results_dir, f"{model_name}_results.json"), "w") as f:
                json.dump(results, f, indent=2)
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
    test_data_path = "/aul/homes/melsh008/First_Case_Scenario/verification_augmented_with_license_id.jsonl"
    results_dir = "/aul/homes/melsh008/First_Case_Scenario/evaluation_results"
    os.makedirs(results_dir, exist_ok=True)
    test_df, test_dataset = load_test_data_jsonl(test_data_path)

    models = [
        {"name": "phi3", "path": "/aul/homes/melsh008/First_Case_Scenario/phi3_qlora_verification"},
        {"name": "albert", "path": "/aul/homes/melsh008/First_Case_Scenario/albert_verification_finetuned"},
        {"name": "distilbert", "path": "/aul/homes/melsh008/First_Case_Scenario/distilbert_verification_finetuned"},
        {"name": "mobilebert", "path": "/aul/homes/melsh008/First_Case_Scenario/mobilebert_verification_finetuned"},
        {"name": "mobilellama", "path": "/aul/homes/melsh008/First_Case_Scenario/mobilellama_verification_finetuned"},
        {"name": "tinyllama", "path": "/aul/homes/melsh008/First_Case_Scenario/tinyllama_verification_finetuned"},
    ]

    wandb.init(
        project="risk-model-evaluation",
        name="verification_model_comparison",
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
        json.dump(
            all_results,
            f,
            default=lambda obj: obj.tolist() if isinstance(obj, np.ndarray) else
            float(obj) if isinstance(obj, (np.float32, np.float64)) else
            int(obj) if isinstance(obj, np.integer) else obj,
            indent=2
        )

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
    summary_df.to_csv(os.path.join(results_dir, "verification_model_comparison_summary.csv"), index=False)
    print("\nModel Performance Summary:")
    print(summary_df.to_string(index=False))
    print(f"\nResults saved to {results_dir}")


if __name__ == "__main__":
    main()

