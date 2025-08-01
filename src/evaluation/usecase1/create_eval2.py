import os
import json
import time
import torch
import numpy as np
import random
import pandas as pd
import matplotlib.pyplot as plt
import wandb
import string
from tqdm import tqdm
from pathlib import Path
from datasets import Dataset, load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoConfig, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, ConfusionMatrixDisplay
from torch.utils.data import DataLoader

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def inject_typos(text, error_rate=0.1):
    chars = list(text)
    num_typos = int(len(chars) * error_rate)
    for _ in range(num_typos):
        idx = random.randint(0, len(chars) - 1)
        if chars[idx].isalpha():
            chars[idx] = random.choice(string.ascii_lowercase)
    return ''.join(chars)

def evaluate_robustness(model, tokenizer, dataset, device, base_accuracy, error_rate=0.1):
    perturbed = dataset.map(lambda ex: {
        "input_text": inject_typos(ex["input_text"], error_rate),
        "label": ex["label"]
    })

    tokenized = perturbed.map(lambda ex: tokenizer(ex["input_text"], truncation=True, padding="max_length", max_length=128), batched=True)
    tokenized.set_format("torch", columns=["input_ids", "attention_mask", "label"])
    loader = DataLoader(tokenized, batch_size=16)

    y_true, y_pred = [], []
    with torch.no_grad():
        for batch in loader:
            labels = batch.pop("label")
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            preds = outputs.logits.argmax(dim=-1)
            y_pred.extend(preds.cpu().numpy())
            y_true.extend(labels.cpu().numpy())

    robust_acc = accuracy_score(y_true, y_pred) * 100
    return robust_acc / base_accuracy if base_accuracy > 0 else 0.0

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

def plot_confusion_matrix(y_true, y_pred, model_name, out_dir):
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap=plt.cm.Blues)
    plt.title(f"Confusion Matrix: {model_name}")
    plt.savefig(os.path.join(out_dir, f"confusion_matrix_{model_name}.png"))
    plt.close()

def measure_model_size(model_path):
    total_size = sum(p.stat().st_size for p in Path(model_path).glob('*') if p.is_file())
    file_sizes = {p.name: p.stat().st_size / (1024 ** 2) for p in Path(model_path).glob('*') if p.is_file()}
    config = AutoConfig.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_config(config)
    param_count = sum(p.numel() for p in model.parameters())
    trainable_param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {
        "disk_size_mb": total_size / (1024 ** 2),
        "file_sizes_mb": file_sizes,
        "parameters": param_count,
        "trainable_parameters": trainable_param_count
    }

def measure_peak_gpu_memory(device):
    if device == "cuda":
        torch.cuda.synchronize()
        peak_mem_bytes = torch.cuda.max_memory_allocated()
        return peak_mem_bytes / (1024 ** 2)
    return 0.0

def evaluate_model(model_name, model_path, test_df, test_dataset, results_dir):
    print(f"\n{'-' * 60}\nEvaluating model: {model_name}\n{'-' * 60}")
    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {device}")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

        if model_name.lower() == 'phi3':
            tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            )
            base_model = AutoModelForCausalLM.from_pretrained(
                "microsoft/phi-3-mini-4k-instruct",
                device_map={"": "cuda:0"},
                trust_remote_code=True,
                quantization_config=bnb_config
            )
            model = PeftModel.from_pretrained(base_model, model_path)
            model.to("cuda:0")
            model.eval()

            def normalize_label(text):
                text = text.lower()
                if "yes" in text:
                    return "yes"
                elif "no" in text:
                    return "no"
                else:
                    return "unknown"

            def generate_batch(model, tokenizer, prompts, max_new_tokens=10):
                inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True).to("cuda:0")
                with torch.no_grad():
                    outputs = model.generate(
                        **inputs,
                        max_new_tokens=max_new_tokens,
                        do_sample=False,
                        num_beams=1,
                        use_cache=True,
                        pad_token_id=tokenizer.eos_token_id
                    )
                decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)
                return [gen.replace(prompt, '').strip() for prompt, gen in zip(prompts, decoded)]

            prompts = test_df["input"].tolist()
            labels = [x.lower().strip() for x in test_df["response"].tolist()]
            batch_size = 8
            preds = []
            start_time = time.time()
            for i in tqdm(range(0, len(prompts), batch_size), desc="Phi3 Evaluation"):
                batch_prompts = [f"<|user|>\nPlease verify if the following information is valid:\n{p}\n<|assistant|>\n" for p in prompts[i:i + batch_size]]
                batch_preds = generate_batch(model, tokenizer, batch_prompts)
                preds.extend([p.lower().split(".")[0] + "." if "." in p else p for p in batch_preds])
            end_time = time.time()

            labels_norm = [normalize_label(l) for l in labels]
            preds_norm = [normalize_label(p) for p in preds]

            accuracy = accuracy_score(labels_norm, preds_norm)
            precision, recall, f1, _ = precision_recall_fscore_support(labels_norm, preds_norm, average="weighted")

            robust_pairs = [(l, p) for l, p in zip(labels_norm, preds_norm) if l in {"yes", "no"} and p in {"yes", "no"}]
            robust_labels = [1 if l == "yes" else 0 for l, _ in robust_pairs]
            robust_preds = [1 if p == "yes" else 0 for _, p in robust_pairs]
            robustness = accuracy_score(robust_labels, robust_preds) if len(robust_pairs) > 0 else 0.0

            num_unknown_preds = sum(1 for p in preds_norm if p == "unknown")
            print(f"[DEBUG] Unknown predictions: {num_unknown_preds} / {len(preds_norm)}")

            eval_runtime = end_time - start_time
            avg_time_per_sample = (eval_runtime / len(prompts)) * 1000

            size_metrics = measure_model_size(model_path)
            latency_metrics = {"mean_ms": avg_time_per_sample}
            peak_memory = measure_peak_gpu_memory("cuda")

            results = {
                "name": model_name,
                "path": model_path,
                "size": size_metrics,
                "accuracy": {
                    "accuracy": accuracy * 100,
                    "f1": f1 * 100,
                    "precision": precision * 100,
                    "recall": recall * 100,
                    "avg_inference_time_ms": avg_time_per_sample,
                    "eval_runtime": eval_runtime,
                    "eval_samples_per_second": len(prompts) / eval_runtime,
                },
                "latency": latency_metrics,
                "memory": {"peak_memory_mb": peak_memory},
                "robustness": {
                    "typos": {"level_0.1": {"robustness_score": robustness}},
                    "baseline_accuracy": accuracy * 100
                }
            }

            with open(os.path.join(results_dir, f"{model_name}_results.json"), "w") as f:
                json.dump(results, f, indent=2)
            return results

        else:
            tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
            model = AutoModelForSequenceClassification.from_pretrained(model_path, num_labels=2).to(device)
            model.eval()

            def preprocess_function(examples):
                return tokenizer(examples["input_text"], truncation=True, padding="max_length", max_length=128)

            test_tokenized = test_dataset.map(preprocess_function, batched=True)
            test_tokenized.set_format("torch", columns=["input_ids", "attention_mask", "label"])

            test_dataloader = DataLoader(test_tokenized, batch_size=16)
            start_time = time.time()
            y_pred, y_true = [], []
            with torch.no_grad():
                for batch in tqdm(test_dataloader, desc=f"Evaluating {model_name}"):
                    labels = batch.pop("label")
                    batch = {k: v.to(device) for k, v in batch.items()}
                    outputs = model(**batch)
                    preds = outputs.logits.argmax(dim=-1)
                    y_pred.extend(preds.cpu().numpy())
                    y_true.extend(labels.cpu().numpy())

            accuracy = accuracy_score(y_true, y_pred)
            precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="weighted")
            total_time = time.time() - start_time
            avg_time_per_sample = total_time / len(y_true) * 1000

            robustness = evaluate_robustness(model, tokenizer, test_dataset, device, accuracy * 100)

            size_metrics = measure_model_size(model_path)
            latency_metrics = {"mean_ms": avg_time_per_sample}
            peak_memory = measure_peak_gpu_memory(device)

            results = {
                "name": model_name,
                "path": model_path,
                "size": size_metrics,
                "accuracy": {
                    "accuracy": accuracy * 100,
                    "f1": f1 * 100,
                    "precision": precision * 100,
                    "recall": recall * 100,
                    "avg_inference_time_ms": avg_time_per_sample,
                },
                "latency": latency_metrics,
                "memory": {"peak_memory_mb": peak_memory},
                "robustness": {
                    "typos": {"level_0.1": {"robustness_score": robustness}},
                    "baseline_accuracy": accuracy * 100
                }
            }

            with open(os.path.join(results_dir, f"{model_name}_results.json"), "w") as f:
                json.dump(results, f, indent=2)
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

    wandb.init(project="risk-model-evaluation", name="verification_model_comparison", config={
        "test_data": test_data_path,
        "num_models": len(models)
    })

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
        json.dump(all_results, f, indent=2)

    summary = [
        {
            "Model": name.upper(),
            "F1 Score": f"{r['accuracy']['f1']:.2f}%",
            "Accuracy": f"{r['accuracy']['accuracy']:.2f}%",
            "Inference Time (ms)": f"{r['accuracy']['avg_inference_time_ms']:.2f}",
            "Model Size (MB)": f"{r['size']['disk_size_mb']:.2f}",
            "Memory (MB)": f"{r['memory']['peak_memory_mb']:.2f}",
            "Robustness Score": f"{r['robustness']['typos']['level_0.1']['robustness_score']:.2f}"
        } for name, r in all_results.items()
    ]

    summary_df = pd.DataFrame(summary)
    wandb.log({"model_comparison_summary": wandb.Table(dataframe=summary_df)})
    summary_df.to_csv(os.path.join(results_dir, "verification_model_comparison_summary.csv"), index=False)
    print("\nModel Performance Summary:")
    print(summary_df.to_string(index=False))
    print(f"\nResults saved to {results_dir}")

if __name__ == "__main__":
    main()
