# ✅ Full ONNX Evaluation Script Mirroring Fine-Tuned Pipeline (Robustness + Metrics)

import os
import time
import json
import random
import pandas as pd
import numpy as np
import torch
import onnxruntime as ort
from datasets import Dataset
from transformers import AutoTokenizer
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import subprocess

# === CONFIG ===
MODEL_PATHS = [
    "/aul/homes/melsh008/temp-storage-space-on-diamond/Quantizedmodels/albert_export_onnx",
    "/aul/homes/melsh008/temp-storage-space-on-diamond/Quantizedmodels/distilbert_export_onnx",
    "/aul/homes/melsh008/temp-storage-space-on-diamond/Quantizedmodels/mobilebert_export_onnx",
    "/aul/homes/melsh008/temp-storage-space-on-diamond/Quantizedmodels/mobilellama_export_onnx",
    "/aul/homes/melsh008/temp-storage-space-on-diamond/Quantizedmodels/tinyllama_onnx_quant",
]

MODEL_FILES = {
    "albert_export_onnx": "albert_int8.onnx",
    "distilbert_export_onnx": "distilbert_int8.onnx",
    "mobilebert_export_onnx": "mobilebert_int8.onnx",
    "mobilellama_export_onnx": "model.onnx",
    "tinyllama_onnx_quant": "tinyllama_int8.onnx"
}

DATA_PATH = "/aul/homes/melsh008/First_Case_Scenario/verification_augmented_with_license_id.jsonl"
RESULTS_DIR = "/aul/homes/melsh008/First_Case_Scenario/evaluation_results"
os.makedirs(RESULTS_DIR, exist_ok=True)
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# === Utilities ===
def load_jsonl(file_path):
    def label_from_response(response_text):
        if response_text.lower().startswith("yes"):
            return 1
        elif response_text.lower().startswith("no"):
            return 0
        else:
            raise ValueError(f"Unknown response: {response_text}")

    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            obj = json.loads(line)
            label = label_from_response(obj["response"])
            prompt = f"Task: {obj.get('task_type', 'triage_verification')}\nInput: {obj['input']}"
            data.append({"text": prompt, "label": label})
    return Dataset.from_list(data)

def get_gpu_memory_mb():
    try:
        result = subprocess.check_output([
            'nvidia-smi', '--query-gpu=memory.used', '--format=csv,nounits,noheader']
        )
        return float(result.decode().strip().split('\n')[0])
    except Exception as e:
        print(f"⚠️ Could not read GPU memory usage via nvidia-smi: {e}")
        return 0.0

# === Fine-Tuned Style Robustness Noise ===
def add_typos(text, typo_rate=0.1):
    text = list(text)
    n_typos = max(1, int(len(text) * typo_rate))
    for _ in range(n_typos):
        idx = random.randint(0, len(text) - 2)
        text[idx], text[idx + 1] = text[idx + 1], text[idx]
    return "".join(text)

def create_noisy_texts(dataset, typo_rate=0.1):
    return [add_typos(sample['text'], typo_rate) for sample in dataset]

# === Load Dataset and Noisy Inputs ===
random.seed(42)
dataset = load_jsonl(DATA_PATH)
labels = torch.tensor(dataset["label"])
noisy_texts = create_noisy_texts(dataset, typo_rate=0.1)

results = []

for model_dir in MODEL_PATHS:
    print("=" * 60)
    print(f"🚀 Evaluating model in: {model_dir}")

    model_name = os.path.basename(model_dir)
    model_file = os.path.join(model_dir, MODEL_FILES.get(model_name, ""))
    if not os.path.exists(model_file):
        print(f"❌ Quantized model file not found: {model_file}")
        continue

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
        providers = ["CUDAExecutionProvider"] if ort.get_device() == "GPU" else ["CPUExecutionProvider"]
        session = ort.InferenceSession(model_file, providers=providers)
        input_names = [inp.name for inp in session.get_inputs()]
    except Exception as e:
        print(f"❌ Failed to load model or tokenizer: {e}")
        continue

    encodings = tokenizer(dataset["text"], truncation=True, padding="max_length", max_length=128, return_tensors="pt")
    noisy_encodings = tokenizer(noisy_texts, truncation=True, padding="max_length", max_length=128, return_tensors="pt")

    batch_size = 8
    num_batches = int(np.ceil(len(labels) / batch_size))

    torch.cuda.empty_cache() if ort.get_device() == "GPU" else None
    start_mem = get_gpu_memory_mb() if ort.get_device() == "GPU" else 0.0

    # === Clean Inference ===
    all_preds = []
    start_time = time.time()
    for i in range(num_batches):
        input_ids = encodings["input_ids"][i*batch_size:(i+1)*batch_size].numpy()
        attention_mask = encodings["attention_mask"][i*batch_size:(i+1)*batch_size].numpy()
        inputs_onnx = {input_names[0]: input_ids}
        if len(input_names) > 1: inputs_onnx[input_names[1]] = attention_mask
        if len(input_names) > 2:
            try:
                input_obj = session.get_inputs()[2]
                dummy_input = np.zeros([d if isinstance(d, int) else 1 for d in input_obj.shape], dtype=np.int64)
                inputs_onnx[input_obj.name] = dummy_input
            except: pass
        try:
            logits = session.run(None, inputs_onnx)[0]
            preds = (logits >= 0.5).astype(int).flatten() if logits.shape[1] == 1 else np.argmax(logits, axis=1)
            all_preds.extend(preds.tolist())
        except Exception as e:
            print(f"❌ Inference failed on batch {i}: {e}")
    end_time = time.time()

    latency_per_sample = (end_time - start_time) / len(labels)
    end_mem = get_gpu_memory_mb() if ort.get_device() == "GPU" else 0.0
    peak_memory_mb = max(end_mem - start_mem, 0.0)
    model_size_mb = os.path.getsize(model_file) / (1024 ** 2)

    accuracy = accuracy_score(labels.numpy(), all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(labels.numpy(), all_preds, average="binary", zero_division=0)

    # === Noisy Inference for Robustness ===
    all_noisy_preds = []
    for i in range(num_batches):
        input_ids = noisy_encodings["input_ids"][i*batch_size:(i+1)*batch_size].numpy()
        attention_mask = noisy_encodings["attention_mask"][i*batch_size:(i+1)*batch_size].numpy()
        inputs_onnx = {input_names[0]: input_ids}
        if len(input_names) > 1: inputs_onnx[input_names[1]] = attention_mask
        if len(input_names) > 2:
            try:
                input_obj = session.get_inputs()[2]
                dummy_input = np.zeros([d if isinstance(d, int) else 1 for d in input_obj.shape], dtype=np.int64)
                inputs_onnx[input_obj.name] = dummy_input
            except: pass
        try:
            logits = session.run(None, inputs_onnx)[0]
            preds = (logits >= 0.5).astype(int).flatten() if logits.shape[1] == 1 else np.argmax(logits, axis=1)
            all_noisy_preds.extend(preds.tolist())
        except Exception as e:
            print(f"❌ Noisy inference failed on batch {i}: {e}")

    robustness_accuracy = accuracy_score(labels.numpy(), all_noisy_preds)
    robustness_score = robustness_accuracy / accuracy if accuracy > 0 else 0.0

    results.append({
        "Model": model_name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
        "Latency_ms": latency_per_sample * 1000,
        "ModelSize_MB": model_size_mb,
        "Memory_MB": peak_memory_mb,
        "Robustness": robustness_score,
        "Noisy_Accuracy": robustness_accuracy
    })

# === Save Results ===
summary_df = pd.DataFrame([{**r, "Model": r["Model"].upper()} for r in results])
summary_csv_path = os.path.join(RESULTS_DIR, "verification_model_comparison_summary.csv")
summary_df.to_csv(summary_csv_path, index=False)
with open(os.path.join(RESULTS_DIR, "all_models_results.json"), "w") as f:
    json.dump(results, f, indent=2)

print("\nModel Evaluation Summary:")
print(summary_df.to_string(index=False))
print(f"\n✅ Saved summary CSV to: {summary_csv_path}")
