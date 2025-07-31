import os
import time
import json
import numpy as np
import torch
import onnxruntime as ort
from datasets import Dataset
from transformers import AutoTokenizer
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# === CONFIG ===
MODEL_PATH = "/aul/homes/melsh008/temp-storage-space-on-diamond/Quantizedmodels/phi3_exported/model.onnx"
TOKENIZER_PATH = "/aul/homes/melsh008/temp-storage-space-on-diamond/Quantizedmodels/phi3_exported"
DATA_PATH = "/aul/homes/melsh008/First_Case_Scenario/verification_augmented_with_license_id.jsonl"
BATCH_SIZE = 64
MAX_LENGTH = 128

# === HELPERS ===
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

def add_noise(text):
    text = text.lower().replace("o", "0").replace("i", "1").replace("e", "3")
    if len(text) > 50 and np.random.rand() < 0.3:
        text = text[:np.random.randint(30, 50)]
    words = text.split()
    if words:
        idx = np.random.randint(0, len(words))
        if len(words[idx]) > 3:
            c = np.random.randint(1, len(words[idx]) - 1)
            words[idx] = words[idx][:c] + "*" + words[idx][c+1:]
    return " ".join(words)

# === LOAD DATA ===
dataset = load_jsonl(DATA_PATH).select(range(5000))  # use only the first 5000 samples
labels = np.array(dataset["label"]).astype(int).flatten()
tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH, local_files_only=True)

# Pre-tokenize and convert to NumPy
encoded = tokenizer(dataset["text"], truncation=True, padding="max_length", max_length=MAX_LENGTH, return_tensors="pt")
encodings_np = {
    "input_ids": encoded["input_ids"].numpy(),
    "attention_mask": encoded["attention_mask"].numpy()
}

# === LOAD ONNX MODEL ===
providers = ["CUDAExecutionProvider"] if ort.get_device() == "GPU" else ["CPUExecutionProvider"]
session = ort.InferenceSession(MODEL_PATH, providers=providers)
input_names = [inp.name for inp in session.get_inputs()]
print("Model input names:", input_names)

# === INFERENCE ===
def predict_np(encodings_np, num_samples):
    preds = []
    for i in range(0, num_samples, BATCH_SIZE):
        inputs = {
            input_names[0]: encodings_np["input_ids"][i:i+BATCH_SIZE],
            input_names[1]: encodings_np["attention_mask"][i:i+BATCH_SIZE]
        }
        if len(input_names) > 2:
            dummy_shape = session.get_inputs()[2].shape
            dummy = np.zeros([d if isinstance(d, int) else 1 for d in dummy_shape], dtype=np.int64)
            inputs[input_names[2]] = dummy
        logits = session.run(None, inputs)[0]
        if logits.shape[1] == 1:
            batch_preds = (logits >= 0.5).astype(int).squeeze(-1)
        else:
            batch_preds = np.argmax(logits, axis=-1)
        preds.extend(batch_preds.tolist())
    return np.array(preds).astype(int).flatten()

# === EVALUATION ===
start = time.time()
all_preds = predict_np(encodings_np)
latency = (time.time() - start) / len(labels)

acc = accuracy_score(labels, all_preds)
prec, rec, f1, _ = precision_recall_fscore_support(labels, all_preds, average="binary", zero_division=0)
mem_used = torch.cuda.max_memory_allocated() / (1024 ** 2) if torch.cuda.is_available() else 0
model_size = os.path.getsize(MODEL_PATH) / (1024 ** 2)

# === ROBUSTNESS ===
noisy_texts = [add_noise(t) for t in dataset["text"]]
noisy_encoded = tokenizer(noisy_texts, truncation=True, padding="max_length", max_length=MAX_LENGTH, return_tensors="pt")
noisy_np = {
    "input_ids": noisy_encoded["input_ids"].numpy(),
    "attention_mask": noisy_encoded["attention_mask"].numpy()
}
noisy_preds = predict_np(noisy_np).astype(int).flatten()
robustness = accuracy_score(labels, noisy_preds)

# === RESULTS ===
print("\n================== Phi-3 ONNX Evaluation ==================")
print(f"Accuracy       : {acc*100:.2f}%")
print(f"Precision      : {prec*100:.2f}%")
print(f"Recall         : {rec*100:.2f}%")
print(f"F1 Score       : {f1*100:.2f}%")
print(f"Latency/sample : {latency*1000:.2f} ms")
print(f"Model Size     : {model_size:.2f} MB")
print(f"Peak Memory    : {mem_used:.2f} MB")
print(f"Robustness Acc : {robustness*100:.2f}%")
print("============================================================")
