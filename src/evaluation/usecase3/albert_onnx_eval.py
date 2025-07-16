import os
import pandas as pd
import time
import psutil 
import numpy as np
import onnxruntime as ort
import matplotlib.pyplot as plt
from transformers import AutoTokenizer
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay

# --- CONFIGURATION ---
onnx_model_path = "/disk/diamond-scratch/cvaro009/data/onnx_models/albert_onnx/model.onnx"
tokenizer_dir ="/disk/diamond-scratch/cvaro009/data/albert"
test_csv_path = "/aul/homes/cvaro009/Desktop/LLMComp2025/data/processed/unified/unified_test.csv"
results_dir = "/aul/homes/cvaro009/Desktop/LLMComp2025/onnx_results/albert_onnx"
os.makedirs(results_dir, exist_ok=True)

# --- LOAD TEST DATA ---
df = pd.read_csv(test_csv_path)
texts = df["input_text"].astype(str).tolist()
labels = df["label"].values

# --- TOKENIZE INPUTS ---
tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir)
encodings = tokenizer(
    texts,
    padding="max_length",
    truncation=True,
    max_length=128,
    return_tensors="np"
)
# Prepare ONNX inputs
session = ort.InferenceSession(onnx_model_path)
input_names = [i.name for i in session.get_inputs()]
test_inputs = {k: v for k, v in encodings.items() if k in input_names}
test_labels = labels

# --- INFERENCE & LATENCY ---
batch_size = 32
all_preds = []
latencies = []
process = psutil.Process(os.getpid())
mem_before = process.memory_info().rss

for i in range(0, len(texts), batch_size):
    batch = {k: v[i:i+batch_size] for k, v in test_inputs.items()}
    start = time.time()
    ort_outs = session.run(None, batch)
    latencies.append(time.time() - start)
    preds = np.argmax(ort_outs[0], axis=1)
    all_preds.extend(preds)

mem_after = process.memory_info().rss
all_preds = np.array(all_preds)


# --- METRICS ---
acc = accuracy_score(test_labels, all_preds)
cm = confusion_matrix(test_labels, all_preds)
avg_latency_ms = np.mean(latencies) * 1000
model_size_mb = os.path.getsize(onnx_model_path) / (1024 * 1024)
memory_usage_mb = (mem_after - mem_before) / (1024 * 1024)

# --- ROBUSTNESS (simple typos) ---
def add_typos(text, level=0.1):
    import random
    chars = list(text)
    n_typos = max(1, int(len(chars) * level))
    for _ in range(n_typos):
        idx = random.randint(0, len(chars)-1)
        chars[idx] = random.choice("abcdefghijklmnopqrstuvwxyz ")
    return "".join(chars)

noisy_texts = [add_typos(t, 0.1) for t in texts]
noisy_encodings = tokenizer(
    noisy_texts,
    padding="max_length",
    truncation=True,
    max_length=128,
    return_tensors="np"
)
noisy_inputs = {k: v for k, v in noisy_encodings.items() if k in input_names}
noisy_preds = []
for i in range(0, len(noisy_texts), batch_size):
    batch = {k: v[i:i+batch_size] for k, v in noisy_inputs.items()}
    ort_outs = session.run(None, batch)
    preds = np.argmax(ort_outs[0], axis=1)
    noisy_preds.extend(preds)
noisy_acc = accuracy_score(test_labels, noisy_preds)
robustness = noisy_acc / acc if acc > 0 else 0

# --- SAVE CONFUSION MATRIX ---
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(cmap=plt.cm.Blues)
plt.title("Confusion Matrix: Albert ONNX")
plt.savefig(os.path.join(results_dir, "confusion_matrix_albert_onnx.png"))
plt.close()

# --- SAVE METRICS ---
metrics = {
    "accuracy": float(acc),
    "avg_latency_ms": float(avg_latency_ms),
    "model_size_mb": float(model_size_mb),
    "memory_usage_mb": float(memory_usage_mb),
    "robustness": float(robustness),
    "confusion_matrix": cm.tolist()
}
with open(os.path.join(results_dir, "metrics.json"), "w") as f:
    import json
    json.dump(metrics, f, indent=2)

print(f"Albert ONNX Accuracy: {acc:.4f}")
print(f"Avg Latency (ms): {avg_latency_ms:.2f}")
print(f"Model Size (MB): {model_size_mb:.2f}")
print(f"Memory Usage (MB): {memory_usage_mb:.2f}")
print(f"Robustness (noisy/clean acc): {robustness:.2f}")
print(f"Results saved in {results_dir}")