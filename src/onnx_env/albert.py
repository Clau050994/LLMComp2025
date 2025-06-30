import os
import torch
import shutil
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Paths
model_dir = "/disk/diamond-scratch/cvaro009/data/albert"
onnx_dir = "/disk/diamond-scratch/cvaro009/data/onnx_models/albert_onnx"
os.makedirs(onnx_dir, exist_ok=True)
onnx_path = os.path.join(onnx_dir, "model.onnx")

# Load model and tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_dir)
model = AutoModelForSequenceClassification.from_pretrained(model_dir)
model.eval()

# Prepare dummy input
dummy_text = "Traveler with French nationality, arriving from Brazil, last visited Morocco."
inputs = tokenizer(dummy_text, return_tensors="pt", max_length=128, padding="max_length", truncation=True)
input_names = ["input_ids", "attention_mask"]
inputs_list = [inputs[name] for name in input_names]

# Export to ONNX
torch.onnx.export(
    model,
    tuple(inputs_list),
    onnx_path,
    input_names=input_names,
    output_names=["logits"],
    dynamic_axes={name: {0: "batch_size"} for name in input_names},
    opset_version=17,
    do_constant_folding=True
)
print(f"Exported to {onnx_path}")

# --- Copy tokenizer files ---
tokenizer_files = [
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.txt",
    "special_tokens_map.json",
    "merges.txt",
    "added_tokens.json"
]
for fname in tokenizer_files:
    src_file = os.path.join(model_dir, fname)
    if os.path.exists(src_file):
        shutil.copy(src_file, onnx_dir)
        print(f"Copied {fname} to {onnx_dir}")