import os
import shutil
from transformers import AutoTokenizer
from optimum.onnxruntime import ORTModelForSequenceClassification

# Paths
model_id = "/disk/diamond-scratch/cvaro009/data/model/mobilellama_risk_assessment"
onnx_dir = "/disk/diamond-scratch/cvaro009/data/onnx_models/mobile_llama_onnx"
os.makedirs(onnx_dir, exist_ok=True)

# Export model to ONNX using Optimum
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = ORTModelForSequenceClassification.from_pretrained(model_id, export=True)
model.save_pretrained(onnx_dir)
tokenizer.save_pretrained(onnx_dir)
print(f"Exported ONNX model and tokenizer to {onnx_dir}")

# --- Copy tokenizer files (if any extra) ---
tokenizer_files = [
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.txt",
    "special_tokens_map.json",
    "merges.txt",
    "added_tokens.json"
]
for fname in tokenizer_files:
    src_file = os.path.join(model_id, fname)
    if os.path.exists(src_file):
        shutil.copy(src_file, onnx_dir)
        print(f"Copied {fname} to {onnx_dir}")