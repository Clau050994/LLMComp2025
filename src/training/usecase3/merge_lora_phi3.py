#!/usr/bin/env python3

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, AutoConfig
from peft import PeftModel
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # Use GPU 1


# === CONFIG ===
trained_model_dir = "/disk/diamond-scratch/cvaro009/data/usecase3/phi3_risk_classification_qlora"
merged_model_dir = "/disk/diamond-scratch/cvaro009/data/usecase3/phi3_merged_fp16"
base_model_id = "microsoft/phi-3-mini-4k-instruct"

# === LOAD TOKENIZER ===
tokenizer = AutoTokenizer.from_pretrained(base_model_id)
tokenizer.save_pretrained(merged_model_dir)

# === LOAD CONFIG ===
config = AutoConfig.from_pretrained(base_model_id, num_labels=3)

# === LOAD BASE MODEL ON CPU ===
base_model = AutoModelForSequenceClassification.from_pretrained(
    base_model_id,
    config=config,
    torch_dtype=torch.float16,
    device_map={"": "cpu"}
)

# === LOAD LoRA + MERGE ON CPU ===
model = PeftModel.from_pretrained(base_model, trained_model_dir)
model = model.merge_and_unload()

# === SAVE MERGED MODEL ===
model.save_pretrained(merged_model_dir)
print(f"✅ Merged model saved to {merged_model_dir}")
