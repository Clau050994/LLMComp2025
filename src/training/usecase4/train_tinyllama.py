#!/usr/bin/env python3
"""
QLoRA + 4-bit TinyLLaMA on BillSum summarization task using TRL's SFTTrainer.
"""

import os
import torch
import wandb
import numpy as np
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    EarlyStoppingCallback,
    BitsAndBytesConfig,
)
from trl import SFTTrainer, SFTConfig
from evaluate import load
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

# 🔧 Environment setup
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
os.environ["WANDB_LOG_MODEL"] = "checkpoint"
os.environ["FLASH_ATTENTION_2_ENABLED"] = "0"  # Disable Flash Attention

# 🌟 Init W&B
wandb.init(
    project="tinyllama_billsum",
    name="tinyllama-billsum-qlora-run",
    config={
        "epochs": 5,
        "lr": 2e-5,
        "batch_size": 4,
        "model": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        "dataset": "BillSum",
        "quantization": "QLoRA 4-bit"
    }
)

print("🚀 Starting TinyLLaMA QLoRA training...")

# 📂 Load dataset
train_path = "/aul/homes/cvaro009/Desktop/LLMComp2025/data/processed/billsum/us_train_data_final_OFFICIAL.jsonl"
val_path = "/aul/homes/cvaro009/Desktop/LLMComp2025/data/processed/billsum/us_test_data_final_OFFICIAL.jsonl"
train_ds = load_dataset("json", data_files=train_path)["train"]
val_ds = load_dataset("json", data_files=val_path)["train"]

# 🤖 Load tokenizer and quantized model
model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
tokenizer = AutoTokenizer.from_pretrained(model_name)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    quantization_config=bnb_config,
    trust_remote_code=True
)

# ✅ Prepare model for QLoRA fine-tuning
model = prepare_model_for_kbit_training(model)
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)
model = get_peft_model(model, lora_config)

model.gradient_checkpointing_enable()
model.config.use_cache = False

# 🔧 Preprocessing
max_total_tokens = 1536
max_input_tokens = 1024
max_output_tokens = 512

def preprocess(example):
    instruction = f"""### Instruction:
Summarize the following legislative document in plain English.

### Input:
{example['text']}

### Response:"""
    summary = example["summary"]

    input_enc = tokenizer(instruction, truncation=True, max_length=max_input_tokens, padding=False)
    output_enc = tokenizer(summary, truncation=True, max_length=max_output_tokens, padding=False)

    input_ids = input_enc["input_ids"] + output_enc["input_ids"]
    attention_mask = input_enc["attention_mask"] + output_enc["attention_mask"]
    labels = [-100] * len(input_enc["input_ids"]) + output_enc["input_ids"]

    return {
        "input_ids": input_ids[:max_total_tokens],
        "attention_mask": attention_mask[:max_total_tokens],
        "labels": labels[:max_total_tokens]
    }

train_ds = train_ds.map(preprocess, remove_columns=train_ds.column_names)
val_ds = val_ds.map(preprocess, remove_columns=val_ds.column_names)

# 🔢 Evaluation
rouge = load("rouge")
bertscore = load("bertscore")

def compute_metrics(eval_preds):
    preds, labels = eval_preds
    if isinstance(preds, tuple):
        preds = preds[0]
    if preds.ndim == 3:
        preds = preds.argmax(axis=-1)

    labels = [[(token if token != -100 else tokenizer.pad_token_id) for token in label] for label in labels]
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    decoded_preds = ["\n".join(p.strip().split(". ")) for p in decoded_preds]
    decoded_labels = ["\n".join(l.strip().split(". ")) for l in decoded_labels]

    rouge_result = rouge.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)
    bert_result = bertscore.compute(predictions=decoded_preds, references=decoded_labels, lang="en")

    return {
        "rouge1": round(rouge_result["rouge1"].mid.fmeasure * 100, 2),
        "rouge2": round(rouge_result["rouge2"].mid.fmeasure * 100, 2),
        "rougeL": round(rouge_result["rougeL"].mid.fmeasure * 100, 2),
        "bertscore_f1": round(np.mean(bert_result["f1"]) * 100, 2),
    }

# 📁 Output directories
output_dir = "/disk/diamond-scratch/cvaro009/data/usecase4/tinyllama_billsum"
final_model_dir = "/disk/diamond-scratch/cvaro009/data/usecase4/tinyllama"
os.makedirs(output_dir, exist_ok=True)
os.makedirs(final_model_dir, exist_ok=True)

# ⚙️ SFT Config
sft_config = SFTConfig(
    output_dir=output_dir,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    num_train_epochs=5,
    learning_rate=2e-5,
    max_seq_length=1536,
    logging_steps=50,
    save_strategy="epoch",
    save_total_limit=2,
    eval_strategy="epoch",
    eval_accumulation_steps=1,
    fp16=torch.cuda.is_available(),
    fp16_full_eval=False,
    report_to="wandb",
    dataset_text_field="input_ids",
    load_best_model_at_end=True,
    metric_for_best_model="rougeL",
    greater_is_better=True,
    save_safetensors=True,
    dataloader_pin_memory=False,
    remove_unused_columns=False
)

# 🏋️ Train
trainer = SFTTrainer(
    model=model,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    args=sft_config,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
)

trainer.train()

# 📂 Save model
trainer.save_model(final_model_dir)
tokenizer.save_pretrained(final_model_dir)

wandb.finish()
print("✅ QLoRA 4-bit training complete. Model saved.")
