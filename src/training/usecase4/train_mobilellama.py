#!/usr/bin/env python3
"""
Fine-tune LLaMA-68M on BillSum summarization task using CausalLM and ROUGE evaluation.
"""

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import pandas as pd
import wandb
import torch
import numpy as np
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
    default_data_collator
)
from evaluate import load
import logging

# ✅ Initialize W&B
wandb.init(
    project="llama68m-summarization",
    name="llama68m-summarization-run",
    config={
        "learning_rate": 2e-5,
        "epochs": 5,
        "batch_size": 4,
        "model": "JackFram/llama-68m"
    }
)

# ✅ Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ✅ Reproducibility
np.random.seed(42)
torch.manual_seed(42)

def main():
    # ✅ Paths
    model_output_dir = "/disk/diamond-scratch/cvaro009/data/llama68m_summarization"
    train_path = "/aul/homes/cvaro009/Desktop/LLMComp2025/data/processed/billsum/us_train_data_final_OFFICIAL.jsonl"
    val_path = "/aul/homes/cvaro009/Desktop/LLMComp2025/data/processed/billsum/us_test_data_final_OFFICIAL.jsonl"
    os.makedirs(model_output_dir, exist_ok=True)

    # ✅ Load dataset
    train_df = pd.read_json(train_path, lines=True)
    val_df = pd.read_json(val_path, lines=True)
    train_dataset = Dataset.from_pandas(train_df)
    val_dataset = Dataset.from_pandas(val_df)

    # ✅ Model + tokenizer
    model_id = "JackFram/llama-68m"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
        use_cache=False
    )
    model.gradient_checkpointing_enable()
    model.config.pad_token_id = tokenizer.pad_token_id

    # ✅ Preprocessing
    max_input_length = 1024
    max_target_length = 256

    def preprocess_function(examples):
        inputs = [
            f"### Instruction:\nSummarize the following legislative document in plain English.\n\n### Input:\n{text}\n\n### Response:"
            for text in examples["text"]
        ]
        model_inputs = tokenizer(
            inputs,
            max_length=max_input_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt"
        )
        labels = tokenizer(
            examples["summary"],
            max_length=max_target_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt"
        )
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    train_tokenized = train_dataset.map(preprocess_function, batched=True, remove_columns=train_dataset.column_names)
    val_tokenized = val_dataset.map(preprocess_function, batched=True, remove_columns=val_dataset.column_names)

    # ✅ Collator
    data_collator = DataCollatorForSeq2Seq(
    tokenizer, model=model, padding=True, ignore_pad_token_for_loss=True
    )

    # ✅ Training args
    training_args = TrainingArguments(
        output_dir=model_output_dir,
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=8,
        gradient_accumulation_steps=2,
        num_train_epochs=5,
        weight_decay=0.01,
        logging_steps=25,
        save_total_limit=2,
        load_best_model_at_end=True,
        report_to="wandb",
        fp16=True,
        warmup_steps=100,
    )

    # ✅ Metrics
    rouge = load("rouge")
    def compute_metrics(eval_preds):
        preds, labels = eval_preds
        if isinstance(preds, tuple):
            preds = preds[0]
        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
        decoded_preds = [pred.strip() for pred in decoded_preds]
        decoded_labels = [label.strip() for label in decoded_labels]
        result = rouge.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)
        return {
            "rouge1": round(result["rouge1"].mid.fmeasure * 100, 2),
            "rouge2": round(result["rouge2"].mid.fmeasure * 100, 2),
            "rougeL": round(result["rougeL"].mid.fmeasure * 100, 2),
        }

    # ✅ Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_tokenized,
        eval_dataset=val_tokenized,
        compute_metrics=compute_metrics,
        data_collator=data_collator,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
    )

    logger.info("🚀 Starting training...")
    trainer.train()
    trainer.save_model(model_output_dir)
    tokenizer.save_pretrained(model_output_dir)
    logger.info("✅ Training complete!")

if __name__ == "__main__":
    main()
