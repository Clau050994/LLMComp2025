#!/usr/bin/env python3
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

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
    default_data_collator,
    BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from evaluate import load
import logging

# ✅ Initialize Weights & Biases
wandb.init(
    project="mobilellama-summarization",
    name="mobilellama-qlora-run",
    config={
        "learning_rate": 2e-5,
        "epochs": 5,
        "batch_size": 4,
        "model": "JackFram/llama-68m",
        "quantization": "QLoRA-4bit"
    }
)

# ✅ Logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
np.random.seed(42)
torch.manual_seed(42)

def main():
    model_output_dir = "/disk/diamond-scratch/cvaro009/data/mobilellama_summarization"
    train_path = "/aul/homes/cvaro009/Desktop/LLMComp2025/data/processed/usecase4/generated/test_with_hs6.json"
    val_path = "/aul/homes/cvaro009/Desktop/LLMComp2025/data/processed/usecase4/generated/validation_with_hs6.json"
    os.makedirs(model_output_dir, exist_ok=True)

    train_df = pd.read_json(train_path, lines=True)
    val_df = pd.read_json(val_path, lines=True)
    if train_df.empty or val_df.empty:
        raise ValueError("🚨 One of your input datasets is empty!")

    train_dataset = Dataset.from_pandas(train_df)
    val_dataset = Dataset.from_pandas(val_df)

    model_id = "JackFram/llama-68m"
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True, legacy=False)
    tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.pad_token_id or tokenizer.eos_token_id

    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )

    base_model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=quant_config,
        device_map="auto",
        trust_remote_code=True
    )

    base_model = prepare_model_for_kbit_training(base_model)
    lora_config = LoraConfig(
        r=8,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(base_model, lora_config)
    model.config.pad_token_id = tokenizer.pad_token_id

    max_seq_length = 512
    def preprocess_function(examples):
        prompts = [
            f"### Instruction:\nSummarize the following customs declaration into plain language.\n\n### Input:\n{inp}\n\n### Response:\n"
            for inp in examples["input"]
        ]
        responses = examples["output"]
        input_ids, labels = [], []

        for prompt, response in zip(prompts, responses):
            full_text = prompt + response
            tokenized = tokenizer(
                full_text,
                max_length=max_seq_length,
                truncation=True,
                padding="max_length"
            )
            prompt_ids = tokenizer(prompt, truncation=True, max_length=max_seq_length)["input_ids"]
            prompt_len = len(prompt_ids)
            label_ids = [
                t if i >= prompt_len and t != tokenizer.pad_token_id else -100
                for i, t in enumerate(tokenized["input_ids"])
            ]
            input_ids.append(tokenized["input_ids"])
            labels.append(label_ids)

        return {
            "input_ids": input_ids,
            "attention_mask": [
                [1 if token != tokenizer.pad_token_id else 0 for token in seq] for seq in input_ids
            ],
            "labels": labels
        }

    train_tokenized = train_dataset.map(preprocess_function, batched=True, remove_columns=train_dataset.column_names)
    val_tokenized = val_dataset.map(preprocess_function, batched=True, remove_columns=val_dataset.column_names)

    training_args = TrainingArguments(
        output_dir=model_output_dir,
        run_name="mobilellama-qlora",
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=2,
        num_train_epochs=5,
        weight_decay=0.01,
        logging_steps=25,
        save_total_limit=2,
        load_best_model_at_end=True,
        report_to="wandb",
        warmup_steps=100,
        fp16=True,
        gradient_checkpointing=True,
        dataloader_num_workers=4,
        prediction_loss_only=True
    )

    rouge = load("rouge")
    def compute_metrics(eval_preds):
        preds, labels = eval_preds
        preds = preds[0] if isinstance(preds, tuple) else preds
        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
        decoded_preds = [p.strip() for p in decoded_preds]
        decoded_labels = [l.strip() for l in decoded_labels]
        result = rouge.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)
        return {
            "rouge1": round(result["rouge1"].mid.fmeasure * 100, 2),
            "rouge2": round(result["rouge2"].mid.fmeasure * 100, 2),
            "rougeL": round(result["rougeL"].mid.fmeasure * 100, 2),
        }

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_tokenized,
        eval_dataset=val_tokenized,
        compute_metrics=compute_metrics,
        data_collator=default_data_collator,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    )

    logger.info("🚀 Starting training...")
    trainer.train()
    logger.info("✅ Training complete!")
    trainer.save_model(model_output_dir)
    tokenizer.save_pretrained(model_output_dir)

if __name__ == "__main__":
    main()
