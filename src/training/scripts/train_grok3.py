#!/usr/bin/env python3
"""
Grok-3 Mini-Fast training script (standard, no class weights, uses pre-trained weights).
"""

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

import pandas as pd
import torch
import numpy as np
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback
)
from sklearn.metrics import precision_recall_fscore_support
import random

# Set random seeds
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

# Output directories
model_output_dir = "/disk/diamond-scratch/cvaro009/data/grok3"
os.makedirs(model_output_dir, exist_ok=True)
logs_dir = os.path.join(model_output_dir, "logs")
os.makedirs(logs_dir, exist_ok=True)

print("Loading Grok-3 Mini-Fast model and tokenizer...")
model_name = "grok-3-mini-fast"  # Replace with the actual Hugging Face model name if different
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=3
)

print("Loading unified datasets...")
unified_dir = "/aul/homes/cvaro009/Desktop/LLMComp2025/data/processed/unified/"
train_df = pd.read_csv(os.path.join(unified_dir, "unified_train.csv"))
val_df = pd.read_csv(os.path.join(unified_dir, "unified_val.csv"))
test_df = pd.read_csv(os.path.join(unified_dir, "unified_test.csv"))

print(f"Loaded {len(train_df)} training examples")
print(f"Loaded {len(val_df)} validation examples")
print(f"Loaded {len(test_df)} test examples")
print(f"Training label distribution: {train_df['label'].value_counts().to_dict()}")
if "example_type" in train_df.columns:
    print(f"Example type distribution: {train_df['example_type'].value_counts().to_dict()}")

train_dataset = Dataset.from_pandas(train_df)
val_dataset = Dataset.from_pandas(val_df)
test_dataset = Dataset.from_pandas(test_df)

def tokenize(batch):
    return tokenizer(batch["input_text"], truncation=True, padding="max_length", max_length=128)

print("Tokenizing datasets...")
train_dataset = train_dataset.map(tokenize, batched=True)
val_dataset = val_dataset.map(tokenize, batched=True)
test_dataset = test_dataset.map(tokenize, batched=True)

train_dataset.set_format("torch", columns=["input_ids", "attention_mask", "label"])
val_dataset.set_format("torch", columns=["input_ids", "attention_mask", "label"])
test_dataset.set_format("torch", columns=["input_ids", "attention_mask", "label"])

def compute_metrics(pred):
    labels = pred.label_ids
    preds = torch.argmax(torch.tensor(pred.predictions), dim=1)
    _, _, f1, _ = precision_recall_fscore_support(
        labels, preds, average="weighted"
    )
    return {"f1": f1}

training_args = TrainingArguments(
    output_dir=model_output_dir,
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=32,
    num_train_epochs=8,
    weight_decay=0.01,
    logging_dir=logs_dir,
    logging_steps=100,
    dataloader_num_workers=1,
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    report_to="none",
    warmup_ratio=0.1,
    gradient_accumulation_steps=2,
    fp16=torch.cuda.is_available(),
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
)

print("Starting training...")
train_result = trainer.train()

print("\nTraining completed!")
print(f"Best validation F1: {trainer.state.best_metric}")
print(f"Best checkpoint: {trainer.state.best_model_checkpoint}")

print(f"Saving model to {model_output_dir}")
trainer.save_model(model_output_dir)
tokenizer.save_pretrained(model_output_dir)

print("\nFinal evaluation on test set:")
test_metrics = trainer.evaluate(test_dataset)
print(f"Test F1: {test_metrics['eval_f1']}")