#!/usr/bin/env python3
"""
Train mBERT (bert-base-multilingual-cased) on a CSV dataset to predict language.
"""

import os
import torch
import time
import psutil
import random
from datasets import load_dataset, ClassLabel
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    pipeline,
)
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# === Environment Config ===
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # Change if needed

# === Load dataset ===
data_path = "/aul/homes/melsh008/temp-storage-space-on-diamond/second_case_scenario/jw300_clean_shuffled.csv"
dataset = load_dataset("csv", data_files=data_path)

# Inspect columns
print("Columns:", dataset["train"].column_names)

# === Encode labels ===
languages = sorted(set(dataset["train"]["language"]))
label_classes = ClassLabel(names=languages)

def encode_label(example):
    example["label"] = label_classes.str2int(example["language"])
    return example

dataset = dataset.map(encode_label)

# === Tokenization ===
tokenizer = AutoTokenizer.from_pretrained("bert-base-multilingual-cased")

def tokenize(batch):
    return tokenizer(
        batch["translation"],  # Train on translation column
        padding="max_length",
        truncation=True,
        max_length=512,
    )

tokenized = dataset.map(tokenize, batched=True)

# === Model ===
model = AutoModelForSequenceClassification.from_pretrained(
    "bert-base-multilingual-cased",
    num_labels=len(languages)
)

# === Metrics ===
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = logits.argmax(-1)
    acc = accuracy_score(labels, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average="macro")
    return {
        "accuracy": acc,
        "f1": f1,
        "precision": precision,
        "recall": recall,
    }

# === Training Arguments ===
training_args = TrainingArguments(
    output_dir="./mbert_output",
    evaluation_strategy="no",  # No validation split, train on all data
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=100,
)

# === Trainer ===
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized["train"],
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

# === Train ===
trainer.train()

# === Save Final Model ===
model_dir = "./mbert_final"
trainer.save_model(model_dir)

# === Extended Metrics ===
pipe = pipeline("text-classification", model=model, tokenizer=tokenizer, device=0 if torch.cuda.is_available() else -1)
sample_text = "Tämä on esimerkkilause kielen tunnistamiseen."
start_time = time.time()
_ = pipe(sample_text)
end_time = time.time()
latency = end_time - start_time

# Model Size
total_size = 0
for dirpath, dirnames, filenames in os.walk(model_dir):
    for f in filenames:
        fp = os.path.join(dirpath, f)
        total_size += os.path.getsize(fp)
model_size_mb = total_size / (1024 * 1024)

# Memory Usage
process = psutil.Process(os.getpid())
memory_mb = process.memory_info().rss / (1024 * 1024)

# Robustness
def add_noise(text):
    words = text.split()
    if len(words) > 3:
        i = random.randint(0, len(words) - 1)
        words[i] = "".join(random.sample(words[i], len(words[i])))
    return " ".join(words)

noisy_text = add_noise(sample_text)
clean_pred = pipe(sample_text)
noisy_pred = pipe(noisy_text)

# === Print Extended Metrics ===
print("\n===== Extended Evaluation Metrics =====")
print(f"Latency (single example): {latency:.4f} sec")
print(f"Model size: {model_size_mb:.2f} MB")
print(f"Memory usage (RSS): {memory_mb:.2f} MB")
print(f"Clean input prediction: {clean_pred}")
print(f"Noisy input prediction: {noisy_pred}")
