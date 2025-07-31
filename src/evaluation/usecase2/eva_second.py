import os
import json
import torch
import numpy as np
import pandas as pd
from datasets import Dataset, load_dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix
)
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer
)

# === Config ===
model_dir = "/aul/homes/melsh008/temp-storage-space-on-diamond/second_case_scenario/distilbert_final"
eval_data_path = "/aul/homes/melsh008/temp-storage-space-on-diamond/second_case_scenario/jw300_clean_shuffled.csv"
text_column = "english"
label_column = "language"

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# === Load model and tokenizer ===
tokenizer = AutoTokenizer.from_pretrained(model_dir)
model = AutoModelForSequenceClassification.from_pretrained(model_dir)

# === Load CSV as pandas, then stratify ===
print("✅ Loading and splitting dataset...")
df = pd.read_csv(eval_data_path)

# Drop missing rows if any
df = df.dropna(subset=[text_column, label_column])

# Perform stratified sampling (20% of the full dataset)
_, eval_df = train_test_split(
    df,
    test_size=0.2,
    stratify=df[label_column],
    random_state=42
)

# Convert to Hugging Face Dataset
dataset = Dataset.from_pandas(eval_df)

# === Encode labels ===
label_classes = sorted(set(dataset[label_column]))
label2id = {label: idx for idx, label in enumerate(label_classes)}

def encode_label(example):
    example["labels"] = label2id[example[label_column]]
    return example

dataset = dataset.map(encode_label)

# === Tokenize ===
def tokenize(batch):
    return tokenizer(batch[text_column], truncation=True, padding="max_length", max_length=128)

dataset = dataset.map(tokenize, batched=True)
dataset.set_format("torch", columns=["input_ids", "attention_mask", "labels"])

# === Evaluate ===
trainer = Trainer(model=model)

print("✅ Running classification evaluation...")
predictions = trainer.predict(dataset)
pred_labels = np.argmax(predictions.predictions, axis=1)
true_labels = predictions.label_ids

# === Metrics ===
acc = accuracy_score(true_labels, pred_labels)
precision, recall, f1, _ = precision_recall_fscore_support(
    true_labels, pred_labels, average="weighted", zero_division=0
)
cm = confusion_matrix(true_labels, pred_labels)

# === Save results ===
results = {
    "classification": {
        "accuracy": round(acc, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "confusion_matrix": cm.tolist()
    }
}

output_path = "/aul/homes/melsh008/temp-storage-space-on-diamond/second_case_scenario/eval_metrics_classification_only.json"
with open(output_path, "w") as f:
    json.dump(results, f, indent=2)

print("\n✅ Classification evaluation metrics saved to:", output_path)
