import os
import json
import torch
import numpy as np
from datasets import Dataset
from torch.serialization import add_safe_globals
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix


from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer
)


# === Environment Configuration ===
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# === Check for GPU ===
if not torch.cuda.is_available():
    raise SystemError("\U0001f6ab GPU not available. Ensure CUDA is properly configured.")
else:
    print(f"\u2705 GPU: {torch.cuda.get_device_name(torch.cuda.current_device())}")

# === Label Mapping ===
def label_from_response(text):
    text = text.lower()
    return 1 if text.startswith("yes") else 0

# === Load JSONL Dataset ===
def load_jsonl(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            item = json.loads(line)
            prompt = f"Task: {item.get('task_type', 'triage_verification')}\nInput: {item['input']}"
            label = label_from_response(item["response"])
            data.append({"prompt": prompt, "label": int(label)})
    return data

raw_data = load_jsonl("verification_augmented_with_license_id.jsonl")
train_data, val_data = train_test_split(raw_data, test_size=0.2, random_state=42, stratify=[d['label'] for d in raw_data])

train_dataset = Dataset.from_list(train_data)
val_dataset = Dataset.from_list(val_data)

# === Load TinyLLaMA (no token required) ===
repo_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

tokenizer = AutoTokenizer.from_pretrained(repo_id)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForSequenceClassification.from_pretrained(repo_id, num_labels=2).to("cuda")
model.config.pad_token_id = tokenizer.pad_token_id

# === Tokenize ===
def tokenize(example):
    return tokenizer(example["prompt"], padding="max_length", truncation=True, max_length=128)

train_dataset = train_dataset.map(tokenize, batched=True)
val_dataset = val_dataset.map(tokenize, batched=True)

train_dataset = train_dataset.rename_column("label", "labels")
val_dataset = val_dataset.rename_column("label", "labels")

train_dataset.set_format("torch", columns=["input_ids", "attention_mask", "labels"])
val_dataset.set_format("torch", columns=["input_ids", "attention_mask", "labels"])

# === Metrics ===
def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)

    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average="weighted", zero_division=0)
    acc = accuracy_score(labels, preds)
    cm = confusion_matrix(labels, preds)

    # Optionally print confusion matrix during training
    print("\nConfusion Matrix:")
    print(cm)

    return {
        "accuracy": acc,
        "f1": f1,
        "precision": precision,
        "recall": recall,
        "confusion_matrix": cm.tolist()  # to make it JSON serializable
    }

# === Training Config ===
training_args = TrainingArguments(
    output_dir="./tinyllama_verification_finetuned",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=5,
    weight_decay=0.01,
    logging_dir="./tinyllama_logs",
    logging_steps=10,
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    report_to="none",
    warmup_steps=100,
    gradient_accumulation_steps=1,
    fp16=False,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)


add_safe_globals([
    np.core.multiarray._reconstruct,
    np.dtype,
    np.core.multiarray.scalar,
    np.ndarray,
    np.dtypes.UInt32DType  # ✅ Add this to support loading RNG state
])

print("\U0001f680 Training TinyLLaMA...")
trainer.train(resume_from_checkpoint=True)

print("\u2705 Training complete. Saving model...")
trainer.save_model(training_args.output_dir)
tokenizer.save_pretrained(training_args.output_dir)
