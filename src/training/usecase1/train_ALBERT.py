import os
import json
import torch
from datasets import Dataset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from sklearn.model_selection import train_test_split
from transformers import (
    AlbertTokenizerFast,
    AlbertForSequenceClassification,
    TrainingArguments,
    Trainer,
    TrainerCallback
)
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# === Environment Configuration ===
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# === Check GPU ===
if not torch.cuda.is_available():
    raise SystemError("\ud83d\udeab GPU not available. Ensure CUDA is properly configured.")
print(f"\u2705 GPU: {torch.cuda.get_device_name(torch.cuda.current_device())}")

# === Custom Callback for Logging Avg Train Loss ===
class LossLoggingCallback(TrainerCallback):
    def on_epoch_end(self, args, state, control, **kwargs):
        if state.log_history:
            for entry in reversed(state.log_history):
                if 'loss' in entry:
                    print(f"[LOG] Avg Train Loss (Epoch {int(state.epoch)}): {entry['loss']:.4f}")
                    break


# === Label Mapping ===
def label_from_response(text):
    return 1 if text.lower().startswith("yes") else 0

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

raw_data = load_jsonl("/aul/homes/melsh008/First_Case_Scenario/verification_augmented_with_license_id.jsonl")
train_data, val_data = train_test_split(raw_data, test_size=0.2, random_state=42, stratify=[d['label'] for d in raw_data])
train_dataset = Dataset.from_list(train_data)
val_dataset = Dataset.from_list(val_data)

# === Load Tokenizer and Model ===
tokenizer = AlbertTokenizerFast.from_pretrained("albert-base-v2")
model = AlbertForSequenceClassification.from_pretrained("albert-base-v2", num_labels=2).to("cuda")

# === Tokenize ===
def tokenize(example):
    return tokenizer(example["prompt"], padding="max_length", truncation=True, max_length=128)

train_dataset = train_dataset.map(tokenize, batched=True)
val_dataset = val_dataset.map(tokenize, batched=True)
train_dataset = train_dataset.rename_column("label", "labels")
val_dataset = val_dataset.rename_column("label", "labels")
train_dataset.set_format("torch", columns=["input_ids", "attention_mask", "labels"])
val_dataset.set_format("torch", columns=["input_ids", "attention_mask", "labels"])

# === Evaluation Metrics ===
def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average="weighted", zero_division=0)
    acc = accuracy_score(labels, preds)
    cm = confusion_matrix(labels, preds)

    cm_path = os.path.join(training_args.output_dir, "confusion_matrix.json")
    with open(cm_path, "w") as f:
        json.dump({"confusion_matrix": cm.tolist()}, f, indent=2)

    print("\n[RESULT] Confusion Matrix:\n", cm)

    # Plot and save confusion matrix image
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["No", "Yes"], yticklabels=["No", "Yes"])
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Confusion Matrix - ALBERT Verification")
    plt.tight_layout()
    plt.savefig(os.path.join(training_args.output_dir, "confusion_matrix.png"))
    plt.close()

    return {"accuracy": acc, "f1": f1, "precision": precision, "recall": recall}

# === Training Arguments ===
training_args = TrainingArguments(
    output_dir="./albert_verification_finetuned",
    evaluation_strategy="epoch",
    save_strategy="no",  
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=10,
    weight_decay=0.01,
    logging_dir="./albert_logs",
    logging_steps=10,
    load_best_model_at_end=False,
    report_to="none",
    warmup_steps=100,
    gradient_accumulation_steps=1,
    fp16=False,
)

# === Trainer Setup ===
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
    callbacks=[LossLoggingCallback()]
)

print("Starting ALBERT training from scratch...")
trainer.train()

# === Final Evaluation and Save ===
metrics = trainer.evaluate()
print("[RESULT] Final Evaluation Metrics:")
for key, value in metrics.items():
    if isinstance(value, float):
        print(f"{key}: {value:.4f}")

metrics_path = os.path.join(training_args.output_dir, "final_metrics.json")
with open(metrics_path, "w") as f:
    json.dump(metrics, f, indent=2)

print("[INFO] Re-saving model and tokenizer...")
trainer.save_model(training_args.output_dir)
tokenizer.save_pretrained(training_args.output_dir)
