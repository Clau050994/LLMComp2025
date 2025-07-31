import os
import json
import torch
from datasets import load_dataset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    TrainerCallback,
)
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import time

# === Environment Configuration ===
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# === Check GPU ===
if not torch.cuda.is_available():
    raise SystemError("❌ GPU not available. Ensure CUDA is properly configured.")
print(f"✅ GPU: {torch.cuda.get_device_name(torch.cuda.current_device())}")

# === Custom Callback for Logging Avg Train Loss ===
class LossLoggingCallback(TrainerCallback):
    def on_epoch_end(self, args, state, control, **kwargs):
        if state.log_history:
            for entry in reversed(state.log_history):
                if "loss" in entry:
                    print(f"[LOG] Avg Train Loss (Epoch {int(state.epoch)}): {entry['loss']:.4f}")
                    break

# === Load Dataset ===
dataset = load_dataset(
    "csv",
    data_files="/aul/homes/melsh008/temp-storage-space-on-diamond/second_case_scenario/jw300_multi_language_large.csv",
    split="train"
)

# === Encode Labels ===
label_list = sorted(set(dataset["language"]))
label2id = {label: idx for idx, label in enumerate(label_list)}
id2label = {idx: label for label, idx in label2id.items()}
print(f"✅ Loaded {len(label_list)} languages.")

dataset = dataset.map(lambda x: {"label": label2id[x["language"]]})

# === Tokenizer and Model ===
tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base")
model = AutoModelForSequenceClassification.from_pretrained(
    "xlm-roberta-base",
    num_labels=len(label_list),
    id2label=id2label,
    label2id=label2id,
).to("cuda")

# === Tokenize ===
def tokenize(batch):
    texts = [str(t) if t is not None else "" for t in batch["translation"]]
    return tokenizer(
        texts,
        padding="max_length",
        truncation=True,
        max_length=512
    )

dataset = dataset.map(tokenize, batched=True)
dataset = dataset.rename_column("label", "labels")
dataset.set_format("torch", columns=["input_ids", "attention_mask", "labels"])

# === Split Dataset ===
dataset = dataset.train_test_split(test_size=0.2, seed=42)
train_dataset = dataset["train"]
val_dataset = dataset["test"]

# === Evaluation Metrics ===
def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)

    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average="weighted", zero_division=0
    )
    acc = accuracy_score(labels, preds)
    cm = confusion_matrix(labels, preds)

    # Latency measurement (single sample)
    start_time = time.time()
    _ = model(
        input_ids=val_dataset[0]["input_ids"].unsqueeze(0).to("cuda"),
        attention_mask=val_dataset[0]["attention_mask"].unsqueeze(0).to("cuda")
    )
    latency = (time.time() - start_time) * 1000  # ms

    # Model size
    model_path = os.path.join(training_args.output_dir, "pytorch_model.bin")
    if os.path.exists(model_path):
        model_size_mb = os.path.getsize(model_path) / (1024 * 1024)
    else:
        model_size_mb = -1

    # Peak GPU memory usage
    if torch.cuda.is_available():
        gpu_mem_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)
    else:
        gpu_mem_mb = -1

    # Multilingual support
    multilingual_languages = len(label_list)

    # Robustness placeholder
    robustness_score = "N/A - Evaluate separately with noisy/adversarial data"

    # Save confusion matrix
    cm_path = os.path.join(training_args.output_dir, "confusion_matrix.json")
    with open(cm_path, "w") as f:
        json.dump({"confusion_matrix": cm.tolist()}, f, indent=2)

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=False, cmap="Blues")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Confusion Matrix - XLM-RoBERTa Language Classification")
    plt.tight_layout()
    plt.savefig(os.path.join(training_args.output_dir, "confusion_matrix.png"))
    plt.close()

    return {
        "accuracy": acc,
        "f1": f1,
        "precision": precision,
        "recall": recall,
        "latency_ms": latency,
        "model_size_mb": model_size_mb,
        "peak_gpu_memory_mb": gpu_mem_mb,
        "multilingual_languages": multilingual_languages,
        "robustness_score": robustness_score,
    }

# === Training Arguments ===
training_args = TrainingArguments(
    output_dir="/aul/homes/melsh008/temp-storage-space-on-diamond/second_case_scenario/xlmr_finetuned",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=3e-5,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    num_train_epochs=8,
    weight_decay=0.01,
    logging_dir="/aul/homes/melsh008/temp-storage-space-on-diamond/second_case_scenario/xlmr_finetuned/logs",
    logging_steps=10,
    load_best_model_at_end=True,
    report_to="none",
    warmup_steps=500,
    gradient_accumulation_steps=8,
    fp16=True,
)


# === Trainer ===
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
    callbacks=[LossLoggingCallback()],
)

checkpoint_path = "/aul/homes/melsh008/temp-storage-space-on-diamond/second_case_scenario/xlmr_finetuned/checkpoint-84000"
print(f"✅ Resuming XLM-RoBERTa training from {checkpoint_path}...")
trainer.train(resume_from_checkpoint=checkpoint_path)


# === Final Evaluation ===
metrics = trainer.evaluate()
print("[RESULT] Final Evaluation Metrics:")
for key, value in metrics.items():
    if isinstance(value, float):
        print(f"{key}: {value:.4f}")
    else:
        print(f"{key}: {value}")

metrics_path = os.path.join(training_args.output_dir, "final_metrics.json")
with open(metrics_path, "w") as f:
    json.dump(metrics, f, indent=2)

print("[INFO] Re-saving model and tokenizer...")
trainer.save_model(training_args.output_dir)
tokenizer.save_pretrained(training_args.output_dir)

print("✅ Training complete.")
