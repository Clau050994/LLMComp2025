import os
import torch
import wandb
import numpy as np
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
)
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# 🧪 Classification metrics
from evaluate import load

# ❌ ROUGE removed - not used for classification
eval_metric = load("accuracy")

# 🔧 Set environment
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ["WANDB_LOG_MODEL"] = "checkpoint"

# 🎯 Start W&B
wandb.init(
    project="distilbert_classification",
    name="distilbert-customs-hs6",
    config={
        "epochs": 3,
        "lr": 2e-5,
        "batch_size": 16,
        "model": "distilbert-base-uncased",
        "dataset": "usecase4_hs6_classification",
        "max_length": 512,
    },
)

# 📂 Load HS6-enriched datasets
train_path = "/aul/homes/cvaro009/Desktop/LLMComp2025/data/processed/usecase4/generated/train_with_hs6.json"
val_path = "/aul/homes/cvaro009/Desktop/LLMComp2025/data/processed/usecase4/generated/validation_with_hs6.json"
train_ds = load_dataset("json", data_files=train_path)["train"]
val_ds = load_dataset("json", data_files=val_path)["train"]

# ❌ Filter out records with no HS6 code
train_ds = train_ds.filter(lambda x: x["hs6_code"] is not None)
val_ds = val_ds.filter(lambda x: x["hs6_code"] is not None)

# 🔢 Encode HS6 codes as classification labels
label2id = {code: idx for idx, code in enumerate(sorted(set(train_ds["hs6_code"])))}
id2label = {v: k for k, v in label2id.items()}

known_codes = set(label2id.keys())
train_ds = train_ds.filter(lambda x: x["hs6_code"] in known_codes)
val_ds = val_ds.filter(lambda x: x["hs6_code"] in known_codes)

train_ds = train_ds.map(lambda x: {"label": label2id[x["hs6_code"]]})
val_ds = val_ds.map(lambda x: {"label": label2id[x["hs6_code"]]})

# 🤖 Load tokenizer and model
model_name = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=len(label2id),
    id2label=id2label,
    label2id=label2id,
).to("cuda" if torch.cuda.is_available() else "cpu")

# ✂️ Tokenization
def tokenize_function(examples):
    return tokenizer(
        examples["input"],
        padding="max_length",
        truncation=True,
        max_length=512,
    )

train_tokenized = train_ds.map(tokenize_function, batched=True)
val_tokenized = val_ds.map(tokenize_function, batched=True)
train_tokenized.set_format("torch", columns=["input_ids", "attention_mask", "label"])
val_tokenized.set_format("torch", columns=["input_ids", "attention_mask", "label"])

# 📏 Metrics for classification
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    preds = np.argmax(predictions, axis=1)
    acc = accuracy_score(labels, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average="weighted")
    return {"accuracy": acc, "precision": precision, "recall": recall, "f1": f1}

# 💾 Output paths
output_dir = "/disk/diamond-scratch/cvaro009/data/usecase4/distilbert_classification"
final_model_dir = "/disk/diamond-scratch/cvaro009/data/usecase4/distilbert"
os.makedirs(output_dir, exist_ok=True)
os.makedirs(final_model_dir, exist_ok=True)

# 🛠️ Training configuration
training_args = TrainingArguments(
    output_dir=output_dir,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=5,
    learning_rate=2e-5,
    warmup_steps=500,
    weight_decay=0.01,
    logging_steps=50,
    save_strategy="epoch",
    eval_strategy="epoch",
    save_total_limit=2,
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    greater_is_better=True,
    fp16=torch.cuda.is_available(),
    report_to="wandb",
    save_safetensors=True,
    remove_unused_columns=False,
)

# 🚀 Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_tokenized,
    eval_dataset=val_tokenized,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
)

# 🔁 Train
trainer.train()

# 📊 Evaluate
eval_results = trainer.evaluate()
print("\n✅ Final Evaluation:")
for k, v in eval_results.items():
    print(f"{k}: {v:.4f}")

# 💾 Save final model
trainer.save_model(final_model_dir)
tokenizer.save_pretrained(final_model_dir)

# 🏁 Finish W&B
wandb.finish()
