#!/usr/bin/env python3
"""
MobileBERT training and detailed evaluation script for traveler risk assessment.
"""

import os
import time
import psutil
import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from transformers import (
    MobileBertTokenizer,
    MobileBertForSequenceClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback
)
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    roc_auc_score,
    f1_score
)
import logging
import wandb
import seaborn as sns
import matplotlib.pyplot as plt

# ===== Environment Setup =====
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# ===== Logging =====
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ===== Weights & Biases =====
wandb.init(
    project="mobilebert-risk-assessment",
    name="mobilebert-training-eval",
    config={
        "learning_rate": 2e-5,
        "epochs": 10,
        "batch_size": 8,
        "max_seq_length": 512,
        "model": "google/mobilebert-uncased"
    }
)

# ===== Compute Metrics =====
def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average="weighted", zero_division=0
    )
    acc = accuracy_score(labels, preds)
    f1_per_class = f1_score(labels, preds, average=None, labels=[0, 1, 2])
    return {
        "accuracy": acc,
        "f1_weighted": f1,
        "precision_weighted": precision,
        "recall_weighted": recall,
        "f1_low_risk": f1_per_class[0],
        "f1_medium_risk": f1_per_class[1],
        "f1_high_risk": f1_per_class[2]
    }

# ===== Plot Confusion Matrix =====
def plot_confusion_matrix(cm, save_path=None):
    plt.figure(figsize=(8,6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Low", "Medium", "High"],
        yticklabels=["Low", "Medium", "High"]
    )
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Confusion Matrix")
    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
        logger.info(f"Confusion matrix saved to {save_path}")
    plt.close()

# ===== Main =====
def main():
    # ===== Paths =====
    data_dir = "/aul/homes/cvaro009/Desktop/LLMComp2025/data/processed/unified"
    output_dir = "/disk/diamond-scratch/cvaro009/data/mobilebert"
    os.makedirs(output_dir, exist_ok=True)

    # ===== Load Data =====
    train_df = pd.read_csv(os.path.join(data_dir, "unified_train.csv"))
    val_df = pd.read_csv(os.path.join(data_dir, "unified_val.csv"))
    test_df = pd.read_csv(os.path.join(data_dir, "unified_test.csv"))

    logger.info(f"Train size: {len(train_df)}, Val size: {len(val_df)}, Test size: {len(test_df)}")
    logger.info(f"Label distribution - Train: {train_df['label'].value_counts().sort_index().tolist()}")

    # NaN checks
    for df, name in zip([train_df, val_df, test_df], ["train", "validation", "test"]):
        assert not df["input_text"].isna().any(), f"Found NaN in {name} data"

    # Ensure labels are integers
    for df in [train_df, val_df, test_df]:
        df["label"] = df["label"].astype(int)

    # ===== Tokenizer =====
    logger.info("Loading MobileBERT tokenizer...")
    tokenizer = MobileBertTokenizer.from_pretrained("google/mobilebert-uncased")

    def tokenize_function(examples):
        return tokenizer(
            examples["input_text"],
            padding="max_length",
            truncation=True,
            max_length=512
        )

    train_dataset = Dataset.from_pandas(train_df).map(tokenize_function, batched=True)
    val_dataset = Dataset.from_pandas(val_df).map(tokenize_function, batched=True)
    test_dataset = Dataset.from_pandas(test_df).map(tokenize_function, batched=True)

    train_dataset.set_format("torch", columns=["input_ids", "attention_mask", "label"])
    val_dataset.set_format("torch", columns=["input_ids", "attention_mask", "label"])
    test_dataset.set_format("torch", columns=["input_ids", "attention_mask", "label"])

    # ===== Load Model =====
    model = MobileBertForSequenceClassification.from_pretrained(
        "google/mobilebert-uncased",
        num_labels=3
    )

    # ===== Training Arguments =====
    training_args = TrainingArguments(
        output_dir=output_dir,
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=16,
        num_train_epochs=10,
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model="f1_weighted",
        warmup_ratio=0.1,
        logging_steps=50,
        report_to="wandb",
        max_grad_norm=1.0,
        dataloader_pin_memory=False,
        save_total_limit=2,
    )

    # ===== Trainer =====
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
    )

    # ===== Debug Batch =====
    logger.info("Sample batch debugging:")
    sample_batch = next(iter(trainer.get_train_dataloader()))
    logger.info(f"Input IDs shape: {sample_batch['input_ids'].shape}")
    logger.info(f"Labels unique: {torch.unique(sample_batch['labels'])}")

    # ===== Sanity Check Forward Pass =====
    logger.info("Testing model with small batch...")
    model.train()
    small_batch = {
        'input_ids': sample_batch['input_ids'][:2],
        'attention_mask': sample_batch['attention_mask'][:2],
        'labels': sample_batch['labels'][:2]
    }
    outputs = model(**small_batch)
    logger.info(f"Test loss: {outputs.loss}")
    logger.info(f"Test logits shape: {outputs.logits.shape}")

    if torch.isnan(outputs.loss):
        logger.error("NaN loss detected in test batch!")
        return

    # ===== Train =====
    logger.info("Training started...")
    trainer.train()
    logger.info("Training completed.")

    # ===== Validation Evaluation =====
    logger.info("Evaluating on validation set...")
    val_results = trainer.evaluate(val_dataset)
    logger.info(f"Validation results: {val_results}")

    # ===== Test Evaluation =====
    logger.info("Evaluating on test set...")
    start_time = time.time()
    predictions = trainer.predict(test_dataset)
    elapsed = time.time() - start_time

    labels = predictions.label_ids
    preds = predictions.predictions.argmax(-1)

    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average="weighted", zero_division=0
    )
    acc = accuracy_score(labels, preds)
    cm = confusion_matrix(labels, preds, labels=[0, 1, 2])

    try:
        probs = torch.softmax(torch.tensor(predictions.predictions), dim=1).numpy()
        roc_auc = roc_auc_score(labels, probs, multi_class="ovr")
    except Exception as e:
        logger.warning(f"Could not compute ROC AUC: {e}")
        roc_auc = 0

    logger.info("\n===== Test Metrics =====")
    logger.info(f"Accuracy: {acc:.4f}")
    logger.info(f"F1 (weighted): {f1:.4f}")
    logger.info(f"Precision: {precision:.4f}")
    logger.info(f"Recall: {recall:.4f}")
    logger.info(f"ROC AUC: {roc_auc:.4f}")
    logger.info(f"Eval time: {elapsed:.2f}s")

    # ===== Confusion Matrix =====
    plot_confusion_matrix(cm, save_path=os.path.join(output_dir, "confusion_matrix.png"))

    # ===== Save Model =====
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    logger.info("Model and tokenizer saved.")

if __name__ == "__main__":
    main()
