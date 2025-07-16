#!/usr/bin/env python3
"""
Training script for ALBERT model on traveler risk assessment.
"""

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

import pandas as pd
import torch
import numpy as np
from datasets import Dataset
from transformers import (
    AlbertTokenizer,
    AlbertForSequenceClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback
)
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import logging
import wandb
# Initialize Weights & Biases
wandb.init(
    project="albert-risk-assessment",
    name="albert-run-maxlen512",
    config={
        "learning_rate": 3e-5,
        "epochs": 10,
        "batch_size": 16,
        "max_seq_length": 512,
        "model": "albert-base-v2"
    }
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set random seeds
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average="weighted", zero_division=0
    )
    acc = accuracy_score(labels, preds)
    return {"accuracy": acc, "f1": f1, "precision": precision, "recall": recall}

def main():
    # Load datasets
    data_dir = "/aul/homes/cvaro009/Desktop/LLMComp2025/data/processed/unified"
    train_df = pd.read_csv(os.path.join(data_dir, "unified_train.csv"))
    val_df = pd.read_csv(os.path.join(data_dir, "unified_val.csv"))
    test_df = pd.read_csv(os.path.join(data_dir, "unified_test.csv"))
    
    logger.info("Label distribution in training set:")
    logger.info(train_df['label'].value_counts().to_dict())
    
    train_df['label'] = train_df['label'].astype(int)
    val_df['label'] = val_df['label'].astype(int)
    test_df['label'] = test_df['label'].astype(int)
    
    train_dataset = Dataset.from_pandas(train_df)
    val_dataset = Dataset.from_pandas(val_df)
    test_dataset = Dataset.from_pandas(test_df)
    
    # Initialize tokenizer
    logger.info("Loading ALBERT tokenizer...")
    tokenizer = AlbertTokenizer.from_pretrained("albert-base-v2")
    
    def tokenize_function(examples):
        return tokenizer(
            examples["input_text"],
            padding="max_length",
            truncation=True,
            max_length=512,
        )
    
    logger.info("Tokenizing datasets...")
    train_tokenized = train_dataset.map(tokenize_function, batched=True)
    val_tokenized = val_dataset.map(tokenize_function, batched=True)
    test_tokenized = test_dataset.map(tokenize_function, batched=True)
    
    train_tokenized.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
    val_tokenized.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
    test_tokenized.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
    
    model_output_dir = "/disk/diamond-scratch/cvaro009/data/albert"
    os.makedirs(model_output_dir, exist_ok=True)
    
    logger.info("Loading ALBERT model...")
    model = AlbertForSequenceClassification.from_pretrained(
        "albert-base-v2",
        num_labels=3
    )
    
    logger.info(f"Model initialized with {model.config.num_labels} output classes")
    
    training_args = TrainingArguments(
        output_dir=model_output_dir,
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=3e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=32,
        num_train_epochs=10,
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        warmup_ratio=0.1,
        logging_steps=100,
        report_to="wandb",
        max_grad_norm=1.0
    )
    
    logger.info("Initializing trainer...")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_tokenized,
        eval_dataset=val_tokenized,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
    )
    
    logger.info("Starting ALBERT training...")
    trainer.train()
    
    logger.info("Evaluating on test set...")
    test_results = trainer.evaluate(test_tokenized)
    
    logger.info("\nALBERT Results:")
    logger.info(f"Best validation metric: {trainer.state.best_metric:.4f}")
    logger.info(f"Test accuracy: {test_results['eval_accuracy']:.4f}")
    logger.info(f"Test F1: {test_results['eval_f1']:.4f}")
    
    trainer.save_model(model_output_dir)
    tokenizer.save_pretrained(model_output_dir)
    logger.info(f"Model and tokenizer saved to {model_output_dir}")

if __name__ == "__main__":
    main()