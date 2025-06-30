#!/usr/bin/env python3
"""
Training script for TinyLLaMA (TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T)
on the traveler risk classification task, with PyTorch 2.6+ compatibility.
"""

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

# PyTorch 2.6+ safe globals fix
import torch
from torch.serialization import add_safe_globals
import numpy as np
import random

add_safe_globals([
    np.ndarray,
    np._core.multiarray._reconstruct,
    np.dtype,
    np.core.multiarray,
    np.array,
    np._globals
])

import pandas as pd
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    LlamaForSequenceClassification,
    LlamaConfig,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
    DataCollatorWithPadding
)
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import logging
import glob

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

def find_latest_complete_checkpoint(model_dir):
    """Find the latest complete checkpoint (with all necessary files)."""
    checkpoints = glob.glob(os.path.join(model_dir, "checkpoint-*"))
    if not checkpoints:
        return None
    valid_checkpoints = []
    for checkpoint in checkpoints:
        if os.path.exists(os.path.join(checkpoint, "optimizer.pt")):
            checkpoint_num = int(checkpoint.split("-")[-1])
            valid_checkpoints.append((checkpoint_num, checkpoint))
    if valid_checkpoints:
        valid_checkpoints.sort(reverse=True)
        logger.info(f"Found valid checkpoints: {[num for num, _ in valid_checkpoints]}")
        _, latest_checkpoint = valid_checkpoints[0]
        return latest_checkpoint
    return None

def custom_load_rng_state(checkpoint_path):
    """Load RNG state with PyTorch 2.6+ compatibility."""
    rng_file = os.path.join(checkpoint_path, "rng_state.pth")
    if not os.path.isfile(rng_file):
        logger.info(f"No RNG state file found at {rng_file}")
        return False
    try:
        logger.info(f"Loading RNG state from {rng_file}")
        checkpoint_rng_state = torch.load(rng_file, weights_only=False)
        random.setstate(checkpoint_rng_state["python"])
        np.random.set_state(checkpoint_rng_state["numpy"])
        torch.random.set_rng_state(checkpoint_rng_state["cpu"])
        if torch.cuda.is_available():
            for i, device_rng in enumerate(checkpoint_rng_state["cuda"]):
                if i < torch.cuda.device_count():
                    torch.cuda.set_rng_state(device_rng, i)
        logger.info("RNG state loaded successfully")
        return True
    except Exception as e:
        logger.warning(f"Failed to load RNG state: {e}")
        return False

def main():
    # Load datasets
    data_dir = "/aul/homes/cvaro009/Desktop/LLMComp2025/data/processed/unified"
    train_df = pd.read_csv(os.path.join(data_dir, "unified_train.csv"))
    val_df = pd.read_csv(os.path.join(data_dir, "unified_val.csv"))
    test_df = pd.read_csv(os.path.join(data_dir, "unified_test.csv"))

    logger.info("Label distribution in training set:")
    logger.info(train_df['label'].value_counts().to_dict())

    # Convert to datasets
    train_dataset = Dataset.from_pandas(train_df)
    val_dataset = Dataset.from_pandas(val_df)
    test_dataset = Dataset.from_pandas(test_df)

    # Load tokenizer
    logger.info("Loading TinyLLaMA tokenizer...")
    model_id = "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T"
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    # Set padding token for LLaMA models
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # Tokenize function
    def tokenize_function(examples):
        return tokenizer(
            examples["input_text"],
            truncation=True,
            max_length=128,
            padding=False,  # We'll use a data collator for dynamic padding
        )

    # Tokenize datasets
    logger.info("Tokenizing datasets...")
    train_tokenized = train_dataset.map(tokenize_function, batched=True)
    val_tokenized = val_dataset.map(tokenize_function, batched=True)
    test_tokenized = test_dataset.map(tokenize_function, batched=True)

    # Format for PyTorch
    train_tokenized.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
    val_tokenized.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
    test_tokenized.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])

    # Model directory (FIXED PATH)
    model_output_dir = "/disk/diamond-scratch/cvaro009/data/tinyllama"
    os.makedirs(model_output_dir, exist_ok=True)

    # Find latest checkpoint to resume from
    resume_from_checkpoint = find_latest_complete_checkpoint(model_output_dir)

    if resume_from_checkpoint:
        logger.info(f"Will resume training from checkpoint: {resume_from_checkpoint}")
        try:
            model_path = os.path.join(resume_from_checkpoint, "model.safetensors")
            if os.path.exists(model_path):
                logger.info(f"Checkpoint has model.safetensors file")
                step = int(os.path.basename(resume_from_checkpoint).split("-")[-1])
                logger.info(f"Resuming from step {step}")
            else:
                logger.warning("Checkpoint missing model file, starting from scratch")
                resume_from_checkpoint = None
        except Exception as e:
            logger.warning(f"Error checking checkpoint: {e}")
    else:
        logger.info("No valid checkpoint found. Starting training from scratch.")

    # Load model - configure LlamaForSequenceClassification
    logger.info("Loading TinyLLaMA model...")
    config = LlamaConfig.from_pretrained(model_id)
    config.num_labels = 3
    config.pad_token_id = tokenizer.pad_token_id

    model = LlamaForSequenceClassification.from_pretrained(
        model_id,
        config=config,
        torch_dtype=torch.float32
    )
    model.config.pad_token_id = tokenizer.pad_token_id

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    training_args = TrainingArguments(
        output_dir=model_output_dir,
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=4,
        gradient_accumulation_steps=8,
        num_train_epochs=8,
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        warmup_ratio=0.1,
        logging_steps=50,
        report_to="none",
        fp16=False,
        bf16=False,
        max_grad_norm=1.0,
        save_total_limit=2,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_tokenized,
        eval_dataset=val_tokenized,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
        data_collator=data_collator,
    )

    # Patch the _load_rng_state method to avoid PyTorch 2.6+ issues
    original_load_rng_state = trainer._load_rng_state
    def dummy_load_rng_state(*args, **kwargs):
        logger.info("Skipping RNG state loading")
        return
    trainer._load_rng_state = dummy_load_rng_state

    # Try loading RNG state separately (if needed)
    if resume_from_checkpoint:
        try:
            custom_load_rng_state(resume_from_checkpoint)
        except Exception as e:
            logger.warning(f"Failed to load RNG state: {e}")
            logger.warning("Continuing without RNG state")

    logger.info("Starting TinyLLaMA training...")
    try:
        trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    finally:
        trainer._load_rng_state = original_load_rng_state

    logger.info("Evaluating on test set...")
    test_results = trainer.evaluate(test_tokenized)

    logger.info("\nTinyLLaMA Results:")
    logger.info(f"Best validation metric: {trainer.state.best_metric:.4f}")
    logger.info(f"Test accuracy: {test_results['eval_accuracy']:.4f}")
    logger.info(f"Test F1: {test_results['eval_f1']:.4f}")

    trainer.save_model(model_output_dir)
    tokenizer.save_pretrained(model_output_dir)
    logger.info(f"Model and tokenizer saved to {model_output_dir}")

if __name__ == "__main__":
    main()