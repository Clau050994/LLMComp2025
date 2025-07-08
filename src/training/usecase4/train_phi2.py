#!/usr/bin/env python3
"""
Training script for Phi-2 (microsoft/phi-2) on traveler risk classification.
LoRA + 4-bit quantization (QLoRA-style).
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
    EarlyStoppingCallback,
    DataCollatorWithPadding,
    AutoConfig,
    BitsAndBytesConfig
)
from peft import (
    prepare_model_for_kbit_training,
    LoraConfig,
    get_peft_model,
    TaskType
)
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Set seeds
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
    # Paths
    model_output_dir = "/disk/diamond-scratch/cvaro009/data/phi2_risk_classification_qlora"
    train_path = "/aul/homes/cvaro009/Desktop/LLMComp2025/data/processed/unified/unified_train.csv"
    val_path = "/aul/homes/cvaro009/Desktop/LLMComp2025/data/processed/unified/unified_val.csv"
    test_path = "/aul/homes/cvaro009/Desktop/LLMComp2025/data/processed/unified/unified_test.csv"

    os.makedirs(model_output_dir, exist_ok=True)

    logger.info("Loading datasets...")
    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)
    test_df = pd.read_csv(test_path)

    logger.info(f"Training examples: {len(train_df)}")
    logger.info(f"Validation examples: {len(val_df)}")
    logger.info(f"Test examples: {len(test_df)}")
    logger.info("Label distribution in training set:")
    logger.info(train_df['label'].value_counts().to_dict())

    assert set(train_df["label"].unique()) <= {0, 1, 2}, "Unexpected labels!"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    train_dataset = Dataset.from_pandas(train_df)
    val_dataset = Dataset.from_pandas(val_df)
    test_dataset = Dataset.from_pandas(test_df)

    model_id = "microsoft/phi-2"
    logger.info(f"Preparing to train {model_id} with LoRA + 4-bit quantization")

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    def tokenize_function(examples):
        return tokenizer(
            examples["input_text"],
            truncation=True,
            max_length=1024,  # Shorter context for speed
        )

    logger.info("Tokenizing datasets...")
    train_tokenized = train_dataset.map(tokenize_function, batched=True)
    val_tokenized = val_dataset.map(tokenize_function, batched=True)
    test_tokenized = test_dataset.map(tokenize_function, batched=True)

    train_tokenized.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
    val_tokenized.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
    test_tokenized.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])

    logger.info("Loading model with 4-bit quantization...")
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    config = AutoConfig.from_pretrained(
        model_id,
        num_labels=3,
        pad_token_id=tokenizer.pad_token_id
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        model_id,
        config=config,
        quantization_config=quantization_config,
        device_map="auto"
    )

    model = prepare_model_for_kbit_training(model)
    model.gradient_checkpointing_enable()

    logger.info("Applying LoRA adapters...")
    lora_config = LoraConfig(
        r=8,                 # Rank
        lora_alpha=16,       # Alpha
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.SEQ_CLS
    )
    model = get_peft_model(model, lora_config)

    logger.info("LoRA applied. Model ready.")
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Total params: {total_params:,}")
    logger.info(f"Trainable params: {trainable_params:,}")

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    training_args = TrainingArguments(
        output_dir=model_output_dir,
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=2e-4,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=16,
        gradient_accumulation_steps=2,
        num_train_epochs=5,
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        logging_steps=50,
        report_to="none",
        fp16=False,  # Not needed; bfloat16 in 4-bit quantization
        bf16=True,   # Enable bfloat16 if available
        max_grad_norm=1.0
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

    logger.info("Starting training...")
    trainer.train()

    logger.info("Evaluating on test set...")
    test_results = trainer.evaluate(test_tokenized)

    logger.info("\nPhi-2 QLoRA Classification Results:")
    logger.info(f"Best validation metric: {trainer.state.best_metric:.4f}")
    logger.info(f"Test accuracy: {test_results['eval_accuracy']:.4f}")
    logger.info(f"Test F1: {test_results['eval_f1']:.4f}")

    trainer.save_model(model_output_dir)
    tokenizer.save_pretrained(model_output_dir)
    logger.info(f"Model and tokenizer saved to {model_output_dir}")

if __name__ == "__main__":
    main()
    logger.info("Training script completed successfully.")
