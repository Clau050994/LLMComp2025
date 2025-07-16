#!/usr/bin/env python3
"""
Training script for MobileLLaMA (JackFram/llama-68m) on the traveler risk classification task.
This script trains a lightweight LLaMA model for sequence classification.
"""

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["CUDA_VISIBLE_DEVICES"] = "2"  # Adjust as needed

import pandas as pd
import wandb
import torch
import numpy as np
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
from pathlib import Path

# Initialize Weights & Biases
wandb.init(
    project="mobilellama-risk-assessment",
    name="mobilellama-training-eval",
    config={
        "learning_rate": 2e-5,
        "epochs": 10,
        "batch_size": 16,
        "max_seq_length": 1024,
        "model": "JackFram/llama-68m"
    }
)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Set random seeds
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

def compute_metrics(pred):
    """Compute metrics for evaluation"""
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average="weighted", zero_division=0
    )
    acc = accuracy_score(labels, preds)
    return {"accuracy": acc, "f1": f1, "precision": precision, "recall": recall}

def main():
    # Define paths
    model_output_dir = "/disk/diamond-scratch/cvaro009/data/mobilellama"
    
    # FIXED PATHS - using the correct unified directory
    train_path = "Desktop/LLMComp2025/data/processed/unified/unified_train.csv"
    val_path = "/aul/homes/cvaro009/Desktop/LLMComp2025/data/processed/unified/unified_val.csv"
    test_path = "/aul/homes/cvaro009/Desktop/LLMComp2025/data/processed/unified/unified_test.csv"
    
    # Create output directory
    os.makedirs(model_output_dir, exist_ok=True)
    
    try:
        logger.info("Loading datasets...")
        
        # Load datasets with header
        train_df = pd.read_csv(train_path)
        val_df = pd.read_csv(val_path)
        test_df = pd.read_csv(test_path)
        
        # Log dataset info
        logger.info(f"Training examples: {len(train_df)}")
        logger.info(f"Validation examples: {len(val_df)}")
        logger.info(f"Test examples: {len(test_df)}")
        
        # Show label distribution
        logger.info("Label distribution in training set:")
        logger.info(train_df['label'].value_counts().to_dict())
        
        # Convert to datasets
        train_dataset = Dataset.from_pandas(train_df)
        val_dataset = Dataset.from_pandas(val_df)
        test_dataset = Dataset.from_pandas(test_df)
    
    except Exception as e:
        logger.error(f"Error loading datasets: {e}")
        raise
    
    # MobileLLaMA model ID
    model_id = "JackFram/llama-68m"
    logger.info(f"Preparing to train MobileLLaMA from {model_id}")
    
    try:
        # Load tokenizer
        logger.info("Loading tokenizer for MobileLLaMA...")
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
                max_length=1024,  
                padding=False,  
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
        
        # Load model
        logger.info("Loading MobileLLaMA model...")
        
        # Configure model with proper settings for classification
        config = LlamaConfig.from_pretrained(model_id)
        config.num_labels = 3  # Low, Medium, High risk
        config.pad_token_id = tokenizer.pad_token_id
        
        # Initialize model with float32 to avoid gradient unscaling errors
        model = LlamaForSequenceClassification.from_pretrained(
            model_id,
            config=config,
            torch_dtype=torch.float32
        )
        
        # Log model information
        total_params = sum(p.numel() for p in model.parameters())
        logger.info(f"Model loaded with {total_params:,} parameters")
        
        # Create a data collator that will handle padding
        data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir=model_output_dir,
            eval_strategy="epoch",
            save_strategy="epoch",
            learning_rate=2e-5,
            per_device_train_batch_size=16,  
            per_device_eval_batch_size=32,
            gradient_accumulation_steps=2,
            num_train_epochs=10,  
            weight_decay=0.01,
            load_best_model_at_end=True,
            metric_for_best_model="f1",
            warmup_ratio=0.1,
            logging_steps=50,
            report_to="wandb",
            fp16=False, 
            max_grad_norm=1.0,
        )
        
        # Initialize trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_tokenized,
            eval_dataset=val_tokenized,
            compute_metrics=compute_metrics,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
            data_collator=data_collator,
        )
        
        # Train model
        logger.info("Starting training of MobileLLaMA...")
        trainer.train()
        
        # Evaluate model
        logger.info("Evaluating on test set...")
        test_results = trainer.evaluate(test_tokenized)
        
        # Log results
        logger.info("\nMobileLLaMA Results:")
        logger.info(f"Best validation metric: {trainer.state.best_metric:.4f}")
        logger.info(f"Test accuracy: {test_results['eval_accuracy']:.4f}")
        logger.info(f"Test F1: {test_results['eval_f1']:.4f}")
        
        # Save model
        trainer.save_model(model_output_dir)
        tokenizer.save_pretrained(model_output_dir)
        logger.info(f"Model and tokenizer saved to {model_output_dir}")

    except Exception as e:
        logger.error(f"Error during training: {e}")
        raise

if __name__ == "__main__":
    main()
    logger.info("Training script completed successfully.")
    # Ensure the script exits cleanly