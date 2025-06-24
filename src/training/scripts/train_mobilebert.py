#!/usr/bin/env python3
"""
Fixed training script for MobileBERT model on traveler risk assessment
"""

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

import pandas as pd
import torch
import numpy as np
from datasets import Dataset
from transformers import (
    MobileBertTokenizer, 
    MobileBertForSequenceClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback
)
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import logging

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
    data_dir = "data/processed/unified"
    
    train_df = pd.read_csv(os.path.join(data_dir, "unified_complete_labeled.csv"), 
                          header=None, names=["input_text", "label", "example_type"])
    val_df = pd.read_csv(os.path.join(data_dir, "unified_val.csv"))
    test_df = pd.read_csv(os.path.join(data_dir, "unified_test.csv"))
    
    # Print label distribution
    logger.info("Label distribution in training set:")
    logger.info(train_df['label'].value_counts().to_dict())
    
    # Ensure labels are integers
    train_df['label'] = train_df['label'].astype(int)
    val_df['label'] = val_df['label'].astype(int)
    test_df['label'] = test_df['label'].astype(int)
    
    # Convert to datasets
    train_dataset = Dataset.from_pandas(train_df)
    val_dataset = Dataset.from_pandas(val_df)
    test_dataset = Dataset.from_pandas(test_df)
    
    # Initialize tokenizer
    logger.info("Loading MobileBERT tokenizer...")
    tokenizer = MobileBertTokenizer.from_pretrained("google/mobilebert-uncased")
    
    # Tokenize function
    def tokenize_function(examples):
        return tokenizer(
            examples["input_text"],
            padding="max_length",
            truncation=True,
            max_length=128
        )
    
    # Tokenize datasets
    logger.info("Tokenizing datasets...")
    train_tokenized = train_dataset.map(tokenize_function, batched=True)
    val_tokenized = val_dataset.map(tokenize_function, batched=True)
    test_tokenized = test_dataset.map(tokenize_function, batched=True)
    
    # Format for PyTorch
    train_tokenized.set_format(type="torch", columns=["input_ids", "attention_mask", "label"], 
                              output_all_columns=True)
    val_tokenized.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
    test_tokenized.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
    
    # Define model directory
    model_output_dir = os.path.join("models", "traveler_classification", "mobilebert_risk_assessment")
    os.makedirs(model_output_dir, exist_ok=True)
    
    # Load model
    logger.info("Loading MobileBERT model...")
    model = MobileBertForSequenceClassification.from_pretrained(
        "google/mobilebert-uncased",
        num_labels=3
    )
    
    # Simply verify the model has the right number of classes
    logger.info(f"Model initialized with {model.config.num_labels} output classes")
    
    # Create custom trainer for example type weighting
    class WeightedTrainer(Trainer):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.example_type_weights = {
                "food": 2.0,
                "culture": 1.8,
                "history": 1.8,
                "media": 1.7,
                "interest": 1.7,
                "confounding": 1.7,
                "syntactic": 1.5,
                "descriptor": 1.5,
                "noisy": 1.3,
                "origin": 1.2,
                "combined": 1.5,
                "other": 1.0
            }
        
        def compute_loss(self, model, inputs, return_outputs=False):
            # Forward pass with proper inputs
            labels = inputs.pop("labels")
            outputs = model(**inputs)
            logits = outputs.logits
            
            # Re-add labels to inputs for later use
            inputs["labels"] = labels
            
            # Standard cross-entropy loss with reduction='none' for weighting
            loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
            loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
            
            # Apply example type weighting if available
            if self.training and hasattr(inputs, "example_type"):
                weights = torch.ones_like(labels, dtype=torch.float)
                
                # Apply weights based on example type
                for i, example_type in enumerate(inputs.get("example_type", [])):
                    if example_type in self.example_type_weights:
                        weights[i] *= self.example_type_weights[example_type]
                    
                    # Additional weight for medium risk examples
                    if labels[i] == 1:  # Medium risk
                        weights[i] *= 1.5
                
                # Apply weights
                loss = (loss * weights).mean()
            else:
                # If no example_type or not training, use regular mean
                loss = loss.mean()
            
            return (loss, outputs) if return_outputs else loss
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=model_output_dir,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=3e-5,  # Slightly higher learning rate
        per_device_train_batch_size=16,
        per_device_eval_batch_size=32,
        num_train_epochs=8,
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        warmup_ratio=0.1,
        logging_steps=100,
        report_to="none",
        # Add gradient clipping for stability
        max_grad_norm=1.0
    )
    
    # Try with standard Trainer first
    logger.info("Initializing trainer...")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_tokenized,
        eval_dataset=val_tokenized,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
    )
    
    # Train model
    logger.info("Starting MobileBERT training...")
    trainer.train()
    
    # Evaluate model
    logger.info("Evaluating on test set...")
    test_results = trainer.evaluate(test_tokenized)
    
    # Print results
    logger.info("\nMobileBERT Results:")
    logger.info(f"Best validation metric: {trainer.state.best_metric:.4f}")
    logger.info(f"Test accuracy: {test_results['eval_accuracy']:.4f}")
    logger.info(f"Test F1: {test_results['eval_f1']:.4f}")
    
    # Save model
    trainer.save_model(model_output_dir)
    tokenizer.save_pretrained(model_output_dir)
    logger.info(f"Model and tokenizer saved to {model_output_dir}")

if __name__ == "__main__":
    main()