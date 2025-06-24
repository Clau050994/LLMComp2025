#!/usr/bin/env python3
"""
Training script for smaller language models on traveler risk assessment.
Supports: TinyLLaMA-1.1B, ALBERT, MobileBERT, MobileLLaMA, and Grok-3 mini-fast.
"""

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["CUDA_VISIBLE_DEVICES"] = "2"  # Adjust GPU device as needed

import pandas as pd
import torch
import numpy as np
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback
)
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import datetime
import argparse
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("training_small_models.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Set random seeds for reproducibility
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# Model configurations
MODEL_CONFIGS = {
    "tinyllama": {
        "name": "TinyLLaMA-1.1B",
        "model_id": "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T",
        "max_length": 512
    },
    "albert": {
        "name": "ALBERT",
        "model_id": "albert-base-v2",
        "max_length": 128
    },
    "mobilebert": {
        "name": "MobileBERT",
        "model_id": "google/mobilebert-uncased",
        "max_length": 128
    },
    "mobilellama": {
        "name": "MobileLLaMA",
        "model_id": "lmsys/mobilellama-1b-chat-v1",
        "max_length": 512
    },
    "grok": {
        "name": "Grok-3 mini-fast",
        "model_id": "xai-org/grok-3-mini-fast",
        "max_length": 1024
    }
}

def compute_metrics(pred):
    """Compute metrics for evaluation"""
    labels = pred.label_ids
    preds = torch.argmax(torch.tensor(pred.predictions), dim=1).numpy()
    
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average="weighted"
    )
    acc = accuracy_score(labels, preds)
    
    return {
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }

class SyntaxAwareTrainer(Trainer):
    """Custom trainer that applies weights based on example type"""
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
        self.epoch_num = 0
    
    def compute_loss(self, model, inputs, return_outputs=False):
        # Extract relevant batch information
        labels = inputs.get("labels")
        batch_indices = inputs.get("idx", [])
        
        # Forward pass
        outputs = model(**{k: v for k, v in inputs.items() if k != 'idx'})
        logits = outputs.get("logits")
        
        # Standard cross-entropy loss
        loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
        loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        
        # Create weights tensor with base weight of 1.0
        weights = torch.ones_like(labels, dtype=torch.float)
        
        # Apply weights based on example type
        if len(batch_indices) > 0:
            for i, idx in enumerate(batch_indices.tolist()):
                if idx < len(self.train_dataset):
                    example_type = self.train_dataset[idx].get('example_type', 'other')
                    if example_type in self.example_type_weights:
                        weights[i] *= self.example_type_weights[example_type]
                    
                    # Additional weight for medium risk examples
                    if labels[i] == 1:  # Medium risk
                        weights[i] *= 1.5
        
        # Apply weights and calculate mean loss
        weighted_loss = (loss * weights).mean()
        
        return (weighted_loss, outputs) if return_outputs else weighted_loss
    
    def on_epoch_begin(self, args=None, state=None, control=None, **kwargs):
        self.epoch_num += 1
        logger.info(f"Starting epoch {self.epoch_num}")
        return super().on_epoch_begin(args, state, control, **kwargs)

def load_datasets(data_dir="data/processed/unified"):
    """Load the unified datasets"""
    logger.info("Loading datasets...")
    
    train_df = pd.read_csv(os.path.join(data_dir, "unified_complete_labeled.csv"), 
                          header=None, names=["input_text", "label", "example_type"])
    val_df = pd.read_csv(os.path.join(data_dir, "unified_val.csv"))
    test_df = pd.read_csv(os.path.join(data_dir, "unified_test.csv"))
    
    logger.info(f"Loaded {len(train_df)} training examples")
    logger.info(f"Loaded {len(val_df)} validation examples")
    logger.info(f"Loaded {len(test_df)} test examples")
    
    # Print label distribution
    logger.info(f"Training label distribution: {train_df['label'].value_counts().to_dict()}")
    
    # Convert to datasets
    train_dataset = Dataset.from_pandas(train_df)
    val_dataset = Dataset.from_pandas(val_df)
    test_dataset = Dataset.from_pandas(test_df)
    
    return train_dataset, val_dataset, test_dataset

def prepare_tokenized_datasets(tokenizer, train_dataset, val_dataset, test_dataset, max_length):
    """Tokenize the datasets"""
    logger.info(f"Tokenizing datasets with max length {max_length}...")
    
    def tokenize_function(examples):
        return tokenizer(examples["input_text"], truncation=True, 
                        padding="max_length", max_length=max_length)
    
    # Tokenize datasets
    train_dataset = train_dataset.map(tokenize_function, batched=True)
    val_dataset = val_dataset.map(tokenize_function, batched=True)
    test_dataset = test_dataset.map(tokenize_function, batched=True)
    
    # Set format for PyTorch with output_all_columns=True to preserve indices
    train_dataset.set_format("torch", columns=["input_ids", "attention_mask", "label"], 
                            output_all_columns=True)
    val_dataset.set_format("torch", columns=["input_ids", "attention_mask", "label"],
                          output_all_columns=True)
    test_dataset.set_format("torch", columns=["input_ids", "attention_mask", "label"],
                           output_all_columns=True)
    
    return train_dataset, val_dataset, test_dataset

def train_model(model_key, train_dataset, val_dataset, test_dataset, num_epochs=10):
    """Train the specified model"""
    model_config = MODEL_CONFIGS[model_key]
    model_name = model_config["name"]
    model_id = model_config["model_id"]
    max_length = model_config["max_length"]
    
    logger.info(f"Preparing to train {model_name} from {model_id}")
    
    # Create a descriptive model directory name based on model and task
    model_output_dir = os.path.join("models", "traveler_classification", 
                                   f"{model_key}_risk_assessment")

    # Create directory if it doesn't exist                            
    os.makedirs(model_output_dir, exist_ok=True)
    
    # Load tokenizer
    logger.info(f"Loading tokenizer for {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    
    # Handle special cases for tokenizers
    if "llama" in model_key.lower() or "grok" in model_key.lower():
        # For LLaMA and Grok models, make sure tokenizer handles eos/bos tokens properly
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
    
    # Prepare tokenized datasets
    train_tokenized, val_tokenized, test_tokenized = prepare_tokenized_datasets(
        tokenizer, train_dataset, val_dataset, test_dataset, max_length
    )
    
    # Load model for sequence classification
    logger.info(f"Loading {model_name} model...")
    
    # For decoder-only models (LLaMa, Grok), we need to use different model class
    if "llama" in model_key.lower() or "grok" in model_key.lower():
        # For decoder-only models - use a strategy appropriate for classification
        model = AutoModelForSequenceClassification.from_pretrained(
            model_id,
            num_labels=3,  # Low, Medium, High risk
            trust_remote_code=True,  # Required for Grok
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
        )
    else:
        # For BERT-based encoder models (ALBERT, MobileBERT)
        model = AutoModelForSequenceClassification.from_pretrained(
            model_id,
            num_labels=3  # Low, Medium, High risk
        )
    
    # Training arguments
    gradient_accumulation_steps = 4 if "llama" in model_key.lower() or "grok" in model_key.lower() else 2
    batch_size = 8 if "llama" in model_key.lower() or "grok" in model_key.lower() else 16
    
    training_args = TrainingArguments(
        output_dir=model_output_dir,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size*2,
        num_train_epochs=num_epochs,
        weight_decay=0.01,
        logging_dir=os.path.join(model_output_dir, "logs"),
        logging_steps=100,
        dataloader_num_workers=1,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        report_to="none",
        warmup_ratio=0.1,
        gradient_accumulation_steps=gradient_accumulation_steps,
        fp16=torch.cuda.is_available(),
    )
    
    # Initialize trainer
    trainer = SyntaxAwareTrainer(
        model=model,
        args=training_args,
        train_dataset=train_tokenized,
        eval_dataset=val_tokenized,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
    )
    
    # Train model
    logger.info(f"Starting training of {model_name}...")
    trainer.train()
    
    # Evaluate model
    logger.info(f"Evaluating {model_name} on test set...")
    test_results = trainer.evaluate(test_tokenized)
    
    # Log results
    logger.info(f"\n{model_name} Results:")
    logger.info(f"Best validation F1: {trainer.state.best_metric:.4f}")
    logger.info(f"Test F1: {test_results['eval_f1']:.4f}")
    logger.info(f"Test Accuracy: {test_results['eval_accuracy']:.4f}")
    
    # Save model and tokenizer
    trainer.save_model(model_output_dir)
    tokenizer.save_pretrained(model_output_dir)
    logger.info(f"Model and tokenizer saved to {model_output_dir}")
    
    # Return results for comparison
    return {
        "model": model_key,
        "best_val_f1": trainer.state.best_metric,
        "test_f1": test_results["eval_f1"],
        "test_accuracy": test_results["eval_accuracy"],
        "test_precision": test_results["eval_precision"],
        "test_recall": test_results["eval_recall"],
    }

def main():
    parser = argparse.ArgumentParser(description="Train small language models on traveler risk assessment")
    parser.add_argument("--model", type=str, required=True, choices=list(MODEL_CONFIGS.keys()),
                      help="Model to train (tinyllama, albert, mobilebert, mobilellama, grok)")
    parser.add_argument("--epochs", type=int, default=8, help="Number of training epochs")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()
    
    # Set random seed
    set_seed(args.seed)
    
    # Load datasets
    train_dataset, val_dataset, test_dataset = load_datasets()
    
    # Train the selected model
    results = train_model(args.model, train_dataset, val_dataset, test_dataset, num_epochs=args.epochs)
    
    # Print results summary
    logger.info("\nTraining completed!")
    logger.info(f"Model: {MODEL_CONFIGS[args.model]['name']}")
    logger.info(f"Best validation F1: {results['best_val_f1']:.4f}")
    logger.info(f"Test F1: {results['test_f1']:.4f}")
    logger.info(f"Test Accuracy: {results['test_accuracy']:.4f}")

if __name__ == "__main__":
    import random
    main()