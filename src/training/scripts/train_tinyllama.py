import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

import pandas as pd
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
    
    # Label distribution
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
    
    # Model directory
    model_output_dir = os.path.join("models", "traveler_classification", "tinyllama_risk_assessment")
    os.makedirs(model_output_dir, exist_ok=True)
    
    # Load model - FIXED: Properly configure LlamaForSequenceClassification 
    logger.info("Loading TinyLLaMA model...")
    
    # First modify the config with num_labels before loading the model
    config = LlamaConfig.from_pretrained(model_id)
    config.num_labels = 3  # Set the number of labels in config
    config.pad_token_id = tokenizer.pad_token_id  # Set pad token ID
    
    # IMPORTANT: Load model with float32 to avoid FP16 gradient unscaling issues
    model = LlamaForSequenceClassification.from_pretrained(
        model_id,
        config=config,
        torch_dtype=torch.float32  # Use float32 to avoid gradient unscaling errors
    )
    
    # Ensure model knows about padding token
    model.config.pad_token_id = tokenizer.pad_token_id
    
    # Create data collator for dynamic padding
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    
    # Training arguments - FIXED: Disable fp16 training
    training_args = TrainingArguments(
        output_dir=model_output_dir,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=2,  # Reduced batch size
        per_device_eval_batch_size=4,
        gradient_accumulation_steps=8,  # Increased accumulation steps
        num_train_epochs=6,
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        warmup_ratio=0.1,
        logging_steps=50,
        report_to="none",
        fp16=False,  # Disable mixed precision training
        bf16=False,  # Disable bfloat16 training
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
    logger.info("Starting TinyLLaMA training...")
    trainer.train()
    
    # Evaluate model
    logger.info("Evaluating on test set...")
    test_results = trainer.evaluate(test_tokenized)
    
    # Log results
    logger.info("\nTinyLLaMA Results:")
    logger.info(f"Best validation metric: {trainer.state.best_metric:.4f}")
    logger.info(f"Test accuracy: {test_results['eval_accuracy']:.4f}")
    logger.info(f"Test F1: {test_results['eval_f1']:.4f}")
    
    # Save model
    trainer.save_model(model_output_dir)
    tokenizer.save_pretrained(model_output_dir)
    logger.info(f"Model and tokenizer saved to {model_output_dir}")

if __name__ == "__main__":
    main()