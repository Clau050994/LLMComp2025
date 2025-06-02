"""
SageMaker training script for SLMs.
This script is used as the entry point for SageMaker training jobs.
"""

import os
import sys
import logging
import argparse
from datetime import datetime

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer, 
    TrainingArguments,
    DataCollatorForLanguageModeling
)
import datasets
from datasets import load_dataset

# Set up logging
logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[logging.StreamHandler(sys.stdout)]
)

def parse_args():
    """Parse training arguments."""
    parser = argparse.ArgumentParser()
    
    # Model parameters
    parser.add_argument("--model_id", type=str, required=True, 
                      help="Model identifier (HF Hub ID or local path)")
    
    # Dataset parameters
    parser.add_argument("--dataset", type=str, required=True,
                      help="Dataset name to use for training")
    parser.add_argument("--dataset_config", type=str, default=None,
                      help="Dataset configuration name")
    parser.add_argument("--text_column", type=str, default="text",
                      help="Column name for text data")
    parser.add_argument("--max_seq_length", type=int, default=512,
                      help="Maximum sequence length")
    
    # Training parameters
    parser.add_argument("--epochs", type=int, default=3,
                      help="Number of training epochs")
    parser.add_argument("--per_device_train_batch_size", type=int, default=8,
                      help="Batch size per device for training")
    parser.add_argument("--per_device_eval_batch_size", type=int, default=8,
                      help="Batch size per device for evaluation")
    parser.add_argument("--learning_rate", type=float, default=5e-5,
                      help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.01,
                      help="Weight decay")
    parser.add_argument("--warmup_steps", type=int, default=0,
                      help="Number of warmup steps")
    
    # SageMaker parameters
    parser.add_argument("--output_data_dir", type=str, default=os.environ.get("SM_OUTPUT_DATA_DIR"))
    parser.add_argument("--model_dir", type=str, default=os.environ.get("SM_MODEL_DIR"))
    parser.add_argument("--n_gpus", type=str, default=os.environ.get("SM_NUM_GPUS"))
    parser.add_argument("--training_dir", type=str, default=os.environ.get("SM_CHANNEL_TRAIN"))
    parser.add_argument("--validation_dir", type=str, default=os.environ.get("SM_CHANNEL_VALIDATION"))
    
    return parser.parse_args()

def prepare_dataset(args):
    """
    Prepare the dataset for training and evaluation.
    
    Args:
        args: Command line arguments
    
    Returns:
        tuple: (train_dataset, eval_dataset)
    """
    logger.info(f"Loading dataset: {args.dataset}")
    
    # Check if data is being provided through SageMaker channels
    if args.training_dir and os.path.exists(args.training_dir):
        logger.info(f"Loading data from SageMaker channel: {args.training_dir}")
        # This is to handle pre-uploaded datasets in SageMaker
        train_dataset = datasets.load_from_disk(args.training_dir)
        
        if args.validation_dir and os.path.exists(args.validation_dir):
            eval_dataset = datasets.load_from_disk(args.validation_dir)
        else:
            # Split training data if validation not provided
            train_val = train_dataset.train_test_split(test_size=0.1)
            train_dataset = train_val["train"]
            eval_dataset = train_val["test"]
    else:
        # Load from HF datasets
        if args.dataset_config:
            dataset = load_dataset(args.dataset, args.dataset_config)
        else:
            dataset = load_dataset(args.dataset)
            
        # Use default splits or create them
        if "train" in dataset:
            train_dataset = dataset["train"]
            eval_dataset = dataset["validation"] if "validation" in dataset else dataset["test"]
        else:
            # Split if no predefined splits
            train_val = dataset.train_test_split(test_size=0.1)
            train_dataset = train_val["train"]
            eval_dataset = train_val["test"]
    
    logger.info(f"Train dataset size: {len(train_dataset)}")
    logger.info(f"Evaluation dataset size: {len(eval_dataset)}")
    
    return train_dataset, eval_dataset

def tokenize_function(examples, tokenizer, text_column, max_length):
    """Tokenize examples."""
    return tokenizer(
        examples[text_column], 
        padding="max_length",
        truncation=True,
        max_length=max_length
    )

def main():
    """Main training function."""
    args = parse_args()
    
    # Load model and tokenizer
    logger.info(f"Loading model and tokenizer from: {args.model_id}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    model = AutoModelForCausalLM.from_pretrained(args.model_id)
    
    # Prepare datasets
    train_dataset, eval_dataset = prepare_dataset(args)
    
    # Tokenize datasets
    logger.info("Tokenizing datasets...")
    train_dataset = train_dataset.map(
        lambda examples: tokenize_function(examples, tokenizer, args.text_column, args.max_seq_length),
        batched=True,
        remove_columns=[col for col in train_dataset.column_names if col != args.text_column]
    )
    eval_dataset = eval_dataset.map(
        lambda examples: tokenize_function(examples, tokenizer, args.text_column, args.max_seq_length),
        batched=True,
        remove_columns=[col for col in eval_dataset.column_names if col != args.text_column]
    )
    
    # Create data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False  # Not using masked language modeling
    )
    
    # Set up training arguments
    training_args = TrainingArguments(
        output_dir=args.model_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_steps=args.warmup_steps,
        logging_dir=f"{args.output_data_dir}/logs",
        logging_steps=100,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        save_total_limit=3,
        report_to=["tensorboard"]
    )
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator
    )
    
    # Start training
    logger.info("Starting training...")
    trainer.train()
    
    # Evaluate
    logger.info("Evaluating model...")
    eval_results = trainer.evaluate()
    logger.info(f"Evaluation results: {eval_results}")
    
    # Save model and tokenizer
    logger.info(f"Saving model to {args.model_dir}")
    trainer.save_model(args.model_dir)
    tokenizer.save_pretrained(args.model_dir)
    
    # Save evaluation results
    with open(os.path.join(args.model_dir, "eval_results.txt"), "w") as f:
        for key, value in eval_results.items():
            f.write(f"{key} = {value}\n")
            
    logger.info("Training completed successfully")

if __name__ == "__main__":
    main()
