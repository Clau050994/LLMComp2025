import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["CUDA_VISIBLE_DEVICES"] = "2"  

import pandas as pd
import torch
import re
import numpy as np
from datasets import Dataset
from transformers import (
    DistilBertTokenizerFast,
    DistilBertForSequenceClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback
)
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import datetime

# Set random seeds for reproducibility
import random
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

# Add timestamp to output directory for versioning
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
model_output_dir = os.path.join("models", "traveler_classification", f"distilbert_syntax_aware_{timestamp}")
os.makedirs(model_output_dir, exist_ok=True)
logs_dir = os.path.join(model_output_dir, "logs")
os.makedirs(logs_dir, exist_ok=True)

# Starting from scratch with base DistilBERT model
print("Loading base DistilBERT model...")
model_name = "distilbert-base-uncased"
tokenizer = DistilBertTokenizerFast.from_pretrained(model_name)
model = DistilBertForSequenceClassification.from_pretrained(
    model_name,
    num_labels=3  # Low, Medium, High risk
)

# Load the unified datasets
print("Loading unified datasets...")
unified_dir = "data/processed/unified"
train_df = pd.read_csv(os.path.join(unified_dir, "unified_complete_labeled.csv"), 
                      header=None, names=["input_text", "label", "example_type"])
val_df = pd.read_csv(os.path.join(unified_dir, "unified_val.csv"))
test_df = pd.read_csv(os.path.join(unified_dir, "unified_test.csv"))

# Print dataset stats
print(f"Loaded {len(train_df)} training examples")
print(f"Loaded {len(val_df)} validation examples")
print(f"Loaded {len(test_df)} test examples")
print(f"Training label distribution: {train_df['label'].value_counts().to_dict()}")
print(f"Example type distribution: {train_df['example_type'].value_counts().to_dict()}")

# Convert to datasets
train_dataset = Dataset.from_pandas(train_df)
val_dataset = Dataset.from_pandas(val_df)
test_dataset = Dataset.from_pandas(test_df)

# Tokenize function
def tokenize(batch):
    return tokenizer(batch["input_text"], truncation=True, padding="max_length", max_length=128)

# Tokenize datasets
print("Tokenizing datasets...")
train_dataset = train_dataset.map(tokenize, batched=True)
val_dataset = val_dataset.map(tokenize, batched=True)
test_dataset = test_dataset.map(tokenize, batched=True)

# Set format for PyTorch with output_all_columns=True to preserve indices
train_dataset.set_format("torch", columns=["input_ids", "attention_mask", "label"], 
                        output_all_columns=True)  # This preserves the index
val_dataset.set_format("torch", columns=["input_ids", "attention_mask", "label"],
                      output_all_columns=True)
test_dataset.set_format("torch", columns=["input_ids", "attention_mask", "label"],
                       output_all_columns=True)

# Define simplified compute_metrics function that only reports F1 score
def compute_metrics(pred):
    labels = pred.label_ids
    preds = torch.argmax(torch.tensor(pred.predictions), dim=1)
    _, _, f1, _ = precision_recall_fscore_support(
        labels, preds, average="weighted"
    )
    return {"f1": f1}

# Training arguments with simplified logging
training_args = TrainingArguments(
    output_dir=model_output_dir,
    evaluation_strategy="epoch", 
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=32,
    num_train_epochs=8,  
    weight_decay=0.01,   
    logging_dir=logs_dir,
    logging_steps=100,  # Less frequent logging
    dataloader_num_workers=1,
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    report_to="none",  # Disable wandb/tensorboard reporting
    warmup_ratio=0.1,
    gradient_accumulation_steps=2,
    fp16=torch.cuda.is_available(),
)

# Custom simplified trainer with basic weighting
class SyntaxAwareTrainer(Trainer):
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
        print(f"Starting epoch {self.epoch_num} of {int(training_args.num_train_epochs)}")
        return super().on_epoch_begin(args, state, control, **kwargs)

# Initialize trainer with early stopping
trainer = SyntaxAwareTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
)

# Train model
print("Starting training...")
train_result = trainer.train()

# Print simplified final results
print("\nTraining completed!")
print(f"Best validation F1: {trainer.state.best_metric}")
print(f"Best checkpoint: {trainer.state.best_model_checkpoint}")

# Save model and tokenizer
print(f"Saving model to {model_output_dir}")
trainer.save_model(model_output_dir)
tokenizer.save_pretrained(model_output_dir)

# Final evaluation on test set
print("\nFinal evaluation on test set:")
test_metrics = trainer.evaluate(test_dataset)
print(f"Test F1: {test_metrics['eval_f1']}")