import os
import torch
import wandb
from datasets import load_dataset
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification, 
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback
)
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import numpy as np

# 🔧 Environment setup
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ["WANDB_LOG_MODEL"] = "checkpoint"

# 🎯 Init W&B
wandb.init(
    project="distilbert_classification",
    name="distilbert-classification-run",
    config={
        "epochs": 3,
        "lr": 2e-5,
        "batch_size": 16,
        "model": "distilbert-base-uncased",
        "dataset": "text_classification",
        "max_length": 512
    }
)

# Load separate JSONL files for train and validation
print("📂 Loading datasets...")
try:
    # Load separate train and validation JSONL files
    train_path = "/aul/homes/cvaro009/Desktop/LLMComp2025/data/processed/billsum/us_train_data_final_OFFICIAL.jsonl"
    val_path = "/aul/homes/cvaro009/Desktop/LLMComp2025/data/processed/billsum/us_test_data_final_OFFICIAL.jsonl"
    
    # Load each JSONL file separately
    train_ds = load_dataset("json", data_files=train_path)["train"]
    val_ds = load_dataset("json", data_files=val_path)["train"]
    
    print(f"✅ Loaded train: {len(train_ds)} samples, val: {len(val_ds)} samples")
    
    # Check data format
    print("📊 Dataset inspection:")
    sample = train_ds[0]
    print(f"Train columns: {train_ds.column_names}")
    print(f"Val columns: {val_ds.column_names}")
    print(f"Sample keys: {list(sample.keys())}")
    
    # Print sample content
    for key, value in sample.items():
        if isinstance(value, str):
            print(f"{key}: {value[:100]}..." if len(value) > 100 else f"{key}: {value}")
        else:
            print(f"{key}: {value}")
            
except Exception as e:
    print(f"❌ Error loading datasets: {e}")
    import traceback
    traceback.print_exc()
    raise


# Load tokenizer & model for classification
print("🤖 Loading model and tokenizer...")
model_name = "distilbert-base-uncased"
try:
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # For classification - specify number of labels
    unique_labels = set(train_ds["label"])
    num_labels = len(unique_labels)
    print(f"📊 Found {num_labels} unique labels: {unique_labels}")
    
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, 
        num_labels=num_labels
    )
    
    print(f"✅ Model loaded with {num_labels} classes")
    
    if torch.cuda.is_available():
        print("🔧 Using GPU for training")
        model = model.to("cuda")
        print(f"✅ Model moved to GPU: {next(model.parameters()).device}")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    raise

# Add GPU info
print(f"🔧 CUDA Available: {torch.cuda.is_available()}")
print(f"🔧 CUDA Device Count: {torch.cuda.device_count()}")
if torch.cuda.is_available():
    print(f"🔧 Current CUDA Device: {torch.cuda.current_device()}")
    print(f"🔧 Device Name: {torch.cuda.get_device_name()}")
    print(f"🔧 Model Device: {next(model.parameters()).device}")

# Tokenization function
def tokenize_function(examples):
    """Tokenize the input text for classification"""
    # UPDATE: Use the correct text field name from your dataset
    text_field = "input_text" if "input_text" in examples else "text"
    
    return tokenizer(
        examples[text_field],
        truncation=True,
        padding="max_length",
        max_length=512,
        return_tensors="pt"
    )

# Tokenize datasets
print("🔤 Tokenizing datasets...")
try:
    train_tokenized = train_ds.map(tokenize_function, batched=True)
    val_tokenized = val_ds.map(tokenize_function, batched=True)
    
    # Set format for PyTorch
    train_tokenized.set_format("torch", columns=["input_ids", "attention_mask", "label"])
    val_tokenized.set_format("torch", columns=["input_ids", "attention_mask", "label"])
    
    print("✅ Tokenization completed")
except Exception as e:
    print(f"❌ Error tokenizing datasets: {e}")
    raise

# Metrics computation
def compute_metrics(eval_pred):
    """Compute accuracy, precision, recall, F1 for classification"""
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    
    accuracy = accuracy_score(labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predictions, average='weighted'
    )
    
    return {
        "accuracy": accuracy,
        "f1": f1,
        "precision": precision,
        "recall": recall
    }

# Create output directories
output_dir = "/disk/diamond-scratch/cvaro009/data/usecase4/distilbert_classification"
final_model_dir = "/disk/diamond-scratch/cvaro009/data/usecase4/distilbert"

print("📁 Creating output directories...")
try:
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(final_model_dir, exist_ok=True)
    
    # Test write permissions
    test_file = os.path.join(output_dir, "test_write.txt")
    with open(test_file, "w") as f:
        f.write("test")
    os.remove(test_file)
    print(f"✅ Output directories created and writable")
    print(f"📂 Training output: {output_dir}")
    print(f"📂 Final model: {final_model_dir}")
except Exception as e:
    print(f"❌ Error with output directories: {e}")
    raise

# Training arguments
print("⚙️ Configuring training...")
training_args = TrainingArguments(
    output_dir=output_dir,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    learning_rate=2e-5,
    warmup_steps=500,
    weight_decay=0.01,
    logging_steps=50,
    save_strategy="epoch",
    save_total_limit=2,
    evaluation_strategy="epoch",
    fp16=torch.cuda.is_available(),
    report_to="wandb",
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    greater_is_better=True,
    save_safetensors=True,
    dataloader_pin_memory=False,
    remove_unused_columns=False,
)

print("✅ Training configuration ready")

# Create trainer
print("🏃 Creating trainer...")
try:
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_tokenized,
        eval_dataset=val_tokenized,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
        callbacks=[
            EarlyStoppingCallback(early_stopping_patience=2)
        ]
    )
    print("✅ Trainer created successfully")
except Exception as e:
    print(f"❌ Error creating trainer: {e}")
    raise

# Train the model
print("🎯 Starting training...")
try:
    trainer.train()
    print("✅ Training completed successfully!")
except Exception as e:
    print(f"❌ Training failed: {e}")
    import traceback
    traceback.print_exc()
    raise

# Evaluate on validation set
print("📊 Final evaluation...")
try:
    eval_results = trainer.evaluate()
    print("✅ Final evaluation results:")
    for key, value in eval_results.items():
        print(f"  {key}: {value:.4f}")
except Exception as e:
    print(f"❌ Error during evaluation: {e}")

# Save final model
print("💾 Saving final model...")
try:
    trainer.save_model(final_model_dir)
    tokenizer.save_pretrained(final_model_dir)
    print(f"✅ Model saved to {final_model_dir}")
    
    # Verify files were saved
    saved_files = os.listdir(final_model_dir)
    print(f"📄 Saved files: {saved_files}")
    
    # Check file sizes
    for file in saved_files:
        file_path = os.path.join(final_model_dir, file)
        if os.path.isfile(file_path):
            size_mb = os.path.getsize(file_path) / (1024 * 1024)
            print(f"  📄 {file}: {size_mb:.2f} MB")
            
except Exception as e:
    print(f"❌ Error saving model: {e}")
    import traceback
    traceback.print_exc()

# Check if checkpoints were saved during training
print("🔍 Checking training checkpoints...")
try:
    checkpoint_dirs = [d for d in os.listdir(output_dir) if d.startswith("checkpoint-")]
    if checkpoint_dirs:
        print(f"✅ Found {len(checkpoint_dirs)} checkpoints: {checkpoint_dirs}")
        for checkpoint in checkpoint_dirs:
            checkpoint_path = os.path.join(output_dir, checkpoint)
            files = os.listdir(checkpoint_path)
            print(f"  📁 {checkpoint}: {len(files)} files")
    else:
        print("⚠️ No checkpoints found during training")
except Exception as e:
    print(f"❌ Error checking checkpoints: {e}")

# Finish W&B run
print("🏁 Finishing W&B run...")
wandb.finish()
print("✅ Script completed!")