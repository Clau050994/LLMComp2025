#!/usr/bin/env python3
"""
Enhanced evaluation script for MobileBERT model with additional metrics:
- Detailed error analysis
- Performance by example category
- Model size and RAM utilization
- Confusion matrix visualization
- Standard benchmarks comparison
"""

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["CUDA_VISIBLE_DEVICES"] = "2"  
import sys
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer
)
from sklearn.metrics import (
    accuracy_score, 
    precision_recall_fscore_support, 
    confusion_matrix, 
    classification_report,
    roc_auc_score,
    f1_score
)
import psutil  # For RAM utilization
import argparse
from pathlib import Path


def load_model_and_tokenizer(model_path):
    """Load the model and tokenizer from the specified path"""
    print(f"Loading model from {model_path}...")
    try:
        model = AutoModelForSequenceClassification.from_pretrained(model_path)
        tokenizer = AutoTokenizer.from_pretrained(model_path)
    except Exception as e:
        print(f"Error loading from path directly: {e}")
        # If loading fails, try with original MobileBERT
        print("Trying with original MobileBERT tokenizer...")
        model = AutoModelForSequenceClassification.from_pretrained(model_path)
        tokenizer = AutoTokenizer.from_pretrained("google/mobilebert-uncased")
        
    # Get model size
    model_size_mb = sum(p.numel() * p.element_size() for p in model.parameters()) / (1024 * 1024)
    print(f"Model loaded: {model_size_mb:.2f} MB")
    
    return model, tokenizer


def load_dataset(data_path, has_header=True):
    """Load and prepare dataset from CSV file"""
    if has_header:
        df = pd.read_csv(data_path)
    else:
        df = pd.read_csv(data_path, header=None, names=["input_text", "label", "example_type"])
    
    # Ensure input_text column exists
    if "text" in df.columns and "input_text" not in df.columns:
        df["input_text"] = df["text"]
    
    return Dataset.from_pandas(df)


def tokenize_dataset(dataset, tokenizer):
    """Tokenize the dataset using the provided tokenizer"""
    def tokenize_function(examples):
        return tokenizer(
            examples["input_text"], 
            truncation=True, 
            padding="max_length", 
            max_length=128
        )
    
    tokenized_dataset = dataset.map(tokenize_function, batched=True)
    tokenized_dataset.set_format(
        "torch", 
        columns=["input_ids", "attention_mask", "label"],
        output_all_columns=True
    )
    return tokenized_dataset


def compute_metrics(pred):
    """Compute comprehensive metrics for model evaluation"""
    labels = pred.label_ids
    preds = torch.argmax(torch.tensor(pred.predictions), dim=1).numpy()
    
    # Calculate precision, recall, and F1 for each class
    precision, recall, f1, support = precision_recall_fscore_support(
        labels, preds, average=None, labels=[0, 1, 2]
    )
    
    # Calculate weighted averages
    precision_avg, recall_avg, f1_avg, _ = precision_recall_fscore_support(
        labels, preds, average="weighted"
    )
    
    # Calculate accuracy
    acc = accuracy_score(labels, preds)
    
    # Calculate confusion matrix
    cm = confusion_matrix(labels, preds, labels=[0, 1, 2])
    
    # Calculate ROC AUC (one-vs-rest for multiclass)
    try:
        # Get probability predictions for ROC AUC calculation
        probs = torch.softmax(torch.tensor(pred.predictions), dim=1).numpy()
        roc_auc = roc_auc_score(labels, probs, multi_class='ovr')
    except:
        roc_auc = 0  # In case of error, skip ROC AUC
    
    return {
        "accuracy": acc,
        "f1_weighted": f1_avg,
        "precision_weighted": precision_avg,
        "recall_weighted": recall_avg,
        "f1_low_risk": f1[0],
        "f1_medium_risk": f1[1],
        "f1_high_risk": f1[2],
        "confusion_matrix": cm,
        "roc_auc": roc_auc
    }


def evaluate_model(model, eval_dataset, device=None):
    """Evaluate model on dataset and return predictions and metrics"""
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    model.to(device)
    model.eval()
    
    # Create trainer for evaluation
    trainer = Trainer(model=model)
    
    # Track RAM usage before evaluation
    before_ram = psutil.Process().memory_info().rss / (1024 * 1024)
    
    # Evaluate
    start_time = time.time()
    eval_results = trainer.evaluate(eval_dataset)
    
    # Get predictions
    output = trainer.predict(eval_dataset)
    
    # Calculate elapsed time
    elapsed_time = time.time() - start_time
    
    # Track RAM usage after evaluation
    after_ram = psutil.Process().memory_info().rss / (1024 * 1024)
    ram_used = after_ram - before_ram
    
    # Get predictions
    logits = output.predictions
    preds = np.argmax(logits, axis=1)
    labels = output.label_ids
    
    # Calculate metrics
    metrics = compute_metrics(output)
    metrics["eval_time_seconds"] = elapsed_time
    metrics["ram_usage_mb"] = ram_used
    
    return preds, labels, metrics


def plot_confusion_matrix(cm, save_path=None):
    """Plot and save confusion matrix"""
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm, 
        annot=True, 
        fmt='d',
        cmap='Blues',
        xticklabels=['Low Risk', 'Medium Risk', 'High Risk'],
        yticklabels=['Low Risk', 'Medium Risk', 'High Risk']
    )
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('MobileBERT Confusion Matrix')
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        print(f"Confusion matrix saved to {save_path}")
    
    plt.close()


def analyze_errors(df, preds, examples_to_show=10):
    """Analyze and print examples of misclassified instances"""
    df['predicted'] = preds
    df['is_error'] = df['label'] != df['predicted']
    
    errors = df[df['is_error']].copy()
    
    print(f"\nTotal errors: {len(errors)} out of {len(df)} ({len(errors)/len(df)*100:.2f}%)")
    
    if 'example_type' in df.columns:
        print("\nErrors by example type:")
        error_by_type = errors.groupby('example_type').size().reset_index()
        error_by_type.columns = ['example_type', 'error_count']
        
        total_by_type = df.groupby('example_type').size().reset_index()
        total_by_type.columns = ['example_type', 'total_count']
        
        error_analysis = error_by_type.merge(total_by_type, on='example_type')
        error_analysis['error_rate'] = error_analysis['error_count'] / error_analysis['total_count'] * 100
        
        # Sort by error rate descending
        error_analysis = error_analysis.sort_values('error_rate', ascending=False)
        
        print(error_analysis)
    
    print(f"\nShowing {min(examples_to_show, len(errors))} random error examples:")
    sample_errors = errors.sample(min(examples_to_show, len(errors)))
    
    for _, row in sample_errors.iterrows():
        print(f"Text: {row['input_text']}")
        print(f"True label: {row['label']}, Predicted: {row['predicted']}")
        if 'example_type' in row:
            print(f"Example type: {row['example_type']}")
        print("-" * 80)


def evaluate_performance_by_category(df, preds):
    """Evaluate performance broken down by example type"""
    if 'example_type' not in df.columns:
        print("No 'example_type' column found in dataset.")
        return
    
    df['predicted'] = preds
    
    print("\nPerformance by example type:")
    
    categories = df['example_type'].unique()
    results = []
    
    for category in categories:
        subset = df[df['example_type'] == category]
        if len(subset) > 0:
            f1 = f1_score(subset['label'], subset['predicted'], average='weighted')
            accuracy = accuracy_score(subset['label'], subset['predicted'])
            
            results.append({
                'Category': category,
                'Count': len(subset),
                'F1 Score': f1,
                'Accuracy': accuracy
            })
    
    results_df = pd.DataFrame(results)
    # Sort by count descending
    results_df = results_df.sort_values('Count', ascending=False)
    
    print(results_df)


def benchmark_comparison(metrics):
    """Compare model performance against common benchmarks"""
    print("\nModel Performance in Context:")
    
    benchmarks = {
        "Random Baseline (3 classes)": 0.33,
        "Majority Class Baseline": 0.39,  # Assuming distribution from training
        "MobileBERT on GLUE": 0.832,
        "BERT-base on GLUE": 0.849,
        "DistilBERT on GLUE": 0.824
    }
    
    print(f"Our MobileBERT model F1: {metrics['f1_weighted']:.4f}")
    for benchmark, score in benchmarks.items():
        print(f"{benchmark}: {score:.4f} ({(metrics['f1_weighted']-score)*100:+.1f}% diff)")


def main():
    parser = argparse.ArgumentParser(description="Evaluate MobileBERT model on test datasets")
    parser.add_argument("--model_path", type=str, required=True, 
                        help="Path to the model directory")
    parser.add_argument("--test_file", type=str, required=True, 
                        help="Path to test dataset CSV")
    parser.add_argument("--output_dir", type=str, default="evaluation_results",
                        help="Directory to save evaluation results")
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer(args.model_path)
    
    # Load test dataset
    print(f"Loading test data from {args.test_file}...")
    has_header = not args.test_file.endswith("_labeled.csv")  # Assume labeled files have no header
    test_dataset = load_dataset(args.test_file, has_header=has_header)
    
    # Tokenize dataset
    test_dataset = tokenize_dataset(test_dataset, tokenizer)
    
    # Evaluate model
    print("Evaluating model...")
    preds, labels, metrics = evaluate_model(model, test_dataset)
    
    # Print metrics
    print("\nEvaluation Results for MobileBERT:")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"F1 Score (weighted): {metrics['f1_weighted']:.4f}")
    print(f"Precision (weighted): {metrics['precision_weighted']:.4f}")
    print(f"Recall (weighted): {metrics['recall_weighted']:.4f}")
    print(f"F1 Score by class: Low={metrics['f1_low_risk']:.4f}, Medium={metrics['f1_medium_risk']:.4f}, High={metrics['f1_high_risk']:.4f}")
    print(f"ROC AUC Score: {metrics['roc_auc']:.4f}")
    print(f"Evaluation time: {metrics['eval_time_seconds']:.2f} seconds")
    print(f"Memory usage: {metrics['ram_usage_mb']:.2f} MB")
    
    # Plot confusion matrix
    cm_path = os.path.join(args.output_dir, "mobilebert_confusion_matrix.png")
    plot_confusion_matrix(metrics['confusion_matrix'], save_path=cm_path)
    
    # Convert test dataset to DataFrame for analysis
    test_df = test_dataset.to_pandas()
    
    # Analyze errors
    analyze_errors(test_df, preds)
    
    # Evaluate performance by category
    evaluate_performance_by_category(test_df, preds)
    
    # Compare against benchmarks
    benchmark_comparison(metrics)
    
    # Save detailed results to CSV
    results_path = os.path.join(args.output_dir, "mobilebert_evaluation_metrics.csv")
    metrics_df = pd.DataFrame({k: [v] if not isinstance(v, np.ndarray) else [str(v)] 
                              for k, v in metrics.items()})
    metrics_df.to_csv(results_path, index=False)
    print(f"\nDetailed metrics saved to {results_path}")


if __name__ == "__main__":
    main()