#!/usr/bin/env python3
"""
Phi-3-mini fine-tuning script with QLoRA (4-bit quantization), a custom training loop, and test set evaluation with accuracy and F1.
All file paths are hardcoded; other hyperparameters can be changed at the top.
"""

import os
import torch
import pandas as pd
import numpy as np
from tqdm.auto import tqdm
from datetime import datetime
from sklearn.metrics import accuracy_score, f1_score

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    get_linear_schedule_with_warmup
)

from peft import (
    get_peft_model,
    LoraConfig,
    prepare_model_for_kbit_training,
    TaskType
)
from flash_attn import flash_attn_func
print("FlashAttention is ready to use!")


# Hardcoded paths and hyperparameters
TRAIN_FILE = "/aul/homes/cvaro009/Desktop/LLMComp2025/data/processed/unified/unified_train.csv"
VAL_FILE = "/aul/homes/cvaro009/Desktop/LLMComp2025/data/processed/unified/unified_val.csv"
TEST_FILE = "/aul/homes/cvaro009/Desktop/LLMComp2025/data/processed/unified/unified_test.csv"
OUTPUT_DIR = "/disk/diamond-scratch/cvaro009/data/phi3"
MODEL_ID = "microsoft/Phi-3-mini-128k-instruct"
NUM_EPOCHS = 3
LEARNING_RATE = 2e-4
BATCH_SIZE = 2
GRADIENT_ACCUMULATION = 8
MAX_LENGTH = 512
LORA_R = 8
LORA_ALPHA = 16
LORA_DROPOUT = 0.05
ATTN_IMPLEMENTATION = "eager"

def format_instruction(text, label):
    risk_level = ["Low Risk", "Medium Risk", "High Risk"][int(label)]
    return (
        "<|user|>\n"
        f"Assess the risk level of the following traveler scenario:\n"
        f"{text}\n"
        "Classify as Low Risk, Medium Risk, or High Risk.\n"
        "<|assistant|>\n"
        f"Based on the information provided, I classify this traveler as {risk_level}."
    )

def process_dataset(df, tokenizer, max_length):
    if 'text' in df.columns and 'input_text' not in df.columns:
        df['input_text'] = df['text']
    formatted_texts = [format_instruction(row['input_text'], row['label']) for _, row in df.iterrows()]
    encodings = []
    for text in tqdm(formatted_texts, desc="Tokenizing"):
        encoded = tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=max_length,
            return_tensors="pt"
        )
        input_ids = encoded['input_ids'][0]
        eos_positions = (input_ids == tokenizer.eos_token_id).nonzero(as_tuple=True)[0]
        if len(eos_positions) > 0:
            human_end_pos = eos_positions[0].item() + 1
        else:
            human_end_pos = len(input_ids) // 2
        labels = input_ids.clone()
        labels[:human_end_pos] = -100
        encodings.append({
            'input_ids': encoded['input_ids'][0],
            'attention_mask': encoded['attention_mask'][0],
            'labels': labels
        })
    return encodings

class TextDataset(torch.utils.data.Dataset):
    def __init__(self, encodings):
        self.encodings = encodings
    def __len__(self):
        return len(self.encodings)
    def __getitem__(self, idx):
        return self.encodings[idx]

def custom_collate_fn(batch):
    return {
        'input_ids': torch.stack([item['input_ids'] for item in batch]),
        'attention_mask': torch.stack([item['attention_mask'] for item in batch]),
        'labels': torch.stack([item['labels'] for item in batch]),
    }

def evaluate_with_metrics(model, data_loader, tokenizer, device):
    model.eval()
    losses, preds, labels = [], [], []
    for batch in tqdm(data_loader, desc="Evaluating"):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        label_ids = batch["labels"].to(device)

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=label_ids
        )
        losses.append(outputs.loss.item())

        # Generate predictions
        gen_ids = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=30,
            pad_token_id=tokenizer.eos_token_id
        )
        decoded_preds = tokenizer.batch_decode(gen_ids, skip_special_tokens=True)
        for pred_text, label_id in zip(decoded_preds, label_ids):
            label_tokens = label_id[label_id != -100]
            if len(label_tokens) > 0:
                label = label_tokens[0].item()
            else:
                label = -1
            labels.append(label)
            if "High Risk" in pred_text:
                preds.append(2)
            elif "Medium Risk" in pred_text:
                preds.append(1)
            elif "Low Risk" in pred_text:
                preds.append(0)
            else:
                preds.append(-1)  # For failed cases

    model.train()
    valid_preds = [p for p in preds if p in [0, 1, 2]]
    valid_labels = [l for p, l in zip(preds, labels) if p in [0, 1, 2]]

    acc = accuracy_score(valid_labels, valid_preds) if valid_labels else 0.0
    f1 = f1_score(valid_labels, valid_preds, average="macro") if valid_labels else 0.0
    return {
        "loss": np.mean(losses),
        "accuracy": acc,
        "f1": f1
    }

def train_loop(model, train_loader, val_loader, optimizer, scheduler, device, num_epochs, gradient_accumulation, output_dir, tokenizer):
    global_step = 0
    model.train()
    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")
        train_losses = []
        for step, batch in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}")):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            loss = outputs.loss / gradient_accumulation
            loss.backward()
            train_losses.append(loss.item() * gradient_accumulation)
            if (step + 1) % gradient_accumulation == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.3)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1
            if (step + 1) % 100 == 0:
                print(f"Step {step+1}, Loss: {np.mean(train_losses[-100:]):.4f}")
        # --- Validation with metrics ---
        metrics = evaluate_with_metrics(model, val_loader, tokenizer, device)
        print(f"Val Loss: {metrics['loss']:.4f}, Accuracy: {metrics['accuracy']:.4f}, F1: {metrics['f1']:.4f}")
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        checkpoint_dir = os.path.join(output_dir, f"checkpoint-epoch-{epoch+1}-{timestamp}")
        os.makedirs(checkpoint_dir, exist_ok=True)
        model.save_pretrained(checkpoint_dir)
        print(f"Saved checkpoint for epoch {epoch+1} to {checkpoint_dir}")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    # Save final model
    final_dir = os.path.join(output_dir, "final")
    os.makedirs(final_dir, exist_ok=True)
    model.save_pretrained(final_dir)
    print(f"Training complete! Model saved to {final_dir}")

if __name__ == "__main__":
    print("=" * 80)
    print("Optimized Phi-3-mini (128K context) Training with QLoRA (4-bit)")
    print("=" * 80)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"Output will be saved to: {OUTPUT_DIR}")

    # QLoRA: 4-bit quantization config
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    ATTN_IMPLEMENTATION = "flash_attention_2"
    print("Loading Phi-3-mini model and tokenizer...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=quantization_config,
        attn_implementation=ATTN_IMPLEMENTATION,
        device_map="auto",
        trust_remote_code=True
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # QLoRA preparation
    model = prepare_model_for_kbit_training(model)
    lora_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    )
    model.gradient_checkpointing_enable()
    model = get_peft_model(model, lora_config)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Percentage of parameters being trained: {trainable_params/total_params*100:.2f}%")

    print(f"Loading training data from: {TRAIN_FILE}")
    print(f"Loading validation data from: {VAL_FILE}")
    train_df = pd.read_csv(TRAIN_FILE)
    val_df = pd.read_csv(VAL_FILE)
    print(f"Training examples: {len(train_df)}")
    print(f"Validation examples: {len(val_df)}")
    print("Processing training data...")
    train_encodings = process_dataset(train_df, tokenizer, MAX_LENGTH)
    print("Processing validation data...")
    val_encodings = process_dataset(val_df, tokenizer, MAX_LENGTH)
    train_dataset = TextDataset(train_encodings)
    val_dataset = TextDataset(val_encodings)
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=custom_collate_fn
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=custom_collate_fn
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    total_steps = len(train_loader) * NUM_EPOCHS // GRADIENT_ACCUMULATION
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.03 * total_steps),
        num_training_steps=total_steps
    )

    train_loop(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        device=model.device,
        num_epochs=NUM_EPOCHS,
        gradient_accumulation=GRADIENT_ACCUMULATION,
        output_dir=OUTPUT_DIR,
        tokenizer=tokenizer
    )

    # --- TEST SET EVALUATION ---
    print(f"Loading test data from: {TEST_FILE}")
    test_df = pd.read_csv(TEST_FILE)
    print(f"Test examples: {len(test_df)}")
    print("Processing test data...")
    test_encodings = process_dataset(test_df, tokenizer, MAX_LENGTH)
    test_dataset = TextDataset(test_encodings)
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=custom_collate_fn
    )
    print("Evaluating on test set with full metrics...")
    test_metrics = evaluate_with_metrics(model, test_loader, tokenizer, model.device)
    print(f"\nTest Loss: {test_metrics['loss']:.4f}")
    print(f"Test Accuracy: {test_metrics['accuracy']:.4f}")
    print(f"Test F1 Score: {test_metrics['f1']:.4f}")