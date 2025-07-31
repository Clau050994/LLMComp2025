import os
import torch
import json
import numpy
from datasets import Dataset
from sklearn.model_selection import train_test_split
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    TrainerCallback
)

# ––– Patch for PyTorch 2.6+ to avoid UnpicklingError –––
import torch.serialization
import torch.serialization
torch.serialization.add_safe_globals([
    numpy._core.multiarray._reconstruct,
    numpy.ndarray,
    numpy.dtype
])
torch.serialization.add_safe_globals([numpy.ndarray])


# ––– Environment & GPU –––
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
assert torch.cuda.is_available(), "🚫 GPU not available"
print("✅ GPU:", torch.cuda.get_device_name(0))

# ––– Logging callback –––
class LossLoggingCallback(TrainerCallback):
    def on_epoch_end(self, args, state, control, **kwargs):
        for entry in reversed(state.log_history):
            if 'loss' in entry:
                print(f"📉 Avg Train Loss (Epoch {int(state.epoch)}): {entry['loss']:.4f}")
                break

# ––– Load & prepare dataset –––
def label_from_response(text):
    return 1 if text.lower().startswith("yes") else 0

def load_jsonl(path):
    data = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            j = json.loads(line)
            prompt = f"Task: {j.get('task_type','triage_verification')}\nInput: {j['input']}"
            data.append({"prompt": prompt, "label": label_from_response(j["response"])})
    return data

raw = load_jsonl("verification_augmented_with_license_id.jsonl")
train, val = train_test_split(raw, test_size=0.2, random_state=42, stratify=[d["label"] for d in raw])
train_ds, val_ds = Dataset.from_list(train), Dataset.from_list(val)

# ––– Load model & tokenizer –––
repo = "mtgv/MobileLLaMA-1.4B-Chat"
tokenizer = AutoTokenizer.from_pretrained(repo)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForSequenceClassification.from_pretrained(repo, num_labels=2).to("cuda")
model.config.pad_token_id = tokenizer.pad_token_id

# ––– Tokenization –––
def tokenize_fn(batch):
    return tokenizer(batch["prompt"], truncation=True, padding="max_length", max_length=128)

train_ds = train_ds.map(tokenize_fn, batched=True).rename_column("label", "labels")
val_ds = val_ds.map(tokenize_fn, batched=True).rename_column("label", "labels")

train_ds.set_format("torch", columns=["input_ids", "attention_mask", "labels"])
val_ds.set_format("torch", columns=["input_ids", "attention_mask", "labels"])

# ––– Training args –––
training_args = TrainingArguments(
    output_dir="./mobilellama_verification_finetuned",
    evaluation_strategy="no",
    save_strategy="epoch",
    save_total_limit=1,
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    num_train_epochs=5,
    weight_decay=0.01,
    logging_steps=50,
    load_best_model_at_end=False,
    report_to="none",
    warmup_steps=100,
    gradient_accumulation_steps=1,
    fp16=True,
)

# ––– Trainer setup –––
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    tokenizer=tokenizer,
    callbacks=[LossLoggingCallback()]
)

# ––– Training –––
resume_path = "./mobilellama_verification_finetuned/checkpoint-12002"
print("🚀 Starting MobileLLaMA training…")

if os.path.exists(resume_path):
    trainer.train(resume_from_checkpoint=resume_path)
else:
    print("⚠️ No valid checkpoint found — starting from scratch.")
    trainer.train()

# ––– Save final model –––
print("✅ Training complete. Saving final model…")
trainer.save_model(training_args.output_dir)
tokenizer.save_pretrained(training_args.output_dir)
