import os
import json
import torch
from datasets import Dataset
from sklearn.model_selection import train_test_split
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    BitsAndBytesConfig,
    DataCollatorForLanguageModeling,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import set_seed
from transformers import TrainerCallback

class MetricsCallback(TrainerCallback):
    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        print(f"Step {state.global_step} evaluation metrics: {metrics}")

set_seed(42)


os.environ["CUDA_VISIBLE_DEVICES"] = "0"
print("GPU:", torch.cuda.get_device_name(0))

# Load JSONL dataset
def load_jsonl(path):
    data = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            j = json.loads(line)
            prompt = (
                f"<|user|>\n"
                f"Task: {j.get('task_type','triage_verification')}\n"
                f"Input: {j['input']}\n"
                f"<|assistant|>\n{j['response']}"
            )
            data.append({"text": prompt})
    return data

raw = load_jsonl("/aul/homes/melsh008/First_Case_Scenario/verification_augmented_with_license_id.jsonl")
train_data, val_data = train_test_split(raw, test_size=0.2, random_state=42)
train_ds = Dataset.from_list(train_data)
val_ds = Dataset.from_list(val_data)

# tokenizer loader
model_name = "microsoft/phi-3-mini-4k-instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

# Load model in qlora
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)

base_model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    trust_remote_code=True,
    device_map="auto",
)

# training prep
base_model = prepare_model_for_kbit_training(base_model)

# This is the most likely correct target modules??? I think, maybe add more later
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["Wqkv", "o_proj", "q_proj", "v_proj", "k_proj", "gate_proj", "down_proj", "up_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)
model = get_peft_model(base_model, lora_config)

# Tokenization
def tokenize_fn(batch):
    return tokenizer(
        batch["text"],
        truncation=True,
        padding="max_length",
        max_length=512,
        return_attention_mask=True,   # <-- ADD THIS LINE
    )


train_ds = train_ds.map(tokenize_fn, batched=True)
val_ds = val_ds.map(tokenize_fn, batched=True)

train_ds.set_format("torch", columns=["input_ids", "attention_mask"])
val_ds.set_format("torch", columns=["input_ids", "attention_mask"])

collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# Training args
args = TrainingArguments(
    output_dir="./phi3_qlora_verification",
    learning_rate=2e-4,
    num_train_epochs=5,
    per_device_train_batch_size=8,
    gradient_accumulation_steps=4,
    fp16=True,
    gradient_checkpointing=True,  
    logging_steps=20,
    save_steps=500,
    eval_steps=500,
    save_total_limit=2,
    report_to="wandb",             # OPTIONAL: enable Wandb logging, if you want
)


# Trainer
trainer = Trainer(
    model=model,
    tokenizer=tokenizer,
    args=args,
    data_collator=collator,
    train_dataset=train_ds,
    eval_dataset=val_ds,
)

#training progress tracking
print("🚀 Starting QLoRA training...")
trainer.train()

# Save
print("Training complete. Saving LoRA adapters and tokenizer...")
model.save_pretrained("./phi3_qlora_verification")
tokenizer.save_pretrained("./phi3_qlora_verification")