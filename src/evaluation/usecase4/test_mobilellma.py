#!/usr/bin/env python3
import json
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
from evaluate import load
import pandas as pd

# Paths
MODEL_DIR = "/disk/diamond-scratch/cvaro009/data/usecase4/mobilellama_summarization"
TEST_PATH = "/aul/homes/cvaro009/Desktop/LLMComp2025/data/processed/usecase4/test.jsonl"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, legacy=False)
tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token
tokenizer.pad_token_id = tokenizer.pad_token_id or tokenizer.eos_token_id

# Load model with 4-bit quantization
quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16
)
base_model = AutoModelForCausalLM.from_pretrained(
    "JackFram/llama-68m",
    device_map="auto",
    quantization_config=quant_config,
    trust_remote_code=True
)
model = PeftModel.from_pretrained(base_model, MODEL_DIR)
model.eval()

# Load ROUGE
rouge = load("rouge")

# Inference function
def generate_summary(input_text: str, max_new_tokens=128) -> str:
    prompt = f"""### Instruction:
Summarize the following customs declaration into plain language.

### Input:
{input_text}

### Response:
"""
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True).to(DEVICE)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id
        )
    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return decoded.split("### Response:")[-1].strip()

# Load test set
with open(TEST_PATH, "r") as f:
    test_data = [json.loads(line) for line in f]

# Run evaluation
predictions = []
references = []

for item in tqdm(test_data, desc="Evaluating"):
    input_text = item["input"]
    reference = item["output"]
    generated = generate_summary(input_text)
    predictions.append(generated)
    references.append(reference)

# Compute ROUGE
results = rouge.compute(predictions=predictions, references=references, use_stemmer=True)
print("\n📊 Evaluation Results:")
for key, value in results.items():
    print(f"{key}: {value.mid.fmeasure:.4f}")

# Optionally save
pd.DataFrame({
    "input": [d["input"] for d in test_data],
    "reference": references,
    "prediction": predictions
}).to_csv("mobilellama_eval_results.csv", index=False)
