import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

#MODEL_NAME =  Change as needed
MODEL_DIR = f"/disk/diamond-scratch/cvaro009/data/model/distilbert"

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
print("Loading model...")
model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
model.eval()
print("Ready!")

# --- Add this mapping ---
label_map = {0: "low", 1: "medium", 2: "high"}


def predict(text):
    inputs = tokenizer(
        text,
        padding="max_length",
        truncation=True,
        max_length=128,
        return_tensors="pt"
    )
    with torch.no_grad():
        outputs = model(**inputs)
        pred = torch.argmax(outputs.logits, dim=1).item()
    return pred


print("Type your input and press Enter (type 'exit' to quit):")
while True:
    user_input = input("You: ")
    if user_input.strip().lower() == "exit":
        break
    pred = predict(user_input)
    print(f"Model prediction: {pred}")