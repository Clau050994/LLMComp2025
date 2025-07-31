import torch
# from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
from transformers import MobileBertTokenizerFast, MobileBertForSequenceClassification


# Load trained model and tokenizer (distilbert)
# model_path = "./distilbert_verification_finetuned"
# tokenizer = DistilBertTokenizerFast.from_pretrained(model_path)
# model = DistilBertForSequenceClassification.from_pretrained(model_path).to("cuda")
# model.eval()

#mobilebert
model_path = "./mobilebert_verification_finetuned"
tokenizer = MobileBertTokenizerFast.from_pretrained(model_path)
model = MobileBertForSequenceClassification.from_pretrained(model_path).to("cuda")
model.eval()

def predict(prompt):
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, padding="max_length", max_length=128).to("cuda")
    with torch.no_grad():
        outputs = model(**inputs)
        pred = torch.argmax(outputs.logits, dim=1).item()
    return pred

# Loop for real-time input
while True:
    print("\n🔍 Enter identity info (type 'exit' to quit):")
    name = input("Name: ")
    if name.lower() == "exit":
        break
    passport = input("Passport: ")
    email = input("Email: ")
    phone = input("Phone: ")
    ssn = input("SSN: ")

    # Build natural language prompt (NO address)
    prompt = (
        f"The traveler's name is {name}, their passport number is {passport}, "
        f"email is {email}, phone number is {phone}, and their SSN is {ssn}. "
        f"Is this a valid identity?"
    )

    prediction = predict(prompt)
    label = "✅ Valid" if prediction == 1 else "❌ Invalid"
    print(f"\nPrediction: {label}")
    print(f"Prompt: {prompt}")
    print("-" * 60)
