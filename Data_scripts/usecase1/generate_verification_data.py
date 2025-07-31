import pandas as pd
import json
import random
from faker import Faker
from tqdm import tqdm

faker = Faker()
random.seed(42)

# === CONFIGURATION ===
INPUT_CSV_PATH = "train.csv"  # path to your PII-NER CSV
OUTPUT_JSONL_PATH = "verification_augmented_with_license_id.jsonl"

# === COUNTRY-CODE MAPPING ===
region_country_map = {
    "IN": "+91", "US": "+1", "FR": "+33", "UK": "+44", "DE": "+49", "XX": None
}

# === CARD TYPES AND LENGTHS ===
card_issuers = [
    ("VISA 13 digit", 13), ("VISA 16 digit", 16), ("VISA 19 digit", 19),
    ("Mastercard", 16), ("American Express", 15), ("Discover", 16),
    ("JCB 15 digit", 15), ("JCB 16 digit", 16), ("Diners Club / Carte Blanche", 14),
    ("Maestro", 12)
]

# === GENERATOR FUNCTION ===
def generate_verification_sample(name_hint=None):
    name = name_hint or faker.name()
    card_label, digits = random.choice(card_issuers)
    cardholder = faker.name()
    card_number = ''.join(random.choices("0123456789", k=digits))
    expiry = f"{random.randint(1,12):02d}/{random.randint(24,34)}"
    email = faker.email()
    ssn = faker.ssn()
    
    # Phone and region logic
    phone_cc = random.choice(list(region_country_map.values()))
    if phone_cc is None:
        phone_cc = f"+{random.randint(2,999)}"
    phone = f"{phone_cc} {random.randint(1000000000,9999999999)}"
    license_region = random.choice(list(region_country_map.keys()))
    license_id = f"{license_region}-{random.randint(1000000,9999999)}"

    # Security code
    security_code = (
        f"CID: {random.randint(1000, 9999)}"
        if "American Express" in card_label
        else f"CVC: {random.randint(100, 999)}"
    )

    input_text = (
        f"Name: {name}\n"
        f"Card Info: {card_label}\n{cardholder}\n{card_number} {expiry}\n"
        f"{security_code}\n"
        f"Email: {email}\n"
        f"SSN: {ssn}\n"
        f"Phone: {phone}\n"
        f"License ID: {license_id}"
    )

    # === LABEL LOGIC ===
    if name != cardholder:
        response = "No. Cardholder name does not match Name field."
    elif region_country_map.get(license_region) and region_country_map[license_region] != phone_cc:
        response = "No. License ID region does not match the phone number's country code."
    else:
        response = "Yes. All fields appear consistent and valid."

    return {
        "task_type": "triage_verification",
        "input": input_text,
        "response": response
    }

# === MAIN SCRIPT ===
def main():
    df = pd.read_csv(INPUT_CSV_PATH)
    output_data = []

    for _, row in tqdm(df.iterrows(), total=len(df)):
        try:
            assistant_json = json.loads(row["assistant"])
            name = assistant_json.get("NAME", [None])[0]
        except Exception:
            name = None

        record = generate_verification_sample(name_hint=name)
        output_data.append(record)

    with open(OUTPUT_JSONL_PATH, "w") as f:
        for entry in output_data:
            f.write(json.dumps(entry) + "\n")

    print(f"✅ Successfully created {len(output_data)} records at: {OUTPUT_JSONL_PATH}")

if __name__ == "__main__":
    main()
