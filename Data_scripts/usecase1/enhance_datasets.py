import json
from datetime import datetime

# === Configuration ===
input_path = "verification_augmented_with_license_id.jsonl"
output_path = "verification_enriched_for_training.jsonl"

# === Helper function: parse dates safely ===
def parse_date_safe(date_str):
    """
    Try to parse a date string in common formats.
    Returns a datetime object or None if parsing fails.
    """
    formats = [
        "%Y-%m-%d",
        "%d/%m/%Y",
        "%m/%d/%Y",
        "%Y/%m/%d",
        "%d-%m-%Y",
    ]
    for fmt in formats:
        try:
            return datetime.strptime(date_str.strip(), fmt)
        except Exception:
            continue
    return None

# === Processing ===
today = datetime.today()
cleaned_records = []

with open(input_path, "r", encoding="utf-8") as infile:
    for idx, line in enumerate(infile, 1):
        record = json.loads(line)

        # Default expired flag
        record["expired"] = False
        expired_flag = "Unknown"

        # Parse expiration date
        expiration_date_raw = record.get("expiration_date")
        if expiration_date_raw:
            expiration_dt = parse_date_safe(expiration_date_raw)
            if expiration_dt:
                if expiration_dt < today:
                    record["expired"] = True
                    expired_flag = "True"
                else:
                    expired_flag = "False"

        # Safely extract other fields
        holder_name = record.get("holder_name", "Unknown Name")
        doc_type = record.get("document_type", "Unknown Document")
        issue_date = record.get("issuance_date", "Unknown")
        expire_date = expiration_date_raw or "Unknown"
        country = record.get("issuing_country", "Unknown Country")

        # Compose enriched text input
        input_text = (
            f"Document Type: {doc_type}. "
            f"Issuing Country: {country}. "
            f"Holder Name: {holder_name}. "
            f"Issued: {issue_date}. "
            f"Expires: {expire_date}. "
            f"Expired: {expired_flag}."
        )

        # Add input_text to record
        record["input_text"] = input_text

        cleaned_records.append(record)

        if idx % 1000 == 0:
            print(f"Processed {idx} records...")

# === Save cleaned dataset ===
with open(output_path, "w", encoding="utf-8") as outfile:
    for rec in cleaned_records:
        json.dump(rec, outfile)
        outfile.write("\n")

print()
print(f"✅ Finished processing {len(cleaned_records)} records.")
print(f"✅ Output saved to '{output_path}'.")
