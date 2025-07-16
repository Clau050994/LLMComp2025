import pandas as pd
import os
import json

# === 1. Load your raw HTS dataset ===
raw_csv_path = "/aul/homes/cvaro009/Desktop/LLMComp2025/data/raw/usecase_4/hts_2025_revision_13_csv.csv"
df = pd.read_csv(raw_csv_path)

# === 2. Extract 6-digit HS code ===
df["hs6"] = df["HTS Number"].astype(str).str.replace(".", "", regex=False).str[:6]

# === 3. Keep only valid rows (non-null hs6 + Description) ===
df = df[~df["hs6"].isna() & ~df["Description"].isna()]

# === 4. Normalize values ===
df["hs6"] = df["hs6"].str.strip()
df["Description"] = df["Description"].str.strip().str.rstrip(":")

# === 5. Remove vague or generic descriptions ===
# Lowercase match to filter known non-informative values
bad_terms = ["other", "male", "female", "males", "females", "dairy", "horses", "cattle"]
df = df[~df["Description"].str.lower().isin(bad_terms)]
df = df[~df["Description"].str.lower().str.startswith("weighing")]

# === 6. Only keep valid 6-digit codes ===
df = df[df["hs6"].str.match(r"^\d{6}$")]

# === 7. Deduplicate by HS6 code, keep first meaningful description ===
df = df.drop_duplicates(subset="hs6", keep="first")

# === 8. Save cleaned mapping ===
output_dir = "/aul/homes/cvaro009/Desktop/LLMComp2025/data/processed/usecase4"
os.makedirs(output_dir, exist_ok=True)

csv_path = os.path.join(output_dir, "hs_code_map.csv")
json_path = os.path.join(output_dir, "hs_code_map.json")

df.to_csv(csv_path, index=False)
df.set_index("hs6")["Description"].to_json(json_path, orient="index", indent=2)

print(f"✅ Cleaned HS6 mapping saved to:\n📄 {csv_path}\n📄 {json_path}")

