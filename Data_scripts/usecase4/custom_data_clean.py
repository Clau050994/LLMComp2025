import pandas as pd
import os
import json

# === Paths ===
input_dir = "/aul/homes/cvaro009/Desktop/LLMComp2025/data/raw/usecase_4"
output_dir = "/aul/homes/cvaro009/Desktop/LLMComp2025/data/processed/usecase4"
hs_map_path = os.path.join(output_dir, "hs_code_map.json")
os.makedirs(output_dir, exist_ok=True)

# === Load HS6 mapping ===
with open(hs_map_path, "r") as f:
    hs_map = json.load(f)

# === Processing Function ===
def clean_and_process(df):
    df = df[["HS6 Code", "Item Price", "Net Mass", "Country of Origin"]].copy()
    df.columns = ["hs_code", "price", "weight", "origin"]
    df.dropna(subset=["hs_code", "price", "weight", "origin"], inplace=True)

    # Normalize and filter
    df["hs_code"] = df["hs_code"].astype(str).str.zfill(6)
    df["price"] = pd.to_numeric(df["price"], errors="coerce")
    df["weight"] = pd.to_numeric(df["weight"], errors="coerce")
    df = df[
        (df["price"] > 0) & (df["price"] < 1e7) &
        (df["weight"] > 0) & (df["weight"] < 1e5)
    ]

    # Use mapped description if available
    df["desc"] = df["hs_code"].map(hs_map).fillna("Unknown item")

    # Generate input/output
    df["input"] = df.apply(
        lambda row: f"""Item: {row['desc']}\nQuantity: {row['weight']} kg\nDeclared Value: {row['price']} KRW\nCountry of Origin: {row['origin']}""",
        axis=1
    )

    df["output"] = df.apply(
        lambda row: f"This item ({row['desc']}) from {row['origin']} may be subject to import duty based on its value and type.",
        axis=1
    )

    return df[["input", "output"]]

# === File Map ===
file_map = {
    "train": "df_syn_train_eng.csv",
    "validation": "df_syn_valid_eng.csv",
    "test": "df_syn_test_eng.csv"
}

# === Process All Splits ===
for split_name, file_name in file_map.items():
    input_path = os.path.join(input_dir, file_name)
    output_path = os.path.join(output_dir, f"{split_name}.jsonl")

    print(f"🔄 Processing {input_path}...")
    df_raw = pd.read_csv(input_path)
    df_clean = clean_and_process(df_raw)
    df_clean.to_json(output_path, orient="records", lines=True, force_ascii=False)
    print(f"✅ Saved: {output_path}")

print("\n🎉 All datasets processed and saved to:", output_dir)
