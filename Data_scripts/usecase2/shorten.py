import pandas as pd

# === CONFIG ===
INPUT_FILE = "/aul/homes/melsh008/temp-storage-space-on-diamond/second_case_scenario/jw300_clean_shuffled.csv"
OUTPUT_FILE = "/aul/homes/melsh008/temp-storage-space-on-diamond/second_case_scenario/jw300_subset_50k.csv"
NUM_ROWS = 50000

# === Load dataset ===
df = pd.read_csv(INPUT_FILE)

# === Slice first 50,000 rows ===
df_subset = df.head(NUM_ROWS)

# === Save the subset ===
df_subset.to_csv(OUTPUT_FILE, index=False, encoding="utf-8")

print(f"✅ Saved first {NUM_ROWS} rows to '{OUTPUT_FILE}'")
