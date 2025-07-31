import pandas as pd

# Load the 50k subset
df = pd.read_csv("/aul/homes/melsh008/temp-storage-space-on-diamond/second_case_scenario/jw300_subset_50k.csv")

# Count unique languages
unique_langs = df['language'].unique()
print(f"✅ Number of unique languages: {len(unique_langs)}")
print(f"🗣️ Languages included: {sorted(unique_langs)}")
