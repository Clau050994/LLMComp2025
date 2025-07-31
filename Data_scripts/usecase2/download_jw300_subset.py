from datasets import load_dataset, get_dataset_config_names
import csv
import os

# ✅ Set dataset cache to your big disk folder
custom_cache_dir = "/aul/homes/melsh008/temp-storage-space-on-diamond/second_case_scenario/hf_cache"
os.makedirs(custom_cache_dir, exist_ok=True)

# ✅ Automatically get all available language configs
available_configs = get_dataset_config_names(
    "sentence-transformers/parallel-sentences-jw300",
    cache_dir=custom_cache_dir
)

# If you want to skip some languages, list them here
exclude_languages = {"en-zh"}  # (not available anyway)
language_configs = [
    c for c in available_configs
    if "-" in c and c not in exclude_languages
]

print(f"✅ Found {len(language_configs)} configs:")
print(language_configs)

# ✅ Max examples per language
examples_per_language = 10_000

subset = []

print("Downloading...")
for config in language_configs:
    lang_code = config.split("-")[1]
    print(f"🔹 Downloading {config}...")

    ds = load_dataset(
        "sentence-transformers/parallel-sentences-jw300",
        config,
        split="train",
        cache_dir=custom_cache_dir  # ✅ Use the custom cache dir for downloads
    )

    count = 0
    for row in ds:
        if count >= examples_per_language:
            break
        
        # ✅ Clean the text
        english = (row["english"] or "").strip()
        non_english = (row["non_english"] or "").strip()

        # ✅ Skip rows with empty sentences
        if not english or not non_english:
            continue

        subset.append({
            "english": english,
            "non_english": non_english,
            "language": lang_code
        })
        count += 1

    print(f"✅ {config}: collected {count} examples.")

print("Downloaded total examples:", len(subset))

# ✅ Output directory
output_dir = "/aul/homes/melsh008/temp-storage-space-on-diamond/second_case_scenario"
os.makedirs(output_dir, exist_ok=True)

# ✅ Save to CSV in specified location
csv_filename = os.path.join(output_dir, "jw300_multi_language_large.csv")
with open(csv_filename, "w", encoding="utf-8", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["english", "translation", "language"])
    for row in subset:
        writer.writerow([row["english"], row["non_english"], row["language"]])

print(f"✅ Done. Saved to {csv_filename}.")
