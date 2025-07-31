import re
import pandas as pd

# === CONFIG ===
INPUT_FILE = "localization_translations_expanded.csv"
OUTPUT_FILE = "localization_translations_expanded_cleaned.csv"

# === Load the CSV ===
df = pd.read_csv(INPUT_FILE, low_memory=True, encoding="utf-8")



def clean_translation(text, en_text):
    if pd.isna(text):
        return ""
    if pd.isna(en_text):
        en_text = ""
    text_str = str(text).strip()
    en_str = str(en_text).strip()
    # If the translation is exactly the same as the English, remove it
    if en_str and text_str == en_str:
        return ""
    
    # Remove suspicious repeated punctuation
    text_str = text_str.replace(". . .", "").replace("..", ".").strip()
    
    # Remove trailing clutter (example: many repeated dots)
    while text_str.endswith("."):
        text_str = text_str[:-1].strip()
    

    if "InfoFinland" in text_str or "www.watchtool" in text_str or "Pornografigg" in text_str:
        return ""
    
    return text_str


# === Process each language column ===
language_columns = [col for col in df.columns if col not in ["id", "en"]]

for lang in language_columns:
    print(f"Cleaning column: {lang}")
    cleaned = []
    for i, row in df.iterrows():
        cleaned_text = clean_translation(row[lang], row["en"])
        cleaned.append(cleaned_text)
    df[lang] = cleaned

# === Save cleaned CSV ===
df.to_csv(OUTPUT_FILE, index=False, encoding="utf-8")
print(f"\n Cleaning complete. Saved cleaned file to '{OUTPUT_FILE}'")
