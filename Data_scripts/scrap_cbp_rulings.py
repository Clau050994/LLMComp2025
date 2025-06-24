import pandas as pd
import requests
import os
import re
from PyPDF2 import PdfReader

# Paths
RAW_PATH = os.path.expanduser("~/Desktop/LLMComp2025/data/raw/latest_rulings.csv")
PROCESSED_PATH = os.path.expanduser("~/Desktop/LLMComp2025/data/processed/cbp_rulings.csv")
PDF_DIR = os.path.expanduser("~/Desktop/LLMComp2025/data/tmp_pdfs")
os.makedirs(PDF_DIR, exist_ok=True)

# Clean header from PDF
def clean_pdf_text(text):
    lines = text.splitlines()
    lines = [line.strip() for line in lines if line.strip()]
    start_keywords = ["RE:", "ISSUE:", "FACTS:", "DISCUSSION:", "ANALYSIS:", "LAW AND ANALYSIS:"]
    start_index = 0
    for i, line in enumerate(lines):
        if any(keyword in line for keyword in start_keywords):
            start_index = i
            break
    return "\n".join(lines[start_index:]).strip()

# Extract structured sections
def extract_sections(text):
    sections = {
        "issue": "",
        "facts": "",
        "analysis": "",
        "conclusion": ""
    }
    patterns = {
        "issue": r"ISSUE:(.*?)(?=FACTS:|DISCUSSION:|ANALYSIS:|LAW AND ANALYSIS:|CONCLUSION:|$)",
        "facts": r"FACTS:(.*?)(?=DISCUSSION:|ANALYSIS:|LAW AND ANALYSIS:|CONCLUSION:|$)",
        "analysis": r"(?:DISCUSSION:|ANALYSIS:|LAW AND ANALYSIS:)(.*?)(?=CONCLUSION:|$)",
        "conclusion": r"CONCLUSION:(.*)"
    }
    for key, pattern in patterns.items():
        match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
        if match:
            sections[key] = match.group(1).strip()
    return sections

# Download PDF and extract cleaned + structured text
def download_and_extract_text(url, ruling_number):
    try:
        pdf_path = os.path.join(PDF_DIR, f"{ruling_number}.pdf")
        response = requests.get(url, timeout=15)
        with open(pdf_path, "wb") as f:
            f.write(response.content)

        reader = PdfReader(pdf_path)
        raw_text = "\n".join([page.extract_text() or "" for page in reader.pages])
        cleaned_text = clean_pdf_text(raw_text)
        return cleaned_text if cleaned_text else "[Empty after cleanup]"
    except Exception as e:
        return f"Error extracting PDF: {str(e)}"

# Main
def main():
    df = pd.read_csv(RAW_PATH)
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")

    if "url" not in df.columns or "ruling_number" not in df.columns:
        print("❌ Found columns:", df.columns.tolist())
        raise ValueError("CSV must have 'url' and 'ruling_number' columns.")

    print(f"🔍 Processing {len(df)} rulings...")

    # Get full cleaned text
    df["ruling_full_text"] = df.apply(
        lambda row: download_and_extract_text(str(row["url"]).strip(), str(row["ruling_number"])), axis=1
    )

    # Extract structured sections
    section_data = df["ruling_full_text"].apply(extract_sections).apply(pd.Series)
    df = pd.concat([df, section_data], axis=1)

    df.drop(columns=["url", "collection", "date_modified"], errors="ignore", inplace=True)
    df.to_csv(PROCESSED_PATH, index=False)
    print(f"✅ Full rulings with structured sections saved to: {PROCESSED_PATH}")

if __name__ == "__main__":
    main()
