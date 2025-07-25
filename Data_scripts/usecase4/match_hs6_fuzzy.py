import json
from rapidfuzz import process
from pathlib import Path
import os

# 📍 Paths
root_dir = Path("/aul/homes/cvaro009/Desktop/LLMComp2025")
data_dir = root_dir / "data" / "processed" / "usecase4"
hs6_map_path = data_dir / "hs_code_map.json"
output_dir = data_dir / "generated"
output_dir.mkdir(parents=True, exist_ok=True)

# 📂 Input JSONs
input_files = {
    "train": data_dir / "train.jsonl",
    "validation": data_dir / "validation.jsonl",
    "test": data_dir / "test.jsonl"
}

# 📥 Load HS6 code descriptions
with open(hs6_map_path, "r", encoding="utf-8") as f:
    hs6_map = json.load(f)

# 🔍 Fuzzy match function
def match_hs6(item_desc):
    result = process.extractOne(item_desc, hs6_map.values())
    if result:
        best_match, score = result[0], result[1]
        if score > 80:
            for code, desc in hs6_map.items():
                if desc == best_match:
                    return code, desc
    return None, "Unknown"


# 🔄 Process each file
for split, path in input_files.items():
    print(f"📂 Processing {path}...")

    if not path.exists():
        print(f"❌ File not found: {path}")
        continue

    with open(path, "r", encoding="utf-8") as f:
        data = [json.loads(line.strip()) for line in f if line.strip()]

    new_data = []
    for i, record in enumerate(data):
        try:
            if "input" not in record:
                raise ValueError(f"Missing 'input' field in record {i}")

            item_line = record["input"].split("\n")[0].strip()
            if not item_line.lower().startswith("item:"):
                raise ValueError(f"Malformed item line: '{item_line}'")

            item = item_line[len("Item:"):].strip()
            hs6_code, hs6_desc = match_hs6(item)
            record["hs6_code"] = hs6_code
            record["hs6_desc"] = hs6_desc
            new_data.append(record)

        except Exception as e:
            print(f"❌ Error processing record {i}: {e}")
            continue

    # ✍️ Save with HS6 match
    output_path = output_dir / f"{split}_with_hs6.json"
    with open(output_path, "w", encoding="utf-8") as f:
        for line in new_data:
            f.write(json.dumps(line, ensure_ascii=False) + "\n")

    print(f"✅ Saved enriched file: {output_path}")
