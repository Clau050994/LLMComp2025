import json
import random

input_path = "verification_augmented_with_license_id.jsonl"
output_path = "verification_augmented_with_diverse_noise.jsonl"

def random_typo(word):
    if len(word) < 4:
        return word
    idx = random.randint(1, len(word)-2)
    return word[:idx] + "*" + word[idx+1:]

def corrupt_text(text):
    text = text.strip()
    # Decide randomly which corruptions to apply
    corruptions = []

    if random.random() < 0.3:
        corruptions.append("lowercase")

    if random.random() < 0.2:
        corruptions.append("ocr")

    if random.random() < 0.2:
        corruptions.append("truncate")

    if random.random() < 0.2:
        corruptions.append("typo")

    if random.random() < 0.2:
        corruptions.append("repeat")

    # Apply selected corruptions
    if "lowercase" in corruptions:
        text = text.lower()

    if "ocr" in corruptions:
        text = text.replace("o", "0").replace("i", "1").replace("e", "3")

    if "truncate" in corruptions and len(text) > 40:
        cutoff = random.randint(25, 40)
        text = text[:cutoff]

    if "typo" in corruptions:
        words = text.split()
        if words:
            idx = random.randint(0, len(words)-1)
            words[idx] = random_typo(words[idx])
            text = " ".join(words)

    if "repeat" in corruptions:
        words = text.split()
        if len(words) > 2:
            idx = random.randint(0, len(words)-2)
            words.insert(idx, words[idx])
            text = " ".join(words)

    return text

with open(input_path, "r", encoding="utf-8") as f_in, \
     open(output_path, "w", encoding="utf-8") as f_out:

    for line in f_in:
        obj = json.loads(line)
        original_text = obj["input"]
        noisy_text = corrupt_text(original_text)
        obj["input"] = noisy_text
        f_out.write(json.dumps(obj, ensure_ascii=False) + "\n")

print(f"✅ Saved diversified noisy dataset to {output_path}")
