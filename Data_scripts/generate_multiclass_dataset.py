import pandas as pd
import random
from tqdm import tqdm
import string

# Load country risk scores
risk_df = pd.read_csv("data/processed/country_risk_scores.csv")
country_tier = dict(zip(risk_df["country"], risk_df["risk_tier"]))

# Label logic
def label_multiclass(nat, orig, last, risk_map):
    risks = [risk_map.get(nat, "low"),
             risk_map.get(orig, "low"),
             risk_map.get(last, "low")]
    high, med = risks.count("high"), risks.count("medium")
    if high >= 2 or (high == 1 and med >= 1):
        return 2  # High Risk
    if med >= 1:
        return 1  # Medium Risk
    return 0      # Safe

# Generate natural language text with variations
def generate_text_with_variation(nat, orig, last):
    templates = [
        f"Passenger from {nat}, traveling from {orig}, last visited {last}.",
        f"Traveler with {nat} passport arriving from {orig} with recent travel to {last}.",
        f"Citizen of {nat} on flight from {orig}, previously stayed in {last}.",
        f"A {nat} national traveling via {orig} after visiting {last}.",
        f"{nat} traveler coming through {orig} airport having visited {last} recently.",
        f"Individual with {nat} citizenship coming from {orig} with prior stay in {last}.",
        f"Person holding {nat} passport, departed from {orig}, previously in {last}.",
        f"Visitor from {nat} entering via {orig} after spending time in {last}.",
        f"Tourist from {nat} arrived from {orig}, was in {last} before that.",
        f"{nat} resident who traveled from {orig} having visited {last} within past month.",
        # Additional templates for variety
        f"Entry scan: {nat} national, {orig} departure, {last} travel history.",
        f"Border control identified traveler from {nat} via {orig} with prior {last} visit.",
        f"{nat} passport holder arriving on flight from {orig}, declared visit to {last}.",
        f"Immigration records: {nat} citizen entering from {orig} after {last} visit.",
        f"Travel profile: {nat} origin, entering via {orig}, {last} in travel history."
    ]
    return random.choice(templates)

# Add noise to some examples to improve robustness - increased probability to 0.5
def add_noise(text, probability=0.5):
    if random.random() > probability:
        return text
        
    noise_types = [
        # Original noise types
        lambda t: t + " " + "".join(random.choices(string.ascii_lowercase + string.digits + "!@#", k=5)),
        lambda t: "".join(random.choices(string.ascii_lowercase, k=3)) + " " + t,
        lambda t: t.replace("from", "frm").replace("visiting", "visitng").replace("traveled", "travled"),
        lambda t: t.replace("a", "@").replace("e", "3").replace("i", "1").replace("o", "0"),
        lambda t: " ".join([w for w in t.split() if random.random() > 0.1]),  # randomly drop words
        
        # New, more aggressive noise types
        lambda t: " ".join([w for w in t.split() if random.random() > 0.2]),  # More aggressive word dropping
        lambda t: t.replace(" ", "").replace(".", "")[:len(t)//3] + t[len(t)//3:],  # Remove spaces in first third
        lambda t: "".join([c for c in t if random.random() > 0.1]),  # Randomly drop characters
        lambda t: t + " " + t[:20],  # Add partial repetition
        lambda t: t.replace("passport", "pasport").replace("national", "natnl").replace("citizen", "citizn")  # Domain-specific typos
    ]
    
    # Apply 1-2 noise functions for more variation
    if random.random() < 0.3:
        # Apply two noise functions
        noise_fn1 = random.choice(noise_types)
        noise_fn2 = random.choice(noise_types)
        return noise_fn2(noise_fn1(text))
    else:
        # Apply one noise function
        noise_fn = random.choice(noise_types)
        return noise_fn(text)

# Generate samples
countries = list(country_tier.keys())

# Split countries for true test/train separation in future
random.seed(42)  # For reproducibility
test_countries_set = set(random.sample(countries, int(len(countries) * 0.2)))
training_countries = [c for c in countries if c not in test_countries_set]

SAMPLE_SIZE = 50000  # Increased to 50,000
rows = []

print(f"Generating {SAMPLE_SIZE} training examples...")
for _ in tqdm(range(SAMPLE_SIZE)):
    # Use only training countries for 80% of data, all countries for 20%
    country_pool = training_countries if random.random() < 0.8 else countries
    
    nat, orig, last = random.sample(country_pool, 3)
    
    # Add slight variations to risk levels (5% chance)
    def vary_risk(risk_tier, variation_prob=0.05):
        if random.random() < variation_prob:
            if risk_tier == "high":
                return random.choice(["medium", "high"])
            elif risk_tier == "medium":
                return random.choice(["low", "high", "medium"])
            else:  # low
                return random.choice(["medium", "low"])
        return risk_tier
    
    # Get risk levels with occasional variation
    nat_risk = vary_risk(country_tier.get(nat, "low"))
    orig_risk = vary_risk(country_tier.get(orig, "low"))
    last_risk = vary_risk(country_tier.get(last, "low"))
    
    # Calculate label based on possibly varied risks
    risks = [nat_risk, orig_risk, last_risk]
    high, med = risks.count("high"), risks.count("medium")
    
    if high >= 2 or (high == 1 and med >= 1):
        label = 2  # High Risk
    elif med >= 1:
        label = 1  # Medium Risk
    else:
        label = 0  # Safe
    
    # Generate text with natural language variation
    text = generate_text_with_variation(nat, orig, last)
    
    # Add noise to 50% of examples (increased from 30%)
    text = add_noise(text)
    
    rows.append({"input_text": text, "label": label})

# Save
out_df = pd.DataFrame(rows)
out_df.to_csv("data/processed/traveler_text_large_dataset.csv", index=False)
print("✅ Generated data/processed/traveler_text_large_dataset.csv with", len(out_df), "rows.")

# Also create a test set with some truly new countries
test_rows = []
TEST_SIZE = 2000  # Increased test size

print("Generating separate test set...")
for _ in tqdm(range(TEST_SIZE)):
    # Ensure some examples use test-only countries
    use_test_countries = random.random() < 0.3
    
    if use_test_countries:
        # Use at least one test-only country
        test_country_count = random.randint(1, 3)
        test_countries = random.sample(list(test_countries_set), test_country_count)
        other_countries = random.sample(training_countries, 3 - test_country_count)
        nat, orig, last = random.sample(test_countries + other_countries, 3)
    else:
        # Use any countries
        nat, orig, last = random.sample(countries, 3)
    
    # Calculate label based on normal risks (no variation for test set)
    risks = [country_tier.get(nat, "low"), 
             country_tier.get(orig, "low"), 
             country_tier.get(last, "low")]
             
    high, med = risks.count("high"), risks.count("medium")
    
    if high >= 2 or (high == 1 and med >= 1):
        label = 2
    elif med >= 1:
        label = 1
    else:
        label = 0
    
    # Use natural language variation but no noise for test set
    text = generate_text_with_variation(nat, orig, last)
    
    test_rows.append({"input_text": text, "label": label})

# Save test set
test_df = pd.DataFrame(test_rows)
test_df.to_csv("data/processed/traveler_text_natural_language.csv", index=False)
print("✅ Generated data/processed/traveler_text_large_test.csv with", len(test_df), "rows.")

# Create a separate noisy test set to evaluate robustness
noisy_test_rows = []
NOISY_TEST_SIZE = 1000

print("Generating noisy test set for robustness evaluation...")
for _ in tqdm(range(NOISY_TEST_SIZE)):
    # Sample from the clean test set
    sample_idx = random.randint(0, len(test_rows) - 1)
    sample = test_rows[sample_idx].copy()
    
    # Add noise to the text
    sample["input_text"] = add_noise(sample["input_text"], probability=1.0)  # Always add noise
    
    noisy_test_rows.append(sample)

# Save noisy test set
noisy_test_df = pd.DataFrame(noisy_test_rows)
noisy_test_df.to_csv("data/processed/traveler_text_noisy_set.csv", index=False)
print("✅ Generated data/processed/traveler_text_noisy_set.csv with", len(noisy_test_df), "rows.")