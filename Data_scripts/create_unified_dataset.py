#!/usr/bin/env python3
"""
Create a 50,000-row unified dataset for traveler risk classification,
with splits optimized for model learning.
Labels are always based on the origin country risk tier if present, otherwise 0.
"""

import pandas as pd
import os
from sklearn.model_selection import train_test_split
import numpy as np
import random
import re

# Set random seed for reproducibility
random.seed(42)
np.random.seed(42)

print("Creating comprehensive unified dataset...")

# Paths
data_root = "data/processed"
output_dir = os.path.join(data_root, "unified")
os.makedirs(output_dir, exist_ok=True)

def load_risk_categories():
    """Load countries from the risk score dataset and categorize them"""
    risk_score_path = os.path.join(data_root, "country_risk_scores.csv")
    if not os.path.exists(risk_score_path):
        risk_score_path = os.path.join("data/raw", "country_risk_scores.csv")
    risk_df = pd.read_csv(risk_score_path)
    tier_to_label = {"low": 0, "medium": 1, "high": 2}
    country_label = {row["country"]: tier_to_label[row["risk_tier"]] for _, row in risk_df.iterrows()}
    high_risk = risk_df[risk_df["risk_tier"] == "high"]["country"].tolist()
    medium_risk = risk_df[risk_df["risk_tier"] == "medium"]["country"].tolist()
    low_risk = risk_df[risk_df["risk_tier"] == "low"]["country"].tolist()
    print(f"Loaded {len(high_risk)} high, {len(medium_risk)} medium, {len(low_risk)} low risk countries")
    return (high_risk, medium_risk, low_risk), country_label

print("Loading source dataset...")
base_df = pd.read_csv(os.path.join(data_root, "traveler_text_large_dataset.csv"))
print(f"Base dataset: {len(base_df)} examples")

(risk_lists, country_label) = load_risk_categories()
high_risk, medium_risk, low_risk = risk_lists
all_countries = set(high_risk + medium_risk + low_risk)

origin_patterns = [
    r'from ([A-Za-z \-]+)',
    r'citizen of ([A-Za-z \-]+)',
    r'national of ([A-Za-z \-]+)',
    r'passport from ([A-Za-z \-]+)',
    r'born in ([A-Za-z \-]+)',
    r'resident of ([A-Za-z \-]+)',
    r'traveling from ([A-Za-z \-]+)',
    r'originating from ([A-Za-z \-]+)',
    r'departed from ([A-Za-z \-]+)',
    r'arriving from ([A-Za-z \-]+)',
    r'with ([A-Za-z \-]+) citizenship',
    r'([A-Za-z \-]+) passport holder',
    r'([A-Za-z \-]+) national',
]
origin_regex = re.compile('|'.join(origin_patterns), re.IGNORECASE)

def extract_origin_country(text):
    match = origin_regex.search(text)
    if match:
        for group in match.groups():
            if group:
                country = group.strip()
                for c in all_countries:
                    if c.lower() in country.lower():
                        return c
    return None

def get_label_from_origin(text):
    country = extract_origin_country(text)
    if country and country in country_label:
        return country_label[country]
    return 0

def generate_syntactic_examples(num_per_country=30):
    origin_verbs = [
        "from {}", "citizen of {}", "national of {}", "passport from {}", "born in {}", 
        "resident of {}", "traveling from {}", "{} passport holder", "departed from {}", 
        "arriving from {}", "originating from {}", "with {} citizenship"
    ]
    preference_verbs = [
        "enjoys {} cuisine", "likes food from {}", "prefers {} dishes", "is a fan of {} cooking",
        "likes to eat {} food", "interested in {} culture", "studies {} literature",
        "watches movies from {}", "listens to music from {}", "reads books about {}",
        "follows {} sports", "collects art from {}", "admires {} architecture",
        "appreciates {} history", "speaks the language of {}", "is learning the {} language",
        "knows phrases in {}", "can understand {}", "has friends in {}", "knows people from {}",
        "has colleagues from {}", "has relatives who visited {}"
    ]
    templates = [
        "Traveler {origin_verb} who {preference_verb}.",
        "Person {origin_verb} and {preference_verb}.",
        "Individual {origin_verb} that {preference_verb}.",
        "A {origin_verb} traveler who {preference_verb}.",
        "Traveler who {preference_verb} and is {origin_verb}.",
        "Person who {preference_verb}, traveling {origin_verb}.",
        "Individual interested in {preference_verb} with passport {origin_verb}.",
        "The traveler {origin_verb} mentioned they {preference_verb}.",
        "A visitor who {preference_verb} showed documents {origin_verb}.",
        "Traveler {origin_verb} who {preference_verb} and also {preference_verb2}.",
        "Person {origin_verb} that {preference_verb} while being interested in {preference_verb2}."
    ]
    examples = []
    for countries in [high_risk, medium_risk, low_risk]:
        for origin_country in countries:
            confound_countries = list(all_countries - {origin_country})
            for _ in range(num_per_country):
                if random.random() < 0.3 and len(confound_countries) >= 2:
                    template = random.choice([t for t in templates if "{preference_verb2}" in t])
                    origin_verb = random.choice(origin_verbs).format(origin_country)
                    confound1, confound2 = random.sample(confound_countries, 2)
                    preference_verb = random.choice(preference_verbs).format(confound1)
                    preference_verb2 = random.choice(preference_verbs).format(confound2)
                    text = template.format(
                        origin_verb=origin_verb,
                        preference_verb=preference_verb,
                        preference_verb2=preference_verb2
                    )
                else:
                    template = random.choice([t for t in templates if "{preference_verb2}" not in t])
                    origin_verb = random.choice(origin_verbs).format(origin_country)
                    confound = random.choice(confound_countries)
                    preference_verb = random.choice(preference_verbs).format(confound)
                    text = template.format(
                        origin_verb=origin_verb,
                        preference_verb=preference_verb
                    )
                label = get_label_from_origin(text)
                examples.append({
                    "input_text": text,
                    "label": label,
                    "example_type": "syntactic"
                })
    return pd.DataFrame(examples)

def generate_noisy_examples(num_per_country=20):
    alt_spellings = {
        "United States": ["USA", "U.S.A.", "US", "Amerika"],
        "United Kingdom": ["UK", "Britain", "Great Britain"],
        "Russia": ["Rusia", "Rossiya", "Russian Federation"],
        "China": ["PRC", "Zhongguo"],
        "Afghanistan": ["Afganistan", "Afghānistān"],
        "Egypt": ["Misr", "Al-Misr"],
        "Iraq": ["Irak", "Al-Iraq"],
        "Syria": ["Suriya", "Al-Sham"],
        "Japan": ["Nihon", "Nippon"],
        "Germany": ["Deutschland", "Allemagne"],
        "France": ["La France", "République française"],
        "Canada": ["Kanada", "Le Canada"],
        "Mexico": ["México", "Méjico"],
        "Brazil": ["Brasil", "Brazilia"],
        "India": ["Bharat", "Hindustan"]
    }
    templates = [
        "Tr@veler from {country}, {risk_level} risk profile.",
        "P@ss3nger with {country} p@ssport, {risk_level} r!sk.",
        "C!tizen of {country}, {risk_level} risk n@tion.",
        "Person from {alt_spelling} ({country}), {risk_level} risk.",
        "Tr@veler h0lding {country} p@ssport.",
        "Vis!tor from {country} with recent tr@vel.",
        "P@ss3ng3r fr0m {country} @rriving t0d@y.",
        "Tr@v3l3r w1th {country} c1t1z3nsh1p.",
        "1nd1v1du@l fr0m {country} @t b0rd3r ch3ckp01nt."
    ]
    examples = []
    for countries, risk_level in [
        (high_risk, "high"),
        (medium_risk, "medium"),
        (low_risk, "low")
    ]:
        for country in countries:
            for _ in range(num_per_country):
                template = random.choice(templates)
                alt_spelling = random.choice(alt_spellings.get(country, [country]))
                text = template.format(
                    country=country, 
                    risk_level=risk_level,
                    alt_spelling=alt_spelling
                )
                if random.random() < 0.3:
                    text = text.replace('a', '@', random.randint(0, 2))
                    text = text.replace('e', '3', random.randint(0, 2))
                    text = text.replace('i', '1', random.randint(0, 2))
                    text = text.replace('o', '0', random.randint(0, 2))
                    text = text.replace('s', '5', random.randint(0, 2))
                if random.random() < 0.25:
                    text = re.sub(r'(\w)', r'\1 ', text, random.randint(1, 3))
                if random.random() < 0.25:
                    text = re.sub(r' ', '', text, random.randint(1, 3))
                label = get_label_from_origin(text)
                examples.append({
                    "input_text": text,
                    "label": label,
                    "example_type": "noisy"
                })
    return pd.DataFrame(examples)

def generate_descriptor_examples():
    descriptors = {
        "high_risk": [
            "active conflict zone", "region with frequent terrorist attacks",
            "area with ongoing civil war", "country under military dictatorship",
            "nation with widespread violence", "territory with significant insurgent activity",
            "location with high kidnapping rates", "area with foreign travel warnings",
            "region experiencing armed rebellion", "state with collapsed government",
            "territory with active militants", "area under martial law"
        ],
        "medium_risk": [
            "region with occasional unrest", "area with political tensions",
            "country with border disputes", "nation with economic sanctions",
            "territory with moderate security concerns", "area with limited press freedom",
            "nation with sporadic protests", "country with some instability",
            "region with ethnic tensions", "area with travel advisories",
            "location with increased crime rates", "territory with political uncertainty"
        ],
        "low_risk": [
            "stable democracy", "peaceful nation", "economically prosperous region",
            "country with strong rule of law", "area with low crime rates",
            "nation with good healthcare system", "region with positive diplomatic relations",
            "territory with strong human rights record", "country with high tourism safety",
            "area with reliable infrastructure", "nation with stable government",
            "region with strong social services"
        ]
    }
    templates = [
        "Traveler coming from a {descriptor}.",
        "Individual who has citizenship in a {descriptor}.",
        "Passenger who recently visited a {descriptor}.",
        "Person whose passport is from a {descriptor}.",
        "Citizen originating from a {descriptor}.",
        "Tourist who lives in a {descriptor}.",
        "Visitor whose home is in a {descriptor}.",
        "Passenger whose nation is described as a {descriptor}.",
        "Records indicate they've been staying in a {descriptor}.",
        "The documents show residence in a {descriptor}.",
        "Their travel pattern indicates origin in a {descriptor}."
    ]
    examples = []
    for risk_level, desc_list in descriptors.items():
        label = 2 if risk_level == "high_risk" else 1 if risk_level == "medium_risk" else 0
        for descriptor in desc_list:
            for template in templates:
                text = template.format(descriptor=descriptor)
                examples.append({
                    "input_text": text,
                    "label": label,
                    "example_type": "descriptor"
                })
    return pd.DataFrame(examples)

def generate_combined_examples(num_per_country=10):
    descriptors = {
        "high_risk": [
            "active conflict zone", "region with frequent terrorist attacks",
            "area with ongoing civil war", "country under military dictatorship"
        ],
        "medium_risk": [
            "region with occasional unrest", "area with political tensions",
            "country with border disputes", "nation with economic sanctions"
        ],
        "low_risk": [
            "stable democracy", "peaceful nation", 
            "economically prosperous region", "country with strong rule of law"
        ]
    }
    alt_spellings = {
        "United States": "USA", "Russia": "Rusia", "Afghanistan": "Afganistan", 
        "Iraq": "Irak", "Syria": "Suriya", "Japan": "Nihon", "Canada": "Kanada"
    }
    templates = [
        "Tr@v3l3r fr0m {origin} wh0 3nj0ys {confound} cu1s1n3.",
        "P@ss3ng3r w1th {origin} p@ssp0rt wh0 stud13s {confound} l1t3r@tur3.",
        "Person from a {descriptor} who has friends in {confound}.",
        "Traveler from a {descriptor} interested in {confound} music.",
        "Tr@v3l3r fr0m a {descriptor} wh0 c0ll3cts {confound} @rt.",
        "V1s1t0r w1th {alt_spelling} ({origin}) p@ssp0rt wh0 w@tch3s {confound} m0v13s."
    ]
    examples = []
    for countries, risk_desc, desc_key, label in [
        (high_risk, "high risk", "high_risk", 2),
        (medium_risk, "medium risk", "medium_risk", 1),
        (low_risk, "low risk", "low_risk", 0)
    ]:
        for origin in countries[:15]:
            confound_countries = list(all_countries - {origin})
            descriptor = random.choice(descriptors[desc_key])
            alt_spelling = alt_spellings.get(origin, origin)
            for _ in range(num_per_country):
                template = random.choice(templates)
                confound = random.choice(confound_countries)
                if "{descriptor}" in template and "{alt_spelling}" in template:
                    text = template.format(
                        descriptor=descriptor,
                        alt_spelling=alt_spelling,
                        origin=origin,
                        confound=confound
                    )
                elif "{descriptor}" in template:
                    text = template.format(
                        descriptor=descriptor,
                        confound=confound
                    )
                else:
                    text = template.format(
                        origin=origin,
                        confound=confound,
                        alt_spelling=alt_spelling
                    )
                label = get_label_from_origin(text)
                examples.append({
                    "input_text": text,
                    "label": label,
                    "example_type": "combined"
                })
    return pd.DataFrame(examples)

# Generate datasets
print("Generating comprehensive dataset...")
syntactic_df = generate_syntactic_examples(num_per_country=30)
print(f"Generated {len(syntactic_df)} syntactic examples")
noise_df = generate_noisy_examples(num_per_country=20)
print(f"Generated {len(noise_df)} noise examples")
descriptor_df = generate_descriptor_examples()
print(f"Generated {len(descriptor_df)} descriptor examples")
combined_df = generate_combined_examples(num_per_country=10)
print(f"Generated {len(combined_df)} combined challenge examples")

# Combine all datasets
all_dfs = [base_df, syntactic_df, noise_df, descriptor_df, combined_df]
unified_df = pd.concat(all_dfs, ignore_index=True)

# Shuffle and sample exactly 50,000 rows
unified_df = unified_df.sample(n=50000, random_state=42).reset_index(drop=True)

# Split
train_df, test_df = train_test_split(
    unified_df, 
    test_size=0.2, 
    stratify=unified_df["label"],
    random_state=42
)
train_df, val_df = train_test_split(
    train_df,
    test_size=5000,  # 5,000 validation rows
    stratify=train_df["label"],
    random_state=42
)

print(f"\nFinal splits: {len(train_df)} training, {len(val_df)} validation, {len(test_df)} test examples")

# Save datasets
train_path = os.path.join(output_dir, "unified_train.csv")
val_path = os.path.join(output_dir, "unified_val.csv")
test_path = os.path.join(output_dir, "unified_test.csv")
full_path = os.path.join(output_dir, "unified_complete.csv")

train_df.to_csv(train_path, index=False)
val_df.to_csv(val_path, index=False)
test_df.to_csv(test_path, index=False)
unified_df.to_csv(full_path, index=False)

print("\nDatasets saved:")
print(f"- Training: {train_path}")
print(f"- Validation: {val_path}")
print(f"- Test: {test_path}")
print(f"- Complete: {full_path}")

print("\nUnified dataset creation complete!")