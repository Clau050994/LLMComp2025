#!/usr/bin/env python3
"""
Create a comprehensive unified dataset that handles all traveler risk classification challenges
including confounding contexts, country relationships, and syntactic understanding.
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

# Create output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

def load_risk_categories():
    """Load countries from the risk score dataset and categorize them"""
    try:
        # Load the risk score dataset
        risk_score_path = os.path.join(data_root, "country_risk_scores.csv")
        if not os.path.exists(risk_score_path):
            # Try alternate location
            risk_score_path = os.path.join("data/raw", "country_risk_scores.csv")
        
        if not os.path.exists(risk_score_path):
            print(f"Warning: Could not find risk score dataset at {risk_score_path}")
            print("Using fallback country lists...")
            return default_risk_categories()
            
        risk_df = pd.read_csv(risk_score_path)
        
        # Categorize countries based on risk score
        high_risk = risk_df[risk_df["risk_score"] >= 7]["country"].tolist()
        medium_risk = risk_df[(risk_df["risk_score"] >= 4) & (risk_df["risk_score"] < 7)]["country"].tolist()
        low_risk = risk_df[risk_df["risk_score"] < 4]["country"].tolist()
        
        print(f"Loaded {len(high_risk)} high risk, {len(medium_risk)} medium risk, and {len(low_risk)} low risk countries")
        return high_risk, medium_risk, low_risk
        
    except Exception as e:
        print(f"Error loading risk score dataset: {e}")
        print("Using fallback country lists...")
        return default_risk_categories()

def default_risk_categories():
    """Provide default risk categories if risk score dataset isn't available"""
    high_risk_countries = [
        "Iraq", "Syria", "Afghanistan", "Yemen", "Somalia", "North Korea", "Sudan", 
        "Libya", "South Sudan", "Central African Republic", "Venezuela", "Mali", 
        "Democratic Republic of Congo", "Eritrea", "Burundi", "Myanmar", "Niger",
        "Iran", "Lebanon", "Palestine", "Belarus", "Zimbabwe"
    ]

    medium_risk_countries = [
        "Russia", "Pakistan", "Turkey", "Egypt", "Nigeria", "Colombia", "Ukraine", 
        "Ethiopia", "Algeria", "Honduras", "El Salvador", "Azerbaijan", "Cuba", 
        "Bosnia", "Guatemala", "Kenya", "Mexico", "Brazil", "India", "Philippines",
        "South Africa", "Peru", "Bolivia", "Bangladesh", "Indonesia", "Thailand", "Vietnam"
    ]

    low_risk_countries = [
        "Canada", "Japan", "Australia", "New Zealand", "Switzerland", "Norway", "Singapore", 
        "Denmark", "Finland", "Iceland", "Austria", "Portugal", "Czech Republic", "Taiwan", 
        "South Korea", "United Kingdom", "Germany", "France", "Spain", "Belgium", 
        "Netherlands", "Ireland", "Estonia", "Uruguay", "Costa Rica", "Chile", "Sweden",
        "Luxembourg", "United States", "Poland", "Italy", "Greece", "Croatia", "Slovenia"
    ]
    
    print("Using default country categories")
    return high_risk_countries, medium_risk_countries, low_risk_countries

# Load base dataset
print("Loading source dataset...")
base_df = pd.read_csv(os.path.join(data_root, "traveler_text_large_dataset.csv"))
print(f"Base dataset: {len(base_df)} examples")

# Define syntactic relation patterns
def generate_syntactic_examples():
    """Generate examples that focus on syntactic relationships between countries and verbs"""
    high_risk, medium_risk, low_risk = load_risk_categories()
    
    # Define verb categories that clearly distinguish intent/relevance
    origin_verbs = [
        "from {}", 
        "citizen of {}", 
        "national of {}", 
        "passport from {}", 
        "born in {}", 
        "resident of {}", 
        "traveling from {}", 
        "{} passport holder",
        "departed from {}",
        "arriving from {}",
        "originating from {}",
        "with {} citizenship"
    ]

    preference_verbs = [
        # Food related
        "enjoys {} cuisine", 
        "likes food from {}", 
        "prefers {} dishes",
        "is a fan of {} cooking",
        "likes to eat {} food",
        
        # Cultural interests
        "interested in {} culture", 
        "studies {} literature",
        "watches movies from {}",
        "listens to music from {}",
        "reads books about {}",
        "follows {} sports",
        "collects art from {}",
        "admires {} architecture",
        "appreciates {} history",
        
        # Language
        "speaks the language of {}",
        "is learning the {} language",
        "knows phrases in {}",
        "can understand {}",
        
        # Personal connections
        "has friends in {}",
        "knows people from {}",
        "has colleagues from {}",
        "has relatives who visited {}"
    ]
    
    # Templates for creating examples with different syntactic structures
    templates = [
        # Origin first, preference second (standard order)
        "Traveler {origin_verb} who {preference_verb}.",
        "Person {origin_verb} and {preference_verb}.",
        "Individual {origin_verb} that {preference_verb}.",
        "A {origin_verb} traveler who {preference_verb}.",
        
        # Preference first, origin second (reversed order)
        "Traveler who {preference_verb} and is {origin_verb}.",
        "Person who {preference_verb}, traveling {origin_verb}.",
        "Individual interested in {preference_verb} with passport {origin_verb}.",
        
        # Embedded structures
        "The traveler {origin_verb} mentioned they {preference_verb}.",
        "A visitor who {preference_verb} showed documents {origin_verb}.",
        
        # Multiple confounding mentions
        "Traveler {origin_verb} who {preference_verb} and also {preference_verb2}.",
        "Person {origin_verb} that {preference_verb} while being interested in {preference_verb2}."
    ]
    
    examples = []
    
    # Create examples for each risk level
    for countries, risk_level, label in [
        (high_risk, "high", 2),
        (medium_risk, "medium", 1),
        (low_risk, "low", 0)
    ]:
        for origin_country in countries:
            # Sample other countries from different risk levels for confounding
            confound_countries = []
            if label != 2:  # For low/medium risk origins, include high risk confounds
                confound_countries.extend(random.sample(high_risk, min(5, len(high_risk))))
            if label != 1:  # For low/high risk origins, include medium risk confounds
                confound_countries.extend(random.sample(medium_risk, min(5, len(medium_risk))))
            if label != 0:  # For medium/high risk origins, include low risk confounds
                confound_countries.extend(random.sample(low_risk, min(5, len(low_risk))))
                
            # Create examples with this origin country
            for _ in range(10):  # Create multiple examples per country
                # Pick template
                if len(confound_countries) >= 2 and random.random() < 0.3:
                    # Use template with multiple confounds
                    template = random.choice([t for t in templates if "{preference_verb2}" in t])
                    
                    # Format origin verb
                    origin_pattern = random.choice(origin_verbs)
                    origin_verb = origin_pattern.format(origin_country)
                    
                    # Format two different preference verbs with two different countries
                    confound1, confound2 = random.sample(confound_countries, 2)
                    pref_pattern1 = random.choice(preference_verbs)
                    pref_pattern2 = random.choice(preference_verbs)
                    preference_verb = pref_pattern1.format(confound1)
                    preference_verb2 = pref_pattern2.format(confound2)
                    
                    # Create the example
                    text = template.format(
                        origin_verb=origin_verb,
                        preference_verb=preference_verb,
                        preference_verb2=preference_verb2
                    )
                else:
                    # Use standard template with one confound
                    template = random.choice([t for t in templates if "{preference_verb2}" not in t])
                    
                    # Format origin verb
                    origin_pattern = random.choice(origin_verbs)
                    origin_verb = origin_pattern.format(origin_country)
                    
                    # Format preference verb
                    confound = random.choice(confound_countries)
                    pref_pattern = random.choice(preference_verbs)
                    preference_verb = pref_pattern.format(confound)
                    
                    # Create the example
                    text = template.format(
                        origin_verb=origin_verb,
                        preference_verb=preference_verb
                    )
                
                examples.append({
                    "input_text": text,
                    "label": label,
                    "example_type": "syntactic"
                })
    
    return pd.DataFrame(examples)

def generate_noise_examples():
    """Generate noisy examples with misspellings, alternate spellings, etc."""
    high_risk, medium_risk, low_risk = load_risk_categories()
    
    # Alternative spellings/formats for some countries
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
    
    # Templates for noisy examples
    templates = [
        # Basic templates with noise placeholders
        "Tr@veler from {country}, {risk_level} risk profile.",
        "P@ss3nger with {country} p@ssport, {risk_level} r!sk.",
        "C!tizen of {country}, {risk_level} risk n@tion.",
        "Person from {alt_spelling} ({country}), {risk_level} risk.",
        "Tr@veler h0lding {country} p@ssport.",
        "Vis!tor from {country} with recent tr@vel.",
        
        # Templates with character substitutions 
        "P@ss3ng3r fr0m {country} @rriving t0d@y.",
        "Tr@v3l3r w1th {country} c1t1z3nsh1p.",
        "1nd1v1du@l fr0m {country} @t b0rd3r ch3ckp01nt."
    ]
    
    examples = []
    
    # Create examples for each risk level
    for countries, risk_level, label in [
        (high_risk, "high", 2),
        (medium_risk, "medium", 1),
        (low_risk, "low", 0)
    ]:
        # Create multiple examples per country
        for country in countries:
            for _ in range(5):  # 5 examples per country
                template = random.choice(templates)
                
                # Get alternative spelling if available
                alt_spelling = random.choice(alt_spellings.get(country, [country]))
                
                # Apply the template
                text = template.format(
                    country=country, 
                    risk_level=risk_level,
                    alt_spelling=alt_spelling
                )
                
                # Apply random additional transformations
                if random.random() < 0.3:
                    # Replace some letters with numbers/symbols
                    text = text.replace('a', '@', random.randint(0, 2))
                    text = text.replace('e', '3', random.randint(0, 2))
                    text = text.replace('i', '1', random.randint(0, 2))
                    text = text.replace('o', '0', random.randint(0, 2))
                    text = text.replace('s', '5', random.randint(0, 2))
                
                if random.random() < 0.25:
                    # Add random spaces
                    text = re.sub(r'(\w)', r'\1 ', text, random.randint(1, 3))
                
                if random.random() < 0.25:
                    # Remove random spaces
                    text = re.sub(r' ', '', text, random.randint(1, 3))
                
                examples.append({
                    "input_text": text,
                    "label": label,
                    "example_type": "noisy"
                })
    
    return pd.DataFrame(examples)

def generate_descriptor_examples():
    """Generate examples using indirect descriptors instead of country names"""
    # Risk level descriptors
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
    
    # Templates for descriptor examples
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
    
    # Create examples for each risk level
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

def generate_combined_examples():
    """Generate examples that combine multiple challenge types"""
    high_risk, medium_risk, low_risk = load_risk_categories()
    
    # Templates for combined challenges
    templates = [
        # Noisy text + confounding context
        "Tr@v3l3r fr0m {origin} wh0 3nj0ys {confound} cu1s1n3.",
        "P@ss3ng3r w1th {origin} p@ssp0rt wh0 stud13s {confound} l1t3r@tur3.",
        
        # Descriptor + confounding context
        "Person from a {descriptor} who has friends in {confound}.",
        "Traveler from a {descriptor} interested in {confound} music.",
        
        # Complex cases with multiple challenges
        "Tr@v3l3r fr0m a {descriptor} wh0 c0ll3cts {confound} @rt.",
        "V1s1t0r w1th {alt_spelling} ({origin}) p@ssp0rt wh0 w@tch3s {confound} m0v13s."
    ]
    
    # Risk level descriptors
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
    
    # Alternative spellings
    alt_spellings = {
        "United States": "USA", "Russia": "Rusia", "Afghanistan": "Afganistan", 
        "Iraq": "Irak", "Syria": "Suriya", "Japan": "Nihon", "Canada": "Kanada"
    }
    
    examples = []
    
    # Create examples for each risk level
    for countries, risk_desc, desc_key, label in [
        (high_risk, "high risk", "high_risk", 2),
        (medium_risk, "medium risk", "medium_risk", 1),
        (low_risk, "low risk", "low_risk", 0)
    ]:
        # For each country in this risk level
        for origin in countries[:15]:  # Limit to 15 countries per risk level
            # Choose countries from other risk levels for confounding
            confound_countries = []
            if label != 2:  # Include high risk confounds for non-high risk origins
                confound_countries.extend(random.sample(high_risk, min(2, len(high_risk))))
            if label != 1:  # Include medium risk confounds for non-medium risk origins
                confound_countries.extend(random.sample(medium_risk, min(2, len(medium_risk))))
            if label != 0:  # Include low risk confounds for non-low risk origins
                confound_countries.extend(random.sample(low_risk, min(2, len(low_risk))))
            
            # Pick descriptor for this risk level
            descriptor = random.choice(descriptors[desc_key])
            
            # Get alternative spelling if available
            alt_spelling = alt_spellings.get(origin, origin)
            
            # Create examples
            for _ in range(3):  # 3 examples per origin country
                template = random.choice(templates)
                confound = random.choice(confound_countries)
                
                # Format template
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
                
                examples.append({
                    "input_text": text,
                    "label": label,
                    "example_type": "combined"
                })
    
    return pd.DataFrame(examples)

# Generate datasets
print("Generating comprehensive dataset...")
syntactic_df = generate_syntactic_examples()
print(f"Generated {len(syntactic_df)} syntactic examples")

noise_df = generate_noise_examples()
print(f"Generated {len(noise_df)} noise examples")

descriptor_df = generate_descriptor_examples()  
print(f"Generated {len(descriptor_df)} descriptor examples")

combined_df = generate_combined_examples()
print(f"Generated {len(combined_df)} combined challenge examples")

# Combine all datasets
all_dfs = [base_df, syntactic_df, noise_df, descriptor_df, combined_df]
unified_df = pd.concat(all_dfs, ignore_index=True)
print(f"\nFull unified dataset: {len(unified_df)} examples")

# Check class distribution
class_counts = unified_df["label"].value_counts()
print("\nClass distribution in unified dataset:")
for label, count in sorted(class_counts.items()):
    print(f"Class {label}: {count} examples ({count/len(unified_df)*100:.1f}%)")

# Create training and test splits
train_df, test_df = train_test_split(
    unified_df, 
    test_size=0.2, 
    stratify=unified_df["label"],
    random_state=42
)

train_df, val_df = train_test_split(
    train_df,
    test_size=0.125,  # 10% of total = 12.5% of train
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

# Create specialized test sets for evaluation
print("\nCreating specialized test sets for targeted evaluation...")

# Extract confounding examples from test set
confounding_test = test_df[test_df["input_text"].str.contains("enjoys|likes|studies|interested|watches|reads|cuisine|food|culture|music", regex=True, case=False)]
if len(confounding_test) > 0:
    confounding_path = os.path.join(output_dir, "unified_confounding.csv")
    confounding_test.to_csv(confounding_path, index=False)
    print(f"- Created confounding test set: {len(confounding_test)} examples")

# Extract descriptor examples from test set
descriptor_test = test_df[test_df["example_type"] == "descriptor"]
if len(descriptor_test) > 0:
    descriptor_path = os.path.join(output_dir, "unified_descriptors.csv")
    descriptor_test.to_csv(descriptor_path, index=False)
    print(f"- Created descriptor test set: {len(descriptor_test)} examples")

# Extract noisy examples from test set
noisy_test = test_df[test_df["example_type"].isin(["noisy", "combined"])]
if len(noisy_test) > 0:
    noisy_path = os.path.join(output_dir, "unified_noisy.csv")
    noisy_test.to_csv(noisy_path, index=False)
    print(f"- Created noisy test set: {len(noisy_test)} examples")

print("\nUnified dataset creation complete!")