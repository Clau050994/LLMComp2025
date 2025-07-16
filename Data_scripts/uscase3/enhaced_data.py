import pandas as pd
import random
import re
import os

# Load your existing training data
train_df = pd.read_csv("data/processed/traveler_text_large_dataset.csv")
print(f"Original training set: {len(train_df)} examples")

# Create directory for enhanced data if it doesn't exist
os.makedirs("data/processed/enhanced", exist_ok=True)

# 1. Add confounding context examples
def create_confounding_examples():
    high_risk_countries = ["Iraq", "Syria", "Afghanistan", "Yemen", "Somalia", "North Korea"]
    medium_risk_countries = ["Russia", "Pakistan", "Turkey", "Egypt", "Nigeria", "Colombia"]
    low_risk_countries = ["Canada", "Japan", "Australia", "New Zealand", "Switzerland", "Norway"]
    
    templates = [
        "Traveler from {origin} who studied literature about {confound} at university.",
        "Citizen of {origin} who exports goods to {confound}, arriving from {origin}.",
        "Passenger with {origin} passport interested in the culture of {confound}.",
        "{origin} national who recently read an article about {confound}.",
        "Individual from {origin} planning a future vacation to {confound}."
    ]
    
    confounding_examples = []
    
    # High risk countries
    for origin in high_risk_countries:
        for template in templates:
            # Add high risk origin with low risk confound
            confound = random.choice(low_risk_countries)
            text = template.format(origin=origin, confound=confound)
            confounding_examples.append({
                "input_text": text,
                "label": 2  # High risk
            })
    
    # Medium risk countries
    for origin in medium_risk_countries:
        for template in templates:
            # Add medium risk origin with high risk confound
            confound = random.choice(high_risk_countries)
            text = template.format(origin=origin, confound=confound)
            confounding_examples.append({
                "input_text": text,
                "label": 1  # Medium risk
            })
    
    # Low risk countries
    for origin in low_risk_countries:
        for template in templates:
            # Add low risk origin with high risk confound
            confound = random.choice(high_risk_countries)
            text = template.format(origin=origin, confound=confound)
            confounding_examples.append({
                "input_text": text,
                "label": 0  # Low risk
            })
    
    return pd.DataFrame(confounding_examples)

# 2. Create indirect descriptor examples
def create_descriptor_examples():
    descriptors = {
        "high_risk": [
            "active conflict zone", "region with frequent terrorist attacks",
            "area with ongoing civil war", "country under military dictatorship",
            "nation with widespread violence", "territory with significant insurgent activity",
            "location with high kidnapping rates", "area with foreign travel warnings"
        ],
        "medium_risk": [
            "region with occasional unrest", "area with political tensions",
            "country with border disputes", "region with economic sanctions",
            "territory with moderate security concerns", "area with limited press freedom",
            "nation with sporadic protests", "country with some instability"
        ],
        "low_risk": [
            "stable democracy", "peaceful nation", "economically prosperous region",
            "country with strong rule of law", "area with low crime rates",
            "nation with good healthcare system", "region with positive diplomatic relations",
            "territory with strong human rights record", "country with high tourism safety"
        ]
    }
    
    templates = [
        "Traveler coming from a {descriptor}.",
        "Individual who has citizenship in a {descriptor}.",
        "Passenger who recently visited a {descriptor}.",
        "Person who has been living in a {descriptor}.",
        "Traveler with residency in a {descriptor}.",
        "Citizen of a {descriptor}."
    ]
    
    descriptor_examples = []
    
    for risk_level, desc_list in descriptors.items():
        label = 2 if risk_level == "high_risk" else 1 if risk_level == "medium_risk" else 0
        for descriptor in desc_list:
            for template in templates:
                text = template.format(descriptor=descriptor)
                descriptor_examples.append({
                    "input_text": text,
                    "label": label
                })
    
    return pd.DataFrame(descriptor_examples)

# Generate new examples
confounding_df = create_confounding_examples()
descriptor_df = create_descriptor_examples()

print(f"Generated {len(confounding_df)} confounding context examples")
print(f"Generated {len(descriptor_df)} indirect descriptor examples")

# Save separate datasets for analysis
confounding_df.to_csv("data/processed/enhanced/confounding_examples.csv", index=False)
descriptor_df.to_csv("data/processed/enhanced/descriptor_examples.csv", index=False)

# Create enhanced training set by combining original and new examples
enhanced_df = pd.concat([train_df, confounding_df, descriptor_df], ignore_index=True)
print(f"Enhanced training set: {len(enhanced_df)} examples")

# Save enhanced training set
enhanced_df.to_csv("data/processed/enhanced/traveler_text_enhanced.csv", index=False)

# Create validation set with similar distribution of challenging examples
# to track improvement on these specific challenges
val_size = min(len(confounding_df) // 5, len(descriptor_df) // 5)
val_df = pd.concat([
    confounding_df.sample(val_size),
    descriptor_df.sample(val_size)
], ignore_index=True)

val_df.to_csv("data/processed/enhanced/traveler_text_understanding_val.csv", index=False)
print(f"Created validation set with {len(val_df)} examples")

print("Data enhancement complete!")