import pandas as pd

# Load your data
df = pd.read_csv("data/processed/unified/unified_complete_labeled.csv", header=None, names=["input_text", "label", "example_type"])

# Show a few random samples for manual inspection
print(df.sample(10))

# Check label distribution by example_type
print("\nLabel distribution by example_type:")
print(df.groupby("example_type")["label"].value_counts())

# Check for sentences with multiple country mentions
import re

def count_countries(text):
    # Example: crude country detection (replace with your country list for accuracy)
    countries = ["China", "Beijing", "Pakistan", "Iran", "Syria", "France", "Eiffel Tower"]
    return sum(1 for c in countries if c.lower() in text.lower())

df["country_mentions"] = df["input_text"].apply(count_countries)
print("\nRows with more than one country mention:")
print(df[df["country_mentions"] > 1][["input_text", "label", "example_type"]].head(10))

# Check if non-origin examples with country mentions are labeled as 0
non_origin = df[(df["example_type"] != "origin") & (df["country_mentions"] > 0)]
print("\nNon-origin examples with country mentions (should be label 0):")
print(non_origin[non_origin["label"] != 0][["input_text", "label", "example_type"]])

# Check if origin examples are labeled as risky (1 or 2)
origin = df[df["example_type"] == "origin"]
print("\nOrigin examples label distribution:")
print(origin["label"].value_counts())