import pandas as pd

# Load dataset
df = pd.read_csv("data/raw/world_population.csv")

# Print actual column names (optional debug)
# print(df.columns.tolist())

# Select the correct columns (remove trailing space)
df = df[["Country/Territory", "2022 Population"]]

# Rename columns to standardized format
df = df.rename(columns={
    "Country/Territory": "country",
    "2022 Population": "population"
})

# Clean country names
df["country"] = df["country"].str.strip()

# Save cleaned file
df.to_csv("data/processed/country_population_clean.csv", index=False)

print("✅ Cleaned population data saved to data/external/country_population_clean.csv")
