import pandas as pd
import numpy as np

# === Load terrorism dataset (filtered from 2000–2017) ===
df = pd.read_csv("/aul/homes/cvaro009/Desktop/LLMComp2025/data/processed/filtered_2000_2017.csv", encoding="ISO-8859-1", low_memory=False)

# === Severity weight mappings ===
attack_weights = {
    "Bombing/Explosion": 3,
    "Armed Assault": 2,
    "Assassination": 4,
    "Hijacking": 5,
    "Hostage Taking (Kidnapping)": 4,
    "Facility/Infrastructure Attack": 2,
    "Unknown": 1
}

target_weights = {
    "Government (General)": 3,
    "Military": 3,
    "Police": 3,
    "Private Citizens & Property": 2,
    "Business": 2,
    "Religious Figures/Institutions": 3,
    "Educational Institution": 2,
    "Unknown": 1
}

weapon_weights = {
    "Explosives/Bombs/Dynamite": 3,
    "Firearms": 2,
    "Incendiary": 2,
    "Melee": 1,
    "Chemical": 4,
    "Sabotage Equipment": 3,
    "Unknown": 1
}

# === Compute weighted severity with time decay (centered at 2025) ===
df["attack_score"] = df["attacktype1_txt"].map(attack_weights).fillna(1)
df["target_score"] = df["targsubtype1_txt"].map(target_weights).fillna(1)
df["weapon_score"] = df["weaptype1_txt"].map(weapon_weights).fillna(1)

df["recency_weight"] = np.exp(-(2025 - df["iyear"]) / 2)
df["weighted_score"] = (df["attack_score"] + df["target_score"] + df["weapon_score"]) * df["recency_weight"]

# === Aggregate score by country ===
risk_df = df.groupby("country_txt")["weighted_score"].sum().reset_index()
risk_df.columns = ["country", "risk_score"]

# === Normalize by population ===
pop_df = pd.read_csv("/aul/homes/cvaro009/Desktop/LLMComp2025/data/processed/country_population_clean.csv")
pop_df["country"] = pop_df["country"].str.strip()

# === Fix country name mismatches before merging ===
name_fixes = {
    "Bosnia-Herzegovina": "Bosnia and Herzegovina",
    "Macedonia": "North Macedonia",
    "Kosovo": "Kosovo",
    "West Bank and Gaza Strip": "Palestine",
    "Serbia-Montenegro": "Serbia",
    "Swaziland": "Eswatini",
    "St. Lucia": "Saint Lucia",
    "Yugoslavia": None,
    "International": None,
    "East Timor": "Timor-Leste",
    "Slovak Republic": "Slovakia"
}
risk_df["country"] = risk_df["country"].replace(name_fixes)
risk_df = risk_df[risk_df["country"].notnull()]  # Drop unresolved rows

# === Merge and calculate per-million ===
risk_df = pd.merge(risk_df, pop_df, on="country", how="left")


# Avoid division by zero
risk_df["population"] = risk_df["population"].replace(0, np.nan)
risk_df["risk_per_million"] = (risk_df["risk_score"] / risk_df["population"]) * 1_000_000

# === Assign tiers based on percentiles ===
risk_df = risk_df.sort_values("risk_per_million", ascending=False).reset_index(drop=True)
n = len(risk_df)
risk_df["risk_tier"] = "low"
risk_df.loc[:int(n * 0.15), "risk_tier"] = "high"
risk_df.loc[int(n * 0.15):int(n * 0.4), "risk_tier"] = "medium"

# === Save final output ===
risk_df.to_csv("/aul/homes/cvaro009/Desktop/LLMComp2025/data/processed/improved_data/risk_score.csv", index=False)
print("✅ Country risk scores saved to 'data/processed/country_risk_scores.csv'")
