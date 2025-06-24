import pandas as pd
import numpy as np

# === Load terrorism dataset (already filtered 2000–2017) ===
df = pd.read_csv("data/processed/filtered_2000_2017.csv", encoding="ISO-8859-1", low_memory=False)

# === Assign weights to attack, target, weapon types ===
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

df["attack_score"] = df["attacktype1_txt"].map(attack_weights).fillna(1)
df["target_score"] = df["targsubtype1_txt"].map(target_weights).fillna(1)
df["weapon_score"] = df["weaptype1_txt"].map(weapon_weights).fillna(1)

# === Compute severity with exponential time decay ===
df["recency_weight"] = np.exp(-(2020 - df["iyear"]) / 2)
df["weighted_score"] = (df["attack_score"] + df["target_score"] + df["weapon_score"]) * df["recency_weight"]

# === Aggregate scores by country ===
risk_df = df.groupby("country_txt")["weighted_score"].sum().reset_index()
risk_df.columns = ["country", "risk_score"]

# === Load population and normalize ===
pop_df = pd.read_csv("data/external/country_population_clean.csv")
pop_df["country"] = pop_df["country"].str.strip()

# Merge and normalize
risk_df = pd.merge(risk_df, pop_df, on="country", how="left")
risk_df["population"] = risk_df["population"].replace(0, np.nan)
risk_df["risk_per_million"] = (risk_df["risk_score"] / risk_df["population"]) * 1_000_000

# === Assign risk tiers using percentiles ===
risk_df = risk_df.sort_values("risk_per_million", ascending=False).reset_index(drop=True)
n = len(risk_df)
risk_df["risk_tier"] = "low"
risk_df.loc[:int(n * 0.15), "risk_tier"] = "high"
risk_df.loc[int(n * 0.15):int(n * 0.4), "risk_tier"] = "medium"

# Create risk map
country_risk_dict = dict(zip(risk_df["country"], risk_df["risk_tier"]))

# === Simulated traveler data ===
traveler_df = pd.DataFrame({
    "traveler_id": [1, 2, 3, 4, 5],
    "nationality": ["Iraq", "France", "USA", "Yemen", "Mexico"],
    "origin": ["Syria", "Germany", "Canada", "Iraq", "France"],
    "last_visited": ["Afghanistan", "Mexico", "UK", "Yemen", "Italy"]
})

# === Traveler flagging logic ===
def label_traveler(nat, orig, last, risk_map):
    risks = [risk_map.get(nat, "low"), risk_map.get(orig, "low"), risk_map.get(last, "low")]
    high, med = risks.count("high"), risks.count("medium")
    return 1 if high >= 2 or (high == 1 and med >= 1) else 0

traveler_df["flag_label"] = traveler_df.apply(
    lambda row: label_traveler(row["nationality"], row["origin"], row["last_visited"], country_risk_dict),
    axis=1
)

# === Save outputs ===
risk_df.to_csv("data/processed/country_risk_scores1.csv", index=False)
traveler_df.to_csv("data/processed/traveler_flags1.csv", index=False)

print("✅ Risk scores and traveler flags saved to 'data/processed/'")
