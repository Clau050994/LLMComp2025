import os
import pandas as pd

# Construct path
data_path = os.path.join("data", "raw", "globalterrorismdb.csv")
print("Trying to load file from:", data_path)


# Load dataset
df = pd.read_csv(data_path,encoding="ISO-8859-1",low_memory=False)

# Filter by year and select only specific columns
filtered_df = df[
    (df["iyear"] >= 2000) & (df["iyear"] <= 2017)
][["eventid", "iyear", "iday", "country_txt", "attacktype1_txt", "targtype1_txt", "weaptype1_txt"]]


# Save the result to processed directory
output_path = os.path.join("data", "processed", "filtered_2000_2017.csv")
filtered_df.to_csv(output_path, index=False)

