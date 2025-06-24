import pandas as pd

# Load the dataset
train_file = "data/processed/unified/unified_complete_labeled.csv"
print(f"Cleaning {train_file}...")

# Load with header=None
df = pd.read_csv(train_file, header=None)
df.columns = ["input_text", "label", "example_type"]

# Remove header row if it exists
if any(col == 'label' for col in df['label']):
    df = df[df['label'] != 'label']
    print("Removed header row")

# Convert labels to integers
df['label'] = pd.to_numeric(df['label'], errors='coerce')

# Check for any NaN values after conversion
if df['label'].isna().any():
    print(f"Warning: Found {df['label'].isna().sum()} rows with invalid label values")
    # Drop rows with NaN labels
    df = df.dropna(subset=['label'])
    print(f"Dropped rows with invalid labels. New dataset size: {len(df)}")

# Convert to int
df['label'] = df['label'].astype(int)

# Verify the distribution
print("Label distribution after cleaning:")
print(df['label'].value_counts().to_dict())
print("Example type distribution after cleaning:")
print(df['example_type'].value_counts().to_dict())

# Save the cleaned dataset
df.to_csv(train_file, header=False, index=False)
print(f"Saved cleaned dataset with {len(df)} examples")

# Also check and clean validation and test files
for file in ["data/processed/unified/unified_val.csv", "data/processed/unified/unified_test.csv"]:
    val_df = pd.read_csv(file)
    
    # Ensure label is an integer
    if val_df['label'].dtype != 'int64':
        val_df['label'] = val_df['label'].astype(int)
        print(f"Converted labels to integers in {file}")
        
    # Save with proper headers
    val_df.to_csv(file, index=False)
    print(f"Cleaned {file}")