import pandas as pd

# Check validation file
val_file = "data/processed/unified/unified_val.csv"
val_df = pd.read_csv(val_file)
print("Validation file columns:", val_df.columns.tolist())
print("Validation file sample:\n", val_df.head(3))
print("Validation labels distribution:", val_df['label'].value_counts().to_dict() if 'label' in val_df.columns else "No label column found")

# Check test file
test_file = "data/processed/unified/unified_test.csv"
test_df = pd.read_csv(test_file)
print("\nTest file columns:", test_df.columns.tolist())
print("Test file sample:\n", test_df.head(3))
print("Test labels distribution:", test_df['label'].value_counts().to_dict() if 'label' in test_df.columns else "No label column found")

# Check if example_type exists in validation/test files
print("\nExample type in validation:", 'example_type' in val_df.columns)
print("Example type in test:", 'example_type' in test_df.columns)