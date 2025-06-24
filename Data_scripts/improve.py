#!/usr/bin/env python3

import pandas as pd
import re
import os

# Define pattern dictionaries for different categories
food_patterns = ["food", "cuisine", "dishes", "restaurant", "meal", "cooking", "eats"]
culture_patterns = ["culture", "music", "art", "literature", "architecture", "dance", "festival", "tradition"]
history_patterns = ["history", "historical", "ancient", "studies", "researches", "scholar"]
media_patterns = ["watches", "movies", "listens", "songs", "reads", "books", "follows", "shows", "media"]
interest_patterns = ["interested in", "appreciates", "admires", "fan of", "enjoys", "likes", "collects"]

# Origin patterns indicate the actual traveler origin
origin_patterns = ["from", "passport", "citizen", "national", "born in", "resident", 
                   "traveling", "departed", "arriving", "originating"]

def label_example(row):
    text = row['input_text'].lower()
    
    # For unlabeled examples or NaN labels
    if pd.isna(row['example_type']):
        # First check if this is clearly an origin statement
        if any(pattern in text for pattern in origin_patterns):
            # Still check if it has confounding elements
            if any(pattern in text for pattern in food_patterns + culture_patterns + 
                  history_patterns + media_patterns + interest_patterns):
                # Contains origin but also confounding elements - mark as confounding
                return "confounding"
            return "origin"
            
        # For examples without explicit origin patterns but with other indicators
        if any(pattern in text for pattern in food_patterns):
            return "food"
        if any(pattern in text for pattern in culture_patterns):
            return "culture"
        if any(pattern in text for pattern in history_patterns):
            return "history"
        if any(pattern in text for pattern in media_patterns):
            return "media"
        if any(pattern in text for pattern in interest_patterns):
            return "interest"
            
        # Default case for anything else
        return "other"
    
    return row['example_type']

def process_file(filepath):
    print(f"Processing: {filepath}")
    df = pd.read_csv(filepath)
    
    # Apply the labeling function
    df['example_type'] = df.apply(label_example, axis=1)
    
    # Print statistics
    print("Example type distribution:")
    print(df['example_type'].value_counts())
    
    # Save with the same filename (overwrite)
    df.to_csv(filepath, index=False)
    print(f"Labeled {len(df)} examples and saved back to {filepath}\n")

if __name__ == "__main__":
    # Process validation and test files
    val_file = "data/processed/unified/unified_val.csv"
    test_file = "data/processed/unified/unified_test.csv"
    
    process_file(val_file)
    process_file(test_file)
    
    print("All datasets labeled successfully!")