import pandas as pd
import re

# Load the dataset
df = pd.read_csv('data/processed/unified/unified_complete.csv', header=None, 
                names=['input_text', 'risk', 'example_type'])

# Define pattern dictionaries for different categories
food_patterns = ["food", "cuisine", "dishes", "restaurant", "meal", "cooking", "eats"]
culture_patterns = ["culture", "music", "art", "literature", "architecture", "dance", "festival", "tradition"]
history_patterns = ["history", "historical", "ancient", "studies", "researches", "scholar"]
media_patterns = ["watches", "movies", "listens", "songs", "reads", "books", "follows", "shows", "media"]
interest_patterns = ["interested in", "appreciates", "admires", "fan of", "enjoys", "likes", "collects"]

# Origin patterns indicate the actual traveler origin
origin_patterns = ["from", "passport from", "citizen of", "national of", "born in", "resident of", 
                  "traveling from", "departed from", "arriving from", "originating from"]

# Function to label examples based on content patterns
def label_example(row):
    if not pd.isna(row['example_type']):
        # If already labeled as syntactic, keep but refine the label
        if row['example_type'] == 'syntactic':
            text = row['input_text'].lower()
            
            # Check for food references
            if any(pattern in text for pattern in food_patterns):
                return "food"
                
            # Check for cultural references
            if any(pattern in text for pattern in culture_patterns):
                return "culture"
                
            # Check for historical references
            if any(pattern in text for pattern in history_patterns):
                return "history"
                
            # Check for media consumption
            if any(pattern in text for pattern in media_patterns):
                return "media"
                
            # Check for general interest without specific category
            if any(pattern in text for pattern in interest_patterns):
                return "interest"
                
            # Default case - keep syntactic label
            return "syntactic"
        else:
            return row['example_type']
    
    # For unlabeled examples
    text = row['input_text'].lower()
    
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

# Apply the labeling function
df['example_type'] = df.apply(label_example, axis=1)

# Print statistics
print("Example type distribution:")
print(df['example_type'].value_counts())

# Save the updated dataset
df.to_csv('data/processed/unified/unified_complete_labeled.csv', index=False, header=False)
print(f"Saved labeled dataset with {len(df)} examples")