import json
import os
import re

# Define paths
base_dir = "/Users/suvajitmajumder/mtsa-rlvr/MTSA"
input_path = os.path.join(base_dir, "datasets/attack_target/train_attack_target.json")
output_path = os.path.join(base_dir, "datasets/attack_target/bio_attack_target.json")

# Define bio-related keywords
bio_keywords = [
    "virus", "poison", "bio", "disease", "health", "bacteria", 
    "pandemic", "chemical", "toxin", "biological", "pathogen", 
    "epidemic", "drug", "medicine", "vaccine", "synthesize",
    "substance", "deadly"
]

def is_bio_related(text):
    text = text.lower()
    for keyword in bio_keywords:
        if keyword in text:
            return True
    return False

def filter_dataset():
    print(f"Loading dataset from {input_path}...")
    with open(input_path, 'r') as f:
        data = json.load(f)
    
    print(f"Total entries: {len(data)}")
    
    filtered_data = []
    for entry in data:
        # Check both goal and target_response if available, though usually goal is enough
        text_to_check = entry.get('goal', '') + " " + entry.get('target_response', '')
        
        if is_bio_related(text_to_check):
            filtered_data.append(entry)
            
    print(f"Filtered {len(filtered_data)} bio-related entries.")
    
    # Save filtered dataset
    print(f"Saving to {output_path}...")
    with open(output_path, 'w') as f:
        json.dump(filtered_data, f, indent=4)
    print("Done.")

if __name__ == "__main__":
    filter_dataset()
