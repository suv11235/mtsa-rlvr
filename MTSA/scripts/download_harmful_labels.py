from huggingface_hub import list_repo_files, hf_hub_download
import pandas as pd
import json
import os

def download_and_convert():
    repo_id = "coderchen01/HarmfulGeneration-HarmBench"
    print(f">>> Listing files in {repo_id}...")
    
    files = list_repo_files(repo_id, repo_type="dataset")
    print(f"Files found: {files}")
    
    # Look for a parquet file in a 'data' folder or similar
    parquet_files = [f for f in files if f.endswith(".parquet")]
    if not parquet_files:
        print("No parquet files found.")
        return
        
    target_file = parquet_files[0] # Take the first one for now
    print(f">>> Downloading {target_file}...")
    
    file_path = hf_hub_download(
        repo_id=repo_id, 
        filename=target_file,
        repo_type="dataset"
    )

    # Load Parquet
    df = pd.read_parquet(file_path)
    print(f"Columns found: {df.columns.tolist()}")
    
    # Map to our format
    attack_data = []
    # Try different column names based on common HarmBench exports
    behavior_col = 'behavior' if 'behavior' in df.columns else ('goal' if 'goal' in df.columns else df.columns[0])
    answer_col = 'answer' if 'answer' in df.columns else ('target_response' if 'target_response' in df.columns else df.columns[1])
    
    print(f"Using columns: prompt='{behavior_col}', completion='{answer_col}'")

    for _, row in df.iterrows():
        item = {
            "goal": row[behavior_col],
            "target_response": row[answer_col]
        }
        attack_data.append(item)
    
    output_path = "/Users/suvajitmajumder/mtsa-rlvr/MTSA/datasets/attack_target/train_attack_target_labels.json"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(attack_data, f, indent=4)
        
    print(f">>> Successfully saved {len(attack_data)} expert labels to {output_path}")

if __name__ == "__main__":
    download_and_convert()
