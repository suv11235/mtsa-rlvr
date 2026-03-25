from huggingface_hub import HfApi, hf_hub_download
import os

# Configuration
ADAPTER_REPO = "suv11235/red_team_model_SFT_mtsa"
BASE_MODEL = "meta-llama/Llama-3.1-8B-Instruct"

FILES_TO_UPLOAD = [
    "config.json",
    "tokenizer.model",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json"
]

api = HfApi()

for filename in FILES_TO_UPLOAD:
    try:
        print(f">>> Fetching {filename} from base model...")
        file_path = hf_hub_download(repo_id=BASE_MODEL, filename=filename)
        
        print(f">>> Uploading {filename} to adapter repo...")
        api.upload_file(
            path_or_fileobj=file_path,
            path_in_repo=filename,
            repo_id=ADAPTER_REPO,
            repo_type="model"
        )
    except Exception as e:
        print(f"!!! Error processing {filename}: {e}")

print(">>> Upload Loop Complete!")
