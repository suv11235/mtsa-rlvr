from huggingface_hub import HfApi
import sys
import os

if len(sys.argv) < 3:
    print("Usage: python script/push_to_hf.py <repo_id> <folder_path>")
    sys.exit(1)

repo_id = sys.argv[1]
folder_path = sys.argv[2]

api = HfApi()

print(f"Creating repo: {repo_id}")
try:
    api.create_repo(repo_id=repo_id, repo_type="model", exist_ok=True, private=False)
except Exception as e:
    print(f"Repo existence check/creation note: {e}")

print(f"Uploading folder: {folder_path}")
api.upload_folder(
    folder_path=folder_path,
    repo_id=repo_id,
    repo_type="model",
    ignore_patterns=["checkpoint-*", "global_step*", "optimizer.pt", "scheduler.pt", "rng_state.pth", "training_args.bin"]
)
print("Upload complete!")
