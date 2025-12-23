from huggingface_hub import HfApi
import sys
import os

repo_id = "suv11235/mtsa-attacker-qwen2.5-7b"
folder_path = "/workspace/mtsa-rlvr/MTSA/model_output/red_team_model_data20251223035214"

api = HfApi()

print(f"Creating repo: {repo_id}")
try:
    api.create_repo(repo_id=repo_id, repo_type="model", exist_ok=True)
except Exception as e:
    print(f"Repo existence check/creation note: {e}")

print(f"Uploading folder: {folder_path}")
api.upload_folder(
    folder_path=folder_path,
    repo_id=repo_id,
    repo_type="model",
    ignore_patterns=["checkpoint-*", "optimizer.pt", "scheduler.pt", "rng_state.pth", "training_args.bin"]
)
print("Upload complete!")
