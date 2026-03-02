# upload_models.py
# Uploads both model checkpoints to Hugging Face Hub

from huggingface_hub import HfApi
import os

# ── replace with your HF username ──
HF_USERNAME  = "debojitbasak"
REPO_NAME    = "minigpt-models"
REPO_ID      = f"{HF_USERNAME}/{REPO_NAME}"

api = HfApi()

# Create repo if it doesn't exist (public)
api.create_repo(repo_id=REPO_ID, repo_type="model", exist_ok=True, private=False)

models = [
    {
        'local_path': 'checkpoints/english_maxed_model.pth',
        'repo_path':  'english_maxed_model.pth',
    },
    {
        'local_path': 'checkpoints/bangla_improved_model.pth',
        'repo_path':  'bangla_improved_model.pth',
    },
]

print("=" * 55)
print("  Uploading Models to Hugging Face Hub")
print("=" * 55)
print(f"  Repo: {REPO_ID}")
print()

for m in models:
    size_mb = os.path.getsize(m['local_path']) / (1024*1024)
    print(f"Uploading {m['local_path']} ({size_mb:.1f}MB)...")

    api.upload_file(
        path_or_fileobj = m['local_path'],
        path_in_repo    = m['repo_path'],
        repo_id         = REPO_ID,
        repo_type       = "model",
    )
    print(f"  ✓ Done!\n")

print("=" * 55)
print("  All models uploaded!")
print(f"  View at: https://huggingface.co/{REPO_ID}")
print("=" * 55)