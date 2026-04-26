"""Phase 3 setup: create model repo + Space, upload training_space_phase3/, start on A10G.
Usage: HF_TOKEN_P2=<your_token> python _phase3_setup.py
"""
import os
from huggingface_hub import HfApi, create_repo

TOKEN = os.environ.get("HF_TOKEN_P2", "")
assert TOKEN, "Set HF_TOKEN_P2 env var before running"
api   = HfApi(token=TOKEN)

MODEL_REPO   = "PriyanshuHF/dead-internet-detective-model-p3"
TRAINER_REPO = "PriyanshuHF/dead-internet-detective-trainer-p3"
FOLDER       = "./training_space_phase3"

# 1. Model repo
try:
    create_repo(MODEL_REPO, token=TOKEN, repo_type="model", exist_ok=True, private=False)
    print(f"MODEL_REPO: ok  → https://huggingface.co/{MODEL_REPO}")
except Exception as e:
    print(f"MODEL_REPO ERR: {e}")

# 2. Space (Docker SDK)
try:
    create_repo(TRAINER_REPO, token=TOKEN, repo_type="space",
                space_sdk="docker", exist_ok=True, private=False)
    print(f"SPACE: ok  → https://huggingface.co/spaces/{TRAINER_REPO}")
except Exception as e:
    print(f"SPACE ERR: {e}")

# 3. Secrets
for key, val in [("HF_TOKEN", TOKEN), ("HF_MODEL_REPO", MODEL_REPO)]:
    try:
        api.add_space_secret(repo_id=TRAINER_REPO, key=key, value=val)
        print(f"SECRET {key}: ok")
    except Exception as e:
        print(f"SECRET {key} ERR: {e}")

# 4. Upload code
try:
    api.upload_folder(
        folder_path=FOLDER,
        repo_id=TRAINER_REPO,
        repo_type="space",
        commit_message="phase3: fix max_new_tokens=128, MAX_EP_STEPS=12, 40 train steps",
        ignore_patterns=["__pycache__", "*.pyc", ".DS_Store"],
    )
    print("UPLOAD: ok")
except Exception as e:
    print(f"UPLOAD ERR: {e}")

# 5. Request A10G hardware
try:
    api.request_space_hardware(repo_id=TRAINER_REPO, hardware="a10g-small")
    print("HARDWARE a10g-small: requested")
except Exception as e:
    print(f"HW ERR: {e}")

print(f"\nDone. Monitor: python _check_p3.py")
print(f"Space URL: https://huggingface.co/spaces/{TRAINER_REPO}")
