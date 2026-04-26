"""
Manual log dump — call when a Space hits phase=='done' but was running pre-patch.
Fetches /status, packages it, uploads training_log.json to the model repo.
"""
import argparse, json, os, requests, tempfile
from huggingface_hub import HfApi

PHASES = {
    "p1": {
        "host": "aryanjain7031-dead-internet-detective-trainer.hf.space",
        "model_repo": "AryanJain7031/dead-internet-detective-model",
        "token": os.environ.get("HF_TOKEN_P1", ""),
        "label": "phase1-easy",
    },
    "p2": {
        "host": "priyanshuhf-dead-internet-detective-trainer-p2.hf.space",
        "model_repo": "PriyanshuHF/dead-internet-detective-model-p2",
        "token": os.environ.get("HF_TOKEN_P2", ""),
        "label": "phase2-mixed",
    },
}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("phase", choices=list(PHASES.keys()))
    args = ap.parse_args()
    cfg = PHASES[args.phase]

    s = requests.get(f"https://{cfg['host']}/status", timeout=20).json()
    payload = {"label": cfg["label"], "final_status": s, "log": s.get("log", [])}

    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
        json.dump(payload, f, indent=2, default=str)
        path = f.name

    api = HfApi(token=cfg["token"])
    api.upload_file(
        path_or_fileobj=path,
        path_in_repo="training_log.json",
        repo_id=cfg["model_repo"],
        repo_type="model",
        commit_message=f"upload {cfg['label']} training log",
    )
    os.unlink(path)
    print(f"OK uploaded → https://huggingface.co/{cfg['model_repo']}/blob/main/training_log.json")
    print(f"   captured {len(s.get('log', []))} log lines, phase={s.get('phase')}, step={s.get('step')}/{s.get('total')}")

if __name__ == "__main__":
    main()
