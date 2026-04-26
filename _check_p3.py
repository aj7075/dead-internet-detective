"""Monitor Phase 3 training."""
import requests, time

slug = "PriyanshuHF/dead-internet-detective-trainer-p3"
host = "priyanshuhf-dead-internet-detective-trainer-p3.hf.space"

r = requests.get(f"https://huggingface.co/api/spaces/{slug}/runtime", timeout=15).json()
sha = (r.get("sha") or "")[:10]
hw  = r.get("hardware", {})
print(f"=== P3 === STAGE: {r.get('stage')}  HW: {hw}  SHA: {sha}")
try:
    d = requests.get(f"https://{host}/status", timeout=10).json()
    print(f"  PHASE: {d.get('phase')}  STEP: {d.get('step')}/{d.get('total')}  REWARD: {d.get('mean_reward')}")
    for l in d.get("log", [])[-10:]:
        print("   ", l)
except Exception as e:
    print("  STATUS ERR:", e)
print("NOW:", time.strftime("%H:%M:%S"))
