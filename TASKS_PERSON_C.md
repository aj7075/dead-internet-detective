# TASKS — Person C
### Dead Internet Detective | Integration + Deployment Layer

---

## Your Role

You wire everything together and get it live. You do NOT modify models.py, generator.py, tools/, or graders/.
If something in those is broken, tell Person A or B — do not patch it yourself.
Your deliverables: running HF Space, working training notebook, before/after reward plots.

---

## Start Command for Claude Code

```
Read TASKS_PERSON_C.md and dead_internet_detective/environment.py only.
Do not read tools/ or graders/ — assume they work.
My job is: complete environment.step(), server, client, inference.py, Dockerfile, HF deployment, training notebook.
```

---

## Before Starting

```bash
git checkout main
git pull origin main          # gets Person A + B's merged work
git checkout -b feat/person-c-server-deploy
```

Confirm `pytest tests/` passes before you write a single line.

---

## Tasks (in order)

### C1 — Complete environment.py step()

Person A left `step()` as a stub. Fill it in.

File: `dead_internet_detective/environment.py`

```python
def step(self, action: Action) -> tuple[Observation, float, bool, dict]:
    if self._state.done:
        raise RuntimeError("Episode already done. Call reset() first.")

    if action.tool not in VALID_TOOLS:
        obs = self._build_observation(tool_result=None,
                                       error=f"Unknown tool: {action.tool}")
        return obs, 0.0, False, {}

    # Route to tool module
    tool_fn = self._tool_registry[action.tool]
    result = tool_fn(action.params, self._state)

    # Accumulate step reward
    step_reward = result.get("step_reward", 0.0)
    self._state.accumulated_step_rewards += step_reward
    self._state.steps_used += 1

    # Check if agent is filing the final report
    done = False
    terminal_reward = 0.0
    if action.tool == "file_report" and result.get("accepted"):
        terminal_reward = compute_terminal_reward(self._state, action.params)
        done = True
        self._state.done = True

    # Check max steps
    if self._state.steps_used >= self._state.max_steps:
        done = True
        self._state.done = True

    total_reward = step_reward + terminal_reward
    obs = self._build_observation(tool_result=result, error=result.get("error"))
    return obs, total_reward, done, {"terminal_reward": terminal_reward}
```

Also implement `_build_observation()` and `_tool_registry` (a dict mapping tool name strings to the imported tool functions from tools/).

### C2 — server/app.py

FastAPI server. Session state lives in module-scope dict with a threading lock.

```python
from fastapi import FastAPI
import threading

app = FastAPI()
sessions: dict[str, State] = {}
_lock = threading.Lock()

@app.post("/reset")
def reset(body: ResetRequest) -> dict:
    env = DeadInternetEnvironment()
    obs = env.reset(task_id=body.task_id, seed=body.seed)
    session_id = str(uuid.uuid4())
    with _lock:
        sessions[session_id] = env._state
    # Store env object too so step() can use same instance
    return {"session_id": session_id, "observation": asdict(obs)}

@app.post("/step")
def step(body: StepRequest) -> dict:
    with _lock:
        env = session_envs[body.session_id]  # store env objects, not just state
    obs, reward, done, info = env.step(Action(**body.action))
    if done:
        with _lock:
            del session_envs[body.session_id]
    return {"observation": asdict(obs), "reward": reward, "done": done, "info": info}

@app.get("/state/{session_id}")
def get_state(session_id: str) -> dict:
    # For graders / debugging — not for the agent
    with _lock:
        return asdict(session_envs[session_id]._state)

@app.get("/health")
def health():
    return {"status": "ok"}
```

Define Pydantic request models `ResetRequest` and `StepRequest`.

### C3 — client/client.py

HTTP client. This file must NEVER import from `dead_internet_detective/`.
Use only `httpx` or `requests`.

```python
class DeadInternetClient:
    def __init__(self, base_url: str):
        self.base_url = base_url.rstrip("/")

    def reset(self, task_id: str = "easy", seed: int = None) -> dict:
        r = requests.post(f"{self.base_url}/reset",
                          json={"task_id": task_id, "seed": seed})
        r.raise_for_status()
        return r.json()

    def step(self, session_id: str, tool: str, params: dict) -> dict:
        r = requests.post(f"{self.base_url}/step",
                          json={"session_id": session_id,
                                "action": {"tool": tool, "params": params}})
        r.raise_for_status()
        return r.json()

    def health(self) -> bool:
        try:
            r = requests.get(f"{self.base_url}/health", timeout=3)
            return r.status_code == 200
        except Exception:
            return False
```

Add `httpx` or `requests` to `requirements.txt`.

### C4 — inference.py

Baseline dumb agent. Run this locally against Person A's server to validate the full pipeline.

```python
# inference.py
# Usage: python inference.py --url http://localhost:8000 --difficulty easy

import argparse
from client.client import DeadInternetClient

def run_baseline_episode(client, difficulty="easy", seed=42):
    """
    Dumb baseline agent:
    - Visits all 3 dossier URLs
    - Calls whois_lookup on each domain
    - Files verdict TRUE with confidence 0.5
    - No citation tracing, no case file updates
    This is the before-training behavior to compare against.
    """
    result = client.reset(task_id=difficulty, seed=seed)
    session_id = result["session_id"]
    obs = result["observation"]

    print(f"\n=== BASELINE AGENT | difficulty={difficulty} | seed={seed} ===")
    print(f"Claim: {obs['claim']}")
    print(f"Dossier: {obs['dossier_urls']}")

    total_step_reward = 0.0
    step = 0

    # Visit each dossier URL
    for url in obs["dossier_urls"]:
        r = client.step(session_id, "visit_page", {"url": url})
        total_step_reward += r["reward"]
        step += 1
        print(f"Step {step}: visit_page({url}) → reward={r['reward']:.3f}")
        if r["done"]:
            break

    # File report immediately
    r = client.step(session_id, "file_report", {
        "verdict": "true",
        "confidence": 0.5,
        "evidence_chain": obs["dossier_urls"],
        "reasoning": "Visited dossier URLs. No contradictions found."
    })
    print(f"\nFinal report filed.")
    print(f"Terminal reward: {r['info'].get('terminal_reward', 0.0):.4f}")
    print(f"Total reward: {r['reward']:.4f}")
    print(f"Done: {r['done']}")
    return r

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://localhost:8000")
    parser.add_argument("--difficulty", default="easy")
    args = parser.parse_args()

    client = DeadInternetClient(args.url)
    assert client.health(), f"Server not reachable at {args.url}"
    run_baseline_episode(client, args.difficulty)
```

**Validate locally before deploying**:
```bash
# Terminal 1 — Person A's server
uvicorn dead_internet_detective.server.app:app --reload --port 8000

# Terminal 2
python inference.py --url http://localhost:8000 --difficulty easy
```

### C5 — Dockerfile

```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
EXPOSE 7860
CMD ["uvicorn", "dead_internet_detective.server.app:app", \
     "--host", "0.0.0.0", "--port", "7860"]
```

HF Spaces requires port 7860 for the Docker SDK. Do not change this.

### C6 — HF Space deployment

```bash
# Create Space on HuggingFace (do this in the HF UI — Docker SDK type)
# Space name: dead-internet-detective
# SDK: Docker

# Push to HF Space (it's a separate git remote)
git remote add space https://huggingface.co/spaces/YOUR_HF_USERNAME/dead-internet-detective
git push space main

# Validate
curl https://YOUR_HF_USERNAME-dead-internet-detective.hf.space/health
# Expected: {"status": "ok"}
```

Update `inference.py` with the Space URL and confirm it runs end-to-end.

### C7 — ingestion/factcheck_rss.py

```python
import feedparser
import json
import os
from pathlib import Path

FEEDS = [
    ("snopes", "https://www.snopes.com/feed/"),
    ("afp", "https://factcheck.afp.com/feed"),
    ("politifact", "https://www.politifact.com/rss/all/"),
]

def ingest_fact_checks(output_dir: str = "data/real_cases"):
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    cases = []
    for source, url in FEEDS:
        feed = feedparser.parse(url)
        for entry in feed.entries[:50]:  # cap at 50 per source
            cases.append({
                "source": source,
                "claim": entry.get("title", ""),
                "verdict": entry.get("tags", [{}])[0].get("term", "unknown").lower(),
                "primary_source_url": entry.get("link", ""),
                "summary": entry.get("summary", "")
            })

    output_path = Path(output_dir) / "factcheck_cases.json"
    with open(output_path, "w") as f:
        json.dump(cases, f, indent=2)
    print(f"Saved {len(cases)} cases to {output_path}")
    return cases

if __name__ == "__main__":
    ingest_fact_checks()
```

Add `feedparser` to `requirements.txt`. Run once and commit the output to `data/real_cases/`.

### C8 — notebooks/training_grpo.ipynb

This is a Colab notebook. Structure it as these cells in order:

**Cell 1 — Install**
```python
!pip install unsloth trl wandb requests
```

**Cell 2 — Config**
```python
HF_SPACE_URL = "https://YOUR_HF_USERNAME-dead-internet-detective.hf.space"
HF_TOKEN = "your_hf_token_here"  # from HF settings
WANDB_KEY = "your_wandb_key_here"
MODEL_NAME = "unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit"
```

**Cell 3 — Load model with Unsloth**
```python
from unsloth import FastLanguageModel
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=MODEL_NAME,
    max_seq_length=2048,
    load_in_4bit=True,
)
model = FastLanguageModel.get_peft_model(model, r=16, lora_alpha=16,
    target_modules=["q_proj", "v_proj"])
```

**Cell 4 — Rollout function**
```python
import requests

def run_episode(model, tokenizer, difficulty="easy", seed=None):
    """Run one full episode. Returns total reward."""
    r = requests.post(f"{HF_SPACE_URL}/reset", json={"task_id": difficulty, "seed": seed})
    data = r.json()
    session_id = data["session_id"]
    obs = data["observation"]

    system_prompt = """You are a disinformation analyst at a media watchdog organization.
You have a 12-tool investigation desk. Maintain your case file carefully.
File your report only when you have followed citations to primary sources.
Respond with exactly one tool call per turn in JSON format: {"tool": "...", "params": {...}}"""

    messages = [{"role": "system", "content": system_prompt},
                {"role": "user", "content": f"CLAIM: {obs['claim']}\nDOSSIER: {obs['dossier_urls']}\nBegin."}]

    total_reward = 0.0
    done = False
    max_turns = 80

    for _ in range(max_turns):
        if done:
            break
        # Generate action
        input_ids = tokenizer.apply_chat_template(messages, return_tensors="pt")
        output = model.generate(input_ids, max_new_tokens=256, temperature=0.7)
        action_text = tokenizer.decode(output[0][input_ids.shape[-1]:], skip_special_tokens=True)

        try:
            action = json.loads(action_text.strip())
        except Exception:
            action = {"tool": "file_report",
                      "params": {"verdict": "unverifiable", "confidence": 0.3,
                                 "evidence_chain": [], "reasoning": "Parse error."}}

        step_r = requests.post(f"{HF_SPACE_URL}/step",
                               json={"session_id": session_id, "action": action})
        step_data = step_r.json()
        total_reward += step_data["reward"]
        done = step_data["done"]

        messages.append({"role": "assistant", "content": action_text})
        messages.append({"role": "user",
                         "content": f"TOOL RESULT: {json.dumps(step_data['observation']['tool_result'])}"})

    return total_reward
```

**Cell 5 — GRPO training loop**
```python
import wandb
wandb.login(key=WANDB_KEY)
wandb.init(project="dead-internet-detective", name="grpo-phase1")

# Simplified GRPO: generate 8 rollouts, compute relative rewards, update
from trl import GRPOConfig, GRPOTrainer

# ... GRPOTrainer setup pointing rollout_fn to run_episode ...
# Log all 5 reward components separately to W&B
# Commit PNG plots to repo after training
```

**Cell 6 — Before/after comparison**
```python
# Run inference.py baseline (dumb agent) — capture output
# Run trained model on same seed — capture output
# Print side-by-side reward breakdown
# This is the demo
```

**HF token budget**: Run a 10-step smoke test first. Full Phase 1 training (~500 steps) should cost ~$15–20 of your $30.

### C9 — README.md

Write the final README. It must include:
- Problem motivation (2 paragraphs)
- Architecture diagram (ASCII is fine)
- Before/after demo output (copy from Cell 6 output)
- Reward curve plots (embed as `![reward curves](plots/reward_curves.png)`)
- Links: HF Space URL, training notebook URL, W&B run URL
- How to run locally (3 commands)

---

## Done Criteria

- `curl https://YOUR_SPACE_URL/health` returns `{"status": "ok"}`
- `python inference.py --url https://YOUR_SPACE_URL --difficulty easy` completes with reward printed
- Training notebook runs without error on Colab (10-step smoke test)
- W&B shows 5 separate reward component curves
- Before/after comparison output exists
- README has plots embedded and all links filled in

Push branch, open PR to main, merge. Project is done.

---

## Do Not Touch

- `models.py` — if you think something is missing, ask Person A
- `tools/` — if a tool crashes, tell Person B
- `graders/` — if a grader gives wrong results, tell Person B
- `generator.py` — if generation is broken, tell Person A
