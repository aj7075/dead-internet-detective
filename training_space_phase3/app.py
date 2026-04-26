"""
Training Space web server — shows live training status at port 7860.
Launches train.py in a background thread on startup.
Also exposes /infer to run the trained LoRA model against the environment.
"""
import threading
import os
import train as trainer
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel
import uvicorn

app = FastAPI()

# ── Inference state (lazy-loaded on first /infer call) ───────────────────────
_infer_model = None
_infer_tokenizer = None
_infer_lock = threading.Lock()
ADAPTER_REPO = "PriyanshuHF/dead-internet-detective-model-p3"
BASE_MODEL   = "NousResearch/Meta-Llama-3.1-8B-Instruct"


def _load_model():
    global _infer_model, _infer_tokenizer
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from peft import PeftModel

    hf_token = os.environ.get("HF_TOKEN")
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(ADAPTER_REPO, token=hf_token)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        quantization_config=bnb,
        device_map="auto",
        attn_implementation="sdpa",
        torch_dtype=torch.bfloat16,
        token=hf_token,
    )
    model = PeftModel.from_pretrained(base, ADAPTER_REPO, token=hf_token)
    model.eval()
    _infer_model = model
    _infer_tokenizer = tokenizer


def _get_model():
    global _infer_model, _infer_tokenizer
    with _infer_lock:
        if _infer_model is None:
            _load_model()
    return _infer_model, _infer_tokenizer


class InferRequest(BaseModel):
    difficulty: str = "easy"
    seed: int = 42
    n_episodes: int = 1


class InferBatchRequest(BaseModel):
    seeds: list[int] = list(range(500, 510))
    difficulty: str = "easy"

HTML = """<!DOCTYPE html>
<html>
<head>
  <title>Dead Internet Detective — Training</title>
  <meta http-equiv="refresh" content="10">
  <style>
    body {{ font-family: monospace; background: #111; color: #eee; padding: 2rem; }}
    h1 {{ color: #4af; }}
    .status {{ font-size: 1.4rem; margin: 1rem 0; }}
    .bar {{ background: #333; border-radius: 4px; height: 24px; width: 100%; max-width: 600px; }}
    .fill {{ background: #4af; height: 24px; border-radius: 4px; transition: width 0.5s; }}
    pre {{ background: #1a1a1a; padding: 1rem; max-height: 400px; overflow-y: auto;
           border-radius: 4px; font-size: 0.85rem; }}
  </style>
</head>
<body>
  <h1>Dead Internet Detective — GRPO Training (A10G)</h1>
  <div class="status">Phase: <b>{phase}</b></div>
  <div class="status">Step: {step} / {total} &nbsp;|&nbsp; Last reward: {reward:.3f}</div>
  <div class="bar"><div class="fill" style="width:{pct:.1f}%"></div></div>
  <br>
  <pre>{logs}</pre>
</body>
</html>"""


@app.get("/", response_class=HTMLResponse)
def index():
    s = trainer.STATUS
    pct = (s["step"] / max(s["total"], 1)) * 100
    logs = "\n".join(s["log"][-50:])
    return HTML.format(
        phase=s["phase"],
        step=s["step"],
        total=s["total"],
        reward=s["mean_reward"],
        pct=pct,
        logs=logs,
    )


@app.get("/status")
def status():
    return JSONResponse(trainer.STATUS)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/infer")
def infer(req: InferRequest):
    """Run one or more episodes with the trained LoRA model. Returns rewards + verdicts."""
    try:
        model, tokenizer = _get_model()
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))

    from client.client import DeadInternetClient
    client = DeadInternetClient(trainer.HF_SPACE_URL)
    if not client.health():
        raise HTTPException(status_code=502, detail=f"Env server unreachable: {trainer.HF_SPACE_URL}")

    results = []
    for _ in range(req.n_episodes):
        r = trainer.run_episode(model, tokenizer, client, req.difficulty, req.seed)
        results.append(r)

    return {
        "model": "PriyanshuHF/dead-internet-detective-model-p3",
        "difficulty": req.difficulty,
        "seed": req.seed,
        "results": results,
        "mean_reward": round(sum(r["total_reward"] for r in results) / len(results), 4),
    }


@app.post("/infer/batch")
def infer_batch(req: InferBatchRequest):
    """Run trained model on multiple seeds. Compare against baseline."""
    try:
        model, tokenizer = _get_model()
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))

    from client.client import DeadInternetClient
    client = DeadInternetClient(trainer.HF_SPACE_URL)
    if not client.health():
        raise HTTPException(status_code=502, detail=f"Env server unreachable: {trainer.HF_SPACE_URL}")

    results = []
    for seed in req.seeds:
        r = trainer.run_episode(model, tokenizer, client, req.difficulty, seed)
        results.append(r)

    rewards = [r["total_reward"] for r in results]
    return {
        "model": "PriyanshuHF/dead-internet-detective-model-p3",
        "difficulty": req.difficulty,
        "seeds": req.seeds,
        "n_episodes": len(results),
        "mean_reward": round(sum(rewards) / len(rewards), 4),
        "min_reward": round(min(rewards), 4),
        "max_reward": round(max(rewards), 4),
        "results": results,
    }


@app.post("/infer/debug")
def infer_debug(req: InferRequest):
    """Run one episode and return full traceback on error."""
    import traceback, sys
    try:
        model, tokenizer = _get_model()
    except Exception as e:
        return {"error": "model_load_failed", "detail": traceback.format_exc()}

    from client.client import DeadInternetClient
    client = DeadInternetClient(trainer.HF_SPACE_URL)

    # Test reset first
    try:
        reset_result = client.reset(task_id=req.difficulty, seed=req.seed)
        session_id = reset_result["session_id"]
        obs = reset_result["observation"]
    except Exception as e:
        return {"error": "reset_failed", "detail": str(e), "traceback": traceback.format_exc()}

    # Test one generate_action
    try:
        action = trainer.generate_action(model, tokenizer, obs)
    except Exception as e:
        return {"error": "generate_action_failed", "detail": str(e), "traceback": traceback.format_exc()}

    # Test one step
    try:
        result = client.step(session_id, action["tool"], action.get("params", {}))
    except Exception as e:
        return {"error": "step_failed", "action": action, "detail": str(e)}

    return {
        "ok": True,
        "action_generated": action,
        "step_reward": result.get("reward"),
        "done": result.get("done"),
        "obs_claim": obs.get("claim", "")[:100],
    }


@app.get("/infer/status")
def infer_status():
    """Check if trained model is loaded and ready."""
    return {
        "adapter_repo": ADAPTER_REPO,
        "model_loaded": _infer_model is not None,
        "env_server": trainer.HF_SPACE_URL,
        "ready": _infer_model is not None,
    }


def _start_training():
    try:
        trainer.run_training()
    except Exception as e:
        trainer._log(f"TRAINING ERROR: {e}")
        trainer.STATUS["phase"] = f"error: {e}"


if __name__ == "__main__":
    skip = os.environ.get("SKIP_TRAINING", "false").lower() == "true"
    if skip:
        trainer.STATUS["phase"] = "inference-only"
        trainer._log("SKIP_TRAINING=true — serving /infer only, no training.")
    else:
        t = threading.Thread(target=_start_training, daemon=True)
        t.start()
    uvicorn.run(app, host="0.0.0.0", port=7860)
