#!/usr/bin/env python3
"""
Evaluate a trained Dead Internet Detective model against the live environment.

Usage:
    python evaluate.py --model p3                          # shorthand
    python evaluate.py --model PriyanshuHF/dead-internet-detective-model-p3
    python evaluate.py --model p1 --difficulties easy medium hard --n-episodes 5
    python evaluate.py --baseline                          # run dumb baseline only
    python evaluate.py --model p3 --vs-baseline            # compare both

Requires CUDA (loads 8B model 4-bit). Set HF_TOKEN env var if models are private.
"""
import argparse, json, os, random, statistics, sys, time, warnings

warnings.filterwarnings("ignore")

HF_SPACE_URL = os.environ.get(
    "HF_SPACE_URL", "https://aryanjain7031-dead-internet-detective.hf.space"
)
HF_TOKEN = os.environ.get("HF_TOKEN", "")

MODEL_SHORTCUTS = {
    "p1": "AryanJain7031/dead-internet-detective-model",
    "p2": "PriyanshuHF/dead-internet-detective-model-p2",
    "p3": "PriyanshuHF/dead-internet-detective-model-p3",
}
BASE_MODEL = "NousResearch/Meta-Llama-3.1-8B-Instruct"

SYSTEM_PROMPT = """You are a disinformation analyst.
At each step respond ONLY with a single JSON object like:
{"tool": "<tool_name>", "params": {<params>}}

Valid tools and their required params:
visit_page:           {"url": "<url>"}
search:               {"query": "<string>"}
wayback_check:        {"url": "<url>"}
whois_lookup:         {"domain": "<domain>"}
author_lookup:        {"author_name": "<name>"}
image_provenance:     {"image_url": "<url>"}
citation_trace:       {"url": "<url>"}
linguistic_fingerprint: {"url": "<url>"}
cross_reference:      {"url_a": "<url>", "url_b": "<url>"}
update_case_file:     {"confirmed_facts": [], "synthetic_sources": [],
                       "credible_sources": [], "confidence": 0.0}
request_expert:       {"domain": "medical|legal|statistical", "question": "<q>"}
file_report:          {"verdict": "true|false|unverifiable",
                       "confidence": 0.0,
                       "evidence_chain": ["<url>"],
                       "reasoning": "<string>"}

Always end every episode by calling file_report."""

MAX_EP_STEPS = 12
MAX_EP_SECS  = 30
MAX_SEQ_LEN  = 2048


# ---------------------------------------------------------------------------
# Baseline agent (no model)
# ---------------------------------------------------------------------------

def run_baseline_episode(client, difficulty, seed):
    result = client.reset(task_id=difficulty, seed=seed)
    session_id = result["session_id"]
    obs = result["observation"]
    total = 0.0
    for url in obs.get("dossier_urls", []):
        r = client.step(session_id, "visit_page", {"url": url})
        total += r["reward"]
        if r["done"]:
            return total
    r = client.step(session_id, "file_report", {
        "verdict": "true", "confidence": 0.5,
        "evidence_chain": obs.get("dossier_urls", []),
        "reasoning": "Visited dossier URLs. No contradictions found.",
    })
    total += r["reward"]
    return total


# ---------------------------------------------------------------------------
# Model agent
# ---------------------------------------------------------------------------

def load_model(adapter_repo):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from peft import PeftModel

    print(f"Loading base model: {BASE_MODEL}")
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, token=HF_TOKEN or None)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        quantization_config=bnb,
        device_map="auto",
        attn_implementation="sdpa",
        torch_dtype=torch.bfloat16,
        token=HF_TOKEN or None,
    )
    print(f"Loading LoRA adapter: {adapter_repo}")
    model = PeftModel.from_pretrained(base, adapter_repo, token=HF_TOKEN or None)
    model.eval()
    return model, tokenizer


def generate_action(model, tokenizer, obs_dict):
    import torch
    prompt = (
        f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{SYSTEM_PROMPT}<|eot_id|>"
        f"<|start_header_id|>user<|end_header_id|>\n\nOBSERVATION:\n"
        f"{json.dumps(obs_dict, default=str)[:3000]}\n\n"
        f"Respond with one JSON action.<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
    )
    inputs = tokenizer(
        prompt, return_tensors="pt", truncation=True,
        max_length=MAX_SEQ_LEN, padding=False,
    ).to(model.device)
    with torch.no_grad():
        out = model.generate(
            **inputs, max_new_tokens=64, do_sample=True,
            temperature=0.7, pad_token_id=tokenizer.eos_token_id,
        )
    torch.cuda.empty_cache()
    decoded = tokenizer.decode(
        out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
    ).strip()
    try:
        s, e = decoded.find("{"), decoded.rfind("}") + 1
        return json.loads(decoded[s:e])
    except Exception:
        return {"tool": "file_report", "params": {
            "verdict": "unverifiable", "confidence": 0.3,
            "evidence_chain": [], "reasoning": f"parse error: {decoded[:80]}"
        }}


def run_model_episode(model, tokenizer, client, difficulty, seed):
    result = client.reset(task_id=difficulty, seed=seed)
    session_id = result["session_id"]
    obs = result["observation"]
    total = 0.0
    t0 = time.time()
    for _ in range(MAX_EP_STEPS):
        if time.time() - t0 > MAX_EP_SECS:
            client.step(session_id, "file_report", {
                "verdict": "unverifiable", "confidence": 0.3,
                "evidence_chain": [], "reasoning": "timeout",
            })
            break
        action = generate_action(model, tokenizer, obs)
        r = client.step(session_id, action["tool"], action.get("params", {}))
        total += r["reward"]
        obs = r.get("observation", obs)
        if r["done"]:
            break
    return total


# ---------------------------------------------------------------------------
# Benchmark runner
# ---------------------------------------------------------------------------

def benchmark(label, episode_fn, client, difficulties, n_episodes, seeds):
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")
    results = {}
    for diff in difficulties:
        rewards = []
        for i in range(n_episodes):
            seed = seeds[i]
            r = episode_fn(diff, seed)
            rewards.append(r)
            print(f"  [{diff}] ep {i+1}/{n_episodes}  seed={seed}  reward={r:.4f}")
        mean = statistics.mean(rewards)
        std  = statistics.stdev(rewards) if len(rewards) > 1 else 0.0
        results[diff] = {"mean": mean, "std": std, "rewards": rewards}
        print(f"  [{diff}] mean={mean:.4f}  std={std:.4f}")
    overall = statistics.mean(v["mean"] for v in results.values())
    print(f"\n  Overall mean reward: {overall:.4f}")
    return results


def print_comparison(baseline_res, model_res):
    print(f"\n{'='*60}")
    print("  COMPARISON: baseline vs model")
    print(f"{'='*60}")
    print(f"  {'Difficulty':<12} {'Baseline':>10} {'Model':>10} {'Delta':>10}")
    print(f"  {'-'*44}")
    for diff in baseline_res:
        b = baseline_res[diff]["mean"]
        m = model_res[diff]["mean"]
        delta = m - b
        sign = "+" if delta >= 0 else ""
        print(f"  {diff:<12} {b:>10.4f} {m:>10.4f} {sign}{delta:>9.4f}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", help="p1/p2/p3 or full HF repo id")
    ap.add_argument("--baseline", action="store_true", help="run baseline agent")
    ap.add_argument("--vs-baseline", action="store_true", help="run both and compare")
    ap.add_argument("--difficulties", nargs="+", default=["easy", "medium", "hard"])
    ap.add_argument("--n-episodes", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0, help="base seed (episodes use seed+i)")
    ap.add_argument("--env-url", default=HF_SPACE_URL)
    args = ap.parse_args()

    if not args.model and not args.baseline:
        ap.error("Specify --model p1/p2/p3 or --baseline (or both with --vs-baseline)")

    sys.path.insert(0, str(__file__).rsplit("\\", 1)[0])
    from client.client import DeadInternetClient

    client = DeadInternetClient(args.env_url)
    if not client.health():
        print(f"ERROR: Environment not reachable at {args.env_url}", file=sys.stderr)
        sys.exit(1)
    print(f"Environment OK: {args.env_url}")

    seeds = [args.seed + i for i in range(args.n_episodes)]
    difficulties = args.difficulties

    baseline_res = None
    model_res    = None

    if args.baseline or args.vs_baseline:
        baseline_res = benchmark(
            "BASELINE AGENT (dumb — visits dossier URLs, always verdicts true)",
            lambda diff, seed: run_baseline_episode(client, diff, seed),
            client, difficulties, args.n_episodes, seeds,
        )

    if args.model:
        repo = MODEL_SHORTCUTS.get(args.model, args.model)
        model, tokenizer = load_model(repo)
        model_res = benchmark(
            f"TRAINED MODEL ({repo})",
            lambda diff, seed: run_model_episode(model, tokenizer, client, diff, seed),
            client, difficulties, args.n_episodes, seeds,
        )

    if baseline_res and model_res:
        print_comparison(baseline_res, model_res)

    print("\nDone.")


if __name__ == "__main__":
    main()
