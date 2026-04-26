#!/usr/bin/env python3
"""
Dead Internet Detective — Evaluation Space
Loads a trained LoRA model and benchmarks it against the live environment.
"""
import gc, json, os, random, sys, time, warnings
warnings.filterwarnings("ignore")

import gradio as gr

sys.path.insert(0, "/app")
from client.client import DeadInternetClient

ENV_URL      = "https://aryanjain7031-dead-internet-detective.hf.space"
HF_TOKEN     = os.environ.get("HF_TOKEN", "")
BASE_MODEL   = "NousResearch/Meta-Llama-3.1-8B-Instruct"
MAX_EP_STEPS = 12
MAX_EP_SECS  = 35

MODEL_OPTIONS = {
    "P3 — phase3-improved (easy+medium+hard, 40 steps)": "PriyanshuHF/dead-internet-detective-model-p3",
    "P2 — phase2-mixed    (easy+medium+hard, 20 steps)": "PriyanshuHF/dead-internet-detective-model-p2",
    "P1 — phase1-easy     (easy only, 15 steps)":        "AryanJain7031/dead-internet-detective-model",
    "Baseline (dumb — no model)":                        "__baseline__",
}

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

# Model cache — keep loaded between runs
_loaded = {"repo": None, "model": None, "tokenizer": None}


def _load_model(repo):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from peft import PeftModel

    if _loaded["repo"] == repo:
        return _loaded["model"], _loaded["tokenizer"]

    # Unload previous
    if _loaded["model"] is not None:
        del _loaded["model"]
        gc.collect()
        torch.cuda.empty_cache()

    bnb = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True,
    )
    tok = AutoTokenizer.from_pretrained(BASE_MODEL, token=HF_TOKEN or None)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    base = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, quantization_config=bnb, device_map="auto",
        attn_implementation="sdpa", torch_dtype=torch.bfloat16, token=HF_TOKEN or None,
    )
    model = PeftModel.from_pretrained(base, repo, token=HF_TOKEN or None)
    model.eval()

    _loaded["repo"] = repo
    _loaded["model"] = model
    _loaded["tokenizer"] = tok
    return model, tok


def _generate_action(model, tokenizer, obs_dict):
    import torch
    prompt = (
        f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{SYSTEM_PROMPT}<|eot_id|>"
        f"<|start_header_id|>user<|end_header_id|>\n\nOBSERVATION:\n"
        f"{json.dumps(obs_dict, default=str)[:3000]}\n\n"
        f"Respond with one JSON action.<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
    )
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True,
                       max_length=2048, padding=False).to(model.device)
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
        return json.loads(decoded[s:e]), decoded
    except Exception:
        return {"tool": "file_report", "params": {
            "verdict": "unverifiable", "confidence": 0.3,
            "evidence_chain": [], "reasoning": f"parse error: {decoded[:80]}"
        }}, decoded


def _run_model_episode(model, tokenizer, client, difficulty, seed):
    res = client.reset(task_id=difficulty, seed=seed)
    sid, obs = res["session_id"], res["observation"]
    lines = [f"**Claim:** {obs.get('claim', '?')}"]
    total, t0 = 0.0, time.time()
    for step_i in range(MAX_EP_STEPS):
        if time.time() - t0 > MAX_EP_SECS:
            client.step(sid, "file_report", {"verdict": "unverifiable", "confidence": 0.3, "evidence_chain": [], "reasoning": "timeout"})
            lines.append(f"  step {step_i+1}: ⏱ timeout")
            break
        action, raw = _generate_action(model, tokenizer, obs)
        r = client.step(sid, action["tool"], action.get("params", {}))
        total += r["reward"]
        lines.append(f"  step {step_i+1}: `{action['tool']}` → reward {r['reward']:+.3f}")
        obs = r.get("observation", obs)
        if r["done"]:
            break
    lines.append(f"**Episode reward: {total:.4f}**")
    return total, "\n".join(lines)


def _run_baseline_episode(client, difficulty, seed):
    res = client.reset(task_id=difficulty, seed=seed)
    sid, obs = res["session_id"], res["observation"]
    lines = [f"**Claim:** {obs.get('claim', '?')}"]
    total = 0.0
    for url in obs.get("dossier_urls", []):
        r = client.step(sid, "visit_page", {"url": url})
        total += r["reward"]
        lines.append(f"  visit_page({url}) → reward {r['reward']:+.3f}")
        if r["done"]:
            break
    r = client.step(sid, "file_report", {"verdict": "true", "confidence": 0.5, "evidence_chain": obs.get("dossier_urls", []), "reasoning": "visited dossier"})
    total += r["reward"]
    lines.append(f"  file_report → reward {r['reward']:+.3f}")
    lines.append(f"**Episode reward: {total:.4f}**")
    return total, "\n".join(lines)


# ---------------------------------------------------------------------------
# Gradio handler (generator — streams output line by line)
# ---------------------------------------------------------------------------

def run_eval(model_label, difficulties_in, n_episodes, seed_base, progress=gr.Progress()):
    difficulties = [d.strip() for d in difficulties_in]
    repo = MODEL_OPTIONS[model_label]
    client = DeadInternetClient(ENV_URL)

    yield "⏳ Checking environment…\n"
    if not client.health():
        yield f"❌ Environment not reachable at {ENV_URL}"
        return

    yield f"✅ Environment OK: {ENV_URL}\n\n"

    if repo != "__baseline__":
        yield f"⏳ Loading model `{repo}`…  (first load ~2–3 min)\n"
        try:
            model, tokenizer = _load_model(repo)
        except Exception as e:
            yield f"❌ Model load failed: {e}"
            return
        yield "✅ Model ready.\n\n"

    all_rewards: dict[str, list[float]] = {d: [] for d in difficulties}
    log_lines = []

    total_eps = len(difficulties) * n_episodes
    ep_idx = 0

    for diff in difficulties:
        log_lines.append(f"\n### Difficulty: `{diff}`\n")
        yield "".join(log_lines)
        for i in range(n_episodes):
            seed = seed_base + ep_idx
            ep_idx += 1
            progress(ep_idx / total_eps, desc=f"{diff} ep {i+1}/{n_episodes}")
            log_lines.append(f"\n**Episode {i+1}/{n_episodes}** (seed={seed})\n")
            yield "".join(log_lines)

            if repo == "__baseline__":
                r, detail = _run_baseline_episode(client, diff, seed)
            else:
                r, detail = _run_model_episode(model, tokenizer, client, diff, seed)

            all_rewards[diff].append(r)
            log_lines.append(detail + "\n")
            yield "".join(log_lines)

    # Summary table
    log_lines.append("\n---\n## Results\n")
    log_lines.append("| Difficulty | Mean Reward | Std |\n|---|---|---|\n")
    overall = []
    for diff in difficulties:
        rs = all_rewards[diff]
        mean = sum(rs) / len(rs)
        std  = (sum((x - mean)**2 for x in rs) / max(len(rs)-1, 1)) ** 0.5
        overall.append(mean)
        log_lines.append(f"| {diff} | {mean:.4f} | ±{std:.4f} |\n")
    log_lines.append(f"\n**Overall mean: {sum(overall)/len(overall):.4f}**\n")
    yield "".join(log_lines)


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------

with gr.Blocks(title="Dead Internet Detective — Evaluator") as demo:
    gr.Markdown(
        "# Dead Internet Detective — Model Evaluator\n"
        "Runs trained GRPO LoRA models against the synthetic disinformation environment.\n\n"
        "> **Hardware:** A10G Small required for P1/P2/P3 models (8B 4-bit). "
        "Baseline runs on CPU.\n"
        "> **Environment:** `aryanjain7031-dead-internet-detective.hf.space`"
    )

    with gr.Row():
        model_dd   = gr.Dropdown(list(MODEL_OPTIONS.keys()), value=list(MODEL_OPTIONS.keys())[0], label="Model")
        diff_check = gr.CheckboxGroup(["easy", "medium", "hard"], value=["easy", "medium", "hard"], label="Difficulties")

    with gr.Row():
        n_ep_sl  = gr.Slider(1, 10, value=3, step=1, label="Episodes per difficulty")
        seed_num = gr.Number(value=42, label="Base seed", precision=0)

    run_btn = gr.Button("▶ Run Evaluation", variant="primary")
    output  = gr.Markdown(label="Output")

    run_btn.click(
        fn=run_eval,
        inputs=[model_dd, diff_check, n_ep_sl, seed_num],
        outputs=output,
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, show_error=True)
