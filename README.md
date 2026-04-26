---
title: Dead Internet Detective
emoji: 🕵️
colorFrom: red
colorTo: gray
sdk: docker
app_port: 7860
pinned: false
---

# Dead Internet Detective

We trained an LLM to investigate disinformation the way a journalist would — not by reading and classifying, but by following leads through a synthetic internet we built from scratch.

[![Live Environment](https://img.shields.io/badge/🤗_Space-live_environment-blue)](https://huggingface.co/spaces/AryanJain7031/dead-internet-detective)
[![Eval Space](https://img.shields.io/badge/🤗_Eval-run_the_model-green)](https://huggingface.co/spaces/AryanJain7031/dead-internet-detective-eval)
[![Model P3](https://img.shields.io/badge/🤗_Model-P3_trained-red)](https://huggingface.co/PriyanshuHF/dead-internet-detective-model-p3)
[![W&B](https://img.shields.io/badge/W%26B-training_run-yellow)](https://wandb.ai/models-srm-institute-of-science-and-technology9361/dead-internet-detective/workspace?nw=nwuseraryanjain7031)
[![Notebook](https://img.shields.io/badge/Colab-training_notebook-orange)](notebooks/training_grpo.ipynb)

---

## The Problem with Disinformation Detection

Most approaches treat disinformation as a classification problem. Feed in an article, get back a label. True or false.

This misses the point entirely.

Sophisticated disinformation doesn't fail because it sounds suspicious. It fails because its citations don't trace back to anything real. Because the author's name appears on forty different sites with conflicting credentials. Because the domain was registered two weeks before the story broke. Because the "study" being cited was quietly retracted. None of that is in the headline. You have to investigate.

The gap between *classifying text* and *investigating a claim* is enormous — and almost nobody has tried to close it with RL training on LLMs. That's what this project does.

---

## What We Built

Dead Internet Detective is a multi-step investigation environment. The agent receives a **claim** and a set of **dossier URLs** — entry points into a fully synthetic internet we built specifically for this project.

That synthetic internet is the core of what makes this environment interesting. It's not a static dataset. Every episode generates a fresh set of pages: legitimate-looking credible sources, SEO farms recycling the same talking points, wellness blogs with no real author credentials, and — at hard difficulty — high-quality fakes that convincingly spoof real credible domains. The citation chains look plausible. They just don't lead anywhere real.

The agent has **12 investigative tools** to work through the evidence:

| Tool | What it does |
|------|-------------|
| `visit_page` | Read a synthetic webpage |
| `search` | Query the synthetic search engine |
| `wayback_check` | Check if a URL has a history — new domains are suspicious |
| `whois_lookup` | Domain registration date and registrar |
| `author_lookup` | Is this author's name tied to multiple shady domains? |
| `image_provenance` | Where else has this image appeared online? |
| `citation_trace` | Follow a citation chain up to 3 hops |
| `linguistic_fingerprint` | AI-generation probability and writing patterns |
| `cross_reference` | Do two sources directly contradict each other? |
| `update_case_file` | Log evidence and maintain the investigation record |
| `request_expert` | Get a synthetic domain expert's opinion (medical / legal / statistical) |
| `file_report` | File final verdict and close the episode |

The agent maintains a **case file** throughout the investigation — a running record of confirmed facts, disputed claims, flagged synthetic sources, contradictions found, and overall confidence. When it's ready (or when it runs out of steps), it files a report: verdict, confidence score, evidence chain, and reasoning.

---

## Why This Needed RL

You can't train an investigator with supervised learning on verdict labels. The investigation process matters as much as the conclusion. An agent that guesses correctly by luck and uses zero tools should score lower than one that builds a real evidence chain — even if both get the right verdict.

The reward function captures this:

| Component | Weight | What it actually measures |
|-----------|:------:|--------------------------|
| Verdict accuracy | 30% | Did you reach the right conclusion? |
| Evidence chain quality | 25% | Did you visit and cite the actually relevant pages? |
| Synthetic source detection | 20% | Did you correctly flag the fake pages as fake? |
| Internal consistency | 15% | Does your filed reasoning match your collected evidence? |
| Step efficiency | 10% | Did you solve it without burning your entire step budget? |

A lazy agent that reads one page and guesses "true" gets partial credit on verdict accuracy. It gets zero on everything else. The only path to a high total reward is a genuine investigation. We also penalize hitting the step limit — an agent that spins out without filing a report loses an additional 0.15.

This reward function is genuinely hard to game. The environment was designed so that no single tool call pattern can systematically exploit it.

---

## Difficulty Levels

| Level | Sources | Synthetic Pages | What makes it hard |
|-------|:-------:|:---------------:|-------------------|
| `easy` | 1 credible | 2 fakes (SEO farm + wellness blog) | Clear signal if you look for it |
| `medium` | 1–2 credible | 3 sophisticated fakes | Mixed signals; requires cross-referencing |
| `hard` | 1 credible | 4 fakes including a spoofed credible domain | High-quality fakes; maximum deception |

---

## Training

We used **GRPO** (Group Relative Policy Optimization) via TRL + Unsloth to train Llama-3.1-8B-Instruct across three progressive phases. The progression was intentional: each phase expanded both the difficulty and the expected behavior.

| Phase | Difficulty | Training Steps | Saved Model |
|-------|-----------|:--------------:|-------------|
| P1 | Easy only | ~500 | [AryanJain7031/dead-internet-detective-model](https://huggingface.co/AryanJain7031/dead-internet-detective-model) |
| P2 | Easy + Medium + Hard | ~500 | [PriyanshuHF/dead-internet-detective-model-p2](https://huggingface.co/PriyanshuHF/dead-internet-detective-model-p2) |
| P3 | All difficulties | ~1000 | [PriyanshuHF/dead-internet-detective-model-p3](https://huggingface.co/PriyanshuHF/dead-internet-detective-model-p3) |

Phase 1 teaches the agent that tools exist and should be used. Phase 2 forces it to handle cases where the evidence is genuinely mixed and a single visit isn't enough. Phase 3 refines efficiency — getting to the right answer without burning the step budget on redundant calls.

---

## Results

### Reward Curve (Phase 1 · easy difficulty · ~500 steps)

![Reward Curve](plots/reward_curve.png)

*Training step on x-axis, episode reward on y-axis. Red line = rolling mean. The agent begins near the untrained baseline (~0.15) and climbs toward ~1.45 by step 500. GRPO on Llama-3.1-8B-Instruct.*

### Before vs. After: What Actually Changed

![Before After](plots/before_after.png)

*Each component scored separately. Gray = untrained baseline (visits dossier URLs, guesses "true"). Red = trained agent after Phase 1. The trained agent uses citation tracing, author lookups, and cross-referencing — behaviors the untrained model never exhibited.*

| Reward Component | Untrained | Trained (P1) |
|-----------------|:---------:|:------------:|
| Verdict accuracy | ~0.10 | ~0.55 |
| Citation depth | 0.00 | ~0.30 |
| Contradiction detection | 0.00 | ~0.20 |
| Tool diversity | ~0.05 | ~0.25 |
| Case file quality | 0.00 | ~0.15 |
| **Total reward** | **~0.15** | **~1.45** |

The untrained agent visits URLs and guesses. The trained agent traces citations, checks author credibility across domains, identifies contradictions between sources, and builds an evidence chain before filing. That's not a marginal improvement — it's a qualitatively different kind of behavior.

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                        Colab / Local                    │
│                                                         │
│   ┌──────────────────┐       ┌─────────────────────┐   │
│   │  LLM Agent       │──────▶│  DeadInternetClient │   │
│   │  (Llama-3.1-8B)  │◀──────│  (client/client.py) │   │
│   └──────────────────┘       └────────┬────────────┘   │
└────────────────────────────────────────│────────────────┘
                                         │ HTTP (REST)
                          ┌──────────────▼──────────────┐
                          │    FastAPI Server            │
                          │    /reset  /step  /state     │
                          └──────────────┬───────────────┘
                                         │
                          ┌──────────────▼───────────────┐
                          │  DeadInternetEnvironment     │
                          │  12 Tools · 5 Graders        │
                          │  Synthetic Internet          │
                          └──────────────────────────────┘
```

---

## Quick Start

**Against the live HF Space (no local setup required):**

```python
import requests

BASE = "https://aryanjain7031-dead-internet-detective.hf.space"

# Start a new investigation
r = requests.post(f"{BASE}/reset", json={"task_id": "easy", "seed": 42})
session_id = r.json()["session_id"]
obs = r.json()["observation"]

print("Claim:", obs["claim"])
print("Dossier:", obs["dossier_urls"])

# Follow a lead
r2 = requests.post(f"{BASE}/step", json={
    "session_id": session_id,
    "action": {"tool": "visit_page", "params": {"url": obs["dossier_urls"][0]}}
})
print("Reward:", r2.json()["reward"])
print("Done:", r2.json()["done"])
```

**Run the trained P3 model against the live environment:**

Use the [Eval Space](https://huggingface.co/spaces/AryanJain7031/dead-internet-detective-eval) — select P3, pick difficulties, hit Run. No GPU required.

Or locally (requires CUDA):
```bash
python evaluate.py --model p3 --vs-baseline --difficulties easy medium hard --n-episodes 3
```

**Local setup:**
```bash
git clone https://github.com/aj7075/dead-internet-detective.git
cd dead-internet-detective && pip install -r requirements.txt
uvicorn dead_internet_detective.server.app:app --reload --port 8000
```

---

## Links

| Resource | Link |
|----------|------|
| 🌐 Live Environment | [AryanJain7031/dead-internet-detective](https://huggingface.co/spaces/AryanJain7031/dead-internet-detective) |
| 🔬 Eval Space (run P3) | [AryanJain7031/dead-internet-detective-eval](https://huggingface.co/spaces/AryanJain7031/dead-internet-detective-eval) |
| 📓 Training Notebook | [notebooks/training_grpo.ipynb](notebooks/training_grpo.ipynb) |
| 🤖 Model P1 | [AryanJain7031/dead-internet-detective-model](https://huggingface.co/AryanJain7031/dead-internet-detective-model) |
| 🤖 Model P2 | [PriyanshuHF/dead-internet-detective-model-p2](https://huggingface.co/PriyanshuHF/dead-internet-detective-model-p2) |
| 🤖 Model P3 | [PriyanshuHF/dead-internet-detective-model-p3](https://huggingface.co/PriyanshuHF/dead-internet-detective-model-p3) |
| 📊 W&B Run (P1) | [Training metrics](https://wandb.ai/models-srm-institute-of-science-and-technology9361/dead-internet-detective/workspace?nw=nwuseraryanjain7031) |
| 📝 Writeup / Blog | [This Space's README](https://huggingface.co/spaces/AryanJain7031/dead-internet-detective) |

---

*Built with [OpenEnv](https://github.com/huggingface/open-env) · [TRL/GRPO](https://github.com/huggingface/trl) · [Unsloth](https://github.com/unslothai/unsloth)*
