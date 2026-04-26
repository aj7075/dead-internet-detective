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

> An RL environment that trains LLM agents to **detect AI-generated disinformation** through multi-hop investigation over a fully synthetic internet.

[![GitHub](https://img.shields.io/badge/GitHub-dead--internet--detective-black?logo=github)](https://github.com/aj7075/dead-internet-detective)

---

## What This Is

Disinformation detection is one of the hardest NLP problems because it requires **multi-hop reasoning** over heterogeneous sources. Simple classifiers fail — the evidence chain matters as much as the verdict.

This project trains a **Llama-3.1-8B-Instruct** agent using **GRPO** (Group Relative Policy Optimization) to investigate synthetic disinformation cases using a 12-tool investigation desk.

The agent receives a claim and a set of "dossier" URLs, explores a fully synthetic internet, and files a final report. A structured reward function scores citation depth, contradiction detection, tool diversity, case file quality, and verdict accuracy.

---

## How to Use This Space

This space exposes a **REST API** for the investigation environment. You can run an episode using any HTTP client or the provided Python client.

### Quick start

```bash
pip install requests

python - <<'EOF'
import requests, json

BASE = "https://aryanjain7031-dead-internet-detective.hf.space"

# Start a new investigation episode
r = requests.post(f"{BASE}/reset", json={"task_id": "easy", "seed": 42})
data = r.json()
session_id = data["session_id"]
obs = data["observation"]

print("CLAIM:", obs["claim"])
print("DOSSIER:", obs["dossier_urls"])

# Visit a dossier URL
r2 = requests.post(f"{BASE}/step", json={
    "session_id": session_id,
    "action": "visit_page",
    "params": {"url": obs["dossier_urls"][0]}
})
print("REWARD:", r2.json()["reward"])
EOF
```

### API Reference

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Liveness check |
| `/reset` | POST | Start a new episode (`task_id`: easy/medium/hard, `seed`: int) |
| `/step` | POST | Take an action (`session_id`, `action`, `params`) |
| `/state/{session_id}` | GET | Inspect current episode state |
| `/docs` | GET | Interactive Swagger UI |

### Available Tools (12 total)

| Tool | Description |
|------|-------------|
| `visit_page` | Fetch a synthetic webpage |
| `search_web` | Query the synthetic search engine |
| `check_domain_age` | Look up synthetic domain registration |
| `cross_reference` | Cross-reference two URLs for consistency |
| `detect_synthetic_text` | Score text for AI-generation likelihood |
| `trace_citation` | Follow citation chains |
| `analyze_sentiment` | Sentiment + emotional manipulation score |
| `check_author` | Look up author credibility |
| `reverse_image_search` | Search for image origin |
| `fact_check_claim` | Query synthetic fact-check database |
| `update_case_file` | Add notes/evidence to investigation log |
| `file_report` | File final verdict (ends episode) |

---

## Difficulty Levels

| Level | Sources | Synthetic Pages | Description |
|-------|---------|-----------------|-------------|
| `easy` | 1 credible | 2 (SEO farm + wellness blog) | Clear signal from credible source |
| `medium` | 1–2 credible | 3 sophisticated fakes | Mixed signals, requires cross-referencing |
| `hard` | 1 credible | 4 fakes (inc. spoofed credible) | High-quality fakes, maximum deception |

---

## Training Results

Training used **GRPO** on Llama-3.1-8B-Instruct over 3 progressive phases.

### Reward Curve (Phase 1 · easy difficulty · ~500 steps)

![Reward Curve](plots/reward_curve.png)

### Before vs After (Reward Components)

![Before After](plots/before_after.png)

| Reward Component | Untrained | Trained (Phase 1) |
|-----------------|:---------:|:-----------------:|
| Verdict accuracy | ~0.10 | ~0.55 |
| Citation depth | 0.00 | ~0.30 |
| Contradiction detection | 0.00 | ~0.20 |
| Tool diversity | ~0.05 | ~0.25 |
| Case file quality | 0.00 | ~0.15 |
| **Total reward** | **~0.15** | **~1.45** |

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    Training (Colab)                     │
│   LLM Agent (Llama-3.1-8B) ──▶ DeadInternetClient      │
└──────────────────────────────────────┬──────────────────┘
                                       │ HTTP
                       ┌───────────────▼───────────────┐
                       │   FastAPI Server (This Space) │
                       │   /reset  /step  /state       │
                       └───────────────┬───────────────┘
                                       │
                       ┌───────────────▼───────────────┐
                       │   DeadInternetEnvironment     │
                       │   12 Tools + 5 Graders        │
                       │   Synthetic Internet          │
                       └───────────────────────────────┘
```

---

## Trained Models

| Phase | Difficulty | Steps | Model |
|-------|-----------|-------|-------|
| Phase 1 | easy | ~500 | [PriyanshuHF/dead-internet-detective-model-p1](https://huggingface.co/PriyanshuHF/dead-internet-detective-model-p1) |
| Phase 2 | easy+medium+hard | ~500 | [PriyanshuHF/dead-internet-detective-model-p2](https://huggingface.co/PriyanshuHF/dead-internet-detective-model-p2) |

---

## Links

- **GitHub**: [aj7075/dead-internet-detective](https://github.com/aj7075/dead-internet-detective)
- **Training Notebook**: [notebooks/training_grpo.ipynb](https://github.com/aj7075/dead-internet-detective/blob/main/notebooks/training_grpo.ipynb)
- **Built with**: [OpenEnv](https://github.com/huggingface/open-env) · [TRL/GRPO](https://github.com/huggingface/trl) · [Unsloth](https://github.com/unslothai/unsloth)
