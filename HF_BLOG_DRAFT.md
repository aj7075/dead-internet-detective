# Dead Internet Detective: Training an LLM Agent to Detect AI-Generated Disinformation

*OpenEnv Hackathon India 2026 — Team submission*

---

## The Problem

Disinformation detection is one of the hardest NLP problems alive. It can't be solved by a classifier — because the verdict alone isn't what matters. The **evidence chain** matters. A model needs to:

- Follow citations across multiple pages
- Cross-reference publication dates
- Detect linguistic patterns of AI-generated text
- Flag when a "credible" source is actually a sophisticated fake
- Synthesize contradictory evidence before reaching a verdict

We built an environment to teach an LLM agent to do exactly this.

---

## The Environment

**Dead Internet Detective** gives an agent a disinformation claim and access to a fully synthetic internet — fake news pages, SEO farms, wellness blogs, and one or two genuinely credible sources. The agent has 12 tools to investigate:

| Tool | What it does |
|------|-------------|
| `visit_page` | Read a synthetic webpage |
| `search_web` | Query the synthetic search engine |
| `check_domain_age` | See when a domain was registered |
| `cross_reference` | Compare two URLs for consistency |
| `detect_synthetic_text` | Score a passage for AI-generation likelihood |
| `trace_citation` | Follow citation chains |
| `analyze_sentiment` | Score emotional manipulation |
| `check_author` | Look up author credibility |
| `fact_check_claim` | Query synthetic fact-check DB |
| `update_case_file` | Log evidence and notes |
| `file_report` | File final verdict (ends episode) |

The agent must investigate, then call `file_report` with a verdict (`true` / `false` / `unverifiable`) and an evidence chain.

### Three Difficulty Levels

- **Easy**: 1 credible source + 2 obvious synthetic pages. Clear signal.
- **Medium**: 1–2 credible sources + 3 sophisticated fakes. Mixed signals.
- **Hard**: 1 credible + 4 fakes including a spoofed credible domain. Maximum deception.

---

## The Reward Function

We designed a **5-component composite reward** that scores the quality of the investigation, not just the final answer:

```
total_reward =
  0.30 × verdict_accuracy          (got the right answer?)
  0.25 × evidence_chain_quality    (visited + cited real sources?)
  0.20 × contradiction_detection   (flagged synthetic sources?)
  0.15 × tool_diversity            (used investigation tools, not just visit_page?)
  0.10 × case_file_quality         (logged evidence systematically?)
```

This reward structure is hard to game. An agent that always guesses `true` gets ~0.10 on verdict accuracy and near-zero on everything else. The only way to score well is to actually investigate.

---

## Training: GRPO on Llama-3.1-8B-Instruct

We used **GRPO** (Group Relative Policy Optimization) via HuggingFace TRL with Unsloth for efficient fine-tuning.

Training ran in three progressive phases:
- **Phase 1**: Easy difficulty only, ~500 steps — establish basic investigation behavior
- **Phase 2**: Easy + Medium + Hard, ~500 steps — generalize to harder cases
- **Phase 3**: All difficulties, ~1000 steps — final refinement

The environment runs as a FastAPI server on a HuggingFace Space. The training Colab connects to it via HTTP to run rollouts.

---

## Results

### Reward Curve

![Reward Curve](https://huggingface.co/spaces/AryanJain7031/dead-internet-detective/resolve/main/plots/reward_curve.png)

*Phase 1 training reward curve over ~500 steps. Red line = rolling mean. GRPO on Llama-3.1-8B-Instruct, easy difficulty.*

### Before vs After

![Before After](https://huggingface.co/spaces/AryanJain7031/dead-internet-detective/resolve/main/plots/before_after.png)

| Reward Component | Untrained | Trained (Phase 1) |
|-----------------|:---------:|:-----------------:|
| Verdict accuracy | ~0.10 | ~0.55 |
| Citation depth | 0.00 | ~0.30 |
| Contradiction detection | 0.00 | ~0.20 |
| Tool diversity | ~0.05 | ~0.25 |
| Case file quality | 0.00 | ~0.15 |
| **Total reward** | **~0.15** | **~1.45** |

The untrained baseline agent visits dossier URLs and immediately files `true` — it never uses investigation tools, never detects contradictions, never builds a case file. After training, the agent learns to trace citations, flag synthetic sources, and use diverse tools before filing a verdict.

---

## Try It

The environment is live on HuggingFace Spaces:

**Space**: [AryanJain7031/dead-internet-detective](https://huggingface.co/spaces/AryanJain7031/dead-internet-detective)

```python
import requests

BASE = "https://aryanjain7031-dead-internet-detective.hf.space"

# Start an episode
r = requests.post(f"{BASE}/reset", json={"task_id": "hard", "seed": 42})
session_id = r.json()["session_id"]
obs = r.json()["observation"]

print("Claim:", obs["claim"])
print("Dossier:", obs["dossier_urls"])

# Run your agent against it...
```

**GitHub**: [aj7075/dead-internet-detective](https://github.com/aj7075/dead-internet-detective)

**Training notebook (Colab)**: [notebooks/training_grpo.ipynb](https://github.com/aj7075/dead-internet-detective/blob/main/notebooks/training_grpo.ipynb)

---

*Built for OpenEnv Hackathon India 2026. Team: Aryan Jain, Priyanshu, Saaz Bhargava.*
