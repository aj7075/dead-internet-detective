# Dead Internet Detective: Training an LLM to Investigate Disinformation

*We built a synthetic internet and trained an 8B model to investigate it like a journalist. Here's what we learned.*

---

## The Problem Nobody Is Actually Solving

Disinformation detection has a dirty secret: most of the research is just text classification in disguise.

You take an article. You feed it to a model. The model outputs a label — true, false, misleading. Maybe a confidence score. The community celebrates, the benchmark goes up, and the actual problem stays unsolved.

Here's the thing: a real disinformation campaign doesn't fail because it *sounds* wrong. It fails because the citations don't trace back to anything real. Because the author's name appears on forty different websites with contradictory credentials. Because the domain was registered two weeks before the story broke. Because the study being cited was quietly retracted three years ago.

None of that is in the headline. A classifier that reads text can't catch any of it.

What you need isn't a classifier. You need an investigator.

---

## What We Actually Built

Dead Internet Detective is a reinforcement learning environment built on top of [OpenEnv](https://github.com/huggingface/open-env). The agent receives a **claim** and a set of **dossier URLs** — entry points into a fully synthetic internet we generated for this project — and has to investigate before filing a verdict.

The synthetic internet is the core innovation. It's not a static dataset of labeled articles. Every episode generates a fresh set of pages:

- Credible-looking sources (synthetic news organizations, research institutions)
- SEO farms that repeat the claim with slight variations
- Wellness blogs with fabricated author credentials
- At hard difficulty: high-quality spoofs of real credible domains, with plausible citation chains that terminate in nowhere

The agent has **12 investigative tools**: it can visit pages, trace citation chains up to 3 hops, look up domain registration dates, check whether an author's name is associated with suspicious domains, run linguistic fingerprints to estimate AI-generation probability, cross-reference two sources for contradictions, and more.

It maintains a **case file** throughout — a running record of confirmed facts, flagged synthetic sources, and contradictions found. When it's done (or out of steps), it files a report: verdict, confidence, evidence chain, and reasoning.

---

## Why This Needed RL

You can't train a genuine investigator with supervised learning on verdict labels.

Consider two agents that both get the right verdict:
- Agent A visits one page and guesses.
- Agent B traces citations, identifies a spoofed domain, finds a contradiction between two sources, flags three pages as AI-generated, and files a detailed evidence chain.

On a classification benchmark, they score the same. In a world where the reasoning matters — which is the real world — they're completely different.

The reward function captures this difference. It has five components:

| Component | Weight |
|-----------|:------:|
| Verdict accuracy | 30% |
| Evidence chain quality | 25% |
| Synthetic source detection | 20% |
| Internal consistency | 15% |
| Step efficiency | 10% |

A lazy agent that reads one page and guesses "true" gets partial credit on verdict accuracy. It gets zero on evidence chain quality, zero on synthetic source detection, zero on internal consistency. The total reward for lazy behavior is low. The only path to a high score is doing the actual work.

We also penalize hitting the step limit — an agent that burns its entire budget without filing a report loses an additional 0.15. This pushes the agent toward efficient investigation rather than just calling every tool it can reach.

This reward function is designed to be genuinely hard to game. There's no shortcut that scores well across all five components without actually investigating.

---

## Training: Three Phases

We used **GRPO** (Group Relative Policy Optimization) via TRL + Unsloth to train Llama-3.1-8B-Instruct. The training loop connects directly to the live environment — the model generates tool calls, the environment executes them, rewards flow back.

We ran three progressive phases:

**Phase 1 — Easy cases only (~500 steps)**
The agent starts here. Easy cases have a clear credible source and two obvious fakes. The goal is to teach the model that tools exist, that using them earns reward, and that filing a report is how an episode ends. Before training, the model basically ignores the tools entirely.

**Phase 2 — All difficulties (~500 steps)**
The agent now has to handle mixed signals. Medium cases have contradictory sources. Hard cases have spoofed credible domains that require whois lookups and citation tracing to identify. The agent has to generalize beyond pattern-matching on easy cases.

**Phase 3 — All difficulties, more steps (~1000 steps)**
Refinement. The agent has learned to investigate — now it needs to do it efficiently. Phase 3 reduces rollouts per step and increases total training time to push the agent toward tighter, more targeted investigation strategies.

---

## Results

The untrained baseline visits dossier URLs and files a report. Total reward: approximately 0.15.

After Phase 1 training:

| Component | Before | After |
|-----------|:------:|:-----:|
| Verdict accuracy | ~0.10 | ~0.55 |
| Citation depth | 0.00 | ~0.30 |
| Contradiction detection | 0.00 | ~0.20 |
| Tool diversity | ~0.05 | ~0.25 |
| Case file quality | 0.00 | ~0.15 |
| **Total** | **~0.15** | **~1.45** |

The total reward improved roughly 10x. But the numbers undersell what actually changed.

The untrained model visits a couple of URLs and guesses. The trained model uses citation tracing, author lookups, domain age checks, and cross-referencing — and it does them in a logical sequence. It's not just scoring higher. It's doing something qualitatively different.

W&B training run: [Phase 1 metrics](https://wandb.ai/models-srm-institute-of-science-and-technology9361/dead-internet-detective/workspace?nw=nwuseraryanjain7031)

---

## Try It

The environment is live at [AryanJain7031/dead-internet-detective](https://huggingface.co/spaces/AryanJain7031/dead-internet-detective). Open the App tab for an interactive demo where you can run a real episode and step through the investigation yourself.

To run the trained P3 model against it, use the [Eval Space](https://huggingface.co/spaces/AryanJain7031/dead-internet-detective-eval) — no GPU required.

```python
import requests

BASE = "https://aryanjain7031-dead-internet-detective.hf.space"

r = requests.post(f"{BASE}/reset", json={"task_id": "hard", "seed": 42})
session_id = r.json()["session_id"]
obs = r.json()["observation"]

print("Claim:", obs["claim"])

# Trace a citation chain
r2 = requests.post(f"{BASE}/step", json={
    "session_id": session_id,
    "action": {"tool": "citation_trace", "params": {"url": obs["dossier_urls"][0]}}
})
print(r2.json()["observation"])
```

---

## Links

- **Live Environment**: [AryanJain7031/dead-internet-detective](https://huggingface.co/spaces/AryanJain7031/dead-internet-detective)
- **Eval Space**: [AryanJain7031/dead-internet-detective-eval](https://huggingface.co/spaces/AryanJain7031/dead-internet-detective-eval)
- **Trained Model P3**: [PriyanshuHF/dead-internet-detective-model-p3](https://huggingface.co/PriyanshuHF/dead-internet-detective-model-p3)
- **Training Notebook**: [notebooks/training_grpo.ipynb](https://github.com/aj7075/dead-internet-detective/blob/main/notebooks/training_grpo.ipynb)
- **W&B Run (P1)**: [Training metrics](https://wandb.ai/models-srm-institute-of-science-and-technology9361/dead-internet-detective/workspace?nw=nwuseraryanjain7031)
- **GitHub**: [aj7075/dead-internet-detective](https://github.com/aj7075/dead-internet-detective)
