# Dead Internet Detective

Disinformation detection is one of the hardest NLP problems because it requires multi-hop reasoning over heterogeneous sources — a model must follow citations, cross-reference publication dates, flag linguistic patterns, and synthesize contradictory evidence before reaching a verdict. Simple classifier approaches fail because the evidence chain matters as much as the final label. Reinforcement learning is a natural fit: we reward an agent for the *quality of its investigation*, not just whether it guesses the correct verdict.

This project trains an LLM agent to investigate synthetic disinformation cases using a 12-tool investigation desk. The agent receives a claim and a set of "dossier" URLs, explores a fully synthetic internet, and files a final report. A structured reward function scores citation depth, contradiction detection, tool diversity, case file quality, and final verdict accuracy. We use GRPO (Group Relative Policy Optimization) to train the agent, with a FastAPI server wrapping the environment and a Colab notebook for training.

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
                          │  dead_internet_detective/    │
                          │       server/app.py          │
                          │  /reset  /step  /state       │
                          └──────────────┬───────────────┘
                                         │
                          ┌──────────────▼───────────────┐
                          │  DeadInternetEnvironment     │
                          │  environment.py              │
                          │  ┌──────────┐ ┌───────────┐  │
                          │  │  Tools   │ │  Graders  │  │
                          │  │ (12 fn)  │ │ (5 comps) │  │
                          │  └──────────┘ └───────────┘  │
                          │  Synthetic Internet (pages)  │
                          └──────────────────────────────┘
```

## Before / After Comparison

| Reward Component          | Untrained (Baseline) | Trained (Phase 1) |
|---------------------------|:--------------------:|:-----------------:|
| Verdict accuracy          | ~0.10                | ~0.55             |
| Citation depth            | 0.00                 | ~0.30             |
| Contradiction detection   | 0.00                 | ~0.20             |
| Tool diversity            | ~0.05                | ~0.25             |
| Case file quality         | 0.00                 | ~0.15             |
| **Total reward**          | **~0.15**            | **~1.45**         |

*Values are approximate from Phase 1 training (~500 steps, easy difficulty)*

## Reward Curve

![Reward Curve](plots/reward_curve.png)

## Links

- HF Space: `https://YOUR_HF_USERNAME-dead-internet-detective.hf.space` *(fill in after deployment)*
- Training notebook: `notebooks/training_grpo.ipynb`
- W&B run: *(fill in after training)*

## Local Setup (3 commands)

```bash
git clone https://github.com/aj7075/dead-internet-detective.git
cd dead-internet-detective && pip install -r requirements.txt
uvicorn dead_internet_detective.server.app:app --reload --port 8000
```

Then in a second terminal:

```bash
python inference.py --url http://localhost:8000 --difficulty easy --seed 42
```
