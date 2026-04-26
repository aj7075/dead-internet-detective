# Dead Internet Detective — Training Guide

## Prerequisites
- HuggingFace account
- W&B account (free at wandb.ai)
- Google account (for Colab)

## Step 1 — Deploy Environment Server (free, no GPU, do this once)
1. Go to huggingface.co/new-space
2. Name: dead-internet-detective, SDK: Docker, Port: 7860
3. Connect this GitHub repo
4. Wait for build (~3 min)
5. Test: curl https://YOUR_USERNAME-dead-internet-detective.hf.space/health
6. Expected: {"status":"ok"}

## Step 2 — Smoke Test on Colab Free ($0)
1. Open notebooks/training_grpo.ipynb in Google Colab
2. Runtime > Change runtime type > T4 GPU
3. Set HF_SPACE_URL = your space URL from Step 1
4. Set WANDB_KEY = your key from wandb.ai/settings
5. Set SMOKE_TEST = True (already default)
6. Run Cells 1 -> 5 (stop before Cell 6 which is full training)
7. Smoke test prints mean reward — must be > 0.01 to continue

## Step 3 — Full Training on Colab Free (free, slow) or HF GPU (fast, ~$15)

### Option A: Colab Free (T4, ~3-4 hours, $0)
1. Set SMOKE_TEST = False
2. Run all cells including Cell 6
3. Leave tab open (use Colab Pro if you have it to avoid disconnects)

### Option B: HuggingFace GPU Space (~$15 per run)
Use only after smoke test passes. Spend from Account 1.

## Step 4 — Read Evaluation Results
After Cell 7 runs, check:
- plots/eval_results.json — raw numbers
- plots/reward_curve.png — visual comparison
- plots/next_run_config.txt — auto-generated suggestions for next run

## Credit Budget
Account 1 ($25): Phase 1 full training run
Account 2 ($25): Phase 2 mixed difficulty
Account 3 ($25): Final run + buffer
Rule: never spend credits without a passing smoke test first.

## Warning Signs
reward always 0.0          -> file_report not being called, check Cell 4
internal_consistency drops -> reward hacking, monitor in W&B
all episodes timeout       -> reduce MAX_EP_STEPS in Cell 2
model collapses to one tool -> increase temperature in generate_action()
