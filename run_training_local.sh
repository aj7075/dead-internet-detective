#!/bin/bash
set -e

echo "============================================"
echo " Dead Internet Detective — Local Training"
echo " RTX 4070 Laptop | WSL2 | 8GB VRAM"
echo "============================================"

# Step 1: Verify GPU
echo "[1/6] Checking GPU..."
python3 -c "
import torch
assert torch.cuda.is_available(), 'CUDA not available. Run: nvidia-smi in WSL2'
name = torch.cuda.get_device_name(0)
vram = torch.cuda.get_device_properties(0).total_memory / 1e9
print(f'GPU: {name}')
print(f'VRAM: {vram:.1f} GB')
assert vram >= 6.0, f'Not enough VRAM: {vram:.1f}GB. Need at least 6GB.'
print('GPU check PASSED')
"

# Step 2: Run test suite
echo "[2/6] Running test suite..."
PYTHONPATH=. python3 -m pytest tests/ -q
echo "Tests PASSED"

# Step 3: Start the environment server in background
echo "[3/6] Starting environment server on port 8000..."
uvicorn dead_internet_detective.server.app:app \
  --host 127.0.0.1 --port 8000 --log-level warning &
SERVER_PID=$!
sleep 3

# Verify server is up
python3 -c "
import requests
r = requests.get('http://localhost:8000/health', timeout=5)
assert r.json() == {'status': 'ok'}, f'Server error: {r.text}'
print('Server ONLINE at http://localhost:8000')
"

# Step 4: Run baseline inference
echo "[4/6] Running baseline inference (dumb agent)..."
python3 inference.py --url http://localhost:8000 --difficulty easy --seed 42
echo "Baseline DONE"

# Step 5: Run smoke test (10 episodes, no weight updates)
echo "[5/6] Running smoke test (10 episodes)..."
python3 -c "
import sys, os, json, time, random, statistics, requests
sys.path.insert(0, '.')
from client.client import DeadInternetClient

client = DeadInternetClient('http://localhost:8000')
assert client.health()

results = []
for i in range(10):
    r = client.reset(task_id='easy', seed=i)
    sid = r['session_id']
    obs = r['observation']
    total = 0.0
    for url in obs.get('dossier_urls', []):
        res = client.step(sid, 'visit_page', {'url': url})
        total += res.get('reward', 0.0)
        if res.get('done'): break
    res = client.step(sid, 'file_report', {
        'verdict': 'unverifiable', 'confidence': 0.4,
        'evidence_chain': obs.get('dossier_urls', []),
        'reasoning': 'smoke test'
    })
    total += res.get('reward', 0.0)
    results.append(total)
    print(f'  ep={i:02d} reward={total:.3f}')

mean = statistics.mean(results)
print(f'Mean reward: {mean:.3f}')
assert mean > 0.01, f'Smoke test FAILED: mean reward {mean:.3f} is near zero'
print('Smoke test PASSED')
"

# Step 6: Launch Jupyter for training notebook
echo "[6/6] Starting Jupyter notebook for training..."
echo ""
echo "============================================"
echo " NEXT STEP:"
echo " Open your browser and go to the URL below"
echo " Open: notebooks/training_grpo.ipynb"
echo " Set WANDB_KEY in Cell 2"
echo " Run cells 1-5 for smoke test"
echo " Set SMOKE_TEST=False and run all cells"
echo "============================================"
jupyter notebook --no-browser --port=8888 &
echo "Jupyter running on http://localhost:8888"
echo "Server PID: $SERVER_PID (kill with: kill $SERVER_PID)"
wait
