"""Baseline dumb agent. Usage: python inference.py --url http://localhost:8000 --difficulty easy --seed 42"""
import argparse
import sys

from client.client import DeadInternetClient


def run_baseline_episode(client: DeadInternetClient, difficulty: str = "easy", seed: int = 42) -> dict:
    result = client.reset(task_id=difficulty, seed=seed)
    session_id = result["session_id"]
    obs = result["observation"]

    print(f"\n=== BASELINE AGENT | difficulty={difficulty} | seed={seed} ===")
    print(f"Claim: {obs['claim']}")
    print(f"Dossier: {obs['dossier_urls']}")

    total_reward = 0.0
    step_num = 0

    for url in obs["dossier_urls"]:
        r = client.step(session_id, "visit_page", {"url": url})
        total_reward += r["reward"]
        step_num += 1
        print(f"Step {step_num}: visit_page({url}) → reward={r['reward']:.4f}")
        if r["done"]:
            print("Episode ended early.")
            return r

    r = client.step(session_id, "file_report", {
        "verdict": "true",
        "confidence": 0.5,
        "evidence_chain": obs["dossier_urls"],
        "reasoning": "Visited dossier URLs. No contradictions found.",
    })
    total_reward += r["reward"]
    step_num += 1

    print(f"\nStep {step_num}: file_report")
    print(f"Terminal reward: {r['info'].get('terminal_reward', 0.0):.4f}")
    print(f"Step reward:     {r['reward'] - r['info'].get('terminal_reward', 0.0):.4f}")
    print(f"Total reward:    {total_reward:.4f}")
    print(f"Done: {r['done']}")
    return r


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://localhost:8000")
    parser.add_argument("--difficulty", default="easy")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    client = DeadInternetClient(args.url)
    if not client.health():
        print(f"ERROR: Server not reachable at {args.url}", file=sys.stderr)
        sys.exit(1)

    run_baseline_episode(client, args.difficulty, args.seed)
