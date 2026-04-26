"""Sweep all 3 difficulties; assert rewards in valid range."""
import argparse
import math
import sys

from client.client import DeadInternetClient


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://localhost:8000")
    args = parser.parse_args()

    client = DeadInternetClient(args.url)
    if not client.health():
        print(f"ERROR: Server not reachable at {args.url}", file=sys.stderr)
        sys.exit(1)

    for difficulty in ["easy", "medium", "hard"]:
        result = client.reset(task_id=difficulty, seed=1)
        session_id = result["session_id"]
        obs = result["observation"]
        dossier_urls = obs["dossier_urls"]

        total_reward = 0.0
        for url in dossier_urls:
            r = client.step(session_id, "visit_page", {"url": url})
            total_reward += r["reward"]
            if r["done"]:
                break

        r = client.step(session_id, "file_report", {
            "verdict": "true",
            "confidence": 0.5,
            "evidence_chain": dossier_urls,
            "reasoning": "Validation sweep.",
        })
        total_reward += r["reward"]

        print(f"difficulty={difficulty} reward={total_reward:.4f}")
        assert math.isfinite(total_reward), f"reward not finite: {total_reward}"
        assert -0.5 <= total_reward <= 1.0, f"reward out of range: {total_reward}"

    print("All 3 difficulties passed.")


if __name__ == "__main__":
    main()
