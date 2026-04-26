from __future__ import annotations

import requests


class DeadInternetClient:
    def __init__(self, base_url: str):
        self.base_url = base_url.rstrip("/")

    def reset(self, task_id: str = "easy", seed: int | None = None) -> dict:
        r = requests.post(f"{self.base_url}/reset", json={"task_id": task_id, "seed": seed})
        r.raise_for_status()
        return r.json()

    def step(self, session_id: str, tool: str, params: dict) -> dict:
        r = requests.post(
            f"{self.base_url}/step",
            json={"session_id": session_id, "action": {"tool": tool, "params": params}},
        )
        r.raise_for_status()
        return r.json()

    def health(self) -> bool:
        try:
            r = requests.get(f"{self.base_url}/health", timeout=3)
            return r.status_code == 200
        except Exception:
            return False
