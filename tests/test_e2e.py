from __future__ import annotations

import socket

import pytest

from dead_internet_detective.environment import DeadInternetEnvironment
from dead_internet_detective.models import Action


def _server_available(host: str = "localhost", port: int = 8000) -> bool:
    try:
        with socket.create_connection((host, port), timeout=1):
            return True
    except OSError:
        return False


# ── no-server tests ───────────────────────────────────────────────────────────

def test_full_easy_episode_no_server():
    env = DeadInternetEnvironment()
    obs = env.reset("easy", seed=42)

    for url in obs.dossier_urls:
        _, _, done, _ = env.step(Action(tool="visit_page", params={"url": url}))
        if done:
            break

    _, reward, done, _ = env.step(Action(tool="file_report", params={
        "verdict": "true",
        "confidence": 0.5,
        "evidence_chain": obs.dossier_urls,
        "reasoning": "Visited all dossier URLs.",
    }))

    assert done is True
    assert -0.5 <= reward <= 1.0


def test_reward_improves_with_investigation():
    seed = 42

    # Episode A: dumb agent — visit dossier, file immediately
    env_a = DeadInternetEnvironment()
    obs_a = env_a.reset("easy", seed=seed)
    reward_a = 0.0
    done_a = False
    for url in obs_a.dossier_urls:
        _, r, done_a, _ = env_a.step(Action(tool="visit_page", params={"url": url}))
        reward_a += r
        if done_a:
            break
    if not done_a:
        _, r, done_a, _ = env_a.step(Action(tool="file_report", params={
            "verdict": "true",
            "confidence": 0.5,
            "evidence_chain": obs_a.dossier_urls,
            "reasoning": "Dumb agent.",
        }))
        reward_a += r

    # Episode B: investigator — visit + citation_trace + flag synthetics + file
    env_b = DeadInternetEnvironment()
    obs_b = env_b.reset("easy", seed=seed)
    reward_b = 0.0

    for url in obs_b.dossier_urls:
        _, r, _, _ = env_b.step(Action(tool="visit_page", params={"url": url}))
        reward_b += r

    for url in obs_b.dossier_urls:
        _, r, _, _ = env_b.step(Action(tool="citation_trace", params={"url": url}))
        reward_b += r

    state_b = env_b.state()
    synthetic_urls = [
        url for url, page in state_b.synthetic_internet.items()
        if page.ground_truth_label != "credible"
    ]
    _, r, _, _ = env_b.step(Action(tool="update_case_file", params={
        "synthetic_sources": synthetic_urls,
        "confidence": 0.8,
    }))
    reward_b += r

    _, r, done_b, _ = env_b.step(Action(tool="file_report", params={
        "verdict": "true",
        "confidence": 0.8,
        "evidence_chain": obs_b.dossier_urls,
        "reasoning": "Investigated citation network. Identified synthetic sources.",
    }))
    reward_b += r

    assert done_b is True
    assert reward_b >= reward_a, (
        f"Investigator reward {reward_b:.4f} should be >= dumb reward {reward_a:.4f}"
    )


def test_step_exhaustion_penalty():
    # Normal baseline: visit dossier + file report
    env_normal = DeadInternetEnvironment()
    obs_normal = env_normal.reset("easy", seed=1)
    reward_normal = 0.0
    done_normal = False
    for url in obs_normal.dossier_urls:
        _, r, done_normal, _ = env_normal.step(
            Action(tool="visit_page", params={"url": url})
        )
        reward_normal += r
        if done_normal:
            break
    if not done_normal:
        _, r, done_normal, _ = env_normal.step(Action(tool="file_report", params={
            "verdict": "true",
            "confidence": 0.5,
            "evidence_chain": obs_normal.dossier_urls,
            "reasoning": "Baseline.",
        }))
        reward_normal += r

    # Exhaustion episode: spam visit_page until done
    env_ex = DeadInternetEnvironment()
    obs_ex = env_ex.reset("easy", seed=1)
    reward_ex = 0.0
    done_ex = False
    first_url = obs_ex.dossier_urls[0]
    while not done_ex:
        _, r, done_ex, _ = env_ex.step(
            Action(tool="visit_page", params={"url": first_url})
        )
        reward_ex += r

    assert done_ex is True
    assert reward_ex < reward_normal, (
        f"Exhaustion reward {reward_ex:.4f} should be < normal reward {reward_normal:.4f}"
    )
