from __future__ import annotations

import pytest

from dead_internet_detective.environment import DeadInternetEnvironment
from dead_internet_detective.models import Action, CaseFile, Observation


def make_env() -> DeadInternetEnvironment:
    return DeadInternetEnvironment()


@pytest.mark.parametrize("difficulty", ["easy", "medium", "hard"])
def test_reset_returns_valid_observation(difficulty):
    env = make_env()
    obs = env.reset(difficulty, seed=42)

    assert isinstance(obs, Observation)
    assert isinstance(obs.claim, str) and obs.claim
    assert isinstance(obs.dossier_urls, list) and len(obs.dossier_urls) > 0
    assert obs.current_page_content is None        # initial — no page fetched yet
    assert isinstance(obs.visited_urls, list)
    assert isinstance(obs.search_results, list)
    assert obs.tool_result is None                 # initial
    assert isinstance(obs.case_file, CaseFile)
    assert obs.steps_remaining > 0
    assert isinstance(obs.last_action_feedback, str)
    assert obs.last_action_error is None
    assert isinstance(obs.partial_score, float)


def test_reset_seed_determinism():
    env = make_env()
    obs1 = env.reset("hard", seed=7)
    obs2 = env.reset("hard", seed=7)
    assert obs1.claim == obs2.claim
    assert sorted(obs1.dossier_urls) == sorted(obs2.dossier_urls)


def test_reset_no_seed_doesnt_crash():
    env = make_env()
    obs = env.reset("easy")
    assert isinstance(obs, Observation)


def test_state_after_reset():
    env = make_env()
    env.reset("medium", seed=1)
    state = env.state()
    assert state.difficulty == "medium"
    assert state.seed == 1
    assert not state.done
    assert state.steps_used == 0


@pytest.mark.parametrize("difficulty", ["easy", "medium", "hard"])
def test_step_stub_doesnt_crash(difficulty):
    env = make_env()
    env.reset(difficulty, seed=0)
    action = Action(tool="visit_page", params={"url": "https://example.com"})
    obs, reward, done, info = env.step(action)

    assert isinstance(obs, Observation)
    assert reward == 0.0
    assert done is False
    assert isinstance(info, dict)


def test_dossier_urls_are_valid_pages():
    env = make_env()
    obs = env.reset("hard", seed=99)
    state = env.state()
    for url in obs.dossier_urls:
        assert url in state.synthetic_internet, f"dossier url {url} not in synthetic_internet"
