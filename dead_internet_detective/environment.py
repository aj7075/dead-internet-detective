from __future__ import annotations

import random
import uuid

from dead_internet_detective import generator
from dead_internet_detective.models import (
    Action,
    CaseFile,
    Observation,
    State,
)

try:
    from openenv import Environment
except ImportError:  # openenv-core not installed yet — stub for tests
    class Environment:  # type: ignore[no-redef]
        pass


class DeadInternetEnvironment(Environment):

    _MAX_STEPS: dict[str, int] = {"easy": 20, "medium": 30, "hard": 40}

    def reset(self, task_id: str = "easy", seed: int | None = None) -> Observation:
        if seed is None:
            seed = random.randint(0, 2**31 - 1)

        pages_by_url, ground_truth = generator.generate_episode(task_id, seed)

        rng = random.Random(seed + 1)  # separate rng so dossier picks don't collide
        all_urls = list(pages_by_url.keys())
        dossier = rng.sample(all_urls, min(3, len(all_urls)))

        # pick a representative claim from the primary source content
        primary = pages_by_url[ground_truth["primary_source_url"]]
        claim = primary.linguistic_features["key_claims"][0] if primary.linguistic_features.get("key_claims") else primary.content[:120]

        self._session_state = State(
            session_id=str(uuid.uuid4()),
            synthetic_internet=pages_by_url,
            case_file=CaseFile.empty(claim),
            visited_urls=[],
            steps_used=0,
            max_steps=self._MAX_STEPS.get(task_id, 20),
            seed=seed,
            ground_truth=ground_truth,
            difficulty=task_id,
        )

        return Observation(
            claim=claim,
            dossier_urls=dossier,
            current_page_content=None,
            visited_urls=[],
            search_results=[],
            tool_result=None,
            case_file=self._session_state.case_file,
            steps_remaining=self._session_state.max_steps,
            last_action_feedback="",
            last_action_error=None,
            partial_score=0.0,
        )

    def step(self, action: Action) -> tuple[Observation, float, bool, dict]:
        # STUB — Person C implements
        obs = Observation(
            claim=self._session_state.case_file.claim,
            dossier_urls=[],
            current_page_content=None,
            visited_urls=self._session_state.visited_urls,
            search_results=[],
            tool_result=None,
            case_file=self._session_state.case_file,
            steps_remaining=self._session_state.max_steps - self._session_state.steps_used,
            last_action_feedback="",
            last_action_error=None,
            partial_score=0.0,
        )
        return obs, 0.0, False, {}

    def state(self) -> State:
        return self._session_state
