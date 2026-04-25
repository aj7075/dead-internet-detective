from __future__ import annotations

import random
import uuid

from dead_internet_detective import generator
from dead_internet_detective.graders import compute_terminal_reward
from dead_internet_detective.models import (
    Action,
    CaseFile,
    Observation,
    State,
    VALID_TOOLS,
)
from dead_internet_detective.tools.browser import visit_page, search, wayback_check
from dead_internet_detective.tools.casefile import update_case_file, request_expert, file_report
from dead_internet_detective.tools.identity import whois_lookup, author_lookup, image_provenance
from dead_internet_detective.tools.reasoning import citation_trace, linguistic_fingerprint, cross_reference

try:
    from openenv import Environment
except ImportError:  # openenv-core not installed yet — stub for tests
    class Environment:  # type: ignore[no-redef]
        pass

_TOOL_REGISTRY: dict = {
    "visit_page": visit_page,
    "search": search,
    "wayback_check": wayback_check,
    "update_case_file": update_case_file,
    "request_expert": request_expert,
    "file_report": file_report,
    "whois_lookup": whois_lookup,
    "author_lookup": author_lookup,
    "image_provenance": image_provenance,
    "citation_trace": citation_trace,
    "linguistic_fingerprint": linguistic_fingerprint,
    "cross_reference": cross_reference,
}


class DeadInternetEnvironment(Environment):

    _MAX_STEPS: dict[str, int] = {"easy": 40, "medium": 60, "hard": 80}

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
        if self._session_state.done:
            raise RuntimeError("Episode already done. Call reset() first.")

        if action.tool not in VALID_TOOLS:
            obs = self._build_observation(tool_result=None,
                                          error=f"Unknown tool: {action.tool}")
            return obs, 0.0, False, {}

        tool_fn = _TOOL_REGISTRY[action.tool]
        result = tool_fn(action.params, self._session_state)

        step_reward = result.get("step_reward", 0.0)
        self._session_state.accumulated_step_rewards += step_reward
        self._session_state.steps_used += 1

        done = False
        terminal_reward = 0.0
        if action.tool == "file_report" and result.get("accepted"):
            terminal_reward = compute_terminal_reward(self._session_state, action.params)
            done = True
            self._session_state.done = True

        if self._session_state.steps_used >= self._session_state.max_steps:
            done = True
            self._session_state.done = True

        total_reward = step_reward + terminal_reward
        obs = self._build_observation(tool_result=result, error=result.get("error"))
        return obs, total_reward, done, {"terminal_reward": terminal_reward}

    def _build_observation(self, tool_result: dict | None, error: str | None = None) -> Observation:
        s = self._session_state
        current_page = None
        search_results = []
        if tool_result:
            current_page = tool_result.get("content")
            search_results = tool_result.get("results", [])
        return Observation(
            claim=s.case_file.claim,
            dossier_urls=[],
            current_page_content=current_page,
            visited_urls=s.visited_urls,
            search_results=search_results,
            tool_result=tool_result,
            case_file=s.case_file,
            steps_remaining=s.max_steps - s.steps_used,
            last_action_feedback=tool_result.get("feedback", "") if tool_result else "",
            last_action_error=error,
            partial_score=s.accumulated_step_rewards,
        )

    def state(self) -> State:
        return self._session_state
