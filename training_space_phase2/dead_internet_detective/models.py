from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

VALID_TOOLS = {
    "visit_page", "search", "wayback_check",
    "whois_lookup", "author_lookup", "image_provenance",
    "citation_trace", "linguistic_fingerprint", "cross_reference",
    "update_case_file", "request_expert", "file_report",
}


@dataclass
class Action:
    tool: str        # one of VALID_TOOLS
    params: dict


@dataclass
class CaseFile:
    claim: str
    confirmed_facts: list[str]
    disputed_facts: list[str]
    synthetic_sources: list[str]   # URLs flagged as synthetic
    credible_sources: list[str]    # URLs flagged as credible
    open_questions: list[str]
    contradiction_log: list[dict]  # each: {source_a, source_b, description}
    confidence: float              # 0.0–1.0

    @classmethod
    def empty(cls, claim: str) -> "CaseFile":
        return cls(
            claim=claim,
            confirmed_facts=[],
            disputed_facts=[],
            synthetic_sources=[],
            credible_sources=[],
            open_questions=[],
            contradiction_log=[],
            confidence=0.0,
        )


@dataclass
class Page:
    url: str
    content: str
    author: str | None
    domain: str
    registration_date: str         # ISO date string e.g. "2024-12-01"
    citation_urls: list[str]
    image_urls: list[str]
    linguistic_features: dict      # must contain: ai_probability (float), key_claims (list), contradicted_by (list)
    wayback_available: bool        # used by wayback_check tool
    ground_truth_label: str        # "credible" | "synthetic" | "synthetic_hard"
    ground_truth_reasoning: str    # explanation — hidden from agent


@dataclass
class Observation:
    claim: str
    dossier_urls: list[str]
    current_page_content: str | None
    visited_urls: list[str]
    search_results: list[dict]     # each: {url, snippet}
    tool_result: Any
    case_file: CaseFile
    steps_remaining: int
    last_action_feedback: str
    last_action_error: str | None
    partial_score: float


@dataclass
class State:
    session_id: str
    synthetic_internet: dict[str, Page]  # keyed by url for O(1) lookup
    case_file: CaseFile
    visited_urls: list[str]
    steps_used: int
    max_steps: int
    seed: int
    ground_truth: dict             # {verdict: str, primary_source_url: str}
    difficulty: str                # "easy" | "medium" | "hard"
    done: bool = False
    accumulated_step_rewards: float = 0.0
    citation_traced_urls: set = field(default_factory=set)
