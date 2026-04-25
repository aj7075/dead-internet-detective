from __future__ import annotations

from dead_internet_detective.models import State


def evidence_chain_quality(state: State, filed_report: dict) -> float:
    chain = filed_report.get("evidence_chain", [])
    if not chain:
        return 0.0
    visited = set(state.visited_urls)
    traced = getattr(state, "citation_traced_urls", set())
    valid = sum(1 for url in chain if url in visited)
    score = valid / len(chain)
    bonus = sum(0.1 for url in chain if url in traced)
    return min(1.0, score + bonus)
