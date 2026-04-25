from __future__ import annotations

from dead_internet_detective.models import State


def internal_consistency(state: State, filed_report: dict) -> float:
    log = state.case_file.contradiction_log
    if not log:
        return 1.0
    reasoning = filed_report.get("reasoning", "")
    for entry in log:
        source_a = entry.get("source_a", "")
        source_b = entry.get("source_b", "")
        if source_a not in reasoning or source_b not in reasoning:
            return 0.0
    return 1.0
