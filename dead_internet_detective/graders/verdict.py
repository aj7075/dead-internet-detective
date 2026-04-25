from __future__ import annotations

from dead_internet_detective.models import State


def verdict_accuracy(state: State, filed_report: dict) -> float:
    verdict = filed_report.get("verdict", "")
    truth = state.ground_truth.get("verdict", "")
    if verdict == truth:
        return 1.0
    if verdict == "unverifiable" and truth == "false":
        return 0.3
    return 0.0
