from __future__ import annotations

from dead_internet_detective.models import State


def step_efficiency(state: State, filed_report: dict) -> float:
    return 1.0 - (state.steps_used / state.max_steps)
