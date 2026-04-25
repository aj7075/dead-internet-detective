from .verdict import verdict_accuracy
from .evidence import evidence_chain_quality
from .detection import synthetic_detection_rate
from .consistency import internal_consistency
from .efficiency import step_efficiency


def compute_terminal_reward(state, filed_report: dict) -> float:
    r = (
        0.30 * verdict_accuracy(state, filed_report) +
        0.25 * evidence_chain_quality(state, filed_report) +
        0.20 * synthetic_detection_rate(state, filed_report) +
        0.15 * internal_consistency(state, filed_report) +
        0.10 * step_efficiency(state, filed_report)
    ) * 0.5

    if state.steps_used >= state.max_steps:
        r -= 0.15

    return max(-0.5, min(1.0, r))
