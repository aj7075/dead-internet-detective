from __future__ import annotations

from dead_internet_detective.models import State

_LIST_FIELDS = {
    "confirmed_facts", "disputed_facts", "synthetic_sources",
    "credible_sources", "open_questions", "contradiction_log",
}

_EXPERT_RESPONSES = {
    "medical": (
        "Based on current clinical evidence, the claim lacks peer-reviewed support. "
        "Consult a licensed physician before drawing conclusions.",
        0.72,
    ),
    "legal": (
        "From a legal standpoint, the claim may constitute defamation if unsubstantiated. "
        "Jurisdictional variance applies.",
        0.68,
    ),
    "statistical": (
        "The sample size referenced is insufficient for the stated confidence interval. "
        "Margin of error likely exceeds ±5%.",
        0.80,
    ),
}


def update_case_file(params: dict, state: State) -> dict:
    cf = state.case_file
    for key, value in params.items():
        if key in _LIST_FIELDS:
            existing: list = getattr(cf, key)
            for item in (value if isinstance(value, list) else [value]):
                if item not in existing:
                    existing.append(item)
        elif key == "confidence":
            cf.confidence = value
    return {"updated": True, "step_reward": 0.01}


def request_expert(params: dict, state: State) -> dict:
    domain = params.get("domain", "")
    if domain not in _EXPERT_RESPONSES:
        return {"error": f"unknown_domain: {domain!r}. Must be medical|legal|statistical", "step_reward": 0.0}
    opinion, confidence = _EXPERT_RESPONSES[domain]
    return {"expert_opinion": opinion, "confidence": confidence, "step_reward": 0.02}


def file_report(params: dict, state: State) -> dict:
    verdict = params.get("verdict", "")
    if verdict not in {"true", "false", "unverifiable"}:
        return {"error": f"invalid_verdict: {verdict!r}", "step_reward": 0.0}
    confidence = max(0.0, min(1.0, float(params.get("confidence", 0.0))))
    params["confidence"] = confidence
    state.done = True
    return {"accepted": True, "step_reward": 0.0}
