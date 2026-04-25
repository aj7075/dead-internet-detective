from __future__ import annotations

from collections import deque

from dead_internet_detective.models import State


def citation_trace(params: dict, state: State) -> dict:
    start_url = params["url"]
    if start_url not in state.synthetic_internet:
        return {"error": "page_not_found", "step_reward": 0.0}

    visited: set[str] = set()
    chain: list[str] = []
    queue: deque[tuple[str, int]] = deque([(start_url, 0)])
    terminates_at_credible = False

    while queue:
        url, hops = queue.popleft()
        if url in visited or hops > 3:
            continue
        visited.add(url)
        chain.append(url)
        if url not in state.synthetic_internet:
            continue
        page = state.synthetic_internet[url]
        if page.ground_truth_label == "credible":
            terminates_at_credible = True
        if hops < 3:
            for next_url in page.citation_urls:
                if next_url not in visited:
                    queue.append((next_url, hops + 1))

    # Track for evidence_chain_quality bonus
    if hasattr(state, "citation_traced_urls"):
        state.citation_traced_urls.add(start_url)

    return {
        "chain": chain,
        "terminates_at_credible": terminates_at_credible,
        "hops": len(chain) - 1,
        "step_reward": 0.03,
    }


def linguistic_fingerprint(params: dict, state: State) -> dict:
    url = params["url"]
    if url not in state.visited_urls:
        return {"error": "page_not_visited"}
    if url not in state.synthetic_internet:
        return {"error": "page_not_found"}
    page = state.synthetic_internet[url]
    return {
        "ai_probability": page.linguistic_features["ai_probability"],
        "patterns": page.linguistic_features.get("patterns", []),
        "step_reward": 0.02,
    }


def cross_reference(params: dict, state: State) -> dict:
    url_a, url_b = params["url_a"], params["url_b"]
    if url_a not in state.synthetic_internet or url_b not in state.synthetic_internet:
        return {"error": "page_not_found", "step_reward": 0.0}
    page_a = state.synthetic_internet[url_a]
    page_b = state.synthetic_internet[url_b]
    claims_a = page_a.linguistic_features.get("key_claims", [])
    claims_b = page_b.linguistic_features.get("key_claims", [])
    contra_a = page_a.linguistic_features.get("contradicted_by", [])
    contra_b = page_b.linguistic_features.get("contradicted_by", [])

    contradiction_found = (
        any(c in contra_b for c in claims_a) or
        any(c in contra_a for c in claims_b)
    )
    description = (
        f"{url_a} and {url_b} make contradicting claims."
        if contradiction_found
        else "No contradiction found."
    )
    return {
        "contradiction_found": contradiction_found,
        "description": description,
        "step_reward": 0.03,
    }
