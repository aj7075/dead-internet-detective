from __future__ import annotations

from dead_internet_detective.models import State


def visit_page(params: dict, state: State) -> dict:
    url = params["url"]
    if url not in state.synthetic_internet:
        return {"error": "page_not_found", "step_reward": 0.0}
    if url in state.visited_urls:
        return {"error": "already_visited", "step_reward": -0.02}
    state.visited_urls.append(url)
    page = state.synthetic_internet[url]
    return {
        "content": page.content,
        "author": page.author,
        "domain": page.domain,
        "step_reward": 0.02,
    }


def search(params: dict, state: State) -> dict:
    query = params["query"].lower()
    scores: list[tuple[int, str]] = []
    for url, page in state.synthetic_internet.items():
        content_lower = page.content.lower()
        count = content_lower.count(query)
        if count > 0:
            scores.append((count, url))
    scores.sort(key=lambda x: x[0], reverse=True)
    results = []
    for _, url in scores[:5]:
        content = state.synthetic_internet[url].content
        idx = content.lower().find(query)
        snippet = content[max(0, idx):max(0, idx) + 150]
        results.append({"url": url, "snippet": snippet})
    return {"results": results, "step_reward": 0.01}


def wayback_check(params: dict, state: State) -> dict:
    url = params["url"]
    if url not in state.synthetic_internet:
        return {"error": "page_not_found", "step_reward": 0.0}
    page = state.synthetic_internet[url]
    return {"available": page.wayback_available, "step_reward": 0.02}
