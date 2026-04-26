from __future__ import annotations

from dead_internet_detective.models import State


def whois_lookup(params: dict, state: State) -> dict:
    domain = params["domain"]
    for page in state.synthetic_internet.values():
        if page.domain == domain:
            return {
                "registration_date": page.registration_date,
                "registrar": "synthetic-registrar.io",
                "step_reward": 0.02,
            }
    return {"error": "domain_not_found", "step_reward": 0.0}


def author_lookup(params: dict, state: State) -> dict:
    name = params["author_name"]
    pages_found = []
    domains = []
    for url, page in state.synthetic_internet.items():
        if page.author == name:
            pages_found.append(url)
            if page.domain not in domains:
                domains.append(page.domain)
    suspicious = len(domains) > 1
    return {
        "pages_found": pages_found,
        "domains": domains,
        "suspicious": suspicious,
        "step_reward": 0.02,
    }


def image_provenance(params: dict, state: State) -> dict:
    image_url = params["image_url"]
    appears_on = [
        url for url, page in state.synthetic_internet.items()
        if image_url in page.image_urls
    ]
    return {
        "appears_on": appears_on,
        "count": len(appears_on),
        "step_reward": 0.02,
    }
