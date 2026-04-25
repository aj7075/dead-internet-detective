from __future__ import annotations

import collections

import pytest

from dead_internet_detective.generator import generate_episode
from dead_internet_detective.models import Page


# ── helpers ───────────────────────────────────────────────────────────────────

def all_pages(pages_by_url: dict) -> list[Page]:
    return list(pages_by_url.values())


def credible_count(pages_by_url: dict) -> int:
    return sum(1 for p in pages_by_url.values() if p.ground_truth_label == "credible")


def has_synthetic_hard(pages_by_url: dict) -> bool:
    return any(p.ground_truth_label == "synthetic_hard" for p in pages_by_url.values())


def citation_circle_exists(pages_by_url: dict) -> bool:
    """True if at least 2 synthetic pages cite each other."""
    synthetic_urls = {url for url, p in pages_by_url.items()
                      if p.ground_truth_label in ("synthetic", "synthetic_hard")}
    for url in synthetic_urls:
        page = pages_by_url[url]
        for cited in page.citation_urls:
            if cited in synthetic_urls and cited != url:
                cited_page = pages_by_url.get(cited)
                if cited_page and url in cited_page.citation_urls:
                    return True
        # also accept one-directional circle across ≥2 synthetic nodes
        citations_in_synthetic = [c for c in page.citation_urls if c in synthetic_urls]
        if citations_in_synthetic:
            return True
    return False


# ── invariant tests ───────────────────────────────────────────────────────────

@pytest.mark.parametrize("seed", range(10))
def test_easy_exactly_one_credible(seed):
    pages, _ = generate_episode("easy", seed)
    assert credible_count(pages) == 1, f"seed={seed}: easy must have exactly 1 credible"


@pytest.mark.parametrize("seed", range(10))
def test_medium_one_or_two_credible(seed):
    pages, _ = generate_episode("medium", seed)
    c = credible_count(pages)
    assert 1 <= c <= 2, f"seed={seed}: medium must have 1–2 credible, got {c}"


@pytest.mark.parametrize("seed", range(10))
def test_hard_exactly_one_credible(seed):
    pages, _ = generate_episode("hard", seed)
    assert credible_count(pages) == 1, f"seed={seed}: hard must have exactly 1 credible"


@pytest.mark.parametrize("seed", range(10))
def test_hard_has_synthetic_hard(seed):
    pages, _ = generate_episode("hard", seed)
    assert has_synthetic_hard(pages), f"seed={seed}: hard must include synthetic_hard"


@pytest.mark.parametrize("seed", range(20))
def test_seo_farm_domain_age(seed):
    pages, _ = generate_episode("easy", seed)
    for page in pages.values():
        if page.ground_truth_label == "synthetic" and page.author is None:
            from datetime import date
            reg = date.fromisoformat(page.registration_date)
            # base_date derived from seed — just verify it's a valid ISO date and within realistic range
            assert reg.year >= 2023
            assert reg.year <= 2025


@pytest.mark.parametrize("seed", range(20))
def test_wellness_blog_has_author(seed):
    pages, _ = generate_episode("easy", seed)
    for page in pages.values():
        if "wellness" in page.domain:
            assert page.author is not None, f"wellness blog must have author, url={page.url}"


@pytest.mark.parametrize("seed", range(20))
def test_citation_circle_exists(seed):
    for diff in ("easy", "medium", "hard"):
        pages, _ = generate_episode(diff, seed)
        synthetic = [p for p in pages.values()
                     if p.ground_truth_label in ("synthetic", "synthetic_hard")]
        if len(synthetic) >= 2:
            # at least one synthetic page cites another synthetic page
            syn_urls = {p.url for p in synthetic}
            found = any(
                any(c in syn_urls for c in p.citation_urls)
                for p in synthetic
            )
            assert found, f"seed={seed} diff={diff}: no citation circle among synthetic pages"


def test_verdict_distribution():
    counts: dict[str, int] = collections.Counter()
    n = 100
    for seed in range(n):
        diff = ["easy", "medium", "hard"][seed % 3]
        _, gt = generate_episode(diff, seed)
        counts[gt["verdict"]] += 1

    assert counts["true"] >= 25, f"true count={counts['true']} too low"
    assert counts["false"] >= 20, f"false count={counts['false']} too low"
    assert counts["unverifiable"] >= 10, f"unverifiable count={counts['unverifiable']} too low"


def test_determinism():
    for seed in (0, 42, 999):
        pages1, gt1 = generate_episode("hard", seed)
        pages2, gt2 = generate_episode("hard", seed)
        assert list(pages1.keys()) == list(pages2.keys()), f"seed={seed}: URL order differs"
        assert gt1 == gt2, f"seed={seed}: ground_truth differs"


def test_ground_truth_keys():
    for diff in ("easy", "medium", "hard"):
        _, gt = generate_episode(diff, 0)
        assert "verdict" in gt
        assert "primary_source_url" in gt
        assert gt["verdict"] in ("true", "false", "unverifiable")


def test_linguistic_features_shape():
    pages, _ = generate_episode("hard", 7)
    for page in pages.values():
        lf = page.linguistic_features
        assert "ai_probability" in lf
        assert "key_claims" in lf
        assert "contradicted_by" in lf
        assert isinstance(lf["ai_probability"], float)
        assert isinstance(lf["key_claims"], list)
        assert isinstance(lf["contradicted_by"], list)
