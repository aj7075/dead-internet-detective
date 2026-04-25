from __future__ import annotations

import pytest

from dead_internet_detective.models import CaseFile, Page, State
from dead_internet_detective.tools.browser import visit_page, search, wayback_check
from dead_internet_detective.tools.identity import whois_lookup, author_lookup, image_provenance
from dead_internet_detective.tools.reasoning import citation_trace, linguistic_fingerprint, cross_reference
from dead_internet_detective.tools.casefile import update_case_file, request_expert, file_report


def make_page(**kwargs) -> Page:
    defaults = dict(
        url="http://example.com",
        content="Some content about health and vaccines.",
        author="Alice",
        domain="example.com",
        registration_date="2023-01-01",
        citation_urls=[],
        image_urls=[],
        linguistic_features={"ai_probability": 0.1, "key_claims": [], "contradicted_by": [], "patterns": []},
        wayback_available=True,
        ground_truth_label="credible",
        ground_truth_reasoning="Well sourced",
    )
    defaults.update(kwargs)
    return Page(**defaults)


def make_state(pages: list[Page] | None = None) -> State:
    if pages is None:
        pages = [make_page()]
    internet = {p.url: p for p in pages}
    return State(
        session_id="test",
        synthetic_internet=internet,
        case_file=CaseFile.empty("Is the claim true?"),
        visited_urls=[],
        steps_used=0,
        max_steps=20,
        seed=0,
        ground_truth={"verdict": "true", "primary_source_url": "http://example.com"},
        difficulty="easy",
    )


# ── B1: browser ──────────────────────────────────────────────────────────────

class TestVisitPage:
    def test_happy_path(self):
        state = make_state()
        result = visit_page({"url": "http://example.com"}, state)
        assert result["step_reward"] == 0.02
        assert "content" in result
        assert "http://example.com" in state.visited_urls

    def test_page_not_found(self):
        state = make_state()
        result = visit_page({"url": "http://missing.com"}, state)
        assert result["error"] == "page_not_found"
        assert result["step_reward"] == 0.0

    def test_revisit_penalty(self):
        state = make_state()
        visit_page({"url": "http://example.com"}, state)
        result = visit_page({"url": "http://example.com"}, state)
        assert result["error"] == "already_visited"
        assert result["step_reward"] == -0.02


class TestSearch:
    def test_finds_matching_pages(self):
        state = make_state()
        result = search({"query": "health"}, state)
        assert result["step_reward"] == 0.01
        assert len(result["results"]) == 1
        assert result["results"][0]["url"] == "http://example.com"

    def test_no_match(self):
        state = make_state()
        result = search({"query": "zzznomatch"}, state)
        assert result["results"] == []

    def test_top5_limit(self):
        pages = [
            make_page(url=f"http://site{i}.com", domain=f"site{i}.com", content="keyword " * (i + 1))
            for i in range(10)
        ]
        state = make_state(pages)
        result = search({"query": "keyword"}, state)
        assert len(result["results"]) == 5

    def test_snippet_length(self):
        state = make_state()
        result = search({"query": "health"}, state)
        assert len(result["results"][0]["snippet"]) <= 150


class TestWaybackCheck:
    def test_happy_path(self):
        state = make_state()
        result = wayback_check({"url": "http://example.com"}, state)
        assert result["available"] is True
        assert result["step_reward"] == 0.02

    def test_not_found(self):
        state = make_state()
        result = wayback_check({"url": "http://missing.com"}, state)
        assert result["error"] == "page_not_found"


# ── B2: identity ──────────────────────────────────────────────────────────────

class TestWhoisLookup:
    def test_happy_path(self):
        state = make_state()
        result = whois_lookup({"domain": "example.com"}, state)
        assert result["registration_date"] == "2023-01-01"
        assert result["step_reward"] == 0.02

    def test_domain_not_found(self):
        state = make_state()
        result = whois_lookup({"domain": "unknown.com"}, state)
        assert "error" in result


class TestAuthorLookup:
    def test_happy_path(self):
        pages = [
            make_page(url="http://a.com", domain="a.com", author="Bob"),
            make_page(url="http://b.com", domain="b.com", author="Bob"),
        ]
        state = make_state(pages)
        result = author_lookup({"author_name": "Bob"}, state)
        assert len(result["pages_found"]) == 2
        assert result["suspicious"] is True  # 2 distinct domains

    def test_single_domain_not_suspicious(self):
        pages = [
            make_page(url="http://a.com/p1", domain="a.com", author="Carol"),
            make_page(url="http://a.com/p2", domain="a.com", author="Carol"),
        ]
        state = make_state(pages)
        result = author_lookup({"author_name": "Carol"}, state)
        assert result["suspicious"] is False

    def test_author_not_found(self):
        state = make_state()
        result = author_lookup({"author_name": "Nobody"}, state)
        assert result["pages_found"] == []


class TestImageProvenance:
    def test_happy_path(self):
        img = "http://cdn.com/img.jpg"
        pages = [
            make_page(url=f"http://site{i}.com", domain=f"site{i}.com", image_urls=[img])
            for i in range(3)
        ]
        state = make_state(pages)
        result = image_provenance({"image_url": img}, state)
        assert result["count"] == 3
        assert result["step_reward"] == 0.02

    def test_not_found(self):
        state = make_state()
        result = image_provenance({"image_url": "http://cdn.com/nope.jpg"}, state)
        assert result["count"] == 0


# ── B3: reasoning ─────────────────────────────────────────────────────────────

class TestCitationTrace:
    def test_terminates_at_credible(self):
        pages = [
            make_page(url="http://a.com", domain="a.com", citation_urls=["http://b.com"],
                      ground_truth_label="synthetic"),
            make_page(url="http://b.com", domain="b.com", citation_urls=[],
                      ground_truth_label="credible"),
        ]
        state = make_state(pages)
        result = citation_trace({"url": "http://a.com"}, state)
        assert result["terminates_at_credible"] is True
        assert "http://a.com" in result["chain"]

    def test_circular_citation(self):
        pages = [
            make_page(url="http://a.com", domain="a.com", citation_urls=["http://b.com"],
                      ground_truth_label="synthetic"),
            make_page(url="http://b.com", domain="b.com", citation_urls=["http://a.com"],
                      ground_truth_label="synthetic"),
        ]
        state = make_state(pages)
        result = citation_trace({"url": "http://a.com"}, state)
        assert result["terminates_at_credible"] is False
        assert isinstance(result["chain"], list)

    def test_not_found(self):
        state = make_state()
        result = citation_trace({"url": "http://missing.com"}, state)
        assert "error" in result


class TestLinguisticFingerprint:
    def test_error_if_not_visited(self):
        state = make_state()
        result = linguistic_fingerprint({"url": "http://example.com"}, state)
        assert result["error"] == "page_not_visited"

    def test_happy_path(self):
        state = make_state()
        state.visited_urls.append("http://example.com")
        result = linguistic_fingerprint({"url": "http://example.com"}, state)
        assert "ai_probability" in result
        assert result["step_reward"] == 0.02


class TestCrossReference:
    def test_detects_contradiction(self):
        pages = [
            make_page(url="http://a.com", domain="a.com",
                      linguistic_features={"ai_probability": 0.1, "key_claims": ["vaccines cause autism"],
                                           "contradicted_by": [], "patterns": []}),
            make_page(url="http://b.com", domain="b.com",
                      linguistic_features={"ai_probability": 0.1, "key_claims": [],
                                           "contradicted_by": ["vaccines cause autism"], "patterns": []}),
        ]
        state = make_state(pages)
        result = cross_reference({"url_a": "http://a.com", "url_b": "http://b.com"}, state)
        assert result["contradiction_found"] is True
        assert result["step_reward"] == 0.03

    def test_no_contradiction(self):
        pages = [
            make_page(url="http://a.com", domain="a.com",
                      linguistic_features={"ai_probability": 0.1, "key_claims": ["earth is round"],
                                           "contradicted_by": [], "patterns": []}),
            make_page(url="http://b.com", domain="b.com",
                      linguistic_features={"ai_probability": 0.1, "key_claims": ["sky is blue"],
                                           "contradicted_by": [], "patterns": []}),
        ]
        state = make_state(pages)
        result = cross_reference({"url_a": "http://a.com", "url_b": "http://b.com"}, state)
        assert result["contradiction_found"] is False

    def test_page_not_found(self):
        state = make_state()
        result = cross_reference({"url_a": "http://missing.com", "url_b": "http://example.com"}, state)
        assert "error" in result


# ── B4: casefile ──────────────────────────────────────────────────────────────

class TestUpdateCaseFile:
    def test_appends_facts(self):
        state = make_state()
        update_case_file({"confirmed_facts": ["fact1"]}, state)
        update_case_file({"confirmed_facts": ["fact2"]}, state)
        assert "fact1" in state.case_file.confirmed_facts
        assert "fact2" in state.case_file.confirmed_facts

    def test_no_duplicates(self):
        state = make_state()
        update_case_file({"confirmed_facts": ["fact1"]}, state)
        update_case_file({"confirmed_facts": ["fact1"]}, state)
        assert state.case_file.confirmed_facts.count("fact1") == 1

    def test_confidence_replaced(self):
        state = make_state()
        update_case_file({"confidence": 0.8}, state)
        assert state.case_file.confidence == 0.8


class TestRequestExpert:
    def test_medical(self):
        state = make_state()
        result = request_expert({"domain": "medical", "question": "Is X safe?"}, state)
        assert "expert_opinion" in result
        assert 0.0 <= result["confidence"] <= 1.0

    def test_invalid_domain(self):
        state = make_state()
        result = request_expert({"domain": "cooking", "question": "How to bake?"}, state)
        assert "error" in result


class TestFileReport:
    def test_sets_done(self):
        state = make_state()
        result = file_report(
            {"verdict": "true", "confidence": 0.9, "evidence_chain": [], "reasoning": "solid"},
            state,
        )
        assert result["accepted"] is True
        assert state.done is True

    def test_invalid_verdict(self):
        state = make_state()
        result = file_report(
            {"verdict": "maybe", "confidence": 0.5, "evidence_chain": [], "reasoning": ""},
            state,
        )
        assert "error" in result
        assert state.done is False

    def test_confidence_clamped(self):
        state = make_state()
        file_report(
            {"verdict": "false", "confidence": 2.5, "evidence_chain": [], "reasoning": "x"},
            state,
        )
        assert state.done is True  # still accepted after clamp
