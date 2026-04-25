from __future__ import annotations

import pytest

from dead_internet_detective.models import CaseFile, Page, State
from dead_internet_detective.graders.verdict import verdict_accuracy
from dead_internet_detective.graders.evidence import evidence_chain_quality
from dead_internet_detective.graders.detection import synthetic_detection_rate
from dead_internet_detective.graders.consistency import internal_consistency
from dead_internet_detective.graders.efficiency import step_efficiency
from dead_internet_detective.graders import compute_terminal_reward


def make_page(url="http://example.com", label="credible", **kwargs) -> Page:
    defaults = dict(
        url=url,
        content="content",
        author="Author",
        domain=url.split("//")[1].split("/")[0],
        registration_date="2023-01-01",
        citation_urls=[],
        image_urls=[],
        linguistic_features={"ai_probability": 0.1, "key_claims": [], "contradicted_by": []},
        wayback_available=True,
        ground_truth_label=label,
        ground_truth_reasoning="reason",
    )
    defaults.update(kwargs)
    return Page(**defaults)


def make_state(**kwargs) -> State:
    pages = kwargs.pop("pages", [make_page()])
    internet = {p.url: p for p in pages}
    defaults = dict(
        session_id="test",
        synthetic_internet=internet,
        case_file=CaseFile.empty("claim"),
        visited_urls=[],
        steps_used=5,
        max_steps=20,
        seed=0,
        ground_truth={"verdict": "true", "primary_source_url": "http://example.com"},
        difficulty="easy",
    )
    defaults.update(kwargs)
    return State(**defaults)


# ── verdict_accuracy ──────────────────────────────────────────────────────────

class TestVerdictAccuracy:
    def test_correct(self):
        state = make_state(ground_truth={"verdict": "true"})
        assert verdict_accuracy(state, {"verdict": "true"}) == 1.0

    def test_wrong(self):
        state = make_state(ground_truth={"verdict": "true"})
        assert verdict_accuracy(state, {"verdict": "false"}) == 0.0

    def test_unverifiable_partial_credit(self):
        state = make_state(ground_truth={"verdict": "false"})
        assert verdict_accuracy(state, {"verdict": "unverifiable"}) == 0.3

    def test_unverifiable_no_credit_when_truth_true(self):
        state = make_state(ground_truth={"verdict": "true"})
        assert verdict_accuracy(state, {"verdict": "unverifiable"}) == 0.0


# ── evidence_chain_quality ────────────────────────────────────────────────────

class TestEvidenceChainQuality:
    def test_empty_chain(self):
        state = make_state()
        assert evidence_chain_quality(state, {"evidence_chain": []}) == 0.0

    def test_all_visited(self):
        state = make_state()
        state.visited_urls = ["http://example.com"]
        score = evidence_chain_quality(state, {"evidence_chain": ["http://example.com"]})
        assert score == 1.0

    def test_none_visited(self):
        state = make_state()
        score = evidence_chain_quality(state, {"evidence_chain": ["http://example.com"]})
        assert score == 0.0

    def test_citation_traced_bonus(self):
        state = make_state()
        state.visited_urls = ["http://example.com"]
        state.citation_traced_urls = {"http://example.com"}
        score = evidence_chain_quality(state, {"evidence_chain": ["http://example.com"]})
        assert score == 1.0  # clamped: 1.0 + 0.1 → 1.0


# ── synthetic_detection_rate ──────────────────────────────────────────────────

class TestSyntheticDetectionRate:
    def test_all_flagged(self):
        pages = [make_page(url="http://bad.com", label="synthetic")]
        state = make_state(pages=pages)
        state.case_file.synthetic_sources.append("http://bad.com")
        assert synthetic_detection_rate(state, {}) == 1.0

    def test_none_flagged(self):
        pages = [make_page(url="http://bad.com", label="synthetic")]
        state = make_state(pages=pages)
        assert synthetic_detection_rate(state, {}) == 0.0

    def test_no_synthetic_pages(self):
        state = make_state()  # default page is credible
        assert synthetic_detection_rate(state, {}) == 1.0

    def test_partial(self):
        pages = [
            make_page(url="http://bad1.com", label="synthetic"),
            make_page(url="http://bad2.com", label="synthetic"),
        ]
        state = make_state(pages=pages)
        state.case_file.synthetic_sources.append("http://bad1.com")
        assert synthetic_detection_rate(state, {}) == 0.5


# ── internal_consistency ──────────────────────────────────────────────────────

class TestInternalConsistency:
    def test_empty_log_returns_one(self):
        state = make_state()
        assert internal_consistency(state, {"reasoning": ""}) == 1.0

    def test_contradiction_addressed(self):
        state = make_state()
        state.case_file.contradiction_log.append({
            "source_a": "http://a.com",
            "source_b": "http://b.com",
            "description": "conflict",
        })
        reasoning = "I found conflict between http://a.com and http://b.com"
        assert internal_consistency(state, {"reasoning": reasoning}) == 1.0

    def test_contradiction_not_mentioned(self):
        state = make_state()
        state.case_file.contradiction_log.append({
            "source_a": "http://a.com",
            "source_b": "http://b.com",
            "description": "conflict",
        })
        assert internal_consistency(state, {"reasoning": "nothing relevant"}) == 0.0

    def test_binary_no_partial_credit(self):
        state = make_state()
        state.case_file.contradiction_log.append({
            "source_a": "http://a.com",
            "source_b": "http://b.com",
            "description": "conflict",
        })
        result = internal_consistency(state, {"reasoning": "only http://a.com mentioned"})
        assert result in (0.0, 1.0)  # must be binary


# ── step_efficiency ───────────────────────────────────────────────────────────

class TestStepEfficiency:
    def test_no_steps_used(self):
        state = make_state(steps_used=0, max_steps=20)
        assert step_efficiency(state, {}) == 1.0

    def test_half_steps(self):
        state = make_state(steps_used=10, max_steps=20)
        assert step_efficiency(state, {}) == 0.5

    def test_all_steps(self):
        state = make_state(steps_used=20, max_steps=20)
        assert step_efficiency(state, {}) == 0.0


# ── compute_terminal_reward ───────────────────────────────────────────────────

class TestComputeTerminalReward:
    def test_perfect_score_bounded(self):
        pages = [make_page(label="credible")]
        state = make_state(pages=pages, steps_used=0, max_steps=20,
                           ground_truth={"verdict": "true"})
        state.visited_urls = ["http://example.com"]
        report = {
            "verdict": "true",
            "confidence": 1.0,
            "evidence_chain": ["http://example.com"],
            "reasoning": "",
        }
        score = compute_terminal_reward(state, report)
        assert -0.5 <= score <= 1.0

    def test_step_exhaustion_penalty(self):
        pages = [make_page(url="http://bad.com", label="synthetic")]
        state = make_state(pages=pages, steps_used=20, max_steps=20,
                           ground_truth={"verdict": "false"})
        report = {"verdict": "true", "confidence": 0.0, "evidence_chain": [], "reasoning": ""}
        score = compute_terminal_reward(state, report)
        assert score < 0
