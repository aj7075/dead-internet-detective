from __future__ import annotations

import random
import string
from datetime import date, timedelta

from dead_internet_detective.models import Page

# ── helpers ──────────────────────────────────────────────────────────────────

_KEYWORDS = [
    "miracle cure", "big pharma secret", "doctors hate this", "proven remedy",
    "natural healing", "detox protocol", "immune boost", "gut flora reset",
]
_CLAIMS = [
    "Study shows 94% efficacy with zero side effects",
    "New research confirms link between 5G and immune suppression",
    "Scientists discover compound reverses aging in 30 days",
    "Government data reveals vaccine shedding phenomenon",
    "Ancient remedy outperforms modern antibiotics",
    "Climate models found to overstate warming by factor of 3",
]
_AUTHORS = [
    "Dr. Sarah Chen", "Prof. Marcus Webb", "Dr. Yusuf Al-Amin",
    "Dr. Elena Petrov", "Prof. Liang Xu", "Dr. Amara Osei",
]
_STOLEN_BYLINES = [
    "John Hopkins Medical Correspondent",
    "Reuters Health & Science Desk",
    "WHO Independent Observer",
    "Nature Editorial Board Member",
]
_DOMAINS_CREDIBLE = [
    "nih.gov", "who.int", "nature.com", "pubmed.ncbi.nlm.nih.gov",
    "cdc.gov", "sciencedirect.com",
]
_TLD_JUNK = [".info", ".xyz", ".top", ".click", ".health", ".online"]


def _rand_str(rng: random.Random, length: int) -> str:
    return "".join(rng.choices(string.ascii_lowercase, k=length))


def _iso(d: date) -> str:
    return d.isoformat()


def _base_date(rng: random.Random) -> date:
    # anchor within 2023–2024
    return date(2023, 1, 1) + timedelta(days=rng.randint(0, 730))


# ── page factories ────────────────────────────────────────────────────────────

def _generate_seo_farm(rng: random.Random, base_date: date) -> Page:
    domain = _rand_str(rng, 8) + rng.choice(_TLD_JUNK)
    reg = base_date + timedelta(days=rng.randint(1, 29))  # invariant 3: <30 days
    slug = "-".join(rng.choices(_KEYWORDS, k=3)).replace(" ", "-")
    url = f"https://{domain}/{slug}"
    claim = rng.choice(_CLAIMS)
    content = (
        f"{' '.join(rng.choices(_KEYWORDS, k=20))}. "
        f"{claim}. "
        f"{'Buy now! ' * 5}"
    )
    return Page(
        url=url,
        content=content,
        author=None,
        domain=domain,
        registration_date=_iso(reg),
        citation_urls=[],           # filled later by _plant_citation_circle
        image_urls=[f"https://{domain}/img/{_rand_str(rng, 6)}.jpg"],
        linguistic_features={
            "ai_probability": round(rng.uniform(0.75, 0.99), 3),
            "key_claims": [claim],
            "contradicted_by": [],
        },
        wayback_available=False,
        ground_truth_label="synthetic",
        ground_truth_reasoning="SEO farm: no author, new domain, keyword stuffed, circular citations.",
    )


def _generate_wellness_blog(rng: random.Random, base_date: date) -> Page:
    domain = "wellness-" + _rand_str(rng, 5) + ".com"
    reg = base_date - timedelta(days=rng.randint(60, 400))
    author = rng.choice(_STOLEN_BYLINES)          # invariant 4: never None
    claim = rng.choice(_CLAIMS)
    stat = f"{rng.randint(60, 99)}% of participants reported improvement"
    url = f"https://{domain}/article/{_rand_str(rng, 8)}"
    content = (
        f"By {author}. "
        f"A groundbreaking study found that {claim.lower()}. "
        f"Furthermore, {stat}. "
        f"Experts in leading institutions agree this changes everything."
    )
    return Page(
        url=url,
        content=content,
        author=author,
        domain=domain,
        registration_date=_iso(reg),
        citation_urls=[],
        image_urls=[f"https://{domain}/wp-content/{_rand_str(rng, 6)}.png"],
        linguistic_features={
            "ai_probability": round(rng.uniform(0.55, 0.85), 3),
            "key_claims": [claim, stat],
            "contradicted_by": [],
        },
        wayback_available=rng.random() > 0.5,
        ground_truth_label="synthetic",
        ground_truth_reasoning="Wellness blog: stolen byline, fabricated statistics, no primary source.",
    )


def _generate_credible_source(rng: random.Random) -> Page:
    domain = rng.choice(_DOMAINS_CREDIBLE)
    author = rng.choice(_AUTHORS)
    claim = rng.choice(_CLAIMS)
    reg = _iso(date(2010, 1, 1) + timedelta(days=rng.randint(0, 3000)))
    url = f"https://{domain}/publications/{_rand_str(rng, 10)}"
    content = (
        f"Abstract. We present findings that contradict unsubstantiated claims circulating online. "
        f"Author: {author}. "
        f"Our peer-reviewed study (n={rng.randint(500, 5000)}) found no evidence supporting: {claim.lower()}. "
        f"Data and methodology available upon request."
    )
    return Page(
        url=url,
        content=content,
        author=author,
        domain=domain,
        registration_date=reg,
        citation_urls=[f"https://doi.org/10.{rng.randint(1000,9999)}/{''.join(rng.choices(string.ascii_lowercase+string.digits, k=8))}"],
        image_urls=[],
        linguistic_features={
            "ai_probability": round(rng.uniform(0.02, 0.18), 3),
            "key_claims": [f"No evidence for: {claim.lower()}"],
            "contradicted_by": [],
        },
        wayback_available=True,
        ground_truth_label="credible",
        ground_truth_reasoning="Peer-reviewed source on established domain with named author and DOI.",
    )


def _generate_sophisticated_fake(rng: random.Random) -> Page:
    """Real DOI format, real author name, but conclusion inverted."""
    domain = "journal-" + _rand_str(rng, 6) + ".org"
    author = rng.choice(_AUTHORS)
    claim = rng.choice(_CLAIMS)
    doi = f"10.{rng.randint(1000,9999)}/{''.join(rng.choices(string.ascii_lowercase+string.digits, k=8))}"
    url = f"https://{domain}/doi/{doi}"
    reg = _iso(date(2022, 1, 1) + timedelta(days=rng.randint(0, 365)))
    content = (
        f"DOI: {doi}. Author: {author}. "
        f"This study confirms: {claim}. "
        f"Methodology: randomized controlled trial, n={rng.randint(100, 300)}. "
        f"Conclusion: Results are statistically significant (p<0.001). "
        f"Note: journal not indexed in PubMed or Scopus."
    )
    return Page(
        url=url,
        content=content,
        author=author,
        domain=domain,
        registration_date=reg,
        citation_urls=[f"https://doi.org/{doi}"],
        image_urls=[],
        linguistic_features={
            "ai_probability": round(rng.uniform(0.40, 0.70), 3),
            "key_claims": [claim],
            "contradicted_by": [],
        },
        wayback_available=rng.random() > 0.6,
        ground_truth_label="synthetic_hard",
        ground_truth_reasoning=(
            "Sophisticated fake: valid DOI format and real author name, "
            "but journal is not indexed; conclusion inverted from actual evidence."
        ),
    )


def _plant_citation_circle(pages: list[Page], rng: random.Random) -> list[Page]:
    """Wire at least 2 synthetic pages so they cite each other."""
    synthetic = [p for p in pages if p.ground_truth_label in ("synthetic", "synthetic_hard")]
    if len(synthetic) < 2:
        return pages
    # pick 2 (or 3 if available) to form the circle
    circle_size = min(3, len(synthetic))
    circle = rng.sample(synthetic, circle_size)
    url_map = {p.url: p for p in pages}
    for i, page in enumerate(circle):
        next_page = circle[(i + 1) % circle_size]
        if next_page.url not in page.citation_urls:
            page.citation_urls.append(next_page.url)
    return pages


# ── verdict distribution ──────────────────────────────────────────────────────

_VERDICT_WEIGHTS = {
    "true": 0.40,
    "false": 0.35,
    "unverifiable": 0.25,
}


def _pick_verdict(rng: random.Random) -> str:
    r = rng.random()
    if r < 0.40:
        return "true"
    elif r < 0.75:
        return "false"
    return "unverifiable"


# ── main entry point ──────────────────────────────────────────────────────────

def generate_episode(difficulty: str, seed: int) -> tuple[dict[str, Page], dict]:
    """
    Returns (pages_by_url, ground_truth).
    ground_truth = {"verdict": "true"|"false"|"unverifiable", "primary_source_url": str}
    """
    rng = random.Random(seed)
    base = _base_date(rng)
    pages: list[Page] = []

    if difficulty == "easy":
        # 1 credible + 2 synthetic (seo/wellness)
        pages.append(_generate_credible_source(rng))
        pages.append(_generate_seo_farm(rng, base))
        pages.append(_generate_wellness_blog(rng, base))

    elif difficulty == "medium":
        # 1–2 credible + mix of synthetic
        n_credible = rng.randint(1, 2)
        for _ in range(n_credible):
            pages.append(_generate_credible_source(rng))
        pages.append(_generate_seo_farm(rng, base))
        pages.append(_generate_wellness_blog(rng, base))
        pages.append(_generate_wellness_blog(rng, base))

    else:  # hard — invariants: exactly 1 credible, at least 1 synthetic_hard
        pages.append(_generate_credible_source(rng))
        pages.append(_generate_sophisticated_fake(rng))   # synthetic_hard
        pages.append(_generate_seo_farm(rng, base))
        pages.append(_generate_wellness_blog(rng, base))
        pages.append(_generate_sophisticated_fake(rng))   # second synthetic_hard

    pages = _plant_citation_circle(pages, rng)

    pages_by_url = {p.url: p for p in pages}

    # primary source = first credible page
    credible_pages = [p for p in pages if p.ground_truth_label == "credible"]
    primary_url = credible_pages[0].url if credible_pages else pages[0].url

    verdict = _pick_verdict(rng)
    ground_truth = {
        "verdict": verdict,
        "primary_source_url": primary_url,
    }

    return pages_by_url, ground_truth
