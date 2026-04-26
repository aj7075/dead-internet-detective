import json
from models import Candidate, ParsedJD, MatchResult
from llm_client import chat_complete

EXPERIENCE_BAND = {
    "Junior": (0, 2),
    "Mid": (2, 5),
    "Senior": (4, 9),
    "Lead": (6, 12),
    "Principal": (8, 20),
}


def _skill_overlap(candidate_skills: list[str], required: list[str], preferred: list[str]) -> tuple[float, list[str], list[str]]:
    c_lower = {s.lower() for s in candidate_skills}
    req_lower = [s.lower() for s in required]
    pref_lower = [s.lower() for s in preferred]

    req_matched = [s for s in req_lower if s in c_lower]
    pref_matched = [s for s in pref_lower if s in c_lower]
    missing = [s for s in req_lower if s not in c_lower]

    req_score = (len(req_matched) / max(len(req_lower), 1)) * 60
    pref_score = (len(pref_matched) / max(len(pref_lower), 1)) * 20

    matched_display = req_matched + pref_matched
    return req_score + pref_score, missing, matched_display


def _experience_score(years: int, jd: ParsedJD) -> float:
    band = EXPERIENCE_BAND.get(jd.experience_level, (jd.min_years, jd.max_years))
    lo, hi = band[0], band[1]
    if lo <= years <= hi:
        return 20.0
    diff = min(abs(years - lo), abs(years - hi))
    return max(0.0, 20.0 - diff * 3)


def _llm_explanation(candidate: Candidate, jd: ParsedJD, skill_score: float, exp_score: float) -> str:
    prompt = f"""Candidate: {candidate.name}, {candidate.experience} years exp, skills: {', '.join(candidate.skills)}, past roles: {', '.join(candidate.past_roles)}.
JD Role: {jd.role}, required: {', '.join(jd.required_skills)}.
Skill score: {skill_score:.0f}/80, Experience score: {exp_score:.0f}/20.
Write a 2-sentence match explanation focusing on fit and gaps. Be specific."""

    return chat_complete(
        [{"role": "user", "content": prompt}],
        temperature=0.4,
    ).strip()


def compute_match(candidate: Candidate, jd: ParsedJD) -> MatchResult:
    skill_score, missing, matched = _skill_overlap(candidate.skills, jd.required_skills, jd.preferred_skills)
    exp_score = _experience_score(candidate.experience, jd)
    total = round(skill_score + exp_score, 1)

    explanation = _llm_explanation(candidate, jd, skill_score, exp_score)

    return MatchResult(
        candidate_id=candidate.id,
        match_score=min(total, 100.0),
        missing_skills=missing,
        matched_skills=matched,
        match_explanation=explanation,
    )
