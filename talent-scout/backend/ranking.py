from models import Candidate, MatchResult, ConversationResult, RankedCandidate
from llm_client import chat_complete

MATCH_WEIGHT = 0.7
INTEREST_WEIGHT = 0.3


def _compute_tags(match: MatchResult, conv: ConversationResult, candidate: Candidate) -> list[str]:
    tags = []
    if conv.intent == "High":
        tags.append("🔥 High Intent")
    elif conv.intent == "Low":
        tags.append("❄️ Low Interest")
    if match.missing_skills:
        tags.append("⚠️ Skill Gap")
    if match.match_score >= 80:
        tags.append("✅ Strong Match")
    if candidate.open_to_work:
        tags.append("🟢 Open to Work")
    if candidate.responsiveness == "high":
        tags.append("⚡ Fast Responder")
    return tags


def _why_candidate(candidate: Candidate, match: MatchResult, conv: ConversationResult) -> str:
    prompt = f"""Write a compelling 2-sentence "Why This Candidate?" summary for a recruiter.
Candidate: {candidate.name}, {candidate.experience} years, skills: {', '.join(candidate.skills[:5])}.
Match score: {match.match_score}/100. Interest level: {conv.intent}. Matched skills: {', '.join(match.matched_skills[:4])}.
Be specific and persuasive."""
    return chat_complete([{"role": "user", "content": prompt}], temperature=0.5).strip()


def rank_candidates(
    candidates: list[Candidate],
    match_results: dict[str, MatchResult],
    conv_results: dict[str, ConversationResult],
    top_k: int = 10,
) -> list[RankedCandidate]:
    ranked = []
    for c in candidates:
        match = match_results.get(c.id)
        conv = conv_results.get(c.id)
        if not match or not conv:
            continue

        final = round(MATCH_WEIGHT * match.match_score + INTEREST_WEIGHT * conv.interest_score, 1)
        tags = _compute_tags(match, conv, c)
        why = _why_candidate(c, match, conv)

        ranked.append(RankedCandidate(
            candidate=c,
            match_score=match.match_score,
            interest_score=conv.interest_score,
            final_score=final,
            matched_skills=match.matched_skills,
            missing_skills=match.missing_skills,
            match_explanation=match.match_explanation,
            intent=conv.intent,
            conversation_summary=conv.conversation_summary,
            messages=conv.messages,
            tags=tags,
            why_this_candidate=why,
        ))

    ranked.sort(key=lambda x: x.final_score, reverse=True)
    return ranked[:top_k]
