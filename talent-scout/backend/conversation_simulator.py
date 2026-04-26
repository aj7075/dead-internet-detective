import json
import re
from models import Candidate, ParsedJD, ConversationMessage, ConversationResult
from llm_client import chat_complete

RECRUITER_OPENER = """Hi {name}! I came across your profile and think you'd be a great fit for a {role} position at our company. The role focuses on {skills}. Would you be open to a quick chat?"""

SYSTEM_CANDIDATE = """You are {name}, a {experience}-year experienced {last_role}. Your skills: {skills}.
You are {responsiveness} to recruiters. Respond realistically and briefly (1-3 sentences).
At the end of each reply include a JSON tag like: <intent>positive|neutral|negative</intent>"""

SYSTEM_RECRUITER = """You are an expert technical recruiter. Keep messages concise and professional. Ask one question at a time. Focus on: experience fit, timeline, compensation expectations."""


def _extract_intent(text: str) -> tuple[str, str]:
    match = re.search(r"<intent>(positive|neutral|negative)</intent>", text, re.IGNORECASE)
    intent = match.group(1) if match else "neutral"
    clean = re.sub(r"<intent>.*?</intent>", "", text).strip()
    return clean, intent


def _intent_to_score(intents: list[str]) -> tuple[str, float]:
    counts = {"positive": intents.count("positive"), "neutral": intents.count("neutral"), "negative": intents.count("negative")}
    pos_ratio = counts["positive"] / max(len(intents), 1)

    if pos_ratio >= 0.6:
        label, score = "High", 75 + pos_ratio * 25
    elif pos_ratio >= 0.3:
        label, score = "Medium", 40 + pos_ratio * 40
    else:
        label, score = "Low", 10 + pos_ratio * 30

    return label, round(min(score, 100), 1)


def simulate_conversation(candidate: Candidate, jd: ParsedJD) -> ConversationResult:
    messages: list[ConversationMessage] = []
    intents: list[str] = []

    last_role = candidate.past_roles[-1] if candidate.past_roles else "professional"
    skills_str = ", ".join(candidate.skills[:4])
    jd_skills_str = ", ".join(jd.required_skills[:3])

    opener = RECRUITER_OPENER.format(name=candidate.name, role=jd.role, skills=jd_skills_str)
    messages.append(ConversationMessage(role="recruiter", content=opener))

    history = [{"role": "user", "content": opener}]
    cand_system = SYSTEM_CANDIDATE.format(
        name=candidate.name,
        experience=candidate.experience,
        last_role=last_role,
        skills=skills_str,
        responsiveness=candidate.responsiveness,
    )

    recruiter_history = [
        {"role": "system", "content": SYSTEM_RECRUITER},
        {"role": "assistant", "content": opener},
    ]

    for turn in range(2):
        cand_messages = [{"role": "system", "content": cand_system}] + history
        cand_reply_raw = chat_complete(cand_messages, temperature=0.75)
        cand_reply, intent = _extract_intent(cand_reply_raw)
        intents.append(intent)
        messages.append(ConversationMessage(role="candidate", content=cand_reply))

        history.append({"role": "assistant", "content": cand_reply_raw})
        recruiter_history.append({"role": "user", "content": cand_reply})

        rec_reply = chat_complete(recruiter_history, temperature=0.5)
        messages.append(ConversationMessage(role="recruiter", content=rec_reply))

        history.append({"role": "user", "content": rec_reply})
        recruiter_history.append({"role": "assistant", "content": rec_reply})

    cand_final = [{"role": "system", "content": cand_system}] + history
    final_raw = chat_complete(cand_final, temperature=0.75)
    final_clean, final_intent = _extract_intent(final_raw)
    intents.append(final_intent)
    messages.append(ConversationMessage(role="candidate", content=final_clean))

    intent_label, interest_score = _intent_to_score(intents)

    summary_prompt = f"""Summarize this recruiter-candidate conversation in 1 sentence. Focus on candidate's interest level and key points raised.\n\nConversation:\n{chr(10).join(f"{m.role}: {m.content}" for m in messages)}"""
    summary = chat_complete([{"role": "user", "content": summary_prompt}], temperature=0.3).strip()

    return ConversationResult(
        candidate_id=candidate.id,
        messages=messages,
        intent=intent_label,
        interest_score=interest_score,
        conversation_summary=summary,
    )
