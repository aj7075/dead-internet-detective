import json
from models import ParsedJD
from llm_client import chat_complete

SYSTEM_PROMPT = """You are an expert recruiter AI. Parse the job description and return ONLY valid JSON with this exact schema:
{
  "role": "string",
  "required_skills": ["skill1", "skill2"],
  "preferred_skills": ["skill1"],
  "experience_level": "Junior|Mid|Senior|Lead|Principal",
  "min_years": 0,
  "max_years": 10
}
Be precise. Extract only what is stated. No extra keys."""


def parse_jd(jd_text: str) -> ParsedJD:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Parse this job description:\n\n{jd_text}"},
    ]
    raw = chat_complete(messages, temperature=0.1, response_format="json")

    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        import re
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        data = json.loads(match.group()) if match else {}

    data.setdefault("role", "Software Engineer")
    data.setdefault("required_skills", [])
    data.setdefault("preferred_skills", [])
    data.setdefault("experience_level", "Mid")
    data.setdefault("min_years", 2)
    data.setdefault("max_years", 6)

    return ParsedJD(**data)
