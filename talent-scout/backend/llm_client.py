import os
import json
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

_client: OpenAI | None = None


def get_client() -> OpenAI:
    global _client
    if _client is None:
        api_key = os.getenv("OPENAI_API_KEY", "sk-mock-key")
        base_url = os.getenv("OPENAI_BASE_URL", None)
        _client = OpenAI(api_key=api_key, base_url=base_url)
    return _client


def chat_complete(
    messages: list[dict],
    model: str | None = None,
    temperature: float = 0.7,
    response_format: str = "text",
) -> str:
    model = model or os.getenv("LLM_MODEL", "gpt-4o-mini")
    use_mock = os.getenv("USE_MOCK_LLM", "false").lower() == "true"

    if use_mock:
        return _mock_response(messages)

    kwargs: dict = {"model": model, "messages": messages, "temperature": temperature}
    if response_format == "json":
        kwargs["response_format"] = {"type": "json_object"}

    resp = get_client().chat.completions.create(**kwargs)
    return resp.choices[0].message.content or ""


def _mock_response(messages: list[dict]) -> str:
    last = messages[-1]["content"] if messages else ""
    if "parse" in last.lower() or "job description" in last.lower():
        return json.dumps({
            "role": "Senior Software Engineer",
            "required_skills": ["Python", "FastAPI", "PostgreSQL", "Docker"],
            "preferred_skills": ["Kubernetes", "Redis", "AWS"],
            "experience_level": "Senior",
            "min_years": 4,
            "max_years": 8,
        })
    if "recruiter" in last.lower() or "outreach" in last.lower():
        return json.dumps({
            "reply": "Thanks for reaching out! This role sounds interesting. I'm currently open to new opportunities.",
            "intent_signal": "positive",
        })
    return "Mock LLM response."
