import json
import os
from pathlib import Path
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from models import RunAgentRequest, RunAgentResponse, Candidate, ParsedJD
from jd_parser import parse_jd
from matching_engine import compute_match
from conversation_simulator import simulate_conversation
from ranking import rank_candidates

app = FastAPI(title="Talent Scout AI", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

DATA_PATH = Path(__file__).parent / "data" / "candidates.json"


def load_candidates() -> list[Candidate]:
    with open(DATA_PATH) as f:
        raw = json.load(f)
    return [Candidate(**c) for c in raw]


@app.get("/")
def root():
    return {"status": "ok", "service": "Talent Scout AI"}


@app.get("/candidates")
def get_candidates():
    return load_candidates()


@app.post("/parse-jd")
def parse_jd_endpoint(body: dict):
    jd_text = body.get("jd_text", "")
    if not jd_text.strip():
        raise HTTPException(400, "jd_text is required")
    return parse_jd(jd_text)


@app.post("/run-agent", response_model=RunAgentResponse)
def run_agent(req: RunAgentRequest):
    if not req.jd_text.strip():
        raise HTTPException(400, "jd_text is required")

    candidates = load_candidates()
    parsed_jd = parse_jd(req.jd_text)

    match_results = {}
    for c in candidates:
        match_results[c.id] = compute_match(c, parsed_jd)

    top_candidates_by_match = sorted(
        candidates,
        key=lambda c: match_results[c.id].match_score,
        reverse=True,
    )[:15]

    conv_results = {}
    for c in top_candidates_by_match:
        conv_results[c.id] = simulate_conversation(c, parsed_jd)

    ranked = rank_candidates(top_candidates_by_match, match_results, conv_results, top_k=req.top_k)

    return RunAgentResponse(
        parsed_jd=parsed_jd,
        ranked_candidates=ranked,
        total_candidates_evaluated=len(candidates),
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
