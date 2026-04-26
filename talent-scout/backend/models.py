from pydantic import BaseModel
from typing import List, Optional


class ParsedJD(BaseModel):
    role: str
    required_skills: List[str]
    preferred_skills: List[str]
    experience_level: str
    min_years: int
    max_years: int


class Candidate(BaseModel):
    id: str
    name: str
    skills: List[str]
    experience: int
    past_roles: List[str]
    open_to_work: bool
    responsiveness: str  # high / medium / low
    location: Optional[str] = None
    summary: Optional[str] = None


class MatchResult(BaseModel):
    candidate_id: str
    match_score: float
    missing_skills: List[str]
    matched_skills: List[str]
    match_explanation: str


class ConversationMessage(BaseModel):
    role: str  # recruiter / candidate
    content: str


class ConversationResult(BaseModel):
    candidate_id: str
    messages: List[ConversationMessage]
    intent: str  # High / Medium / Low
    interest_score: float
    conversation_summary: str


class RankedCandidate(BaseModel):
    candidate: Candidate
    match_score: float
    interest_score: float
    final_score: float
    matched_skills: List[str]
    missing_skills: List[str]
    match_explanation: str
    intent: str
    conversation_summary: str
    messages: List[ConversationMessage]
    tags: List[str]
    why_this_candidate: str


class RunAgentRequest(BaseModel):
    jd_text: str
    top_k: int = 10


class RunAgentResponse(BaseModel):
    parsed_jd: ParsedJD
    ranked_candidates: List[RankedCandidate]
    total_candidates_evaluated: int
