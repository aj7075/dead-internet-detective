export interface ParsedJD {
  role: string;
  required_skills: string[];
  preferred_skills: string[];
  experience_level: string;
  min_years: number;
  max_years: number;
}

export interface ConversationMessage {
  role: "recruiter" | "candidate";
  content: string;
}

export interface RankedCandidate {
  candidate: {
    id: string;
    name: string;
    skills: string[];
    experience: number;
    past_roles: string[];
    open_to_work: boolean;
    responsiveness: string;
    location?: string;
    summary?: string;
  };
  match_score: number;
  interest_score: number;
  final_score: number;
  matched_skills: string[];
  missing_skills: string[];
  match_explanation: string;
  intent: "High" | "Medium" | "Low";
  conversation_summary: string;
  messages: ConversationMessage[];
  tags: string[];
  why_this_candidate: string;
}

export interface RunAgentResponse {
  parsed_jd: ParsedJD;
  ranked_candidates: RankedCandidate[];
  total_candidates_evaluated: number;
}
