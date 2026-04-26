"use client";

import { useState } from "react";
import { RankedCandidate } from "@/app/types";
import ScoreBar from "./ScoreBar";
import ChatModal from "./ChatModal";

interface CandidateCardProps {
  candidate: RankedCandidate;
  rank: number;
}

const intentColors: Record<string, string> = {
  High: "bg-emerald-500/20 text-emerald-400 border border-emerald-500/30",
  Medium: "bg-amber-500/20 text-amber-400 border border-amber-500/30",
  Low: "bg-red-500/20 text-red-400 border border-red-500/30",
};

export default function CandidateCard({ candidate: rc, rank }: CandidateCardProps) {
  const [expanded, setExpanded] = useState(false);
  const [chatOpen, setChatOpen] = useState(false);

  return (
    <>
      <div className="card flex flex-col gap-4 transition-all hover:border-sky-700/60">
        {/* Header */}
        <div className="flex items-start justify-between gap-3">
          <div className="flex items-center gap-3">
            <div className="flex h-10 w-10 flex-shrink-0 items-center justify-center rounded-full bg-sky-500/20 text-sm font-bold text-sky-400">
              #{rank}
            </div>
            <div>
              <h3 className="font-semibold text-white">{rc.candidate.name}</h3>
              <p className="text-xs text-slate-400">
                {rc.candidate.experience} yrs · {rc.candidate.location ?? "Remote"}
              </p>
            </div>
          </div>
          <div className="flex items-center gap-2">
            <span className={`tag ${intentColors[rc.intent]}`}>{rc.intent} Intent</span>
            <div className="rounded-xl bg-sky-500/10 px-3 py-1 text-center">
              <p className="text-xs text-slate-400">Final</p>
              <p className="text-lg font-bold text-sky-400">{rc.final_score.toFixed(0)}</p>
            </div>
          </div>
        </div>

        {/* Tags */}
        <div className="flex flex-wrap gap-2">
          {rc.tags.map((tag, i) => (
            <span key={i} className="tag bg-slate-700/60 text-slate-300 text-xs">
              {tag}
            </span>
          ))}
        </div>

        {/* Scores */}
        <div className="flex flex-col gap-3">
          <ScoreBar label="Match Score" value={rc.match_score} color="sky" />
          <ScoreBar label="Interest Score" value={rc.interest_score} color="emerald" />
          <ScoreBar label="Final Score" value={rc.final_score} color="violet" />
        </div>

        {/* Skills */}
        <div className="flex flex-wrap gap-1">
          {rc.matched_skills.slice(0, 6).map((s) => (
            <span key={s} className="tag bg-emerald-500/10 text-emerald-400 border border-emerald-500/20 text-xs">
              ✓ {s}
            </span>
          ))}
          {rc.missing_skills.slice(0, 3).map((s) => (
            <span key={s} className="tag bg-red-500/10 text-red-400 border border-red-500/20 text-xs">
              ✗ {s}
            </span>
          ))}
        </div>

        {/* Conversation summary */}
        <p className="text-sm text-slate-400 italic">"{rc.conversation_summary}"</p>

        {/* Action row */}
        <div className="flex gap-2">
          <button
            onClick={() => setExpanded(!expanded)}
            className="flex-1 rounded-xl border border-slate-600 py-2 text-sm text-slate-300 hover:bg-slate-700/50 transition"
          >
            {expanded ? "▲ Hide Details" : "▼ Why This Candidate?"}
          </button>
          <button
            onClick={() => setChatOpen(true)}
            className="flex-1 rounded-xl border border-sky-600/50 py-2 text-sm text-sky-400 hover:bg-sky-600/10 transition"
          >
            💬 View Chat
          </button>
        </div>

        {/* Expanded details */}
        {expanded && (
          <div className="flex flex-col gap-3 border-t border-slate-700 pt-4">
            <div>
              <p className="mb-1 text-xs font-semibold uppercase tracking-wider text-slate-500">
                Why This Candidate
              </p>
              <p className="text-sm text-slate-300">{rc.why_this_candidate}</p>
            </div>
            <div>
              <p className="mb-1 text-xs font-semibold uppercase tracking-wider text-slate-500">
                Match Explanation
              </p>
              <p className="text-sm text-slate-300">{rc.match_explanation}</p>
            </div>
            <div>
              <p className="mb-1 text-xs font-semibold uppercase tracking-wider text-slate-500">
                Past Roles
              </p>
              <ul className="list-inside list-disc text-sm text-slate-400">
                {rc.candidate.past_roles.map((r, i) => (
                  <li key={i}>{r}</li>
                ))}
              </ul>
            </div>
          </div>
        )}
      </div>

      {chatOpen && (
        <ChatModal
          name={rc.candidate.name}
          messages={rc.messages}
          onClose={() => setChatOpen(false)}
        />
      )}
    </>
  );
}
