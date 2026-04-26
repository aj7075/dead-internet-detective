"use client";

import { useState } from "react";
import JDInput from "@/components/JDInput";
import CandidateCard from "@/components/CandidateCard";
import { RunAgentResponse, ParsedJD } from "./types";

const API = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

export default function Home() {
  const [jdText, setJdText] = useState("");
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<RunAgentResponse | null>(null);
  const [error, setError] = useState<string | null>(null);

  async function runAgent() {
    if (!jdText.trim()) return;
    setLoading(true);
    setError(null);
    setResult(null);
    try {
      const res = await fetch(`${API}/run-agent`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ jd_text: jdText, top_k: 10 }),
      });
      if (!res.ok) {
        const detail = await res.text();
        throw new Error(detail || `Server error ${res.status}`);
      }
      const data: RunAgentResponse = await res.json();
      setResult(data);
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : "Unknown error");
    } finally {
      setLoading(false);
    }
  }

  return (
    <main className="mx-auto max-w-5xl px-4 py-10">
      {/* Header */}
      <div className="mb-10 text-center">
        <h1 className="mb-2 text-4xl font-bold tracking-tight text-white">
          🎯 Talent Scout <span className="text-sky-400">AI</span>
        </h1>
        <p className="text-slate-400">AI agent that reads a JD, finds matching candidates, simulates recruiter conversations, and ranks the best fits.</p>
      </div>

      <div className="grid gap-8 lg:grid-cols-5">
        {/* Left: JD input */}
        <div className="lg:col-span-2">
          <JDInput value={jdText} onChange={setJdText} onRun={runAgent} loading={loading} />

          {result && (
            <div className="card mt-4">
              <p className="mb-2 text-xs font-semibold uppercase tracking-wider text-slate-500">Parsed JD</p>
              <ParsedJDPanel jd={result.parsed_jd} />
              <p className="mt-3 text-xs text-slate-500">
                Evaluated {result.total_candidates_evaluated} candidates · Showing top {result.ranked_candidates.length}
              </p>
            </div>
          )}
        </div>

        {/* Right: Results */}
        <div className="lg:col-span-3">
          {error && (
            <div className="card border-red-500/30 bg-red-500/10 text-red-400">
              <p className="font-semibold">Error</p>
              <p className="text-sm">{error}</p>
            </div>
          )}

          {loading && (
            <div className="card flex flex-col items-center gap-4 py-12 text-center">
              <div className="flex gap-2">
                {["Parsing JD…", "Matching candidates…", "Simulating conversations…", "Ranking results…"].map((step, i) => (
                  <span key={i} className="rounded-full bg-sky-500/10 px-3 py-1 text-xs text-sky-400 animate-pulse" style={{ animationDelay: `${i * 0.3}s` }}>
                    {step}
                  </span>
                ))}
              </div>
              <p className="text-sm text-slate-400">Agent is working… this takes ~30–60s with a live LLM.</p>
            </div>
          )}

          {result && !loading && (
            <div className="flex flex-col gap-4">
              <h2 className="text-lg font-semibold text-white">
                Top {result.ranked_candidates.length} Candidates
                <span className="ml-2 text-sm font-normal text-slate-400">for {result.parsed_jd.role}</span>
              </h2>
              {result.ranked_candidates.map((rc, i) => (
                <CandidateCard key={rc.candidate.id} candidate={rc} rank={i + 1} />
              ))}
            </div>
          )}

          {!loading && !result && !error && (
            <div className="card flex flex-col items-center gap-3 py-16 text-center text-slate-500">
              <p className="text-4xl">🤖</p>
              <p className="text-lg font-medium">Ready to scout</p>
              <p className="text-sm">Load the demo JD or paste your own, then click Run Agent.</p>
            </div>
          )}
        </div>
      </div>
    </main>
  );
}

function ParsedJDPanel({ jd }: { jd: ParsedJD }) {
  return (
    <div className="flex flex-col gap-2 text-sm">
      <div className="flex justify-between">
        <span className="text-slate-400">Role</span>
        <span className="font-medium text-white">{jd.role}</span>
      </div>
      <div className="flex justify-between">
        <span className="text-slate-400">Level</span>
        <span className="font-medium text-white">{jd.experience_level} ({jd.min_years}–{jd.max_years} yrs)</span>
      </div>
      <div>
        <p className="mb-1 text-slate-400">Required Skills</p>
        <div className="flex flex-wrap gap-1">
          {jd.required_skills.map((s) => (
            <span key={s} className="tag bg-sky-500/10 text-sky-400 text-xs">{s}</span>
          ))}
        </div>
      </div>
      {jd.preferred_skills.length > 0 && (
        <div>
          <p className="mb-1 text-slate-400">Preferred Skills</p>
          <div className="flex flex-wrap gap-1">
            {jd.preferred_skills.map((s) => (
              <span key={s} className="tag bg-violet-500/10 text-violet-400 text-xs">{s}</span>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
