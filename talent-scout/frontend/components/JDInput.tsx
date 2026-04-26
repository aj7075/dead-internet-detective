"use client";

const DEMO_JD = `Senior Backend Engineer – FinTech Payments Platform

We are looking for a Senior Backend Engineer to join our core payments team. You will design and maintain high-throughput payment APIs that process millions of transactions daily.

Requirements:
- 4–8 years of backend engineering experience
- Strong Python skills (FastAPI or Django)
- PostgreSQL and Redis expertise
- Experience with Docker and AWS
- Familiarity with distributed systems

Nice to have:
- Kubernetes and Terraform
- Kafka or event-driven architecture
- Prior fintech or payments domain experience

Location: Remote-friendly. We move fast and ship weekly.`;

interface JDInputProps {
  value: string;
  onChange: (v: string) => void;
  onRun: () => void;
  loading: boolean;
}

export default function JDInput({ value, onChange, onRun, loading }: JDInputProps) {
  return (
    <div className="card flex flex-col gap-4">
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-lg font-semibold text-white">Job Description</h2>
          <p className="text-sm text-slate-400">Paste your JD and let the AI agent find the best candidates.</p>
        </div>
        <button
          onClick={() => onChange(DEMO_JD)}
          className="rounded-lg border border-slate-600 px-3 py-1.5 text-xs text-slate-300 hover:bg-slate-700 transition"
        >
          Load Demo JD
        </button>
      </div>

      <textarea
        className="h-56 w-full resize-none rounded-xl bg-[#0f1117] border border-slate-700 p-4 text-sm text-slate-200 placeholder-slate-600 focus:border-sky-500 focus:outline-none focus:ring-1 focus:ring-sky-500"
        placeholder="Paste job description here…"
        value={value}
        onChange={(e) => onChange(e.target.value)}
      />

      <button
        className="btn-primary flex items-center justify-center gap-2"
        onClick={onRun}
        disabled={loading || !value.trim()}
      >
        {loading ? (
          <>
            <svg className="h-4 w-4 animate-spin" fill="none" viewBox="0 0 24 24">
              <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
              <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8v8z" />
            </svg>
            Running Agent…
          </>
        ) : (
          "🚀 Run Agent"
        )}
      </button>
    </div>
  );
}
