"use client";

interface ScoreBarProps {
  label: string;
  value: number;
  color?: string;
  max?: number;
}

export default function ScoreBar({ label, value, color = "sky", max = 100 }: ScoreBarProps) {
  const pct = Math.min((value / max) * 100, 100);
  const colorMap: Record<string, string> = {
    sky: "bg-sky-500",
    emerald: "bg-emerald-500",
    violet: "bg-violet-500",
    amber: "bg-amber-500",
  };
  const bar = colorMap[color] ?? "bg-sky-500";

  return (
    <div className="w-full">
      <div className="mb-1 flex justify-between text-xs text-slate-400">
        <span>{label}</span>
        <span className="font-semibold text-white">{value.toFixed(0)}</span>
      </div>
      <div className="h-2 w-full overflow-hidden rounded-full bg-slate-700">
        <div
          className={`h-full rounded-full transition-all duration-700 ${bar}`}
          style={{ width: `${pct}%` }}
        />
      </div>
    </div>
  );
}
