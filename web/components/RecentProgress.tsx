"use client";

import { useEffect, useState } from "react";
import { recentAttempts, type AttemptSummary } from "@/lib/attempts";
import { MAX_TOTAL_SCORE } from "@/lib/scoring";

/**
 * The last few readings, newest last so the line reads left to right like a
 * timeline. Hidden entirely until there are at least two — a single bar is a
 * chart of nothing.
 */
export function RecentProgress() {
  const [attempts, setAttempts] = useState<AttemptSummary[] | null>(null);

  useEffect(() => {
    void recentAttempts(8).then(setAttempts);
  }, []);

  if (!attempts || attempts.length < 2) return null;

  const inOrder = [...attempts].reverse();
  const latest = inOrder[inOrder.length - 1];
  const previous = inOrder[inOrder.length - 2];
  const change = (latest.total_score ?? 0) - (previous.total_score ?? 0);

  return (
    <div className="card p-4">
      <div className="flex items-baseline justify-between gap-3">
        <p className="eyebrow">Last {inOrder.length} readings</p>
        <p className="text-xs text-muted">
          {change > 0 ? `+${change} on last time` : change < 0 ? `${change} on last time` : "Same as last time"}
        </p>
      </div>

      <div className="mt-3 flex h-12 items-end gap-1.5">
        {inOrder.map((a, i) => {
          const score = a.total_score ?? 0;
          const height = Math.max(6, (score / MAX_TOTAL_SCORE) * 100);
          const isLatest = i === inOrder.length - 1;
          return (
            <div
              key={`${a.created_at}-${i}`}
              className={`flex-1 rounded-sm ${isLatest ? "bg-accent" : "bg-border"}`}
              style={{ height: `${height}%` }}
              title={`${score} / ${MAX_TOTAL_SCORE}`}
            />
          );
        })}
      </div>

      <p className="mt-2 text-sm">
        <span className="font-semibold tabular-nums">
          {latest.total_score ?? 0}
        </span>
        <span className="text-muted"> / {MAX_TOTAL_SCORE} most recent</span>
      </p>
    </div>
  );
}
