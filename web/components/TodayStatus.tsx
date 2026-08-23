"use client";

import { useStreak } from "@/lib/streak";

/**
 * Today's state, told honestly: the streak records only *whether* a child
 * practised today, not which exercises, so this never claims a count it
 * cannot know.
 */
export function TodayStatus() {
  const { streak, practicedToday } = useStreak();

  return (
    <div className="card flex items-baseline justify-between gap-3 px-4 py-3">
      <p className="text-sm text-muted">
        {practicedToday
          ? "Practised today — streak safe."
          : "Finish one exercise to keep your streak."}
      </p>
      <p className="shrink-0 text-sm">
        <span className="font-semibold tabular-nums">{streak}</span>
        <span className="text-muted"> day{streak === 1 ? "" : "s"}</span>
      </p>
    </div>
  );
}
