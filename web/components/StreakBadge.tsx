"use client";

import { useStreak } from "@/lib/streak";

export function StreakBadge() {
  const { streak, practicedToday } = useStreak();

  return (
    <span
      className={`inline-flex items-center gap-1 rounded-lg border px-2 py-1 text-sm font-semibold tabular-nums ${
        practicedToday
          ? "border-border text-foreground"
          : "border-border text-muted"
      }`}
      title={
        practicedToday
          ? "Practised today"
          : "Do an exercise today to keep your streak"
      }
    >
      <span aria-hidden>{practicedToday ? "●" : "○"}</span>
      {streak}
      <span className="sr-only">day streak</span>
    </span>
  );
}
