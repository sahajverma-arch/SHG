"use client";

import { useStreak } from "@/lib/streak";

export function StreakBadge() {
  const { streak, practicedToday } = useStreak();

  return (
    <div
      className={`icon-badge gap-1.5 border-2 px-3 py-1.5 text-sm font-bold ${
        practicedToday
          ? "border-brand-orange bg-brand-orange/15 text-brand-orange"
          : "border-border bg-surface text-foreground/40"
      }`}
      title={
        practicedToday
          ? "Practiced today — streak safe!"
          : "Do an exercise today to keep your streak"
      }
    >
      <span aria-hidden>🔥</span>
      <span className="tabular-nums">{streak}</span>
    </div>
  );
}
