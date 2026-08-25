"use client";

import { useEffect, useState } from "react";
import {
  MAX_FLUENCY_SCORE,
  MAX_PACE_SCORE,
  MAX_PRONUNCIATION_SCORE,
  MAX_TOTAL_SCORE,
} from "@/lib/scoring";

/**
 * The score as twenty marks in one line, grouped by skill.
 *
 * The bands are named underneath, so giving each its own hue would add colour
 * without adding information. One accent keeps the result page quiet enough to
 * read the marked passage above it.
 */
export function ScoreLine({
  pronunciation,
  fluency,
  pace,
  total,
}: {
  pronunciation: number;
  fluency: number;
  pace: number;
  total: number;
}) {
  const bands = [
    { label: "Sounds", earned: pronunciation, max: MAX_PRONUNCIATION_SCORE },
    { label: "Flow", earned: fluency, max: MAX_FLUENCY_SCORE },
    { label: "Pace", earned: pace, max: MAX_PACE_SCORE },
  ];

  const perfect = total >= MAX_TOTAL_SCORE;
  // Full marks close the gaps: twenty separate marks become one line.
  const [joined, setJoined] = useState(false);
  useEffect(() => {
    if (!perfect) return;
    const timer = setTimeout(() => setJoined(true), MAX_TOTAL_SCORE * 22 + 260);
    return () => clearTimeout(timer);
  }, [perfect]);

  let tickIndex = 0;

  return (
    <div>
      <p className="flex items-baseline gap-1.5">
        <span className="text-4xl font-semibold tabular-nums">{total}</span>
        <span className="text-base text-muted">/ {MAX_TOTAL_SCORE}</span>
      </p>

      <div className="shiro-score mt-4" data-joined={joined} aria-hidden>
        {bands.flatMap((band, bandIndex) =>
          Array.from({ length: band.max }, (_, i) => (
            <span
              key={`${band.label}-${i}`}
              className="shiro-tick"
              data-earned={i < band.earned}
              data-group-end={i === band.max - 1 && bandIndex < bands.length - 1}
              style={{ animationDelay: `${tickIndex++ * 22}ms` }}
            />
          ))
        )}
      </div>

      <div className="mt-2.5 flex justify-between gap-2">
        {bands.map((b) => (
          <p key={b.label} className="text-xs text-muted">
            <span className="font-semibold text-foreground tabular-nums">
              {b.earned}
            </span>
            /{b.max} {b.label}
          </p>
        ))}
      </div>

      {perfect && (
        <p className="mt-4 text-sm font-medium text-good">
          शाबाश! Every word, start to finish.
        </p>
      )}
    </div>
  );
}
