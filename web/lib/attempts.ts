"use client";

import { supabase } from "./supabase";
import { ensureAnonymousSession } from "./session";
import type { ScoreResult } from "./scoring";

export type Passage = { id: string | null; text_hi: string; level?: string | null };

export type AttemptSummary = {
  passage_id: string | null;
  total_score: number | null;
  coverage_percent: number | null;
  created_at: string;
};

/**
 * Save a scored reading.
 *
 * Best-effort: a child should never lose their result screen because the
 * database was unreachable, so failures are swallowed and reported to the
 * console rather than surfaced. Requires a passage row — the built-in
 * fallback passage has no id and is therefore never recorded.
 */
export async function saveAttempt(
  passage: Passage,
  result: ScoreResult
): Promise<void> {
  if (!supabase || !passage.id) return;

  const userId = await ensureAnonymousSession();
  if (!userId) return;

  const { error } = await supabase.from("attempts").insert({
    user_id: userId,
    passage_id: passage.id,
    total_score: result.total_score,
    pronunciation_score: result.pronunciation_score,
    fluency_score: result.fluency_score,
    pace_score: result.pace_score,
    pre_coverage_total: result.pre_coverage_total,
    coverage_percent: result.coverage_percent,
    words_per_minute: result.words_per_minute,
    transcript: result.transcript,
    word_analysis: result.word_analysis,
  });

  if (error) console.error("Could not save this attempt:", error.message);
}

/** Most recent attempts for the signed-in reader, newest first. */
export async function recentAttempts(limit = 10): Promise<AttemptSummary[]> {
  if (!supabase) return [];
  const userId = await ensureAnonymousSession();
  if (!userId) return [];

  const { data } = await supabase
    .from("attempts")
    .select("passage_id, total_score, coverage_percent, created_at")
    .eq("user_id", userId)
    .order("created_at", { ascending: false })
    .limit(limit);

  return (data as AttemptSummary[]) ?? [];
}

/**
 * Choose what to read next.
 *
 * Unread passages come first, in difficulty order, so a reader moves forward
 * instead of meeting the same text every day. Once everything has been read
 * it returns the one left longest — revision, oldest first.
 */
export async function pickPassage(): Promise<Passage | null> {
  if (!supabase) return null;

  const { data: passages } = await supabase
    .from("passages")
    .select("id, text_hi, level, difficulty")
    .order("difficulty", { ascending: true });

  if (!passages || passages.length === 0) return null;

  const userId = await ensureAnonymousSession();
  if (!userId) return passages[0] as Passage;

  const { data: attempts } = await supabase
    .from("attempts")
    .select("passage_id, created_at")
    .eq("user_id", userId)
    .order("created_at", { ascending: false });

  // Newest first, so the first time a passage id appears is its latest read.
  const lastRead = new Map<string, string>();
  for (const a of attempts ?? []) {
    const row = a as { passage_id: string; created_at: string };
    if (!lastRead.has(row.passage_id)) lastRead.set(row.passage_id, row.created_at);
  }

  const unread = passages.find((p) => !lastRead.has((p as Passage).id as string));
  if (unread) return unread as Passage;

  const stalest = [...passages].sort(
    (a, b) =>
      (lastRead.get((a as Passage).id as string) ?? "").localeCompare(
        lastRead.get((b as Passage).id as string) ?? ""
      )
  )[0];
  return stalest as Passage;
}
