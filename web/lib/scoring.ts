export type WordStatus = "correct" | "mispronounced" | "skipped";

export type WordDiffEntry = {
  word: string;
  status: WordStatus;
  /** What the recogniser heard instead — only set for `mispronounced`. */
  heard: string | null;
};

export type FluencySignals = {
  disfluency_count: number;
  seconds_per_correct_word: number;
  transcript_bloat_ratio: number;
  duration_slack: number;
};

export type ScoreResult = {
  transcript: string;
  audio_seconds: number;
  inference_seconds: number;

  /** Out of 20 — pronunciation (8) + fluency (6) + pace (6), scaled by coverage. */
  total_score: number;
  pronunciation_score: number;
  fluency_score: number;
  pace_score: number;

  /** The same three before coverage scaling, for "here's what you lost" copy. */
  pre_coverage_pronunciation_score: number;
  pre_coverage_fluency_score: number;
  pre_coverage_pace_score: number;
  pre_coverage_total: number;

  coverage_percent: number;
  coverage_skill_factor_percent: number;
  alignment_quality: number;
  assessment_confidence: "high" | "medium" | "low";
  assessment_confidence_message: string;

  word_analysis: WordDiffEntry[];
  mispronounced_words: string[];
  skipped_words: string[];
  practice_words: string[];
  vocab_feedback_words: string[];

  words_per_minute: number;
  target_wpm: number;
  level: string;
  fluency_signals: FluencySignals;
};

export type VocabularyWord = { word: string; meaning_english?: string };

export const MAX_TOTAL_SCORE = 20;
export const MAX_PRONUNCIATION_SCORE = 8;
export const MAX_FLUENCY_SCORE = 6;
export const MAX_PACE_SCORE = 6;

export async function scoreAttempt(
  audioBlob: Blob,
  expectedText: string,
  options: { level?: string; vocabulary?: VocabularyWord[] } = {}
): Promise<ScoreResult> {
  const base =
    process.env.NEXT_PUBLIC_SCORING_SERVICE_URL ?? "http://localhost:8000";

  const form = new FormData();
  form.append("audio", audioBlob, "attempt.webm");
  form.append("expected_text", expectedText);
  if (options.level) form.append("level", options.level);
  if (options.vocabulary?.length) {
    form.append("reading_vocabulary", JSON.stringify(options.vocabulary));
  }

  const res = await fetch(`${base}/score`, { method: "POST", body: form });
  if (!res.ok) {
    throw new Error(`Scoring service returned ${res.status}`);
  }
  return res.json();
}
