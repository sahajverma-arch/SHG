export type WordDiffEntry = {
  word: string;
  status: "match" | "substitution" | "missing" | "extra";
};

export type ScoreResult = {
  transcript: string;
  audio_seconds: number;
  inference_seconds: number;
  accuracy_score: number;
  fluency_score: number;
  words_per_minute: number;
  word_diff: WordDiffEntry[];
};

export async function scoreAttempt(
  audioBlob: Blob,
  expectedText: string
): Promise<ScoreResult> {
  const base =
    process.env.NEXT_PUBLIC_SCORING_SERVICE_URL ?? "http://localhost:8000";

  const form = new FormData();
  form.append("audio", audioBlob, "attempt.webm");
  form.append("expected_text", expectedText);

  const res = await fetch(`${base}/score`, { method: "POST", body: form });
  if (!res.ok) {
    throw new Error(`Scoring service returned ${res.status}`);
  }
  return res.json();
}
