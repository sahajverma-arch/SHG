"use client";

import { useEffect, useRef, useState } from "react";
import { Celebrate } from "@/components/Celebrate";
import { scoreAttempt, type ScoreResult } from "@/lib/scoring";
import { supabase } from "@/lib/supabase";
import { useStreak } from "@/lib/streak";

const FALLBACK_PASSAGE = "सूरज पूरब से निकलता है और पश्चिम में डूब जाता है।";

type Passage = { id: string | null; text_hi: string };
type Status = "idle" | "recording" | "scoring" | "done" | "error";

export default function ReadingAloudPage() {
  const { complete } = useStreak();
  const [passage, setPassage] = useState<Passage>({
    id: null,
    text_hi: FALLBACK_PASSAGE,
  });
  const [status, setStatus] = useState<Status>("idle");
  const [result, setResult] = useState<ScoreResult | null>(null);
  const [error, setError] = useState<string | null>(null);

  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const chunksRef = useRef<Blob[]>([]);

  useEffect(() => {
    if (!supabase) return;
    supabase
      .from("passages")
      .select("id, text_hi")
      .limit(1)
      .then(({ data, error: dbError }) => {
        if (!dbError && data && data.length > 0) {
          setPassage(data[0] as Passage);
        }
      });
  }, []);

  async function startRecording() {
    setError(null);
    setResult(null);

    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    const recorder = new MediaRecorder(stream);
    chunksRef.current = [];

    recorder.ondataavailable = (e) => {
      if (e.data.size > 0) chunksRef.current.push(e.data);
    };

    recorder.onstop = async () => {
      stream.getTracks().forEach((t) => t.stop());
      const blob = new Blob(chunksRef.current, { type: "audio/webm" });
      setStatus("scoring");
      try {
        const scored = await scoreAttempt(blob, passage.text_hi);
        setResult(scored);
        setStatus("done");
        complete();
      } catch (err) {
        setError(
          err instanceof Error
            ? err.message
            : "Scoring failed — is the scoring service running on :8000?"
        );
        setStatus("error");
      }
    };

    mediaRecorderRef.current = recorder;
    recorder.start();
    setStatus("recording");
  }

  function stopRecording() {
    mediaRecorderRef.current?.stop();
  }

  function tryAgain() {
    setResult(null);
    setStatus("idle");
    setError(null);
  }

  const recording = status === "recording";

  return (
    <main className="mx-auto max-w-2xl px-6 py-16">
      <h1 className="font-[family-name:var(--font-display)] text-2xl text-brand-pink">
        Reading Aloud
      </h1>
      <p className="mt-2 text-sm text-foreground/60">
        Read the line below, then record yourself.
      </p>

      <p className="mt-8 rounded-3xl border-2 border-brand-pink bg-brand-pink/10 p-8 text-center text-2xl font-bold leading-relaxed text-foreground">
        {passage.text_hi}
      </p>

      {status !== "done" && (
        <div className="mt-6 flex justify-center">
          {!recording ? (
            <button
              onClick={startRecording}
              disabled={status === "scoring"}
              className="rounded-full bg-brand-pink px-6 py-3 text-base font-bold text-white disabled:opacity-50"
            >
              {status === "scoring" ? "Scoring…" : "🎙️ Record"}
            </button>
          ) : (
            <button
              onClick={stopRecording}
              className="rounded-full bg-brand-orange px-6 py-3 text-base font-bold text-white"
            >
              ⏹ Stop
            </button>
          )}
        </div>
      )}

      {error && (
        <p className="mt-4 text-center text-sm font-semibold text-brand-pink">
          {error}
        </p>
      )}

      {result && (
        <div className="mt-8 space-y-6">
          <Celebrate message="शाबाश! Great reading 🎉" />

          <div className="grid grid-cols-3 gap-3 text-center">
            <Stat label="Accuracy" value={`${result.accuracy_score}%`} color="brand-green" />
            <Stat label="Fluency" value={`${result.fluency_score}%`} color="brand-blue" />
            <Stat label="Pace" value={`${result.words_per_minute} wpm`} color="brand-yellow" />
          </div>

          <div>
            <p className="text-sm font-semibold text-foreground/70">
              Word-level diff
            </p>
            <p className="mt-1 text-lg leading-relaxed">
              {result.word_diff.map((w, i) => (
                <span key={i} className={diffClass(w.status)}>
                  {w.word}{" "}
                </span>
              ))}
            </p>
          </div>

          <div>
            <p className="text-sm font-semibold text-foreground/70">Transcript</p>
            <p className="mt-1 text-foreground/60">{result.transcript}</p>
          </div>

          <button
            onClick={tryAgain}
            className="rounded-full bg-brand-pink px-6 py-3 text-base font-bold text-white"
          >
            Try again
          </button>
        </div>
      )}
    </main>
  );
}

function Stat({
  label,
  value,
  color,
}: {
  label: string;
  value: string;
  color: "brand-green" | "brand-blue" | "brand-yellow";
}) {
  const colorClass =
    color === "brand-green"
      ? "border-brand-green text-brand-green"
      : color === "brand-blue"
        ? "border-brand-blue text-brand-blue"
        : "border-brand-yellow text-brand-yellow";

  return (
    <div className={`rounded-2xl border-2 bg-surface p-4 ${colorClass}`}>
      <p className="text-2xl font-extrabold tabular-nums">{value}</p>
      <p className="text-xs font-semibold text-foreground/60">{label}</p>
    </div>
  );
}

function diffClass(status: WordDiffEntry["status"]) {
  switch (status) {
    case "match":
      return "text-brand-green font-semibold";
    case "substitution":
      return "text-brand-orange underline decoration-wavy";
    case "missing":
      return "text-foreground/30 line-through";
    case "extra":
      return "text-foreground/30 italic";
  }
}

type WordDiffEntry = ScoreResult["word_diff"][number];
