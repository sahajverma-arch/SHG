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
  const [speaking, setSpeaking] = useState(false);

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

  useEffect(() => {
    return () => {
      if ("speechSynthesis" in window) window.speechSynthesis.cancel();
    };
  }, []);

  function speakPassage() {
    if (!("speechSynthesis" in window)) {
      setError("Text-to-speech isn't supported in this browser.");
      return;
    }
    window.speechSynthesis.cancel();
    const utterance = new SpeechSynthesisUtterance(passage.text_hi);
    utterance.lang = "hi-IN";
    utterance.rate = 0.9;
    utterance.onstart = () => setSpeaking(true);
    utterance.onend = () => setSpeaking(false);
    utterance.onerror = () => setSpeaking(false);
    window.speechSynthesis.speak(utterance);
  }

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
    <main className="mx-auto max-w-2xl px-6 py-14">
      <div className="flex items-center gap-3">
        <span className="icon-badge h-11 w-11 bg-brand-pink/15 text-xl">
          🎙️
        </span>
        <div>
          <h1 className="font-[family-name:var(--font-display)] text-2xl text-brand-pink">
            Reading Aloud
          </h1>
          <p className="text-sm text-foreground/60">
            Read the line below, then record yourself.
          </p>
        </div>
      </div>

      <div className="mt-8 flex justify-center">
        <button
          onClick={speakPassage}
          disabled={speaking || recording}
          aria-label="Listen to the line"
          className="icon-btn clr-pink h-14 w-14 text-2xl"
        >
          {speaking ? "🔊" : "🔈"}
        </button>
      </div>

      <div className="card relative mt-4 overflow-hidden p-8 text-center">
        <span
          aria-hidden
          className="pointer-events-none absolute -left-2 -top-6 font-[family-name:var(--font-display)] text-8xl text-brand-pink/10"
        >
          “
        </span>
        <p className="relative text-2xl font-bold leading-relaxed text-foreground">
          {passage.text_hi}
        </p>
      </div>

      {status !== "done" && (
        <div className="mt-6 flex justify-center">
          {!recording ? (
            <button
              onClick={startRecording}
              disabled={status === "scoring"}
              className="btn btn-solid clr-pink text-base"
            >
              <span aria-hidden>🎙️</span>
              {status === "scoring" ? "Scoring…" : "Record"}
            </button>
          ) : (
            <button
              onClick={stopRecording}
              className="btn btn-solid clr-orange animate-pulse text-base"
            >
              <span aria-hidden>⏹</span> Stop
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
        <div className="mt-8 space-y-5">
          <Celebrate message="शाबाश! Great reading" />

          <div className="grid grid-cols-3 gap-3 text-center">
            <Stat
              icon="🎯"
              label="Accuracy"
              value={`${result.accuracy_score}%`}
              color="brand-green"
            />
            <Stat
              icon="🌊"
              label="Fluency"
              value={`${result.fluency_score}%`}
              color="brand-blue"
            />
            <Stat
              icon="⏱️"
              label="Pace"
              value={`${result.words_per_minute} wpm`}
              color="brand-yellow"
            />
          </div>

          <div className="card p-5">
            <div className="flex flex-wrap items-center justify-between gap-2">
              <p className="text-sm font-bold text-foreground/70">
                Word-level diff
              </p>
              <div className="flex gap-3 text-[11px] font-semibold text-foreground/50">
                <Legend color="bg-brand-green" label="Match" />
                <Legend color="bg-brand-orange" label="Close" />
                <Legend color="bg-foreground/25" label="Missed" />
              </div>
            </div>
            <p className="mt-3 text-lg leading-relaxed">
              {result.word_diff.map((w, i) => (
                <span key={i} className={diffClass(w.status)}>
                  {w.word}{" "}
                </span>
              ))}
            </p>
          </div>

          <div className="card p-5">
            <p className="text-sm font-bold text-foreground/70">Transcript</p>
            <p className="mt-1 text-foreground/60">{result.transcript}</p>
          </div>

          <button
            onClick={tryAgain}
            className="btn btn-solid clr-pink w-full text-base"
          >
            Try again
          </button>
        </div>
      )}
    </main>
  );
}

function Legend({ color, label }: { color: string; label: string }) {
  return (
    <span className="flex items-center gap-1.5">
      <span className={`h-2 w-2 rounded-full ${color}`} aria-hidden />
      {label}
    </span>
  );
}

function Stat({
  icon,
  label,
  value,
  color,
}: {
  icon: string;
  label: string;
  value: string;
  color: "brand-green" | "brand-blue" | "brand-yellow";
}) {
  const colorClass =
    color === "brand-green"
      ? "text-brand-green"
      : color === "brand-blue"
        ? "text-brand-blue"
        : "text-brand-yellow";

  return (
    <div className="card flex flex-col items-center gap-1 p-4">
      <span className="text-xl" aria-hidden>
        {icon}
      </span>
      <p className={`text-2xl font-extrabold tabular-nums ${colorClass}`}>
        {value}
      </p>
      <p className="text-xs font-semibold text-foreground/50">{label}</p>
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
