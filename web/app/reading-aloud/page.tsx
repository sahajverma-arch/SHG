"use client";

import Link from "next/link";
import { useCallback, useEffect, useRef, useState } from "react";
import { ScoreLine } from "@/components/ScoreLine";
import { pickPassage, saveAttempt, type Passage } from "@/lib/attempts";
import { startLiveReading, words } from "@/lib/liveReading";
import { scoreAttempt, type ScoreResult } from "@/lib/scoring";
import { speakHindi, stopSpeaking, whenVoicesReady } from "@/lib/speech";
import { useStreak } from "@/lib/streak";

const FALLBACK_PASSAGE = "सूरज पूरब से निकलता है और पश्चिम में डूब जाता है।";
const DEFAULT_LEVEL = "P1-P2";
const LEVEL_NAMES: Record<string, string> = {
  "P1-P2": "Foundation",
  "P3-P4": "Building",
  "P5-P6": "Confident",
};

const BAR_COUNT = 17;
// Above this the microphone is hearing a voice rather than room tone.
const SPEECH_RMS = 0.02;
// Stop once a reader has clearly finished: this long quiet, after they began.
const SILENCE_MS = 2600;
// A moment after the last word, so its audio is captured before stopping.
const COMPLETE_GRACE_MS = 900;
// Nothing should record forever if detection fails outright.
const MAX_RECORDING_MS = 180_000;

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
  const [levels, setLevels] = useState<number[]>(() => Array(BAR_COUNT).fill(3));
  const [readWords, setReadWords] = useState(0);

  const recorderRef = useRef<MediaRecorder | null>(null);
  const chunksRef = useRef<Blob[]>([]);
  const audioCtxRef = useRef<AudioContext | null>(null);
  const rafRef = useRef<number | null>(null);
  const stopLiveRef = useRef<(() => void) | null>(null);
  const timersRef = useRef<number[]>([]);
  const heardSpeechRef = useRef(false);
  const lastLoudRef = useRef(0);

  const passageWords = words(passage.text_hi);

  useEffect(() => {
    void pickPassage().then((next) => {
      if (next) setPassage(next);
    });
  }, []);

  useEffect(() => {
    void whenVoicesReady();
  }, []);

  const cleanupCapture = useCallback(() => {
    if (rafRef.current !== null) cancelAnimationFrame(rafRef.current);
    rafRef.current = null;
    audioCtxRef.current?.close().catch(() => {});
    audioCtxRef.current = null;
    stopLiveRef.current?.();
    stopLiveRef.current = null;
    timersRef.current.forEach(clearTimeout);
    timersRef.current = [];
    setLevels(Array(BAR_COUNT).fill(3));
  }, []);

  useEffect(() => {
    return () => {
      stopSpeaking();
      cleanupCapture();
    };
  }, [cleanupCapture]);

  const finishRecording = useCallback(() => {
    if (recorderRef.current?.state === "recording") recorderRef.current.stop();
  }, []);

  async function playPassage() {
    setError(null);
    setSpeaking(true);
    try {
      await speakHindi(passage.text_hi);
    } catch {
      setError("No Hindi voice available. Start the scoring service to listen.");
    } finally {
      setSpeaking(false);
    }
  }

  /**
   * Input level for the meter, and the silence that ends a reading.
   * Silence only counts once a voice has actually been heard, so a slow
   * starter is never cut off before they begin.
   */
  function startMeter(stream: MediaStream) {
    const ctx = new AudioContext();
    audioCtxRef.current = ctx;
    const source = ctx.createMediaStreamSource(stream);
    const analyser = ctx.createAnalyser();
    analyser.fftSize = 1024;
    source.connect(analyser);
    const buffer = new Uint8Array(analyser.frequencyBinCount);
    heardSpeechRef.current = false;
    lastLoudRef.current = performance.now();

    const tick = () => {
      analyser.getByteTimeDomainData(buffer);

      let sum = 0;
      for (let i = 0; i < buffer.length; i++) {
        const v = (buffer[i] - 128) / 128;
        sum += v * v;
      }
      const rms = Math.sqrt(sum / buffer.length);

      const now = performance.now();
      if (rms > SPEECH_RMS) {
        heardSpeechRef.current = true;
        lastLoudRef.current = now;
      } else if (heardSpeechRef.current && now - lastLoudRef.current > SILENCE_MS) {
        finishRecording();
        return;
      }

      const step = Math.floor(buffer.length / BAR_COUNT);
      setLevels(
        Array.from({ length: BAR_COUNT }, (_, i) => {
          let peak = 0;
          for (let j = i * step; j < (i + 1) * step; j++) {
            peak = Math.max(peak, Math.abs(buffer[j] - 128) / 128);
          }
          return 3 + Math.min(1, peak * 2.6) * 21;
        })
      );
      rafRef.current = requestAnimationFrame(tick);
    };
    rafRef.current = requestAnimationFrame(tick);
    return { ctx, source };
  }

  async function startRecording() {
    setError(null);
    setResult(null);
    setReadWords(0);
    stopSpeaking();

    let stream: MediaStream;
    try {
      stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    } catch {
      setError("The microphone is blocked. Allow it in your browser, then try again.");
      setStatus("error");
      return;
    }

    const recorder = new MediaRecorder(stream);
    chunksRef.current = [];
    const { ctx, source } = startMeter(stream);

    // Follow along using the same microphone tap as the meter. If slices fail
    // to transcribe, the silence rule above still ends the recording.
    stopLiveRef.current = startLiveReading(
      ctx,
      source,
      passage.text_hi,
      (count) => setReadWords(count),
      () => {
        timersRef.current.push(
          window.setTimeout(finishRecording, COMPLETE_GRACE_MS)
        );
      }
    );

    timersRef.current.push(window.setTimeout(finishRecording, MAX_RECORDING_MS));

    recorder.ondataavailable = (e) => {
      if (e.data.size > 0) chunksRef.current.push(e.data);
    };

    recorder.onstop = async () => {
      stream.getTracks().forEach((t) => t.stop());
      cleanupCapture();
      setStatus("scoring");
      try {
        const scored = await scoreAttempt(
          new Blob(chunksRef.current, { type: "audio/webm" }),
          passage.text_hi,
          { level: passage.level ?? DEFAULT_LEVEL }
        );
        setResult(scored);
        setStatus("done");
        complete();
        void saveAttempt(passage, scored);
      } catch {
        setError("Couldn't reach the scoring service. Is it running on port 8000?");
        setStatus("error");
      }
    };

    recorderRef.current = recorder;
    recorder.start();
    setStatus("recording");
  }

  async function readAnother() {
    setResult(null);
    setStatus("idle");
    setError(null);
    setReadWords(0);
    const next = await pickPassage();
    if (next) setPassage(next);
  }

  const recording = status === "recording";
  const scoring = status === "scoring";
  const levelName = LEVEL_NAMES[passage.level ?? DEFAULT_LEVEL] ?? "Foundation";
  const progress = passageWords.length
    ? Math.round((readWords / passageWords.length) * 100)
    : 0;

  return (
    <main className="mx-auto w-full max-w-2xl flex-1 px-4 py-6 pb-36 sm:px-6 sm:py-10 sm:pb-10">
      <div className={recording ? "recede" : undefined}>
        <p className="eyebrow">Reading aloud · {levelName}</p>
        <h1 className="mt-1 text-2xl font-semibold sm:text-3xl">पढ़कर सुनाओ</h1>
      </div>

      <div
        className="reading-surface mt-5 px-4 py-5 sm:px-6 sm:py-6"
        data-live={recording}
      >
        {result ? (
          <p className="reading-text">
            {result.word_analysis.map((w, i) => (
              <span key={i} className="me-[0.3em] inline-block text-center align-top">
                <span
                  className={`w ${
                    w.status === "correct"
                      ? "w-correct"
                      : w.status === "mispronounced"
                        ? "w-miss"
                        : "w-skip"
                  }`}
                >
                  {w.word}
                </span>
                {w.heard && <span className="heard">{w.heard}</span>}
              </span>
            ))}
          </p>
        ) : recording ? (
          // While reading, the words already heard carry the line — the same
          // stroke the marked result uses, drawn as they go.
          <p className="reading-text">
            {passageWords.map((word, i) => (
              <span
                key={i}
                className={`me-[0.3em] inline-block border-b-2 transition-colors duration-200 ${
                  i < readWords ? "border-accent" : "border-transparent"
                }`}
              >
                {word}
              </span>
            ))}
          </p>
        ) : (
          <p className="reading-text">{passage.text_hi}</p>
        )}
      </div>

      {result && (
        <p className="mt-2.5 flex flex-wrap gap-x-4 gap-y-1 text-xs text-muted">
          <span>
            <span className="border-b-2 border-good/45">अ</span> read well
          </span>
          <span>
            <span className="border-b-2 border-warn text-warn">अ</span> say again
          </span>
          <span>
            <span className="line-through opacity-55">अ</span> not read
          </span>
        </p>
      )}

      {/* `assertive`, unusually: an error here means the recording did not
          happen, and a child waiting to be told they were heard should not
          have to discover that by silence. */}
      <div aria-live="assertive" role="alert">
        {error && (
          <p className="mt-4 rounded-lg border border-warn/40 px-3 py-2 text-sm text-warn">
            {error}
          </p>
        )}
      </div>

      {result && (
        <div className="fade-up mt-5 space-y-4">
          <div className="card p-4 sm:p-5">
            <ScoreLine
              pronunciation={result.pronunciation_score}
              fluency={result.fluency_score}
              pace={result.pace_score}
              total={result.total_score}
            />
            <p className="mt-4 border-t border-border pt-3 text-sm text-muted">
              {result.assessment_confidence_message}
            </p>
          </div>

          {result.practice_words.length > 0 && (
            <div className="card p-4 sm:p-5">
              <p className="eyebrow">Practise these</p>
              <div className="mt-3 flex flex-wrap gap-2">
                {result.practice_words.map((word) => (
                  <span
                    key={word}
                    className="rounded-lg border border-border px-2.5 py-1 font-[family-name:var(--font-read)] text-base"
                  >
                    {word}
                  </span>
                ))}
              </div>
            </div>
          )}

          <div className="flex gap-3">
            <button onClick={readAnother} className="btn btn-primary flex-1">
              Next passage
            </button>
            <Link href="/" className="btn btn-quiet flex-1">
              Done
            </Link>
          </div>
        </div>
      )}

      {/* On a phone the controls sit in the thumb zone; on a wide screen they
          simply follow the passage. */}
      {status !== "done" && (
        <div className="fixed inset-x-0 bottom-0 z-30 border-t border-border bg-background/95 px-4 pb-[max(0.75rem,env(safe-area-inset-bottom))] pt-3 backdrop-blur sm:static sm:mt-6 sm:border-0 sm:bg-transparent sm:p-0 sm:backdrop-blur-none">
          <div className="mx-auto max-w-2xl">
            {recording ? (
              <>
                {/* A progressbar rather than a live region: this updates every
                    couple of seconds while the child reads, and announcing each
                    change would talk over them. A screen reader can report it
                    on request instead. */}
                <div
                  className="mb-2 h-1 overflow-hidden rounded-full bg-border"
                  role="progressbar"
                  aria-label="Reading progress"
                  aria-valuemin={0}
                  aria-valuemax={passageWords.length}
                  aria-valuenow={readWords}
                  aria-valuetext={`${readWords} of ${passageWords.length} words read`}
                >
                  <div
                    className="h-full rounded-full bg-accent transition-[width] duration-300"
                    style={{ width: `${progress}%` }}
                  />
                </div>
                <div className="mb-2.5 flex items-center justify-between gap-3">
                  <span className="text-xs tabular-nums text-muted">
                    {readWords} / {passageWords.length} words
                  </span>
                  <div className="flex h-5 items-center gap-[3px]" aria-hidden>
                    {levels.map((h, i) => (
                      <span key={i} className="level-bar" style={{ height: `${h}px` }} />
                    ))}
                  </div>
                </div>
                <button onClick={finishRecording} className="btn btn-stop w-full">
                  Stop now
                </button>
              </>
            ) : (
              <div className="flex gap-3">
                <button
                  onClick={startRecording}
                  disabled={scoring}
                  className="btn btn-primary flex-1"
                >
                  {scoring ? "Marking…" : "Start reading"}
                </button>
                <button
                  onClick={playPassage}
                  disabled={speaking || scoring}
                  className="btn btn-quiet"
                >
                  {speaking ? "Playing…" : "Listen"}
                </button>
              </div>
            )}
          </div>
        </div>
      )}
    </main>
  );
}
