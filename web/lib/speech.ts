"use client";

/**
 * Speaking Hindi, in order of preference.
 *
 * 1. The scoring service's /tts — one consistent voice on every device.
 * 2. The browser's own speechSynthesis, *only* if a Hindi voice is installed.
 *
 * The browser route is a genuine fallback, not a default: Windows ships no
 * hi-IN voice, so `speechSynthesis` will happily accept `lang = "hi-IN"` and
 * then read Devanagari with an English voice, or emit nothing at all. Asking
 * for a Hindi voice by name is the only way to know.
 */

const base =
  process.env.NEXT_PUBLIC_SCORING_SERVICE_URL ?? "http://localhost:8000";

let audio: HTMLAudioElement | null = null;

export function hasHindiVoice(): boolean {
  if (typeof window === "undefined" || !("speechSynthesis" in window)) return false;
  return window.speechSynthesis
    .getVoices()
    .some((v) => v.lang?.toLowerCase().startsWith("hi"));
}

/** Voices load asynchronously in most browsers; resolve once they exist. */
export function whenVoicesReady(): Promise<void> {
  if (typeof window === "undefined" || !("speechSynthesis" in window)) {
    return Promise.resolve();
  }
  if (window.speechSynthesis.getVoices().length > 0) return Promise.resolve();
  return new Promise((resolve) => {
    const done = () => resolve();
    window.speechSynthesis.addEventListener("voiceschanged", done, { once: true });
    setTimeout(done, 1500);
  });
}

export function stopSpeaking() {
  if (audio) {
    audio.pause();
    audio = null;
  }
  if (typeof window !== "undefined" && "speechSynthesis" in window) {
    window.speechSynthesis.cancel();
  }
}

/**
 * Speak `text`, resolving when playback finishes.
 * Throws only when neither route can produce Hindi.
 */
export async function speakHindi(text: string): Promise<void> {
  stopSpeaking();

  try {
    const res = await fetch(
      `${base}/tts?text=${encodeURIComponent(text)}&slow=true`
    );
    if (res.ok) {
      const url = URL.createObjectURL(await res.blob());
      const el = new Audio(url);
      audio = el;
      await new Promise<void>((resolve, reject) => {
        el.onended = () => resolve();
        el.onerror = () => reject(new Error("playback failed"));
        el.play().catch(reject);
      });
      URL.revokeObjectURL(url);
      audio = null;
      return;
    }
  } catch {
    // Service unreachable — fall through to the browser voice.
  }

  if (!hasHindiVoice()) {
    throw new Error("no-hindi-voice");
  }

  await new Promise<void>((resolve) => {
    const utterance = new SpeechSynthesisUtterance(text);
    utterance.lang = "hi-IN";
    utterance.rate = 0.85;
    utterance.onend = () => resolve();
    utterance.onerror = () => resolve();
    window.speechSynthesis.speak(utterance);
  });
}
