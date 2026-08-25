"use client";

/**
 * Live reading progress, from our own model.
 *
 * The browser's SpeechRecognition was the obvious route and turned out to be
 * the wrong one: it streams audio to Google rather than recognising on-device,
 * and returns `network` wherever that is unreachable — silently, since the
 * error surfaces only on a handler nobody watches.
 *
 * So progress comes from the same Whisper that marks the reading. Raw PCM is
 * tapped off the microphone, and every few seconds the newest slice is sent to
 * /transcribe as a self-contained WAV. Slices are independent, so cost stays
 * flat however long a child reads — re-sending the whole recording each time
 * would grow until it could not keep up.
 *
 * Scoring is untouched: it still runs on the full recording at the end. This
 * only moves a progress bar, so a rough slice transcript is good enough.
 */

const DEVANAGARI_PUNCT = /[।॥,.!?;:"'()\[\]{}‘’“”‌‍]/g;
const HYPHENS = /[-‐‑‒–—]/g;
const MARKS = /[ऺऻ़ािीुूृॄॅॆेैॉॊोौ्ॎॏऀँंः]/g;

/** Consonant confusions Hindi learners and recognisers both make. */
const FOLD: Record<string, string> = {
  ख: "क", घ: "ग", छ: "च", झ: "ज", ठ: "ट", ढ: "ड", थ: "त", ध: "द", फ: "प", भ: "ब",
  ट: "त", ड: "द", ण: "न", ष: "स", श: "स", व: "ब",
};

// How long a reader waits to see a word light up is this interval plus one
// round trip, and the interval dominates. It cannot simply be made small: each
// flush sends only the audio since the last one, so a short interval means
// short slices, and a slice that cuts a word in half transcribes neither
// half.
//
// Replaying two known readings through the service, counting how many passage
// words the tracker actually matched:
//
//   interval   words tracked   felt lag
//     3.0s      100% / 96%       ~1.95s
//     2.0s      100% / 96%       ~1.39s
//     1.2s       60% / 44%       ~0.87s
//
// So 2s: a third off the lag for no accuracy. Below that the tracker starts
// losing the reading, which looks far worse than a slight delay.
const CHUNK_MS = 2000;
// A slice is hard-capped so that falling behind drops the oldest audio instead
// of letting slices grow into a spiral the tracker never recovers from.
// Progress may skip; it will not stall the recording.
//
// 4s specifically: shorter slices carry less context and are likelier to end
// mid-word, which is what makes Whisper re-decode. Measured over arbitrary cut
// points, 3s slices had a worst case of 8.67s against 0.73s for 4s ones.
const MAX_SLICE_SEC = 4;

// How far ahead a spoken word may match: a short hop for ordinary progress, a
// longer one to recover after dropped audio.
const NEAR_WINDOW = 3;
const FAR_WINDOW = 15;
// Minimum consonant-skeleton length for a word to be allowed a far jump.
const DISTINCTIVE_LEN = 3;

export function words(text: string): string[] {
  return text
    .normalize("NFC")
    .replace(DEVANAGARI_PUNCT, " ")
    .replace(HYPHENS, "")
    .split(/\s+/)
    .filter(Boolean);
}

function skeleton(word: string): string {
  return [...word].map((c) => FOLD[c] ?? c).join("").replace(MARKS, "");
}

function similar(a: string, b: string): boolean {
  if (a === b) return true;
  const sa = skeleton(a);
  const sb = skeleton(b);
  if (sa === sb) return true;
  return sa.length >= 3 && (sa.startsWith(sb) || sb.startsWith(sa));
}

/**
 * How many passage words have been read, given everything heard so far.
 *
 * A forward pointer that may skip a couple of expected words, so one misheard
 * word does not stall the bar for the rest of the passage.
 */
export function readCount(expected: string[], heardText: string): number {
  const heard = words(heardText);
  let pointer = 0;

  for (const spoken of heard) {
    let moved = false;

    // Ordinary progress: the next word, or one just after it if a word was
    // misheard.
    for (let look = 0; look < NEAR_WINDOW && pointer + look < expected.length; look++) {
      if (similar(expected[pointer + look], spoken)) {
        pointer += look + 1;
        moved = true;
        break;
      }
    }
    if (moved) continue;

    // Resynchronise after a gap. A slice can be dropped when the server falls
    // behind, leaving a hole the near window cannot bridge — without this the
    // bar freezes for the rest of the passage.
    //
    // Only distinctive words may jump: Hindi function words (है, में, को, से)
    // collapse to one or two letters and recur constantly, so allowing them to
    // match far ahead would race the bar to the end on a coincidence.
    if (skeleton(spoken).length < DISTINCTIVE_LEN) continue;
    for (
      let look = NEAR_WINDOW;
      look < FAR_WINDOW && pointer + look < expected.length;
      look++
    ) {
      if (similar(expected[pointer + look], spoken)) {
        pointer += look + 1;
        break;
      }
    }
  }
  return Math.min(pointer, expected.length);
}

/** 16-bit PCM WAV — self-contained, and decoded server-side without ffmpeg. */
function encodeWav(samples: Float32Array, sampleRate: number): Blob {
  const buffer = new ArrayBuffer(44 + samples.length * 2);
  const view = new DataView(buffer);
  const ascii = (offset: number, text: string) => {
    for (let i = 0; i < text.length; i++) view.setUint8(offset + i, text.charCodeAt(i));
  };

  ascii(0, "RIFF");
  view.setUint32(4, 36 + samples.length * 2, true);
  ascii(8, "WAVEfmt ");
  view.setUint32(16, 16, true);
  view.setUint16(20, 1, true); // PCM
  view.setUint16(22, 1, true); // mono
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, sampleRate * 2, true);
  view.setUint16(32, 2, true);
  view.setUint16(34, 16, true);
  ascii(36, "data");
  view.setUint32(40, samples.length * 2, true);

  for (let i = 0; i < samples.length; i++) {
    const s = Math.max(-1, Math.min(1, samples[i]));
    view.setInt16(44 + i * 2, s < 0 ? s * 0x8000 : s * 0x7fff, true);
  }
  return new Blob([buffer], { type: "audio/wav" });
}

/**
 * Follow along while a child reads.
 *
 * Taps the microphone source already feeding the level meter, so the browser
 * is never asked for a second capture.
 */
export function startLiveReading(
  ctx: AudioContext,
  source: MediaStreamAudioSourceNode,
  passage: string,
  onProgress: (readWords: number, total: number) => void,
  onComplete: () => void
): () => void {
  const base =
    process.env.NEXT_PUBLIC_SCORING_SERVICE_URL ?? "http://localhost:8000";
  const expected = words(passage);

  let pending: Float32Array[] = [];
  let transcript = "";
  let best = 0;
  let stopped = false;
  let inFlight = false;

  const processor = ctx.createScriptProcessor(4096, 1, 1);
  processor.onaudioprocess = (e) => {
    if (stopped) return;
    pending.push(new Float32Array(e.inputBuffer.getChannelData(0)));
  };
  source.connect(processor);
  // Required for the node to run, but must stay silent — routing the
  // microphone to the speakers would echo the child back at themselves.
  const mute = ctx.createGain();
  mute.gain.value = 0;
  processor.connect(mute);
  mute.connect(ctx.destination);

  async function flush() {
    if (stopped || inFlight || pending.length === 0) return;
    inFlight = true;

    const slices = pending;
    pending = [];
    const total = slices.reduce((n, s) => n + s.length, 0);
    const joined = new Float32Array(total);
    let offset = 0;
    for (const s of slices) {
      joined.set(s, offset);
      offset += s.length;
    }

    // Keep only the most recent audio if we have fallen behind.
    const cap = Math.floor(ctx.sampleRate * MAX_SLICE_SEC);
    const merged = joined.length > cap ? joined.subarray(joined.length - cap) : joined;

    try {
      const form = new FormData();
      form.append("audio", encodeWav(merged, ctx.sampleRate), "slice.wav");
      const res = await fetch(`${base}/transcribe`, { method: "POST", body: form });
      if (res.ok && !stopped) {
        const { text } = (await res.json()) as { text: string };
        if (text) {
          transcript += " " + text;
          // Never go backwards: a bar that retreats reads as the app losing
          // track of the reader.
          const count = Math.max(best, readCount(expected, transcript));
          if (count !== best) {
            best = count;
            onProgress(best, expected.length);
          }
          if (best >= expected.length) onComplete();
        }
      }
    } catch {
      // A dropped slice costs a little progress, never the recording.
    } finally {
      inFlight = false;
    }
  }

  const timer = window.setInterval(flush, CHUNK_MS);

  return () => {
    stopped = true;
    clearInterval(timer);
    processor.onaudioprocess = null;
    try {
      source.disconnect(processor);
      processor.disconnect();
      mute.disconnect();
    } catch {
      /* already torn down */
    }
  };
}
