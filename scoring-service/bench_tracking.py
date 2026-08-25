"""Does a shorter chunk interval track worse?

    python bench_tracking.py <scratch-dir-with-fleurs-samples>
    SCORING_URL=... to point at something other than localhost:8000

Needs the service running: this measures the real round trip, not the model.

Lower CHUNK_MS means the bar moves sooner, but each slice carries less context
and is likelier to be cut mid-word. This replays a known reading through
/transcribe exactly as the client does - drain the buffer, send, accumulate -
and reports both the latency and how much of the passage was recovered.

It is where the table in `web/lib/liveReading.ts` came from, and why CHUNK_MS
is 2000 rather than something smaller:

    interval   words tracked   felt lag
      3.0s      100% / 96%       ~1.95s
      2.0s      100% / 96%       ~1.39s
      1.2s       60% / 44%       ~0.87s

1.2s was tried and reverted. Re-run this after anything that changes ASR
speed - a faster backend does not automatically make a shorter interval safe,
because the limit is how much speech fits in a slice, not how fast it decodes.
"""

import io
import json
import os
import re
import sys
import time
import unicodedata
import urllib.request
import wave

import numpy as np
from pydub import AudioSegment

BASE = os.environ.get("SCORING_URL", "http://127.0.0.1:8000")
# Directory holding meta.json + sample0.wav + sample1.wav (FLEURS Hindi).
ROOT = os.path.join(sys.argv[1] if len(sys.argv) > 1 else ".", "hindi")

MARKS = "\u093a-\u094d\u0951-\u0957\u0962\u0963"
FOLD = str.maketrans({
    "ख": "क", "घ": "ग", "छ": "च", "झ": "ज", "ठ": "ट", "ढ": "ड",
    "थ": "त", "ध": "द", "फ": "प", "भ": "ब", "ट": "त", "ड": "द",
    "ण": "न", "ष": "स", "श": "स", "व": "ब",
})


def skeleton(word: str) -> str:
    word = unicodedata.normalize("NFC", word).translate(FOLD)
    return re.sub(f"[{MARKS}]", "", word)


def tokens(text: str) -> list[str]:
    text = re.sub(r"[।॥,.!?;:\"'()\[\]{}\u200c\u200d]", " ", text)
    return [w for w in text.replace("-", "").split() if w]


def to_wav_bytes(samples, rate=16000):
    pcm = (np.clip(samples, -1, 1) * 32767).astype("<i2")
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(pcm.tobytes())
    return buf.getvalue()


def transcribe(wav_bytes):
    boundary = "----trackbench"
    body = b"".join([
        f"--{boundary}\r\n".encode(),
        b'Content-Disposition: form-data; name="audio"; filename="s.wav"\r\n',
        b"Content-Type: audio/wav\r\n\r\n",
        wav_bytes,
        f"\r\n--{boundary}--\r\n".encode(),
    ])
    req = urllib.request.Request(
        f"{BASE}/transcribe", data=body,
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
    )
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=180) as r:
        return json.loads(r.read().decode()).get("text", ""), time.time() - t0


def read_count(expected, heard):
    """The client's matcher: in order, never backwards, small look-ahead."""
    pointer = 0
    for spoken in heard:
        s = skeleton(spoken)
        for look in range(min(3, len(expected) - pointer)):
            e = skeleton(expected[pointer + look])
            if e == s or (len(s) >= 3 and (e.startswith(s) or s.startswith(e))):
                pointer += look + 1
                break
    return pointer


def replay(wav, expected, chunk_sec, cap_sec=4.0):
    rate = 16000
    heard_text, times = "", []
    step = int(rate * chunk_sec)
    cap = int(rate * cap_sec)
    for start in range(0, len(wav), step):
        piece = wav[start : start + step]
        if len(piece) < rate * 0.2:
            break
        if len(piece) > cap:
            piece = piece[-cap:]
        text, dt = transcribe(to_wav_bytes(piece))
        times.append(dt)
        if text:
            heard_text += " " + text
    counted = read_count(expected, tokens(heard_text))
    arr = np.array(times) if times else np.array([0.0])
    # Perceived lag: half the interval on average, plus the round trip.
    return counted, arr.mean(), arr.max(), chunk_sec / 2 + arr.mean()


def main():
    meta = json.load(io.open(f"{ROOT}\\meta.json", encoding="utf-8"))
    for sample in (0, 1):
        seg = AudioSegment.from_file(f"{ROOT}\\sample{sample}.wav")
        seg = seg.set_frame_rate(16000).set_channels(1)
        wav = np.array(seg.get_array_of_samples()).astype(np.float32) / float(
            1 << (8 * seg.sample_width - 1)
        )
        expected = tokens(meta[sample]["text"])
        print(f"\nsample{sample}: {len(expected)} words, {len(wav)/16000:.1f}s")
        print(f"  {'chunk':>6} {'tracked':>16} {'mean rt':>8} {'worst':>7} {'felt lag':>9}")
        for chunk_sec in (3.0, 2.0, 1.2, 0.8):
            counted, mean, worst, felt = replay(wav, expected, chunk_sec)
            pct = 100 * counted / len(expected)
            print(f"  {chunk_sec:>5.1f}s {counted:>3}/{len(expected):<3} ({pct:>3.0f}%)"
                  f"{'':>4} {mean:>7.2f}s {worst:>6.2f}s {felt:>8.2f}s")


if __name__ == "__main__":
    main()
