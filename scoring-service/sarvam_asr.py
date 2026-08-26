"""Sarvam's Saaras ASR, and the reason to care: `verbatim` mode.

Not a replacement for Whisper. The live tracker runs a slice every 2s against a
local GPU in well under a second, for free and offline; routing that through
the network would be slower, billed, and capped at 30s a request. This exists
for one question Whisper cannot answer.

**The confound.** Whisper does not transcribe sounds, it transcribes its best
guess at *words*. A child says रभीवार and Whisper writes रविवार, because a
language model snaps the sounds to the nearest real word. So the transcript
says the child read it correctly, and `scoring.py` -- which only ever sees the
transcript -- scores it correctly. "The child mispronounced it" and "the model
misheard it" become indistinguishable, which is the ceiling on how diagnostic
the pronunciation score can be. It is the same failure the reference site ships
(see docs/), where four mispronounced words scored 8/8.

Sarvam documents `verbatim` as "word-for-word (with fillers)" -- output that is
not tidied into dictionary words. If that holds for mispronunciations and not
just for filler sounds, the pronunciation tier gets real signal for the first
time. Whether it does is a measurement, not a claim: see eval_asr_modes.py.

    SARVAM_API_KEY=...        in scoring-service/.env, same key as the voice

Billed at ~Rs.30/hour of audio, so a 3s clip is about 2.5 paise. Cheap enough
to sweep, not free enough to leave in a loop.
"""

import io
import json
import mimetypes
import os
import time
import urllib.error
import urllib.request
import uuid
import wave
from pathlib import Path

# Shares the .env loader with the voice side: one key, read once, at import.
from sarvam_tts import api_key

ENDPOINT = os.environ.get("SARVAM_ASR_URL", "https://api.sarvam.ai/speech-to-text")
MODEL = os.environ.get("SARVAM_ASR_MODEL", "saaras:v3")
LANGUAGE = os.environ.get("SARVAM_ASR_LANGUAGE", "hi-IN")
TIMEOUT = float(os.environ.get("SARVAM_ASR_TIMEOUT", "60"))

# The REST endpoint caps a request at 30s of audio. A passage read aloud by a
# child runs well past that, so anything using this for whole attempts has to
# chunk -- another reason it is not simply swapped in for Whisper.
MAX_SECONDS = 30

# A sweep hits the rate limit routinely; these are for that, not for outages.
RETRIES = int(os.environ.get("SARVAM_ASR_RETRIES", "5"))
BACKOFF = float(os.environ.get("SARVAM_ASR_BACKOFF", "2"))

MODES = ("transcribe", "verbatim", "translit", "codemix")


class SarvamASRError(RuntimeError):
    """Raised loudly on purpose.

    The voice path fails soft because a silent downgrade there still speaks.
    Here a silent failure would be a transcript that looks real and is not,
    which is how a measurement quietly becomes wrong. Callers that want to
    continue past an error have to say so.
    """


def _multipart(fields: dict[str, str], filename: str, payload: bytes) -> tuple[bytes, str]:
    boundary = f"----sarvam{uuid.uuid4().hex}"
    content_type = mimetypes.guess_type(filename)[0] or "application/octet-stream"
    parts = bytearray()
    for name, value in fields.items():
        parts += f"--{boundary}\r\n".encode()
        parts += f'Content-Disposition: form-data; name="{name}"\r\n\r\n'.encode()
        parts += f"{value}\r\n".encode()
    parts += f"--{boundary}\r\n".encode()
    parts += (
        f'Content-Disposition: form-data; name="file"; filename="{filename}"\r\n'
    ).encode()
    parts += f"Content-Type: {content_type}\r\n\r\n".encode()
    parts += payload
    parts += f"\r\n--{boundary}--\r\n".encode()
    return bytes(parts), f"multipart/form-data; boundary={boundary}"


def transcribe_bytes(
    audio: bytes,
    filename: str = "clip.wav",
    mode: str = "transcribe",
    language: str | None = None,
) -> str:
    """One clip, one mode, in: the transcript out.

    `mode` is the whole point of this module -- `transcribe` cleans the output
    into words, `verbatim` is documented not to. Comparing the two on the same
    audio is the experiment.
    """
    key = api_key()
    if key is None:
        raise SarvamASRError(
            "No SARVAM_API_KEY. Put it in scoring-service/.env (gitignored)."
        )
    if mode not in MODES:
        raise SarvamASRError(f"Unknown mode {mode!r}; expected one of {MODES}")

    fields = {
        "model": MODEL,
        "language_code": language or LANGUAGE,
        "mode": mode,
    }
    body, content_type = _multipart(fields, filename, audio)

    # 429 is not a failure, it is "later". A sweep over a few dozen clips hits
    # the limit routinely, and treating that as an empty transcript would be
    # worse than crashing: an eval counts a blank as "the engine missed it" and
    # silently reports a worse number than the truth.
    last = None
    for attempt in range(RETRIES):
        request = urllib.request.Request(
            ENDPOINT,
            data=body,
            headers={"api-subscription-key": key, "Content-Type": content_type},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=TIMEOUT) as response:
                payload = json.loads(response.read().decode("utf-8"))
            return (payload.get("transcript") or "").strip()
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", "replace")[:400]
            last = SarvamASRError(f"HTTP {exc.code} from Sarvam: {detail}")
            if exc.code not in (429, 500, 502, 503, 504):
                raise last from exc
            # Honour Retry-After when the server sends one, else back off.
            wait = exc.headers.get("Retry-After")
            delay = float(wait) if wait and wait.isdigit() else BACKOFF * (2**attempt)
        except (urllib.error.URLError, OSError, ValueError) as exc:
            last = SarvamASRError(f"{type(exc).__name__}: {exc}")
            delay = BACKOFF * (2**attempt)
        if attempt < RETRIES - 1:
            time.sleep(delay)

    raise last or SarvamASRError("exhausted retries")


def transcribe_file(path: str | Path, mode: str = "transcribe", **kwargs) -> str:
    path = Path(path)
    return transcribe_bytes(path.read_bytes(), path.name, mode=mode, **kwargs)


# ---------------------------------------------------------------------------
# The 30-second cap
# ---------------------------------------------------------------------------
#
# This is the part that decides whether the app works for the children it is
# for. A 46-word passage is 28s at 100 WPM -- inside the cap -- and 39s at 70
# WPM, which is what a child who is still learning to read actually sounds
# like. Four of the six stored passages break at 70 WPM. Without splitting,
# the app would fail hardest for exactly the readers who need it most.


def wav_bytes(samples, rate: int) -> bytes:
    """Mono float32 in [-1, 1] to 16-bit PCM wav, in memory."""
    import numpy as np

    pcm = (np.clip(samples, -1.0, 1.0) * 32767).astype("<i2")
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(rate)
        handle.writeframes(pcm.tobytes())
    return buffer.getvalue()


def split_for_limit(samples, rate: int, max_seconds: float = MAX_SECONDS - 2):
    """Cut audio into pieces the API will accept, preferring quiet moments.

    A fixed cut every 28s would land mid-word about as often as not, and a word
    sliced in half is transcribed as two wrong words -- which the rubric then
    scores as two mispronunciations the child never made. So the cut is placed
    at the quietest point in a window around the target, which in read speech
    is the gap between words.

    Returns a list of arrays; a short clip comes back as a single piece, so
    callers do not need to special-case the common path.
    """
    import numpy as np

    limit = int(max_seconds * rate)
    if samples.size <= limit:
        return [samples]

    # Search the last quarter of each piece: far enough in to make progress,
    # late enough that a cut is never much earlier than it had to be.
    window = max(int(rate * 0.5), limit // 4)
    pieces = []
    start = 0
    while samples.size - start > limit:
        target = start + limit
        lo = max(start + limit - window, start + rate)
        chunk = samples[lo:target]
        if chunk.size:
            # Energy per 20ms frame; the quietest frame is the word gap.
            frame = max(int(rate * 0.02), 1)
            usable = chunk.size - chunk.size % frame
            if usable >= frame:
                energy = np.abs(chunk[:usable]).reshape(-1, frame).mean(axis=1)
                cut = lo + int(energy.argmin()) * frame
            else:
                cut = target
        else:
            cut = target
        pieces.append(samples[start:cut])
        start = cut
    if start < samples.size:
        pieces.append(samples[start:])
    return pieces


def transcribe_long(samples, rate: int, mode: str = "transcribe") -> str:
    """Transcribe audio of any length by splitting it past the API's cap.

    Pieces are joined with a space rather than concatenated: the rubric
    tokenises on whitespace, and gluing the last word of one piece to the first
    of the next would invent a compound word that scores as two errors.
    """
    pieces = split_for_limit(samples, rate)
    texts = []
    for index, piece in enumerate(pieces):
        text = transcribe_bytes(
            wav_bytes(piece, rate), f"part{index}.wav", mode=mode
        ).strip()
        if text:
            texts.append(text)
    return " ".join(texts)


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print(f"usage: python sarvam_asr.py <audio-file> [{'|'.join(MODES)}]")
        raise SystemExit(2)
    chosen = sys.argv[2] if len(sys.argv) > 2 else "transcribe"
    print(transcribe_file(sys.argv[1], mode=chosen))
