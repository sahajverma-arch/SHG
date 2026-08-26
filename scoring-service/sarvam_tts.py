"""Hindi speech from Sarvam's Bulbul, cached to disk so each passage costs once.

This is the middle tier of three. `/tts` tries, in order:

    1. a pre-rendered IndicF5 clip   - free, instant, but only exists for
                                       passages someone ran prerender_tts.py on
    2. Bulbul (here)                 - any text, ~1s, costs money once per text
    3. edge-tts                      - free, always available, audibly robotic

Tier 2 is the one that closes the gap tier 1 leaves. IndicF5 needs ~30s per
sentence on a GPU, so its clips have to be rendered ahead of time and only six
passages ever were; every other passage silently dropped to edge-tts. Bulbul is
a hosted call, so it renders anything on demand.

**Why the disk cache is not optional.** Bulbul is billed per character
(~Rs.30 per 10k, so roughly Rs.1 for a typical passage). A child pressing
"hear it first" four times must not be four charges, and neither must a demo
re-reading the same passage. Clips land in `tts-cache/sarvam/` under the same
content hash `tts_cache.prerender_key` uses, so a passage is paid for once and
is free forever after - including across restarts. Editing a passage orphans
its clip and re-renders, which is correct: the old audio no longer matches.

**Every failure here is soft.** No key, no network, a 4xx, a malformed body -
all of it returns None and `/tts` falls through to edge-tts. A billing problem
or an expired key must never turn the "hear it first" button into a dead
button. `X-TTS-Source` on the response is how you tell which tier actually
spoke; `/health` reports whether a key is even configured.

The key is read from the environment, or from `scoring-service/.env` (which is
gitignored) so it does not have to be re-exported every launch:

    SARVAM_API_KEY=...

Nothing here imports a third-party HTTP client on purpose. edge-tts already
owns the async HTTP path and pulls in aiohttp; adding httpx for one POST would
be a dependency for nothing. urllib does it, run off the event loop.
"""

import base64
import io
import json
import os
import tempfile
import urllib.error
import urllib.request
import wave
from pathlib import Path

from tts_cache import PRERENDER_DIR, prerender_key

ENDPOINT = os.environ.get("SARVAM_TTS_URL", "https://api.sarvam.ai/text-to-speech")

# bulbul:v3 is the current model; v2 is cheaper (Rs.15 vs Rs.30 per 10k chars)
# and noticeably flatter. For a child copying the reading, prosody is the whole
# point, so v3 is the default and the saving is not worth taking.
MODEL = os.environ.get("SARVAM_TTS_MODEL", "bulbul:v3")

# A taste call, and a one-word change. The app's other voices are female
# (edge-tts hi-IN-SwaraNeural, and the IndicF5 reference prompt), so a female
# speaker keeps the app sounding like one reader rather than three. Bulbul v3
# offers ~36; pick by ear, not from the list.
SPEAKER = os.environ.get("SARVAM_TTS_SPEAKER", "ritu")

# Matches the IndicF5 clips, so tiers 1 and 2 do not differ in bandwidth or
# resampling artefacts on top of already differing in voice.
SAMPLE_RATE = int(os.environ.get("SARVAM_TTS_SAMPLE_RATE", "24000"))

# Bulbul re-performs at a different speed rather than time-stretching a fixed
# take, which is why this can go further than edge-tts's rate could. The -10%
# ceiling there was set by prosody smearing into something synthetic; that is
# not the failure mode here. Still conservative: a model reading a child should
# copy has to stay natural.
PACE_SLOW = float(os.environ.get("SARVAM_TTS_PACE_SLOW", "0.85"))
PACE_NORMAL = float(os.environ.get("SARVAM_TTS_PACE_NORMAL", "1.0"))

# Bulbul v3 accepts 2500; /tts caps text well below that. Guard anyway, so a
# raised cap upstream turns into a clean fallback instead of a 400 from Sarvam.
MAX_CHARS = 2500

# Where clips are read from, and where new ones are written.
#
# These differ in a serverless deployment, and getting that wrong is a bill
# rather than a bug. A function's bundle is read-only and its /tmp does not
# survive a cold start, so a single cache directory would mean re-paying for
# every passage every time an instance spun up -- silently, since the audio
# still plays.
#
# So: reads check the bundled directory too. The six stored passages can be
# rendered once and committed (about 6MB), and then production never calls the
# TTS API for them at all. Writes go somewhere writable for everything else --
# a custom sentence, a new passage -- which /tmp handles for as long as the
# instance lives.
BUNDLED_DIR = PRERENDER_DIR / "sarvam"
CACHE_DIR = Path(
    os.environ.get("SARVAM_TTS_CACHE_DIR")
    or (
        str(Path(tempfile.gettempdir()) / "shg-tts")
        if os.environ.get("VERCEL")
        else str(BUNDLED_DIR)
    )
)

TIMEOUT = float(os.environ.get("SARVAM_TTS_TIMEOUT", "20"))


def _read_env_file() -> None:
    """Load scoring-service/.env into the environment, if it exists.

    Deliberately tiny and not python-dotenv: one `KEY=value` per line, `#`
    comments, no interpolation, no quoting rules to remember. Existing
    environment variables win, so an explicit `SARVAM_API_KEY=... uvicorn ...`
    still overrides the file.
    """
    path = Path(__file__).resolve().parent / ".env"
    try:
        raw = path.read_text(encoding="utf-8")
    except OSError:
        return
    for line in raw.splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        name, _, value = line.partition("=")
        name = name.strip()
        # A key pasted with surrounding quotes is a common way to spend an hour
        # on a 401 that reads as "the key is wrong".
        value = value.strip().strip("\"'")
        if name and name not in os.environ:
            os.environ[name] = value


_read_env_file()


def api_key() -> str | None:
    key = os.environ.get("SARVAM_API_KEY", "").strip()
    return key or None


def configured() -> bool:
    """Whether Bulbul can be tried at all. Reported by /health."""
    return api_key() is not None


def cached_path(text: str, slow: bool) -> Path:
    """Where this exact reading lives on disk.

    Keyed on the pace too: the slow and normal readings are different audio,
    and serving one for the other would be a silent wrong answer rather than a
    cache miss.
    """
    suffix = "slow" if slow else "normal"
    return CACHE_DIR / f"{prerender_key(text)}-{suffix}.wav"


def find_cached(text: str, slow: bool) -> Path | None:
    """Look in the writable cache first, then in whatever shipped with the code.

    Order matters only in that a locally re-rendered clip should win over a
    stale committed one; in practice the two are the same file on a laptop.
    """
    suffix = "slow" if slow else "normal"
    name = f"{prerender_key(text)}-{suffix}.wav"
    for directory in (CACHE_DIR, BUNDLED_DIR):
        path = directory / name
        if path.is_file():
            return path
    return None


def synthesise(text: str, slow: bool = True) -> bytes | None:
    """Render `text` through Bulbul and cache it. None on any failure.

    Callers must treat None as "try the next tier", never as an error worth
    surfacing: see the module docstring on why this fails soft.
    """
    key = api_key()
    if key is None or not text.strip() or len(text) > MAX_CHARS:
        return None

    body = json.dumps(
        {
            "text": text,
            "target_language_code": "hi-IN",
            "speaker": SPEAKER,
            "model": MODEL,
            "pace": PACE_SLOW if slow else PACE_NORMAL,
            "speech_sample_rate": SAMPLE_RATE,
            "output_audio_codec": "wav",
        }
    ).encode("utf-8")

    request = urllib.request.Request(
        ENDPOINT,
        data=body,
        headers={
            "api-subscription-key": key,
            "Content-Type": "application/json",
        },
        method="POST",
    )

    try:
        with urllib.request.urlopen(request, timeout=TIMEOUT) as response:
            payload = json.loads(response.read().decode("utf-8"))
        audios = payload.get("audios") or []
        if not audios:
            return None
        # Long text comes back as several chunks meant to be played in order.
        # They are separate WAV files, so concatenating the bytes would splice
        # a 44-byte header into the middle of the audio: joined properly below.
        clips = [base64.b64decode(chunk) for chunk in audios]
        audio = clips[0] if len(clips) == 1 else _join_wavs(clips)
    except (urllib.error.URLError, OSError, ValueError, KeyError, wave.Error):
        # Includes HTTPError (401 bad key, 402 out of credits, 429 rate limit),
        # timeouts, DNS failure, and anything malformed in the response. They
        # all mean the same thing to the caller, and /tts names the tier that
        # actually spoke, so a silent demotion here is still visible.
        return None

    if not audio:
        return None

    try:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        # Write beside the target and rename, so an interrupted write cannot
        # leave a truncated wav that every later request happily serves.
        target = cached_path(text, slow)
        temporary = target.with_suffix(".wav.part")
        temporary.write_bytes(audio)
        temporary.replace(target)
    except OSError:
        # A read-only cache directory is not a reason to withhold audio we
        # already paid for. Serve it; the next request pays again.
        pass

    return audio


def _join_wavs(clips: list[bytes]) -> bytes:
    """Concatenate several WAV clips into one, keeping a single valid header."""
    out = io.BytesIO()
    writer = None
    try:
        for clip in clips:
            with wave.open(io.BytesIO(clip), "rb") as reader:
                if writer is None:
                    writer = wave.open(out, "wb")
                    writer.setnchannels(reader.getnchannels())
                    writer.setsampwidth(reader.getsampwidth())
                    writer.setframerate(reader.getframerate())
                writer.writeframes(reader.readframes(reader.getnframes()))
    finally:
        if writer is not None:
            writer.close()
    return out.getvalue()


def _main(argv: list[str]) -> int:
    """Warm the cache, and check the key, without starting the service.

        python sarvam_tts.py --file passages.txt
        python sarvam_tts.py "एक वाक्य" "दूसरा वाक्य"

    Two reasons this exists. A key that is wrong fails *soft* through /tts --
    the button still speaks, in the robotic voice -- so the one place it can
    be checked loudly is here. And a demo should not pay for its first play of
    every passage while someone is watching: render them beforehand and the
    button is instant and free.

    Already-cached passages are skipped, so re-running after adding one
    renders only the new one. Both readings are rendered, because the app asks
    for the slow one by default and the normal one from the same screen.
    """
    if not argv:
        print(__doc__)
        return 2

    if api_key() is None:
        print(
            "No SARVAM_API_KEY found.\n"
            "Put it in scoring-service/.env (gitignored):\n"
            "    SARVAM_API_KEY=...",
        )
        return 1

    if argv[0] == "--file":
        if len(argv) < 2:
            print("--file needs a path")
            return 2
        # One passage per line. Devanagari through a Windows shell is a good
        # way to render a subtly mangled passage and not notice, so prefer
        # this over arguments -- same reasoning as prerender_tts.py.
        raw = Path(argv[1]).read_text(encoding="utf-8")
        texts = [
            line.strip()
            for line in raw.splitlines()
            if line.strip() and not line.startswith("#")
        ]
    else:
        texts = [text for text in argv if text.strip()]

    rendered = skipped = failed = 0
    characters = 0
    for text in texts:
        for slow in (True, False):
            label = "slow" if slow else "normal"
            if find_cached(text, slow) is not None:
                skipped += 1
                continue
            if synthesise(text, slow) is None:
                failed += 1
                print(f"  FAILED  [{label}] {text[:40]}")
            else:
                rendered += 1
                characters += len(text)
                print(f"  ok      [{label}] {text[:40]}")

    # Rs.30 per 10k characters for bulbul:v3. Printed because a per-character
    # bill is the kind of thing that should never be a surprise.
    print(
        f"\n{rendered} rendered, {skipped} already cached, {failed} failed"
        f"\n{characters} characters billed (about Rs.{characters * 30 / 10_000:.2f})"
        f"\ncache: {CACHE_DIR}"
    )
    if failed and not rendered:
        print("\nNothing rendered at all - the key, the balance, or the network.")
        return 1
    return 0


if __name__ == "__main__":
    import sys

    raise SystemExit(_main(sys.argv[1:]))
