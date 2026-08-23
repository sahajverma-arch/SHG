import io
import json
import os
import time
import wave
from collections import OrderedDict
from pathlib import Path

import numpy as np
from fastapi import FastAPI, Form, HTTPException, Response, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydub import AudioSegment

from scoring import DEFAULT_LEVEL, score_attempt

SAMPLE_RATE = 16000

# Two ASR backends over the same weights.
#
# `faster-whisper` runs the CTranslate2 build and is the default because the
# transformers decoder is unusably slow on CPU here: 0.44s per generated token,
# ~263s to score 12s of audio, against ~5s for the same model through
# CTranslate2. The encoder is fine either way (1.8s) -- it is token generation
# that collapses.
#
# The `transformers` backend is kept as a fallback: it needs no conversion
# step, and it is what the tests stub.
ASR_BACKEND = os.environ.get("ASR_BACKEND", "faster-whisper")
MODEL_ID = os.environ.get("WHISPER_MODEL_ID", "vasista22/whisper-hindi-small")
CT2_MODEL_PATH = os.environ.get(
    "CT2_MODEL_PATH",
    str(Path(__file__).resolve().parent / "models" / "whisper-hindi-small-ct2"),
)
CT2_COMPUTE_TYPE = os.environ.get("CT2_COMPUTE_TYPE", "int8")
# Greedy decoding. Beam search multiplies decode cost for little gain when the
# expected text is already known and scoring is edit-distance based.
BEAM_SIZE = int(os.environ.get("ASR_BEAM_SIZE", "1"))

app = FastAPI(title="Hindi Pronunciation Scoring Service")

app.add_middleware(
    CORSMiddleware,
    allow_origins=os.environ.get("CORS_ORIGIN", "http://localhost:3000").split(","),
    allow_methods=["POST", "GET"],
    allow_headers=["*"],
)

_asr = None


def _load_faster_whisper():
    """CTranslate2 backend. Needs the converted model — see README."""
    from faster_whisper import WhisperModel

    if not Path(CT2_MODEL_PATH).is_dir():
        raise HTTPException(
            503,
            f"No CTranslate2 model at {CT2_MODEL_PATH}. Convert it (see "
            "scoring-service/README.md) or set ASR_BACKEND=transformers.",
        )

    model = WhisperModel(
        CT2_MODEL_PATH,
        device="cpu",
        compute_type=CT2_COMPUTE_TYPE,
        cpu_threads=os.cpu_count() or 4,
    )

    def run(wav: np.ndarray, _sr: int) -> str:
        segments, _info = model.transcribe(
            wav,
            language="hi",
            beam_size=BEAM_SIZE,
            # Each attempt is one short passage, so carrying context between
            # segments only invites the model to invent continuations.
            condition_on_previous_text=False,
        )
        return "".join(segment.text for segment in segments)

    return run


def _load_transformers():
    """Reference backend. Slow on CPU, but needs no conversion step."""
    from transformers import pipeline

    # First call downloads the model from Hugging Face (about 1 GB) — expect
    # this to take a while the first time the service starts.
    pipe = pipeline("automatic-speech-recognition", model=MODEL_ID, chunk_length_s=30)

    def run(wav: np.ndarray, sr: int) -> str:
        return pipe({"array": wav, "sampling_rate": sr})["text"]

    return run


def get_asr():
    global _asr
    if _asr is None:
        if ASR_BACKEND == "transformers":
            _asr = _load_transformers()
        elif ASR_BACKEND == "faster-whisper":
            _asr = _load_faster_whisper()
        else:
            raise HTTPException(500, f"Unknown ASR_BACKEND {ASR_BACKEND!r}")
    return _asr


def _resample_mono(samples: np.ndarray, source_rate: int) -> np.ndarray:
    if source_rate == SAMPLE_RATE or samples.size == 0:
        return samples
    duration = samples.size / source_rate
    target_len = max(1, int(round(duration * SAMPLE_RATE)))
    return np.interp(
        np.linspace(0, duration, target_len, endpoint=False),
        np.linspace(0, duration, samples.size, endpoint=False),
        samples,
    ).astype(np.float32)


def _decode_pcm_wav(raw_bytes: bytes) -> np.ndarray | None:
    """Decode 16-bit PCM wav in-process.

    pydub shells out to ffmpeg even for wav, so this keeps the common
    already-uncompressed case working on machines without it. Anything else
    (the browser records webm) still needs ffmpeg.
    """
    try:
        with wave.open(io.BytesIO(raw_bytes), "rb") as handle:
            if handle.getsampwidth() != 2:
                return None
            channels = handle.getnchannels()
            rate = handle.getframerate()
            frames = handle.readframes(handle.getnframes())
    except (wave.Error, EOFError, ValueError):
        return None

    samples = np.frombuffer(frames, dtype="<i2").astype(np.float32) / 32768.0
    if channels > 1:
        samples = samples[: samples.size - samples.size % channels]
        samples = samples.reshape(-1, channels).mean(axis=1)
    return _resample_mono(samples, rate)


def load_audio(raw_bytes: bytes) -> tuple[np.ndarray, int]:
    samples = _decode_pcm_wav(raw_bytes)
    if samples is not None:
        return samples, SAMPLE_RATE

    try:
        segment = AudioSegment.from_file(io.BytesIO(raw_bytes))
    except Exception as exc:
        raise HTTPException(400, "Could not decode audio - is ffmpeg installed?") from exc

    segment = segment.set_frame_rate(SAMPLE_RATE).set_channels(1)
    samples = np.array(segment.get_array_of_samples()).astype(np.float32)
    samples /= float(1 << (8 * segment.sample_width - 1))
    return samples, SAMPLE_RATE


# Whisper invents fluent speech out of silence -- a second of nothing comes
# back as a news bulletin. A child who records nothing must be told so, not
# scored against a sentence the model made up, so near-silent audio never
# reaches the model at all.
SILENCE_RMS_THRESHOLD = float(os.environ.get("SILENCE_RMS_THRESHOLD", "0.005"))


def is_silent(wav: np.ndarray) -> bool:
    if wav.size == 0:
        return True
    return float(np.sqrt(np.mean(np.square(wav)))) < SILENCE_RMS_THRESHOLD


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": _asr is not None,
        "backend": ASR_BACKEND,
        "model_id": MODEL_ID if ASR_BACKEND == "transformers" else CT2_MODEL_PATH,
    }


# Browsers are not a reliable way to hear Hindi: Windows ships no hi-IN voice,
# so `speechSynthesis` silently reads Devanagari with an English voice or does
# nothing at all. Synthesising server-side means every device gets the same
# model reading, whatever voices happen to be installed.
#
# gTTS calls Google Translate's undocumented speech endpoint — free and no
# account, matching the no-card constraint in blueprint 04, but it needs
# network access and could change without notice. /tts degrades to 503 and the
# web app falls back to the browser voice.
_TTS_CACHE: "OrderedDict[tuple[str, bool], bytes]" = OrderedDict()
_TTS_CACHE_MAX = 32
_TTS_MAX_CHARS = 800


@app.get("/tts")
def tts(text: str, slow: bool = True):
    text = text.strip()
    if not text:
        raise HTTPException(400, "Nothing to say")
    if len(text) > _TTS_MAX_CHARS:
        raise HTTPException(400, f"Text longer than {_TTS_MAX_CHARS} characters")

    key = (text, slow)
    cached = _TTS_CACHE.get(key)
    if cached is None:
        try:
            from gtts import gTTS

            buffer = io.BytesIO()
            gTTS(text, lang="hi", slow=slow).write_to_fp(buffer)
            cached = buffer.getvalue()
        except Exception as exc:
            # Name the cause: a silent 503 here is indistinguishable from the
            # service being down, and the usual causes (no network, upstream
            # change) need different fixes.
            raise HTTPException(
                503, f"Speech unavailable: {type(exc).__name__}: {exc}"
            ) from exc

        _TTS_CACHE[key] = cached
        while len(_TTS_CACHE) > _TTS_CACHE_MAX:
            _TTS_CACHE.popitem(last=False)
    else:
        _TTS_CACHE.move_to_end(key)

    return Response(
        content=cached,
        media_type="audio/mpeg",
        headers={"Cache-Control": "public, max-age=86400"},
    )


@app.post("/transcribe")
async def transcribe(audio: UploadFile):
    """Transcribe a short slice of audio, nothing more.

    This backs the follow-along progress bar while a child reads. Browsers
    cannot do this themselves — Chrome's Web Speech API streams to Google and
    returns `network` where that is unreachable — so the same model that marks
    the reading also tracks it.

    Kept separate from /score: no rubric, no alignment, just text, so a slice
    stays cheap enough to run every few seconds while recording continues.
    """
    raw = await audio.read()
    if not raw:
        return {"text": ""}

    wav, sr = load_audio(raw)
    if is_silent(wav):
        return {"text": ""}

    return {"text": get_asr()(wav, sr).strip()}


def parse_vocabulary(raw: str | None) -> list | None:
    """Optional target-word list: ["word"] or [{"word": ..., "meaning_english": ...}]."""
    if not raw:
        return None
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, list) else None


@app.post("/score")
async def score(
    audio: UploadFile,
    expected_text: str = Form(...),
    level: str = Form(DEFAULT_LEVEL),
    reading_vocabulary: str | None = Form(None),
):
    raw = await audio.read()
    if not raw:
        raise HTTPException(400, "Empty audio upload")

    wav, sr = load_audio(raw)
    # Measured from the decoded audio rather than taken from the client, so the
    # pace score can't be moved by editing the request.
    audio_seconds = len(wav) / sr

    if is_silent(wav):
        transcript = ""
        inference_seconds = 0.0
    else:
        asr = get_asr()
        start = time.perf_counter()
        transcript = asr(wav, sr).strip()
        inference_seconds = time.perf_counter() - start

    return {
        "transcript": transcript,
        "audio_seconds": round(audio_seconds, 2),
        "inference_seconds": round(inference_seconds, 2),
        **score_attempt(
            expected_text,
            transcript,
            audio_seconds,
            level=level,
            reading_vocabulary=parse_vocabulary(reading_vocabulary),
        ),
    }
