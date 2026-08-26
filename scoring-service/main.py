import io
import json
import os
import sys
import time
import types
import wave
from collections import OrderedDict
from pathlib import Path

import numpy as np
from fastapi import FastAPI, Form, HTTPException, Response, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from pydub import AudioSegment

import sarvam_tts
from scoring import DEFAULT_LEVEL, score_attempt
from tts_cache import PRERENDER_DIR, find_prerendered

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
# Device for the CTranslate2 backend. "auto" uses the GPU when there is one and
# falls back to CPU otherwise, so the same checkout runs on a laptop with a
# discrete card and on a plain CPU box without configuration.
ASR_DEVICE = os.environ.get("ASR_DEVICE", "auto")
# Left unset so the device chooses: float16 on GPU, int8 on CPU. Setting it
# pins both.
CT2_COMPUTE_TYPE = os.environ.get("CT2_COMPUTE_TYPE") or None
# Greedy decoding. Beam search multiplies decode cost for little gain when the
# expected text is already known and scoring is edit-distance based.
BEAM_SIZE = int(os.environ.get("ASR_BEAM_SIZE", "1"))


def _add_cuda_runtime_to_path() -> None:
    """Let CTranslate2 find cuBLAS and cuDNN when pip supplied them.

    The CUDA runtime is not bundled with ctranslate2; it arrives as the
    `nvidia-cublas-cu12` / `nvidia-cudnn-cu12` wheels, which unpack into
    site-packages rather than anywhere the loader looks. CTranslate2 asks for
    `cublas64_12.dll` by bare name, and on Windows that search covers PATH but
    *not* directories added through `os.add_dll_directory`, so PATH is what has
    to be amended — before the first CUDA call, which is why this runs at
    import.
    """
    import site

    roots = list(site.getsitepackages())
    if hasattr(site, "getusersitepackages"):
        roots.append(site.getusersitepackages())

    for root in roots:
        for package in ("cublas", "cudnn"):
            folder = Path(root) / "nvidia" / package / "bin"
            if folder.is_dir() and str(folder) not in os.environ.get("PATH", ""):
                os.environ["PATH"] = f"{folder}{os.pathsep}{os.environ.get('PATH', '')}"


def _resolve_device() -> tuple[str, str]:
    """(device, compute_type) actually usable on this machine."""
    requested = ASR_DEVICE
    if requested in ("auto", "cuda"):
        _add_cuda_runtime_to_path()
        try:
            import ctranslate2

            if ctranslate2.get_cuda_device_count() > 0:
                return "cuda", CT2_COMPUTE_TYPE or "float16"
        except Exception:
            pass
        if requested == "cuda":
            raise HTTPException(
                503,
                "ASR_DEVICE=cuda but no usable CUDA device was found. Install "
                "nvidia-cublas-cu12 and nvidia-cudnn-cu12, or set ASR_DEVICE=cpu.",
            )
    return "cpu", CT2_COMPUTE_TYPE or "int8"

app = FastAPI(title="Hindi Pronunciation Scoring Service")

app.add_middleware(
    CORSMiddleware,
    allow_origins=os.environ.get("CORS_ORIGIN", "http://localhost:3000").split(","),
    allow_methods=["POST", "GET"],
    allow_headers=["*"],
    # CORS hides every response header outside a small safelist unless it is
    # named here. X-TTS-Source exists so the voice in use is checkable rather
    # than guessed at, and it read as null from the browser the moment the app
    # was served from a different origin than this service — which is exactly
    # when you most want to know.
    expose_headers=["X-TTS-Source"],
)

_asr = None
# Filled in when the CTranslate2 backend loads, and reported by /health so a
# demo machine can be checked at a glance rather than by timing it.
_asr_device: str | None = None
_asr_compute_type: str | None = None


def _load_faster_whisper():
    """CTranslate2 backend. Needs the converted model — see README."""
    # Checked before the import so a missing conversion reports itself as such,
    # rather than being masked by whatever the library does on the way in.
    if not Path(CT2_MODEL_PATH).is_dir():
        raise HTTPException(
            503,
            f"No CTranslate2 model at {CT2_MODEL_PATH}. Convert it (see "
            "scoring-service/README.md) or set ASR_BACKEND=transformers.",
        )

    # faster-whisper imports PyAV at module load, only for its own
    # `decode_audio`. Nothing here calls that — audio reaches the backend
    # already decoded to a numpy array by `load_audio` — and PyAV's bundled
    # binaries are refused outright by Windows Application Control on some
    # machines. A stub module is enough to get past the import when the real
    # one will not load.
    try:
        import av  # noqa: F401
    except ImportError:
        sys.modules.setdefault("av", types.ModuleType("av"))

    try:
        from faster_whisper import WhisperModel
    except ImportError as exc:
        # The stub above covers PyAV. This catches the rest: ctranslate2 ships
        # its own compiled extension, and that fails to load for reasons with
        # nothing to do with the request — no wheel for the platform, a missing
        # CUDA runtime, or the same Windows policy refusing a different DLL.
        raise HTTPException(
            503,
            f"faster-whisper could not be loaded ({exc}). Install it, or set "
            "ASR_BACKEND=transformers to use the slower pure-Python backend.",
        ) from exc

    device, compute_type = _resolve_device()
    model = WhisperModel(
        CT2_MODEL_PATH,
        device=device,
        compute_type=compute_type,
        # Physical cores, not logical. CTranslate2's GEMMs are already
        # vectorised, so handing it both hyperthreads per core oversubscribes
        # them and measured slower here. Ignored on GPU.
        cpu_threads=max((os.cpu_count() or 8) // 2, 1),
    )
    global _asr_device, _asr_compute_type
    _asr_device, _asr_compute_type = device, compute_type

    def run(wav: np.ndarray, _sr: int, quick: bool = False) -> str:
        options = {}
        if quick:
            # Whisper re-decodes a segment at rising temperatures whenever the
            # result looks unconfident. A live slice is cut at an arbitrary
            # moment and usually ends mid-word, so it trips that check often —
            # and the retries are what the reader sees as the tracker stalling.
            # Measured over 24 arbitrary cut points of a 3s slice: the worst
            # case falls from 8.67s to 2.39s, with the mean unchanged at ~0.5s.
            # A progress bar does not need the accuracy the retries buy.
            options["temperature"] = 0.0
        segments, _info = model.transcribe(
            wav,
            language="hi",
            beam_size=BEAM_SIZE,
            # Each attempt is one short passage, so carrying context between
            # segments only invites the model to invent continuations.
            condition_on_previous_text=False,
            **options,
        )
        return "".join(segment.text for segment in segments)

    return run


def _load_transformers():
    """Reference backend. Slow on CPU, but needs no conversion step."""
    from transformers import pipeline

    # First call downloads the model from Hugging Face (about 1 GB) — expect
    # this to take a while the first time the service starts.
    pipe = pipeline("automatic-speech-recognition", model=MODEL_ID, chunk_length_s=30)

    def run(wav: np.ndarray, sr: int, quick: bool = False) -> str:
        # `quick` is a latency hint the CTranslate2 backend acts on; this
        # pipeline exposes no equivalent knob, so it is accepted and ignored.
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
        # None until the first transcription loads the model.
        "device": _asr_device,
        "compute_type": _asr_compute_type,
        "tts_voice": TTS_VOICE,
        # 0 means every passage falls back to edge-tts. IndicF5 has to be
        # rendered ahead of time (see README), and without this count an empty
        # cache looks exactly like a working one.
        "tts_prerendered_clips": (
            len(list(PRERENDER_DIR.glob("*.wav"))) if PRERENDER_DIR.is_dir() else 0
        ),
        # False means no SARVAM_API_KEY reached the process, and every passage
        # without a pre-rendered clip is speaking in the robotic voice. That is
        # the same invisible-downgrade trap as the count above: a missing key
        # and a working one look identical from the outside.
        "tts_sarvam_configured": sarvam_tts.configured(),
        "tts_sarvam_voice": f"{sarvam_tts.MODEL}:{sarvam_tts.SPEAKER}",
        "tts_sarvam_cached_clips": (
            len(list(sarvam_tts.CACHE_DIR.glob("*.wav")))
            if sarvam_tts.CACHE_DIR.is_dir()
            else 0
        ),
        "tts_prefer_indicf5": TTS_PREFER_INDICF5,
    }


# Browsers are not a reliable way to hear Hindi: Windows ships no hi-IN voice,
# so `speechSynthesis` silently reads Devanagari with an English voice or does
# nothing at all. Synthesising server-side means every device gets the same
# model reading, whatever voices happen to be installed.
#
# edge-tts drives Microsoft Edge's "Read aloud" service over the same
# unofficial channel the browser feature itself uses — free and no account,
# like gTTS, but a real neural voice instead of Google Translate's older,
# more robotic one. It needs network access and could change without notice.
# /tts degrades to 503 and the web app falls back to the browser voice.
TTS_VOICE = os.environ.get("TTS_VOICE", "hi-IN-SwaraNeural")

# How far to slow the model reading. This is a blunt time-stretch, not a
# re-performance: past about -10% the prosody smears and the voice starts to
# sound synthetic, which is the opposite of what a child should be copying.
# -25% was audibly dragging.
TTS_RATE_SLOW = os.environ.get("TTS_RATE_SLOW", "-10%")
TTS_RATE_NORMAL = os.environ.get("TTS_RATE_NORMAL", "+0%")

# Pre-rendered speech, checked before the network call.
#
# IndicF5 (AI4Bharat, MIT) is a much more natural Hindi voice than anything
# edge-tts offers, but it is a 0.4B flow-matching model: about 30s per
# sentence on a GTX 1650, and minutes on CPU. That is nowhere near a button
# press. The passages are a fixed set, though, so they can be rendered once
# ahead of time and served instantly from disk. See prerender_tts.py.
#
# Anything without a pre-rendered file falls through to Sarvam's Bulbul, and
# then to edge-tts, so a passage nobody has rendered yet still speaks.
_TTS_CACHE: "OrderedDict[tuple[str, bool], bytes]" = OrderedDict()
_TTS_CACHE_MAX = 32
_TTS_MAX_CHARS = 800

# Two good voices for one app is one too many, and Bulbul won the side-by-side
# on 2026-08-26 -- same passage, both voices, picked by ear. It also covers
# every passage rather than the six IndicF5 was ever run on, so the app now
# sounds like one reader instead of two.
#
# IndicF5 stays as the tier behind it rather than being deleted: a clone with
# no SARVAM_API_KEY still gets a good voice for whatever it has rendered, which
# is the difference between a degraded demo and a robotic one.
# TTS_PREFER_INDICF5=1 puts it back in front.
TTS_PREFER_INDICF5 = os.environ.get("TTS_PREFER_INDICF5", "").lower() in {
    "1",
    "true",
    "yes",
}


def _speak(audio: bytes, source: str) -> Response:
    """One audio response, however it was produced.

    Every tier answers with wav now, and every tier names itself in
    X-TTS-Source -- the header is the only way to tell a working cache from a
    silently degraded one.
    """
    return Response(
        content=audio,
        media_type="audio/wav",
        headers={
            "Cache-Control": "public, max-age=86400",
            "X-TTS-Source": source,
        },
    )


@app.get("/tts")
async def tts(text: str, slow: bool = True):
    text = text.strip()
    if not text:
        raise HTTPException(400, "Nothing to say")
    if len(text) > _TTS_MAX_CHARS:
        raise HTTPException(400, f"Text longer than {_TTS_MAX_CHARS} characters")

    # Which voice actually spoke is otherwise invisible. Every fallback here is
    # silent by design — the button still works — but that also means an empty
    # tts-cache/ is indistinguishable from a working one, and someone checking
    # the voice has no way to tell which one they just heard. The header says.
    def indicf5():
        pre = find_prerendered(text)
        if pre is None:
            return None
        return _speak(pre.read_bytes(), "prerendered")

    if TTS_PREFER_INDICF5 and (clip := indicf5()) is not None:
        return clip

    # Bulbul: any passage, not just the six someone pre-rendered. Billed per
    # character, so the disk cache is checked first and every rendered clip is
    # kept -- see sarvam_tts for why that is not just an optimisation.
    voice = f"{sarvam_tts.MODEL}:{sarvam_tts.SPEAKER}"
    hit = sarvam_tts.find_cached(text, slow)
    if hit is not None:
        return _speak(hit.read_bytes(), f"sarvam-cache:{voice}")

    if sarvam_tts.configured():
        # urllib blocks, and blocking the event loop here would stall every
        # other request behind one synthesis.
        spoken = await run_in_threadpool(sarvam_tts.synthesise, text, slow)
        if spoken:
            return _speak(spoken, f"sarvam:{voice}")
        # Falls through on purpose. A bad key or an empty balance costs the
        # child nothing: the tiers below still speak, and the header says so.

    # No key, no network, or nothing rendered for this text: a pre-rendered
    # IndicF5 clip is still far better than edge-tts, so it is tried before
    # giving up on a good voice rather than after.
    if (clip := indicf5()) is not None:
        return clip

    key = (text, slow)
    cached = _TTS_CACHE.get(key)
    if cached is None:
        try:
            from edge_tts import Communicate

            rate = TTS_RATE_SLOW if slow else TTS_RATE_NORMAL
            communicate = Communicate(text, TTS_VOICE, rate=rate)
            chunks = bytearray()
            async for chunk in communicate.stream():
                if chunk["type"] == "audio":
                    chunks.extend(chunk["data"])
            if not chunks:
                raise RuntimeError("no audio returned")
            cached = bytes(chunks)
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
        headers={
            "Cache-Control": "public, max-age=86400",
            "X-TTS-Source": f"edge-tts:{TTS_VOICE}",
        },
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

    return {"text": get_asr()(wav, sr, quick=True).strip()}


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
