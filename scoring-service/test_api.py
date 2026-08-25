"""Route-level tests for /score.

The ASR pipeline is stubbed so these run without downloading the model or
needing a GPU. What they actually exercise is the wiring around it: multipart
parsing, audio decoding, the server-side duration measurement the pace score
depends on, optional-field handling, and the response shape the web app types
against.
"""

import io
import math
import os
import struct
import sys
import types
import wave

import pytest
from fastapi import HTTPException

SAMPLE_RATE = 16000
TRANSCRIPT = "आज रविवार है"


def _install_fake_transformers(transcript=TRANSCRIPT):
    """Stand in for transformers.pipeline before main.py imports it."""
    module = types.ModuleType("transformers")

    def pipeline(*_args, **_kwargs):
        def run(_audio):
            return {"text": transcript}

        return run

    module.pipeline = pipeline
    sys.modules["transformers"] = module


def _wav_bytes(seconds: float, silent: bool = False) -> bytes:
    """Plain PCM wav, decodable without ffmpeg on the machine running tests.

    Audible by default: silence is short-circuited before the model (see
    `is_silent`), so a silent fixture would stop exercising the ASR path.
    """
    frames = int(SAMPLE_RATE * seconds)
    if silent:
        payload = b"\x00\x00" * frames
    else:
        payload = b"".join(
            struct.pack("<h", int(12000 * math.sin(2 * math.pi * 220 * i / SAMPLE_RATE)))
            for i in range(frames)
        )
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SAMPLE_RATE)
        w.writeframes(payload)
    return buf.getvalue()


@pytest.fixture(scope="module")
def client():
    # Pin the transformers backend: it is the one the stub above replaces, so
    # these stay hermetic whatever the deployed default is.
    os.environ["ASR_BACKEND"] = "transformers"
    _install_fake_transformers()
    pytest.importorskip("numpy")
    pytest.importorskip("pydub")
    from fastapi.testclient import TestClient

    import main

    return TestClient(main.app)


def _post(client, seconds=6.0, silent=False, **extra):
    data = {"expected_text": "आज रविवार है और मौसम अच्छा है", **extra}
    files = {"audio": ("attempt.wav", _wav_bytes(seconds, silent), "audio/wav")}
    return client.post("/score", data=data, files=files)


# ---------------------------------------------------------------------------
# /transcribe — the live progress tracker
# ---------------------------------------------------------------------------

def _stub_asr(monkeypatch, transcript=TRANSCRIPT):
    """Replace the loaded ASR and record how each call was made."""
    import main

    calls = []

    def run(wav, sr, quick=False):
        calls.append({"samples": len(wav), "sample_rate": sr, "quick": quick})
        return transcript

    monkeypatch.setattr(main, "_asr", run)
    return calls


def test_transcribe_returns_text_for_a_slice(client, monkeypatch):
    calls = _stub_asr(monkeypatch)
    body = client.post(
        "/transcribe", files={"audio": ("slice.wav", _wav_bytes(2.0), "audio/wav")}
    ).json()
    assert body["text"] == TRANSCRIPT
    assert len(calls) == 1


def test_a_live_slice_asks_for_the_low_latency_decode(client, monkeypatch):
    """Whisper's temperature retries are what the reader sees as a stall.

    A live slice is cut mid-word and trips them often, so /transcribe opts out.
    Measured over arbitrary cut points, that took the worst case from 8.67s to
    2.39s. If this argument ever stops being passed the tracker still works, so
    nothing would fail except the responsiveness — hence the test.
    """
    calls = _stub_asr(monkeypatch)
    client.post("/transcribe", files={"audio": ("slice.wav", _wav_bytes(2.0), "audio/wav")})
    assert calls[0]["quick"] is True


def test_scoring_a_whole_reading_keeps_the_accurate_decode(client, monkeypatch):
    calls = _stub_asr(monkeypatch)
    _post(client)
    assert calls[0]["quick"] is False


def test_transcribe_says_nothing_about_silence(client, monkeypatch):
    """The tracker must not advance on a child who has not started yet."""
    calls = _stub_asr(monkeypatch)
    body = client.post(
        "/transcribe", files={"audio": ("slice.wav", _wav_bytes(2.0, silent=True), "audio/wav")}
    ).json()
    assert body["text"] == ""
    assert calls == [], "silence should never reach the model"


def test_transcribe_tolerates_an_empty_upload(client, monkeypatch):
    _stub_asr(monkeypatch)
    body = client.post("/transcribe", files={"audio": ("slice.wav", b"", "audio/wav")}).json()
    assert body["text"] == ""


# ---------------------------------------------------------------------------
# /tts — which voice actually spoke
# ---------------------------------------------------------------------------

def test_a_prerendered_clip_is_served_and_says_so(client, tmp_path, monkeypatch):
    import main
    import tts_cache

    text = "सूरज पूरब से निकलता है"
    clip = tmp_path / f"{tts_cache.prerender_key(text)}.wav"
    clip.write_bytes(b"RIFF....WAVEfake")
    monkeypatch.setattr(main, "find_prerendered", lambda t: clip if t == text else None)

    res = client.get("/tts", params={"text": text})
    assert res.status_code == 200
    assert res.content == b"RIFF....WAVEfake"
    assert res.headers["x-tts-source"] == "prerendered"


def test_falling_back_to_edge_tts_is_visible(client, monkeypatch):
    """The fallback is silent on purpose — the button still speaks — which is
    exactly why an empty tts-cache/ looked identical to a working one."""
    import main

    monkeypatch.setattr(main, "find_prerendered", lambda _t: None)
    monkeypatch.setitem(main._TTS_CACHE, ("नमस्ते", True), b"fake-mp3")

    res = client.get("/tts", params={"text": "नमस्ते", "slow": "true"})
    assert res.status_code == 200
    assert res.headers["x-tts-source"] == f"edge-tts:{main.TTS_VOICE}"


def test_health_counts_prerendered_clips(client):
    body = client.get("/health").json()
    assert "tts_prerendered_clips" in body
    assert body["tts_voice"]


def test_health_reports_the_active_backend(client):
    body = client.get("/health").json()
    assert body["status"] == "ok"
    assert body["backend"] == "transformers"
    assert body["model_id"]


def test_faster_whisper_backend_reports_a_missing_model_clearly(client):
    """A missing conversion should say what to do, not fail deep in the library."""
    import main

    original_path, original_asr = main.CT2_MODEL_PATH, main._asr
    main.CT2_MODEL_PATH = "does/not/exist"
    try:
        with pytest.raises(Exception) as excinfo:
            main._load_faster_whisper()
        assert "ASR_BACKEND=transformers" in str(excinfo.value.detail)
    finally:
        main.CT2_MODEL_PATH, main._asr = original_path, original_asr


def test_cpu_is_chosen_when_the_gpu_is_not_asked_for(client):
    import main

    original = main.ASR_DEVICE
    main.ASR_DEVICE = "cpu"
    try:
        assert main._resolve_device() == ("cpu", "int8")
    finally:
        main.ASR_DEVICE = original


def test_asking_for_a_gpu_that_is_absent_says_so(client, monkeypatch):
    """Better than falling back silently and looking merely slow."""
    import main

    monkeypatch.setattr(main, "ASR_DEVICE", "cuda")
    monkeypatch.setattr(main, "_add_cuda_runtime_to_path", lambda: None)

    import ctranslate2

    monkeypatch.setattr(ctranslate2, "get_cuda_device_count", lambda: 0)
    with pytest.raises(HTTPException) as excinfo:
        main._resolve_device()
    assert "ASR_DEVICE=cpu" in excinfo.value.detail


def test_unknown_backend_is_rejected(client):
    import main

    original_backend, original_asr = main.ASR_BACKEND, main._asr
    main.ASR_BACKEND, main._asr = "nonsense", None
    try:
        with pytest.raises(Exception):
            main.get_asr()
    finally:
        main.ASR_BACKEND, main._asr = original_backend, original_asr


def test_empty_upload_is_rejected(client):
    files = {"audio": ("empty.wav", b"", "audio/wav")}
    res = client.post("/score", data={"expected_text": "आज"}, files=files)
    assert res.status_code == 400


def test_score_returns_the_full_rubric(client):
    body = _post(client).json()
    for key in (
        "total_score",
        "pronunciation_score",
        "fluency_score",
        "pace_score",
        "pre_coverage_total",
        "coverage_percent",
        "word_analysis",
        "mispronounced_words",
        "skipped_words",
        "assessment_confidence",
        "fluency_signals",
        "transcript",
    ):
        assert key in body, key
    parts = body["pronunciation_score"] + body["fluency_score"] + body["pace_score"]
    assert parts == body["total_score"]


def test_duration_is_measured_from_the_audio_not_the_client(client):
    # The pace score must not be movable by lying about how long the read took.
    short = _post(client, seconds=3.0).json()
    long = _post(client, seconds=30.0).json()
    assert short["audio_seconds"] == pytest.approx(3.0, abs=0.1)
    assert long["audio_seconds"] == pytest.approx(30.0, abs=0.1)
    assert long["fluency_signals"]["duration_slack"] > short["fluency_signals"]["duration_slack"]


def test_level_changes_the_pace_target(client):
    assert _post(client, level="P1-P2").json()["target_wpm"] == 100
    assert _post(client, level="P5-P6").json()["target_wpm"] == 140


def test_optional_fields_may_be_omitted_or_malformed(client):
    assert _post(client).status_code == 200
    # A bad vocabulary payload must not take the whole request down.
    res = _post(client, reading_vocabulary="not json at all")
    assert res.status_code == 200
    assert res.json()["vocab_feedback_words"] == []


def test_silence_never_reaches_the_model(client):
    """Whisper hallucinates fluent sentences from silence.

    One second of real silence transcribes as a news bulletin, so a child who
    records nothing must be told so rather than scored against invented text.
    """
    body = _post(client, seconds=4.0, silent=True).json()
    assert body["transcript"] == ""
    assert body["inference_seconds"] == 0.0
    assert body["total_score"] == 0
    assert body["mispronounced_words"] == []
    assert body["skipped_words"], "every passage word should read as missed"


def test_vocabulary_is_parsed_and_reported(client):
    body = _post(
        client,
        reading_vocabulary='[{"word": "मौसम", "meaning_english": "weather"}]',
    ).json()
    # मौसम is in the passage but not in the stubbed transcript, so it is missed.
    assert "मौसम" in body["vocab_feedback_words"]
