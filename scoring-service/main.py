import io
import os
import time

import numpy as np
from fastapi import FastAPI, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydub import AudioSegment
from transformers import pipeline

from scoring import score_attempt

MODEL_ID = os.environ.get("WHISPER_MODEL_ID", "vasista22/whisper-hindi-large-v2")
SAMPLE_RATE = 16000

app = FastAPI(title="Hindi Pronunciation Scoring Service")

app.add_middleware(
    CORSMiddleware,
    allow_origins=os.environ.get("CORS_ORIGIN", "http://localhost:3000").split(","),
    allow_methods=["POST", "GET"],
    allow_headers=["*"],
)

_asr = None


def get_asr():
    global _asr
    if _asr is None:
        # First call downloads the model from Hugging Face (a few GB) —
        # expect this to take a while the first time the service starts.
        _asr = pipeline(
            "automatic-speech-recognition",
            model=MODEL_ID,
            chunk_length_s=30,
        )
    return _asr


def load_audio(raw_bytes: bytes) -> tuple[np.ndarray, int]:
    try:
        segment = AudioSegment.from_file(io.BytesIO(raw_bytes))
    except Exception as exc:
        raise HTTPException(400, "Could not decode audio — is ffmpeg installed?") from exc

    segment = segment.set_frame_rate(SAMPLE_RATE).set_channels(1)
    samples = np.array(segment.get_array_of_samples()).astype(np.float32)
    samples /= float(1 << (8 * segment.sample_width - 1))
    return samples, SAMPLE_RATE


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": _asr is not None, "model_id": MODEL_ID}


@app.post("/score")
async def score(audio: UploadFile, expected_text: str = Form(...)):
    raw = await audio.read()
    if not raw:
        raise HTTPException(400, "Empty audio upload")

    wav, sr = load_audio(raw)
    audio_seconds = len(wav) / sr

    asr = get_asr()
    start = time.perf_counter()
    result = asr({"array": wav, "sampling_rate": sr})
    inference_seconds = time.perf_counter() - start

    transcript = result["text"].strip()

    return {
        "transcript": transcript,
        "audio_seconds": round(audio_seconds, 2),
        "inference_seconds": round(inference_seconds, 2),
        **score_attempt(expected_text, transcript, audio_seconds),
    }
