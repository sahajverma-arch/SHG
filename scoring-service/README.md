# Scoring service

FastAPI wrapper around `vasista22/whisper-hindi-large-v2` (self-hosted, free, no
account or card — see blueprint §04). Takes a passage + a recording, returns a
transcript plus an edit-distance/pace accuracy score.

This is a v1 proxy for pronunciation scoring, not true phoneme-level
assessment — it scores *which words* were read correctly and *how fast*, not
*how each word was pronounced*. Good enough to prototype the Reading Aloud
module end to end; revisit if/when this moves to Azure Pronunciation
Assessment for real phoneme scoring.

## Setup

Requires [ffmpeg](https://ffmpeg.org/download.html) on PATH (used to decode
whatever audio format the browser records — webm/ogg/wav all work).

```
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env
uvicorn main:app --reload --port 8000
```

The first request downloads the model from Hugging Face — a few GB, so the
first call will be slow. No account or card needed; it's a public model.

## API

`POST /score` — multipart form:
- `audio`: the recorded attempt (any format ffmpeg can read)
- `expected_text`: the passage text the user was asked to read

```json
{
  "transcript": "...",
  "audio_seconds": 8.4,
  "inference_seconds": 1.9,
  "accuracy_score": 87.5,
  "fluency_score": 100.0,
  "words_per_minute": 132.0,
  "word_diff": [{ "word": "...", "status": "match" }]
}
```

`word_diff` status is one of `match`, `substitution`, `missing`, `extra`.
