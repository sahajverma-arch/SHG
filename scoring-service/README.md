# Scoring service

FastAPI wrapper around `vasista22/whisper-hindi-small` (self-hosted, free, no
account or card — see blueprint §04). Takes a passage + a recording, returns a
transcript plus a 20-point rubric score. Override the checkpoint with
`WHISPER_MODEL_ID` — `vasista22/whisper-hindi-large-v2` is more accurate and
much heavier.

This is a proxy for pronunciation scoring, not true phoneme-level assessment:
it scores *which words* were read, *how they differ* from the passage, and
*how fast*, but it cannot hear how a single sound was articulated. A text
pipeline also can't separate "the learner mispronounced it" from "the
recogniser misheard it".

What it does do is stay honest about that. A word read differently is reported
as `mispronounced` — never quietly matched back to `correct`, and never called
`skipped` as though nothing was said — and full marks require a clean read.
See `scoring.py` for the Hindi-specific confusion folding (aspirated/
unaspirated, retroflex/dental) that decides what counts as a near miss.

## Setup

Requires ffmpeg on PATH to decode whatever the browser records (webm/ogg).
Plain PCM `.wav` is decoded in-process and works without it.

```
winget install Gyan.FFmpeg     # Windows — then open a new terminal for PATH
brew install ffmpeg            # macOS
apt install ffmpeg             # Debian/Ubuntu
```

```
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env
```

### Convert the model (one-off)

The default backend runs the CTranslate2 build of the weights, so convert them
once before starting the service:

```
ct2-transformers-converter --model vasista22/whisper-hindi-small ^
  --output_dir models/whisper-hindi-small-ct2 --quantization int8 ^
  --copy_files preprocessor_config.json tokenizer_config.json vocab.json ^
  special_tokens_map.json added_tokens.json normalizer.json

python -c "from transformers import WhisperTokenizerFast as T; T.from_pretrained('vasista22/whisper-hindi-small').backend_tokenizer.save('models/whisper-hindi-small-ct2/tokenizer.json')"
```

The second command is needed because the upstream repo ships a slow tokenizer
and faster-whisper expects `tokenizer.json`. The result is ~240 MB (down from
967 MB) and is gitignored — rebuild it rather than committing it.

Converting from the official `vasista22` weights is deliberate: pre-converted
CTranslate2 copies of this model exist on the Hub, but they are third-party
re-uploads and this is a one-line step.

```
uvicorn main:app --reload --port 8000
```

### Why not just use transformers?

Both backends run the same weights. On CPU the transformers decoder costs
~0.44s per generated token, which is what makes it unusable here — the encoder
is fine in both.

| backend | 12.2s of Hindi audio |
|---|---|
| `transformers` pipeline | 263s (21.5x realtime) |
| `faster-whisper` int8 | **5.0s (0.4x realtime)** |

Set `ASR_BACKEND=transformers` to fall back — it needs no conversion step, so
it is useful for a quick start or to sanity-check a transcript against the
unquantised weights.

## API

`POST /score` — multipart form:
- `audio`: the recorded attempt (any format ffmpeg can read)
- `expected_text`: the passage text the user was asked to read
- `level` *(optional)*: `P1-P2` (default), `P3-P4` or `P5-P6` — sets the
  expected reading pace at 100 / 120 / 140 wpm
- `reading_vocabulary` *(optional)*: JSON list of target words to call out,
  either `["शब्द"]` or `[{"word": "शब्द", "meaning_english": "word"}]`

```json
{
  "transcript": "...",
  "audio_seconds": 38.1,
  "inference_seconds": 1.9,

  "total_score": 19,
  "pronunciation_score": 7,
  "fluency_score": 6,
  "pace_score": 6,
  "pre_coverage_total": 19,
  "coverage_percent": 100,

  "word_analysis": [{ "word": "थके", "status": "mispronounced", "heard": "ठके" }],
  "mispronounced_words": ["थके"],
  "skipped_words": [],
  "practice_words": ["थके"],
  "vocab_feedback_words": [],

  "assessment_confidence": "high",
  "fluency_signals": { "disfluency_count": 0, "duration_slack": 0.86 }
}
```

Scoring is out of 20: pronunciation 8, fluency 6, pace 6. The three are scored
on what was actually read, then scaled by `coverage_percent` — so reading three
words perfectly does not score full marks. The unscaled figures are kept as
`pre_coverage_*`.

`word_analysis` status is one of `correct`, `mispronounced` or `skipped`, and
`heard` carries what the recogniser got instead (only set for
`mispronounced`). `skipped` means nothing was said for that word.

## Tests

```
pip install pytest
python -m pytest
```

`test_scoring.py` is pinned to a real recording — including a regression test
for the case where a learner misreads several words and must not receive full
pronunciation marks. `test_api.py` covers the route with the model stubbed, so
neither file needs the checkpoint, a GPU, or ffmpeg.
