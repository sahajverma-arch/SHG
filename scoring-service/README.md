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

## Speech for "Hear it first"

`GET /tts?text=...&slow=true` returns audio of the passage read aloud. Two
sources, in order:

1. **A pre-rendered clip**, if one exists for that exact text. Served straight
   from `tts-cache/` as wav.
2. **edge-tts**, otherwise — Microsoft Edge's "Read aloud" voices over the same
   unofficial channel the browser feature uses. Free, no account, needs
   network. Returns mp3. On failure the endpoint 503s and the web app falls
   back to the browser's own voice.

`TTS_VOICE` picks the edge-tts voice; `TTS_RATE_SLOW` / `TTS_RATE_NORMAL` set
the rate. Past about -10% a neural voice smears and starts to sound robotic,
which is the opposite of what a child should be copying.

### Pre-rendering with IndicF5

[IndicF5](https://huggingface.co/ai4bharat/IndicF5) (AI4Bharat, MIT) is a much
more natural Hindi voice than anything edge-tts offers. It is also a 0.4B
flow-matching model, and the measurements are what force the design here:

| | per sentence |
| --- | --- |
| edge-tts (network call) | well under 1s |
| IndicF5 on a GTX 1650 | ~35s |
| IndicF5 on CPU | did not finish one in 20 minutes |

So it cannot run per request. The passages are a fixed set, though, so each is
rendered once ahead of time and served instantly from disk.

It needs its **own virtualenv** — it pins `numpy<=1.26.4` and
`transformers<4.50`, both of which break the ASR side — and a **GPU**. Weights
are gated on Hugging Face (free, instant): accept at
[huggingface.co/ai4bharat/IndicF5](https://huggingface.co/ai4bharat/IndicF5),
then `hf auth login`.

```
python -m venv C:\if5-venv          # keep the path SHORT, see below
C:\if5-venv\Scripts\pip install torch --index-url https://download.pytorch.org/whl/cu126
C:\if5-venv\Scripts\pip install torchaudio "transformers<4.50" safetensors
C:\if5-venv\Scripts\pip install soundfile librosa vocos x_transformers torchdiffeq
C:\if5-venv\Scripts\pip install ema_pytorch cached_path jieba pypinyin accelerate
C:\if5-venv\Scripts\pip install hydra-core tomli pydub click tqdm huggingface_hub
C:\if5-venv\Scripts\pip install matplotlib wandb datasets
# Python 3.13+ only — see below
C:\if5-venv\Scripts\pip install audioop-lts
C:\if5-venv\Scripts\pip install --no-deps git+https://github.com/AI4Bharat/IndicF5.git
```

Then, from this directory — build the list from the database, and render it:

```
.venv\Scripts\python dump_passages.py                        # -> passages.txt
C:\if5-venv\Scripts\python prerender_tts.py --file passages.txt
```

`dump_passages.py` reads the `passages` table rather than `supabase/seed.sql`,
because the seed is what was inserted once and the table is what a child is
shown. It also handles the two ways that read looks like an empty table:
`passages_read` requires an authenticated role, so the publishable key alone is
filtered by RLS to `[]` with no error, and the table has no `title` column, so
selecting one is a 400.

One passage per line, UTF-8. Already-rendered passages are skipped, so adding a
line and re-running only renders the new one. Clips land in `tts-cache/`, named
by a hash of the whitespace-normalised text — `tts_cache.py` owns that naming
and is imported by both sides so they cannot drift apart.

`passages.txt` is generated and gitignored. Re-run `dump_passages.py` after
editing a passage: the clip is keyed by the text, so an edit orphans its audio
and the passage quietly falls back to edge-tts. `/health` reports
`tts_prerendered_clips`, and `/tts` answers with an `X-TTS-Source` header, so
which voice actually spoke is checkable rather than guessed at.

### Five things that will bite you

**`audioop-lts` only exists for Python 3.13+, and it takes the whole install
down with it.** It backports `audioop`, which was *removed* in 3.13; on 3.12
that module is still in the standard library and the package is unnecessary.
On 3.12 pip refuses it with `No matching distribution found`, and because pip
resolves a command's packages as one set, every other package named alongside
it is skipped too — so a single line that looks like it failed over one
optional dependency actually leaves you with no numpy, no transformers and no
torchaudio. The CUDA check still passes, which makes it easy to miss. Install
it on its own line, or not at all.

**`pip install torch` gives you the CPU build on Windows.** You have to point
at the CUDA index explicitly. Check with `torch.cuda.is_available()`; on CPU
this is unusable, not merely slow.

**Keep the venv path short.** torch's bundled licence tree is ~176 characters
deep on its own, and Windows `MAX_PATH` is 260 unless long paths are enabled
(they are off by default). A venv much past 60 characters fails to install with
`[WinError 206]`.

**`torch.compile` is a straight loss here.** `model.py` wraps both the vocoder
and the transformer in it. Windows has no Triton, so inductor cannot emit GPU
kernels and spends minutes finding that out — 528s for one sentence, against
~35s without. Dynamo's own `eager` backend is no better: it traces on CPU with
the GPU idle at 0%. `prerender_tts.py` disables it.

**Disabling it silently unloads every weight.** The published checkpoint was
saved *from* a compiled model, so its keys carry an `_orig_mod` segment that
stops matching once the wrapper is gone. `transformers` reports this as a
warning, not an error, and you are left with a randomly initialised model that
still emits fluent, confident, meaningless audio. `prerender_tts.py` strips the
prefix, loads the weights explicitly, and refuses to render if any tensor fails
to match.

**IndicF5 clones its speaker from a reference clip**, so the voice is whoever
is in `tts-prompts/`. Change `REF_NAME`/`REF_TEXT` in `prerender_tts.py` to
change who it sounds like — no retraining involved. The default is the Punjabi
prompt the model card itself uses for Hindi; cross-language transfer is the
point of the model.

## Tests

```
pip install pytest
python -m pytest
```

`test_scoring.py` is pinned to a real recording — including a regression test
for the case where a learner misreads several words and must not receive full
pronunciation marks. `test_api.py` covers the route with the model stubbed, so
neither file needs the checkpoint, a GPU, or ffmpeg.
