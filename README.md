# Prototype

Reading-aloud pronunciation practice, scaffolded per `pronunciation-engine-blueprint.html`
(§02 stack, §04 AI voice module, §05 schema).

```
web/               Next.js app — UI, currently just the Reading Aloud module
scoring-service/   FastAPI + self-hosted Hindi Whisper (vasista22/whisper-hindi-small)
supabase/          schema.sql — run once a Supabase project exists
```

## Run it locally

**1. Scoring service** (start this first)

```
cd scoring-service
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env
```

Then **convert the weights** — this is not optional. The default backend is the
CTranslate2 build, the converted model is gitignored because it is ~240 MB, and
without it every request answers 503 telling you so. The one-off command is in
[scoring-service/README.md](scoring-service/README.md#convert-the-model-one-off);
it takes a few minutes.

If the machine has an NVIDIA GPU, also `pip install -r requirements-gpu.txt`.
`ASR_DEVICE` defaults to `auto`, so the GPU is picked up when it is usable and
CPU is used when it is not — measured 1.6s against 4.2s for 12s of audio.

```
uvicorn main:app --reload --port 8000
```

Needs ffmpeg on PATH — the browser records compressed `webm`, and ffmpeg is
what decodes it. On Windows: `winget install Gyan.FFmpeg`, then open a new
terminal so the updated PATH is picked up. (`brew install ffmpeg` on macOS,
`apt install ffmpeg` on Debian/Ubuntu.)

**2. Web app**

```
cd web
npm install
copy .env.local.example .env.local
npm run dev
```

`.env.local` needs a real Supabase URL and publishable key — see the Supabase
section below. Without them the passage list is empty and the app quietly
serves its one built-in fallback passage.

Open http://localhost:3000/reading-aloud, allow mic access, record a line,
stop — the transcript and a 20-point breakdown (pronunciation 8, fluency 6,
pace 6) come back from the scoring service.

**3. The model reading — optional**

"Listen" works immediately through edge-tts, over the network, no account, but
audibly robotic. Two better voices sit in front of it, and `/tts` tries all
three in order:

| tier | voice | covers | cost |
| --- | --- | --- | --- |
| 1 | Sarvam Bulbul | anything | ~₹1 per passage, once |
| 2 | IndicF5, pre-rendered | only passages someone rendered | GPU time, once |
| 3 | edge-tts | anything | free |

**Bulbul leads because it won a side-by-side against IndicF5** — same passage,
both voices, picked by ear — and because it covers every passage rather than
the handful anyone got round to rendering, so the app sounds like one reader
instead of two. `TTS_PREFER_INDICF5=1` restores the old order.

Tier 2 cannot be produced on demand: IndicF5 is a 0.4B flow-matching model,
~30s a sentence on a GPU, so its clips have to be rendered ahead of time. They
are gitignored, so a fresh clone has none, and producing them needs a GPU,
gated Hugging Face access and a second virtualenv — all of it in
[scoring-service/README.md](scoring-service/README.md#pre-rendering-with-indicf5).
It is kept behind Bulbul rather than dropped so that a clone with no key still
gets a good voice for whatever it has rendered.

**Tier 1 is the one worth setting up**, and it needs none of that. Get a key at
[dashboard.sarvam.ai](https://dashboard.sarvam.ai) (new accounts get ₹100 of
credit) and put it in `scoring-service/.env`, which is gitignored:

```
SARVAM_API_KEY=...
```

Nothing else changes — no new Python packages, no restart flags. Each rendered
clip is cached to `tts-cache/sarvam/`, so a passage is billed once and is free
for every later play, across restarts. Check the key and pre-pay a demo's
passages in one go:

```bash
python sarvam_tts.py --file passages.txt   # prints what it billed
```

Which voice you are actually hearing is checkable rather than guessable:
`/health` reports `tts_prerendered_clips`, `tts_sarvam_configured` and
`tts_sarvam_cached_clips`, and `/tts` answers with an `X-TTS-Source` header
naming the tier that spoke. A wrong key never breaks the button — it falls
through to the tier below and says so in that header.

## Supabase — for the streak to survive a cleared cache

Without it, the streak lives in this browser's `localStorage` — it works,
but a cleared cache or a different device starts back at zero. This can't be
automated end to end: creating the project needs your own sign-up. Once it
exists, everything else (schema, code) is already wired to pick it up.

1. Go to [supabase.com/dashboard](https://supabase.com/dashboard) and sign
   up — free, no card. New project → name it, set a DB password, pick a
   region, create (~2 min to provision).
2. In the project: **SQL Editor** → paste in `supabase/schema.sql` → Run, then
   do the same with `supabase/seed.sql` to load the starter passages. Both are
   safe to re-run, and `schema.sql` ends with a migration block that brings an
   older project up to the current columns.
3. **Authentication → Sign In / Providers → Anonymous** → enable it. This
   lets the app get a persistent user id without building a login screen —
   the streak is tied to that session rather than to `localStorage`.
4. **Project Settings → API** → copy the **Project URL** and the **anon
   public** key.
5. In `web/.env.local` (copy from `.env.local.example` if you haven't),
   fill in `NEXT_PUBLIC_SUPABASE_URL` and `NEXT_PUBLIC_SUPABASE_ANON_KEY`.
6. Restart `npm run dev`. The streak badge now reads/writes the `streaks`
   table instead of `localStorage` — same UI, same behavior, just backed by
   a real database.

Caveats:
- Free projects auto-pause after 7 days with no activity — one click to
  resume from the dashboard if that happens.
- This still won't survive the user clearing cookies/site data, since the
  anonymous session token lives in the browser too. True cross-device
  persistence (open on your phone, see the same streak) needs real login
  (email or magic link) built on top of this — this sets up the table and
  wiring for that, but the login screen itself isn't built.
- Reading Aloud's passage also comes from Supabase once it's connected —
  insert a row into `passages` or it keeps using the placeholder line.

## Scoring

Reading Aloud is scored out of 20 — pronunciation 8, fluency 6, pace 6 — with
the three scaled by how much of the passage was actually attempted, so reading
three words perfectly doesn't score full marks.

A word read differently is reported as `mispronounced` and carries what the
recogniser heard instead. It is deliberately never matched back to `correct`
(which hands full marks to a learner who misread half the line) nor called
`skipped` (which claims they said nothing). This is still a transcript-based
proxy, not phoneme-level assessment — see `scoring-service/README.md`.

## Not done yet

- Real login (email/magic link) — needed for the streak to follow a user
  across devices, not just across sessions on one browser
- Writing scored attempts to the `attempts` table (the columns and the
  nullable `audio_url` are ready; nothing inserts yet, and recordings are
  never uploaded to Storage)
- Phoneme-level pronunciation scoring — the current score can't tell a
  genuine mispronunciation from the recogniser mishearing a word
- The other four modules from the original reference site (all DB-only, no
  AI — see blueprint §03) — out of scope for now, 2 exercises is the target
# SHG
