# Phoneme-level pronunciation scoring for Hindi

**Status: built, measured, and not shipped.** The machinery works and is
tested; the available models are not accurate enough to judge a child's
pronunciation word by word. This document records what was tried and what the
numbers were, so the work is not repeated blindly.

**Picking this up?** Start at [Where to take this next](#where-to-take-this-next) — the most promising idea there was never tested, and the section ends with the dead ends worth skipping.

## Why it was attempted

`scoring.py` scores a transcript, which is the wrong instrument for
pronunciation. A speech recogniser is built to emit real words, so a child who
says **काना** is transcribed as the real word **खाना** and scores full marks.
Blueprint §04 conceded this and settled for it. Reference implementations have
the same gap — thehindilab.com returns `mispronounced_words: []` and 8/8 for a
reading with four clear misreadings.

Phoneme scoring was meant to close it.

## What was built

| file | role |
|---|---|
| `phones.py` | phone comparison cost, tuned so Hindi's aspiration and dental/retroflex contrasts cost full price while notation differences cost nothing |
| `pronunciation.py` | espeak Hindi G2P (expected sounds) + wav2vec2 phone recognition (heard sounds) |
| `gop.py` | CTC forced alignment and Goodness of Pronunciation |
| `eval_gop.py` | the evaluation harness below — run it to reproduce |
| `test_pronunciation.py` | 27 tests over the parts that are correct |

espeak-ng comes from the `espeakng-loader` package rather than a system
install, because installing it system-wide needs administrator rights.

## The measurements

Two approaches, two models, one evaluation set: correct native readings from
Google FLEURS as negatives, and synthesised readings carrying real Hindi
learner errors (aspiration dropped or added, dental/retroflex swapped) as
positives.

### Approach 1 — compare phone strings

Decode the phones freely, compare against the phones the passage calls for.

| model | false positives on **correct** native readings |
|---|---|
| `facebook/wav2vec2-lv-60-espeak-cv-ft` | **33% and 44%** |
| `allosaurus` (Hindi phone inventory) | **70% and 77%** |

The cause is a systematic offset: the recogniser is zero-shot on Hindi and does
not realise its phones the way espeak writes them. For खाना (`kʰ aː n aː`) it
returns `k a n a` — dropping the aspiration on a *correct* recording. It hears
काना as `ɡ ɑ n ɔ`, so it does separate the pair, just one step down from where
espeak puts them.

Two speakers reading the *same correct text* disagreed on **24%–46%** of
phones, while a deliberate mispronunciation moved the error by **11%**. The
noise is larger than the signal.

### Approach 2 — Goodness of Pronunciation

Never let the model choose what was said. Force-align the *expected* phones to
the audio and score each by how confident the model is that it is really there.
This is the established method and it is materially better:

- False positives halved, to **17% and 19%**
- Threshold sweep over both error types (`eval_gop.py`), best operating point:
  **2 of 5 planted errors caught, at 15% false positives — 1.2 false
  accusations per reading.** Settings that bring false positives near zero
  detect nothing.
- At utterance level, a 1–2 word mispronunciation moves the score by **0.12**,
  while two native speakers differ by **1.53**. Roughly 1:13 signal to noise.

The implementation is sound, and that was verified rather than assumed: scoring
one recording against a *completely different* transcript separates cleanly
(**−1.97 vs −5.93, a gap of +3.97**). GOP works. It simply lacks the resolution
to judge one word against ordinary speaker variation.

## Why it is not shipped

A detector that catches 40% of errors while telling a child they mispronounced
more than one correctly-read word per reading makes the product worse. For a
learner, being wrongly corrected is more damaging than an error slipping past —
it teaches them to distrust the feedback.

The transcript scorer already reports mispronunciations honestly within its
limits, and says plainly what those limits are.

## Where to take this next

Ordered by payoff per unit of work. The first two are cheap enough to try in
an afternoon on top of what is already in this directory.

### 1. Ask a two-way question instead of an open one (cheapest, untried)

This is the most promising idea and it was not tested. Everything above asks
an open question — *was this word pronounced correctly?* — and answering it
needs an absolute notion of correct that the models do not have.

The errors children actually make are not open. They are a small, known set of
confusions: aspiration dropped (`खाना` → `काना`), aspiration added, retroflex
for dental (`थके` → `ठके`), voicing swapped. So ask the two-way question:

```
gop(खाना's phones) vs gop(काना's phones)   on the same audio
```

Both sides run through the same speaker, the same microphone, the same room
and the same model bias, so the offset that ruined the absolute score cancels
almost entirely. It also turns a hard problem (open scoring) into an easy one
(pick the likelier of two known candidates).

Everything needed is already here: `gop.gop_per_phone` takes any target
sequence, and `phones.py` knows which contrasts matter. The work is generating
the confusion candidate for each word and comparing. Worth measuring against
the same `eval_gop.py` set before anything else is attempted.

### 2. Score words, not readings — and only a few of them

Nothing needs to judge every word. `main.py` already accepts
`reading_vocabulary`; a passage can nominate three or four target words. A
per-word threshold, calibrated from many children saying *that specific word*,
is a far easier thing to get right than one global threshold that must hold
across every word in Hindi. It also cuts the false-accusation rate directly,
because 90% of the words are never judged at all.

Pair this with the two-way test above and the target words can be chosen for
the contrast they teach.

### 3. Collect labelled data — the thing that actually unblocks everything

No threshold here can be tuned honestly, because there is no ground truth.
Every number in this document rests on synthesised errors and FLEURS
negatives, which is why the conclusion is "not shippable" rather than a
tuned operating point.

What would change that: store each attempt's audio alongside its
`word_analysis`, and have a Hindi teacher mark a few hundred words as
correct or not. Even 200 labelled words makes the difference between guessing
and measuring — it is the missing Hindi `speechocean762`, at the smallest
useful size.

This needs parental consent and a retention policy before a single recording
is kept. Treat that as part of the work, not a formality.

### 4. Buy the capability

**Azure Pronunciation Assessment** does real phoneme scoring with `hi-IN`
support. Blueprint §04 rejected it over the card-at-signup requirement; that
requirement is now the only thing standing between this project and working
pronunciation scoring. The F0 tier does not auto-bill.

Shape it as a third verdict source behind a flag, not a replacement: the
rubric in `scoring.py` still owns coverage, pace and fluency, and only the
*pronunciation* judgement is delegated. That keeps the service working when
the key is absent, which is also what makes it testable without one.

### 5. Fix the model's systematic offset

The phone recogniser is zero-shot on Hindi and mis-realises Hindi phones in a
*consistent* way — it drops aspiration on correct audio. That is a calibration
error, not a capability limit, and calibration errors are learnable.

Adapting `facebook/wav2vec2-lv-60-espeak-cv-ft` on Hindi speech with espeak
labels as targets (IndicTTS, Shrutilipi, FLEURS) should remove most of the
33–44% false positives without needing any mispronunciation labels at all —
the training signal is just "make your phones agree with espeak's on audio
that is known to be correct". This is the single change that would make
absolute scoring viable, and it is a real training job rather than an
afternoon.

### 6. Change what the feedback costs

A softer product contract makes accuracy matter less. Instead of telling a
child they got a word wrong, play the TTS of the word next to their own
recording and let them hear the difference themselves. A wrong suggestion then
costs a few seconds of listening rather than a false accusation, which changes
the accuracy the detector has to reach before it is worth shipping.

Related: a word that scores low across *many* readings by the same child is
signal; a word that scores low once is noise. Longitudinal aggregation is free
once attempts are stored, and it is strictly more reliable than any one-shot
judgement.

### Dead ends — do not spend time here again

- **Free-decoding phones and comparing strings.** Measured at 33–44% and
  70–77% false positives. The noise between two correct speakers is larger
  than the difference a real mispronunciation makes. No threshold fixes this.
- **Allosaurus**, including with the Hindi phone inventory restriction. Worse
  than wav2vec2 on this task by a wide margin.
- **Tuning the GOP thresholds further.** `eval_gop.py` sweeps them; the whole
  frontier was examined. Settings with acceptable false positives detect
  nothing, and settings that detect anything accuse a correctly-read word more
  than once per reading.
- **Hunting for an off-the-shelf Hindi phoneme model.** Hugging Face and GitHub
  were searched; none exists publicly. Check again before assuming it is still
  true, but do not expect to find one.

## Reproducing

```
pip install -r requirements-phonemes.txt
python eval_gop.py <scratch-dir-with-fleurs-samples>
python -m pytest test_pronunciation.py
```
