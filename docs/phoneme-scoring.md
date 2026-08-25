# Phoneme-level pronunciation scoring for Hindi

**Status: built, measured, and not shipped.** The machinery works and is
tested; the available models are not accurate enough to judge a child's
pronunciation word by word. This document records what was tried and what the
numbers were, so the work is not repeated blindly.

**Picking this up?** Read [Approach 3](#approach-3--the-two-way-question) first —
it found a bug that invalidated part of the earlier measurements, and it is the
only approach with a clear path forward. Then
[Where to take this next](#where-to-take-this-next), which ends with the dead
ends worth skipping.

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
| `phones.py` | phone comparison cost, tuned so Hindi's aspiration and dental/retroflex contrasts cost full price while notation differences cost nothing; plus `model_slots`, how this recogniser actually spells Hindi |
| `pronunciation.py` | espeak Hindi G2P (expected sounds) + wav2vec2 phone recognition (heard sounds) |
| `gop.py` | CTC forced alignment, Goodness of Pronunciation, whole-utterance `alignment_score`, and pooled-slot scoring |
| `eval_gop.py` | the one-sided evaluation harness — run it to reproduce |
| `eval_contrastive.py` | the two-way evaluation (Approach 3) |
| `diag_aspiration.py` | the aspiration diagnostic — run this before trusting any new phone model with Hindi |
| `test_pronunciation.py` | 39 tests over the parts that are correct |

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

### Approach 3 — the two-way question

*Added after the first version of this document, which listed this as the most
promising untried idea. It has now been tried: `eval_contrastive.py`.*

Instead of *was this word correct?*, ask *does this audio fit `खाना` or `काना`
better?* Both sides run through the same speaker, microphone and model, so the
offset that ruined the absolute score should cancel.

Building it turned up a real bug in everything above.

**The recogniser never emits espeak's aspirated symbols.** `kʰ`, `bʰ`, `pʰ` and
the rest are all in its 392-symbol vocabulary. Across **545 non-blank frames of
Hindi — real human speech included — it produced an aspirated stop exactly zero
times.** It writes those sounds as an ASCII digraph where it has one (`kh`,
`th`, `ph`), otherwise as the plain stop followed by a separate `h`, and फ
usually as `f`. It decodes भाई as `b h aɪ i` and फल as `f a l`.

So every aspirated phone in every passage was being scored against a symbol the
model does not use — an automatic error on precisely the contrast Hindi reading
practice exists to teach. When the audio genuinely contained an aspirated stop,
the plain counterpart scored better **29 times out of 29** (GOP −8.89 vs −2.30).

`phones.model_slots` fixes this, mapping each expected sound onto the symbols
this model actually produces. The result is worth stating plainly:

| | caught | false positives |
|---|---|---|
| One-sided GOP, before the notation fix | 2 of 5 | 15% (1.2 per reading) |
| One-sided GOP, after the notation fix | 2 of 5 | **17% (1.1 per reading)** |

**The fix changed nothing.** That is the useful finding: the problem was never
the notation, it is the question. Asking "are the expected sounds present?"
cannot separate खाना from काना, because the plain reading is never *ruled out*
by a missing target — it only has to absorb the leftover frames, and a per-phone
mean waters that evidence down across the rest of the word.

Asking which of two specific words fits better, scored over the whole utterance
(`gop.alignment_score`), does separate them:

- On 20 minimal pairs, the correct reading scores higher than the wrong one
  **20 times out of 20**, median gap **+7.87**. The information is unambiguously
  there.
- Uncalibrated, though, the sign is right only **65%** of the time on correct
  readings and 80% on wrong ones — each contrast sits at its own offset.
- Calibrating per contrast class, leaving each item out: **17 of 17 errors
  caught, but 47% of correct readings falsely accused.**
- Worse, thresholds calibrated on synthesised speech do not transfer to real
  speech at all. On FLEURS the same thresholds flag 14 of 16 correctly-read
  words in one class, 3 of 3 in another.
- Normalising within a reading instead (the trick `flag_words` uses, which
  cancels a speaker's own offset) trades it back the other way: **1 of 5 caught
  at 7% false positives** — cleaner than the incumbent, but catching less.

So the two-way question is genuinely better than the open one, and it is still
not deployable. The discriminative signal is real and strong; it sits on top of
a per-speaker, per-word offset of the same size, and there is no labelled data
to learn that offset from. Which is the same wall as before, now located
precisely.

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

### 1. Finish the two-way question — it has signal, it needs an offset

Approach 3 above is now built and measured. Start here, because the hard part
is done and the remaining part is well defined.

What is established: on the same audio, the correct word beats a specific wrong
word 20 times out of 20 with a median gap of +7.87. What blocks it: each
contrast and each speaker sits at its own offset, and no threshold transfers
between them.

Three concrete things to try, in order:

- **A better rival than one rule-generated guess.** `eval_contrastive.rival_for`
  invents a single rival per word by the first applicable swap. Score *several*
  rivals per word and take the best-fitting one — that is both closer to what a
  child actually does and a stronger signal, since the margin to the *nearest*
  wrong word is what matters.
- **Estimate the speaker's offset from the reading itself.** Per-reading
  normalisation already gets false positives down to 7%. It currently needs at
  least four words in a contrast class before it will judge any of them, which
  is why most words in a short passage are skipped entirely. Pooling classes
  with a per-class shift, rather than normalising each separately, should judge
  far more words from the same evidence.
- **Only then, a threshold.** Which needs (3).

### 1b. Do not re-derive the notation bug

`phones.model_slots` exists because this recogniser does not spell Hindi the way
espeak does. Anything new that maps expected phones to model vocabulary ids must
go through it, or it will silently reproduce the "zero aspirated frames in 545"
failure and every measurement on top will be wrong in the same invisible way.

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

Approach 3 measured how consistent: **zero aspirated stops in 545 non-blank
frames**, and the plain counterpart winning 29 times out of 29 on audio that
genuinely contained the aspirated sound. `phones.model_slots` works around the
notation, but the model's probability mass is still in the wrong place, and
that is what caps every method built on it.

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
python eval_gop.py <scratch-dir-with-fleurs-samples>          # one-sided
python eval_contrastive.py <scratch-dir-with-fleurs-samples>  # two-way
python diag_aspiration.py <scratch-dir-with-fleurs-samples>   # is the model deaf to aspiration?
python -m pytest test_pronunciation.py
```

The scratch directory needs `hindi/meta.json` and `hindi/sample0.wav`,
`hindi/sample1.wav` — two FLEURS Hindi clips and their transcripts. Both
evaluations synthesise their own error cases with gTTS and cache them, so the
first run needs a network connection and later runs do not.

Set `PYTHONIOENCODING=utf-8` on Windows, or the Devanagari in the reports will
crash the console rather than the script.
