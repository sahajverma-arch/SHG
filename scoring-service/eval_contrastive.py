"""Ask a two-way question instead of an open one.

The shipped GOP scorer asks "how confident is the model that the sounds this
word calls for are present?", and answers on an absolute scale. That scale
moves with the speaker, the microphone and the room, which is why a threshold
over it flags 17-19% of correctly read words (see docs/phoneme-scoring.md).

This asks instead: given this same audio, does the correct word or one specific
wrong word fit better? Both sides are measured on the same frames through the
same model, so the offset a quiet child or a cheap microphone introduces
applies to both and cancels in the subtraction.

Two separate defects had to be fixed before the question could even be asked,
and this script reports the effect of each:

1. NOTATION - fixed in `phones.model_slots`. espeak writes Hindi's aspirated
   stops as `kʰ`, `bʰ`, `pʰ`. This recogniser has those symbols and never emits
   them: across 545 non-blank frames of Hindi it produced an aspirated stop
   zero times. Scoring `kʰ` against it made every aspirated phone an automatic
   error. On its own, though, this fix changes almost nothing - `eval_gop.py`
   still lands at 2 of 5 caught for 17% false positives - because of:

2. THE QUESTION ITSELF. Fixing the notation does not rescue the one-sided
   scorer: `eval_gop.py` still lands at 2 of 5 caught for 17% false positives,
   the same place it was before. Asking "are the expected sounds present?"
   cannot separate खाना from काना, because the plain reading is never ruled out
   by a missing target - it only has to absorb the leftover frames, and a
   per-phone mean waters that down across the rest of the word. Asking which of
   two specific words fits better, scored over the whole utterance, does
   separate them: 20 of 20 below.

Everything here is measured on synthesised speech. It says the method has
signal; it does not say what it does on a seven-year-old, and nothing in this
repository can say that until labelled child recordings exist.
"""

import io
import json
import os
import sys

import numpy as np
import torch
from pydub import AudioSegment

import gop
import phones
import pronunciation

SCRATCH = sys.argv[1] if len(sys.argv) > 1 else "."
DANDA = "।"

# A real Hindi word and the wrong word a learner produces when one contrast
# slips: aspiration dropped or added, or a dental stop made retroflex. Each
# sits in a carrier sentence so the synthesiser produces connected speech
# rather than an isolated citation form.
PAIRS = [
    ("मेज़ पर खाना रखा है", "खाना", "काना"),
    ("मेरा घर बहुत सुंदर है", "घर", "गर"),
    ("मेरा भाई स्कूल जाता है", "भाई", "बाई"),
    ("मुझे कपड़े धोना है", "धोना", "दोना"),
    ("यह फल बहुत मीठा है", "फल", "पल"),
    ("थाली में चावल रखा है", "थाली", "ताली"),
    ("बारिश में छाता ले जाओ", "छाता", "चाता"),
    ("तिरंगा झंडा ऊपर है", "झंडा", "जंडा"),
    ("यह पानी बहुत ठंडा है", "ठंडा", "टंडा"),
    ("गाँव में ढोल बजता है", "ढोल", "डोल"),
    ("मुझे थोड़ा पानी चाहिए", "पानी", "फानी"),
    ("यह काम बहुत कठिन है", "काम", "खाम"),
    ("आसमान में तारा चमकता है", "तारा", "टारा"),
    ("आज का दिन अच्छा है", "दिन", "डिन"),
    ("यह साल बीत गया", "साल", "शाल"),
    ("वह मीठा गाना गाती है", "गाना", "घाना"),
    ("मुझे गरम चाय पसंद है", "चाय", "छाय"),
    ("नदी का जल साफ है", "जल", "झल"),
    ("उसके बाल बहुत लंबे हैं", "बाल", "भाल"),
    ("मुझे वह किताब देना", "देना", "धेना"),
]

_SWAPS = [
    ("ख", "क"), ("घ", "ग"), ("छ", "च"), ("झ", "ज"), ("ठ", "ट"),
    ("ढ", "ड"), ("थ", "त"), ("ध", "द"), ("फ", "प"), ("भ", "ब"),
    ("क", "ख"), ("ग", "घ"), ("च", "छ"), ("ज", "झ"), ("ट", "ठ"),
    ("ड", "ढ"), ("त", "थ"), ("द", "ध"), ("प", "फ"), ("ब", "भ"),
    ("ट", "त"), ("ड", "द"), ("त", "ट"), ("द", "ड"), ("स", "श"),
]

ASPIRATED = {"kʰ", "ɡʰ", "bʰ", "dʰ", "pʰ", "tʰ", "cʰ", "ɟʰ", "ʈʰ", "ɖʰ"}
RETROFLEX = {"ʈ", "ɖ", "ɳ", "ʈʰ", "ɖʰ", "ʂ"}


def rival_for(word: str) -> str | None:
    for src, dst in _SWAPS:
        if src in word:
            candidate = word.replace(src, dst, 1)
            if candidate != word:
                return candidate
    return None


def contrast_class(correct: str, rival: str) -> str:
    """Which kind of contrast separates these two words.

    The likelihood ratio carries a different constant offset for each kind, so
    a usable detector needs one calibration per class rather than one global
    threshold.
    """
    a = pronunciation.expected_phones(correct)
    b = pronunciation.expected_phones(rival)
    for x, y in zip(a, b):
        if x == y:
            continue
        if x in ASPIRATED and y not in ASPIRATED:
            return "aspiration dropped"
        if y in ASPIRATED and x not in ASPIRATED:
            return "aspiration added"
        if (x in RETROFLEX) != (y in RETROFLEX):
            return "place shifted"
        return "other"
    return "other"


def load_audio(path):
    seg = AudioSegment.from_file(path).set_frame_rate(16000).set_channels(1)
    return np.array(seg.get_array_of_samples()).astype(np.float32) / float(
        1 << (8 * seg.sample_width - 1)
    )


def build():
    pronunciation.ensure_espeak()
    from transformers import AutoModelForCTC, AutoProcessor

    processor = AutoProcessor.from_pretrained(pronunciation.PHONE_MODEL_ID)
    model = AutoModelForCTC.from_pretrained(pronunciation.PHONE_MODEL_ID)
    model.eval()
    return processor, model, processor.tokenizer.get_vocab(), model.config.pad_token_id or 0


def frame_log_probs(processor, model, wav):
    inputs = processor(wav, sampling_rate=16000, return_tensors="pt")
    with torch.no_grad():
        logits = model(inputs.input_values).logits[0]
    return torch.log_softmax(logits, dim=-1).numpy().astype(np.float64)


def hypothesis_score(log_probs, words, index, replacement, vocab, blank_id):
    """Whole-utterance alignment log-likelihood for one reading of the passage."""
    spoken = list(words)
    if replacement is not None:
        spoken[index] = replacement
    per_word = [
        phones.word_slots(pronunciation.expected_phones(w), vocab) for w in spoken
    ]
    if not per_word[index]:
        return None
    slot_ids = [[vocab[s] for s in slot] for slots in per_word for slot in slots]
    pooled = gop.pooled_columns(log_probs, slot_ids, blank_id)
    total = gop.alignment_score(pooled, list(range(len(slot_ids))), len(slot_ids))
    return None if np.isnan(total) else total


def margin(log_probs, words, index, rival, vocab, blank_id):
    """Positive means the audio fits the correct word better than the rival."""
    a = hypothesis_score(log_probs, words, index, None, vocab, blank_id)
    b = hypothesis_score(log_probs, words, index, rival, vocab, blank_id)
    return None if a is None or b is None else a - b


def synth(path, text):
    if not os.path.exists(path):
        from gtts import gTTS

        gTTS(text, lang="hi", slow=False).save(path)
    return path


def main():
    processor, model, vocab, blank_id = build()
    cache = os.path.join(SCRATCH, "contrastive")
    os.makedirs(cache, exist_ok=True)

    rows = []
    print(f"{'word':<7} {'rival':<7} {'contrast':<19} {'correct':>9} {'wrong':>9} {'paired':>8}")
    print("-" * 64)
    for i, (sentence, correct, rival) in enumerate(PAIRS):
        words = sentence.split()
        index = words.index(correct)
        wrong_sentence = " ".join(rival if j == index else w for j, w in enumerate(words))

        lp_c = frame_log_probs(processor, model, load_audio(synth(os.path.join(cache, f"c{i}.mp3"), sentence)))
        lp_w = frame_log_probs(processor, model, load_audio(synth(os.path.join(cache, f"w{i}.mp3"), wrong_sentence)))
        m_c = margin(lp_c, words, index, rival, vocab, blank_id)
        m_w = margin(lp_w, words, index, rival, vocab, blank_id)
        kind = contrast_class(correct, rival)
        rows.append((correct, rival, kind, m_c, m_w))
        pair = f"{m_c - m_w:+.2f}" if m_c is not None and m_w is not None else "n/a"
        print(f"{correct:<7} {rival:<7} {kind:<19} "
              f"{m_c:>+9.2f} {m_w:>+9.2f} {pair:>8}")

    good = [r for r in rows if r[3] is not None and r[4] is not None]
    paired = np.array([r[3] - r[4] for r in good])
    print(f"\nPaired separation: the same contrast scores higher on the correct")
    print(f"reading than the wrong one in {int((paired > 0).sum())}/{len(paired)} cases "
          f"(median gap {np.median(paired):+.2f}).")
    raw_c = np.array([r[3] for r in good])
    raw_w = np.array([r[4] for r in good])
    print(f"Raw sign, no calibration: correct readings {100 * float((raw_c > 0).mean()):.0f}% right, "
          f"wrong readings {100 * float((raw_w < 0).mean()):.0f}% right.")

    # A detector needs one threshold per contrast class. Calibrate each class's
    # offset leaving the item under test out, so no item sets its own threshold.
    print("\nLeave-one-out calibrated detector (threshold = median margin of the")
    print("other correct readings in the same contrast class):")
    caught = missed = false_pos = true_neg = 0
    for i, (correct, rival, kind, m_c, m_w) in enumerate(good):
        others = [r[3] for j, r in enumerate(good) if j != i and r[2] == kind]
        if len(others) < 2:
            continue
        threshold = float(np.median(others))
        if m_c < threshold:
            false_pos += 1
        else:
            true_neg += 1
        if m_w < threshold:
            caught += 1
        else:
            missed += 1
    judged = caught + missed
    print(f"  caught {caught}/{judged} genuine mispronunciations")
    print(f"  falsely accused {false_pos}/{false_pos + true_neg} correct readings "
          f"({100 * false_pos / max(false_pos + true_neg, 1):.0f}%)")

    print("\nReal human speech (FLEURS), read correctly, every word given a rival.")
    print("Each flag here is a child told they were wrong when they were not:")
    for sample in (0, 1):
        meta = json.load(io.open(os.path.join(SCRATCH, "hindi", "meta.json"), encoding="utf-8"))
        words = [w.strip(DANDA + ",.") for w in meta[sample]["text"].split() if w.strip(DANDA + ",.")]
        log_probs = frame_log_probs(
            processor, model, load_audio(os.path.join(SCRATCH, "hindi", f"sample{sample}.wav"))
        )
        by_class = {}
        for index, word in enumerate(words):
            rival = rival_for(word)
            if rival is None:
                continue
            expected = pronunciation.expected_phones(word)
            if phones.word_slots(expected, vocab) == phones.word_slots(
                pronunciation.expected_phones(rival), vocab
            ):
                continue
            m = margin(log_probs, words, index, rival, vocab, blank_id)
            if m is None:
                continue
            by_class.setdefault(contrast_class(word, rival), []).append((word, rival, m))

        total = sum(len(v) for v in by_class.values())
        print(f"  sample{sample}: {total} words judged")
        for kind, items in sorted(by_class.items()):
            values = np.array([m for _w, _r, m in items])
            # Same calibration the detector above would use for this class.
            class_rows = [r[3] for r in good if r[2] == kind]
            threshold = float(np.median(class_rows)) if class_rows else 0.0
            flagged = [f"{w}->{r}" for w, r, m in items if m < threshold]
            print(f"    {kind:<19} n={len(items):<3} median {np.median(values):+7.2f}  "
                  f"threshold {threshold:+7.2f}  flagged {len(flagged)}"
                  + (f"  {', '.join(flagged[:5])}" if flagged else ""))

    return processor, model, vocab, blank_id


def reading_margins(log_probs, words, vocab, blank_id):
    """Every word's margin against a rule-generated rival, for one reading."""
    out = []
    for index, word in enumerate(words):
        rival = rival_for(word)
        if rival is None:
            continue
        if phones.word_slots(pronunciation.expected_phones(word), vocab) == phones.word_slots(
            pronunciation.expected_phones(rival), vocab
        ):
            continue
        m = margin(log_probs, words, index, rival, vocab, blank_id)
        if m is not None:
            out.append((word, rival, contrast_class(word, rival), m))
    return out


def normalised_pass(processor, model, vocab, blank_id, z_threshold=1.5):
    """The offset cancels within a reading - does the signal survive?

    A threshold calibrated on one voice does not transfer to another, which is
    what sinks the calibrated detector above. But a child's own offset applies
    to every word they read, so judging each word against the rest of the same
    reading should cancel it - the trick `gop.flag_words` already uses.

    Scored per contrast class, because the classes sit at different offsets.
    """
    from eval_gop import POSITIVES

    cache = os.path.join(SCRATCH, "contrastive")
    samples = []

    meta = json.load(io.open(os.path.join(SCRATCH, "hindi", "meta.json"), encoding="utf-8"))
    for sample in (0, 1):
        words = [w.strip(DANDA + ",.") for w in meta[sample]["text"].split() if w.strip(DANDA + ",.")]
        wav = load_audio(os.path.join(SCRATCH, "hindi", f"sample{sample}.wav"))
        samples.append((f"fleurs{sample} (real, correct)", words, wav, set()))

    for index, (correct, spoken, corrupted) in enumerate(POSITIVES):
        path = os.path.join(cache, f"rp{index}.mp3")
        if not os.path.exists(path):
            from gtts import gTTS

            gTTS(spoken, lang="hi", slow=False).save(path)
        samples.append((f"positive{index} (synth, {len(corrupted)} wrong)",
                        correct.split(), load_audio(path), corrupted))

    print(f"\nPer-reading normalisation, z < -{z_threshold} inside each contrast class:")
    caught = missed = false_pos = clean = 0
    for label, words, wav, corrupted in samples:
        log_probs = frame_log_probs(processor, model, wav)
        rows = reading_margins(log_probs, words, vocab, blank_id)
        by_class = {}
        for word, rival, kind, m in rows:
            by_class.setdefault(kind, []).append((word, rival, m))

        flagged = []
        for kind, items in by_class.items():
            values = np.array([m for _w, _r, m in items])
            if len(values) < 4:
                continue
            median = float(np.median(values))
            mad = float(np.median(np.abs(values - median))) or 1e-6
            spread = 1.4826 * mad
            for (word, _rival, m) in items:
                if (median - m) / spread >= z_threshold:
                    flagged.append(word)

        hits = [w for w in flagged if w in corrupted]
        wrong = [w for w in flagged if w not in corrupted]
        caught += len(set(hits))
        missed += len(corrupted) - len(set(hits))
        false_pos += len(wrong)
        clean += len(rows) - len(wrong)
        note = f"caught {sorted(set(hits))}" if hits else ""
        print(f"  {label:<34} {len(rows):>3} judged  {len(wrong):>2} false  {note}")

    print(f"\n  caught {caught}/{caught + missed} planted errors")
    print(f"  falsely accused {false_pos} words "
          f"({100 * false_pos / max(false_pos + clean, 1):.0f}% of correct words judged)")


if __name__ == "__main__":
    normalised_pass(*main())
