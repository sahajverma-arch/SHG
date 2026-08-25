"""Evaluate GOP mispronunciation detection, and tune its thresholds.

Two sets, because tuning against either one alone produces a useless detector:

  negatives  correct native readings (FLEURS). Every flag here is a child being
             told they got a word wrong when they did not.
  positives  synthesised readings where specific words carry a real Hindi
             learner error — aspiration dropped or added, retroflex/dental
             swapped. Those words, and ideally only those, should be flagged.

Run directly to sweep thresholds and print the trade-off.
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

# Real Hindi learner errors: aspiration lost or added, dental/retroflex swapped.
POSITIVES = [
    ("सूरज पूरब से निकलता है और पश्चिम में डूब जाता है",
     "सूरज फूरब से निकलता है और पश्चिम में डूब चाता है", {"पूरब", "जाता"}),
    ("मुझे मैदान में खेलना बहुत अच्छा लगता है",
     "मुझे मैदान में खेलना बहुत अच्छा लकता है", {"लगता"}),
    ("दिन में आसमान नीला दिखता है और तारे चमकते हैं",
     "दिन में आसमान नीला दिखता है और तारे चमकदे हैं", {"चमकते"}),
    ("हमारे गाँव के पास एक छोटी नदी बहती है",
     "हमारे गाँव के पास एक छोटी नदी पहती है", {"बहती"}),
]


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


def word_gops(processor, model, vocab, blank, words, wav):
    """Mean GOP per word, or None where the word is too short to judge.

    Targets go through `phones.word_slots`, which spells each expected sound
    the way this recogniser actually writes it. Scoring espeak's `kʰ` directly
    against a model that only ever emits `kh` or `k`+`h` made every aspirated
    phone an automatic error, and aspiration is most of what Hindi reading
    practice is checking.
    """
    inputs = processor(wav, sampling_rate=16000, return_tensors="pt")
    with torch.no_grad():
        logits = model(inputs.input_values).logits[0]
    log_probs = torch.log_softmax(logits, dim=-1).numpy().astype(np.float64)

    per_word = [
        phones.word_slots(pronunciation.expected_phones(w), vocab) for w in words
    ]
    flat = [[vocab[s] for s in slot] for slots in per_word for slot in slots]
    scores = gop.gop_per_slot(log_probs, flat, blank)

    grouped, cursor = [], 0
    for slots in per_word:
        grouped.append(scores[cursor : cursor + len(slots)])
        cursor += len(slots)
    return grouped


def main():
    processor, model, vocab, blank = build()
    from gtts import gTTS

    cache = os.path.join(SCRATCH, "eval")
    os.makedirs(cache, exist_ok=True)

    samples = []  # (label, words, grouped_scores, corrupted_set)

    meta = json.load(io.open(os.path.join(SCRATCH, "hindi", "meta.json"), encoding="utf-8"))
    for i in (0, 1):
        words = [w.strip(DANDA + ",.") for w in meta[i]["text"].split() if w.strip(DANDA + ",.")]
        wav = load_audio(os.path.join(SCRATCH, "hindi", f"sample{i}.wav"))
        samples.append((f"negative/fleurs{i}", words, word_gops(processor, model, vocab, blank, words, wav), set()))

    for index, (correct, spoken, corrupted) in enumerate(POSITIVES):
        path = os.path.join(cache, f"pos{index}.mp3")
        if not os.path.exists(path):
            gTTS(spoken, lang="hi", slow=False).save(path)
        words = correct.split()
        wav = load_audio(path)
        samples.append((f"positive/{index}", words, word_gops(processor, model, vocab, blank, words, wav), corrupted))

    # Also treat a clean synthesis of each positive passage as a negative, so
    # the sweep sees the same voice reading correctly.
    for index, (correct, _spoken, _c) in enumerate(POSITIVES):
        path = os.path.join(cache, f"clean{index}.mp3")
        if not os.path.exists(path):
            gTTS(correct, lang="hi", slow=False).save(path)
        words = correct.split()
        wav = load_audio(path)
        samples.append((f"negative/clean{index}", words, word_gops(processor, model, vocab, blank, words, wav), set()))

    print(f"{'min_ph':>6} {'z':>5} {'floor':>6} | {'caught':>7} {'missed':>6} | {'false pos':>9} {'per read':>9}")
    print("-" * 66)
    best = None
    for min_phones in (2, 3, 4):
        for z in (0.8, 1.0, 1.25, 1.5, 1.75, 2.0):
            for floor in (-1.0, -1.5, -2.0, -2.5):
                caught = missed = false_pos = negatives_words = 0
                reads = 0
                for label, words, grouped, corrupted in samples:
                    flags, means, _ = gop.flag_words(
                        grouped, min_phones=min_phones, z_threshold=z, floor=floor
                    )
                    reads += 1
                    for j, word in enumerate(words):
                        if means[j] is None:
                            continue
                        if word in corrupted:
                            caught += bool(flags[j])
                            missed += not flags[j]
                        else:
                            negatives_words += 1
                            false_pos += bool(flags[j])
                fp_rate = 100 * false_pos / max(negatives_words, 1)
                per_read = false_pos / max(reads, 1)
                recall = caught / max(caught + missed, 1)
                # Prefer high recall, then few false alarms per reading.
                score = recall - 0.12 * per_read
                if best is None or score > best[0]:
                    best = (score, min_phones, z, floor, caught, missed, fp_rate, per_read)
                if z in (1.0, 1.5, 2.0) and floor in (-1.5, -2.5):
                    print(f"{min_phones:>6} {z:>5} {floor:>6} | {caught:>7} {missed:>6} | {fp_rate:>8.0f}% {per_read:>9.2f}")

    print()
    _, min_phones, z, floor, caught, missed, fp_rate, per_read = best
    print(f"BEST: min_phones={min_phones} z={z} floor={floor}")
    print(f"  caught {caught}/{caught+missed} corrupted words, {fp_rate:.0f}% false positives ({per_read:.2f} per reading)")


if __name__ == "__main__":
    main()
