"""Does this phone model represent Hindi aspiration at all?

    python diag_aspiration.py <scratch-dir-with-fleurs-samples>

The contrastive test failed in a very structured way: the plain-consonant
hypothesis won every time, regardless of what was spoken. That points at the
model, not the method. This checks it directly.

It produced the two numbers `phones.py` and `docs/phoneme-scoring.md` both
rest on: across 545 non-blank frames of Hindi — real human speech included —
`facebook/wav2vec2-lv-60-espeak-cv-ft` emitted an aspirated stop **zero**
times, and where the audio genuinely held one, the plain counterpart scored
better **29 times out of 29**.

That is why `phones.model_slots` exists. Re-run this before trusting any new
phone model with Hindi: the same blindness in a different model would be just
as invisible, and every measurement layered on top would be wrong in the same
quiet way.
"""

import collections
import io
import json
import os
import sys

import numpy as np
import torch

import gop
import pronunciation
from eval_contrastive import PAIRS, frame_log_probs, load_audio

SCRATCH = sys.argv[1]
DANDA = "।"

ASPIRATED = ["kʰ", "ɡʰ", "bʰ", "dʰ", "pʰ", "tʰ", "cʰ", "ɟʰ", "ʈʰ", "ɖʰ"]
PLAIN = ["k", "ɡ", "b", "d", "p", "t", "c", "ɟ", "ʈ", "ɖ"]


def main():
    pronunciation.ensure_espeak()
    from transformers import AutoModelForCTC, AutoProcessor

    processor = AutoProcessor.from_pretrained(pronunciation.PHONE_MODEL_ID)
    model = AutoModelForCTC.from_pretrained(pronunciation.PHONE_MODEL_ID)
    model.eval()
    vocab = processor.tokenizer.get_vocab()
    inv = {i: p for p, i in vocab.items()}
    blank = model.config.pad_token_id or 0

    clips = []
    meta = json.load(io.open(os.path.join(SCRATCH, "hindi", "meta.json"), encoding="utf-8"))
    for s in (0, 1):
        words = [w.strip(DANDA + ",.") for w in meta[s]["text"].split() if w.strip(DANDA + ",.")]
        clips.append((f"fleurs{s} (real human)", words,
                      load_audio(os.path.join(SCRATCH, "hindi", f"sample{s}.wav"))))
    cache = os.path.join(SCRATCH, "contrastive")
    for i, (sentence, _c, _r) in enumerate(PAIRS):
        path = os.path.join(cache, f"c{i}.mp3")
        if os.path.exists(path):
            clips.append((f"tts c{i}", sentence.split(), load_audio(path)))

    # 1. How often does the model ever *choose* an aspirated phone?
    print("=" * 68)
    print("1. Argmax decode: how often the model picks an aspirated symbol")
    print("=" * 68)
    counts = collections.Counter()
    total_nonblank = 0
    for name, _words, wav in clips:
        lp = frame_log_probs(processor, model, wav)
        for idx in lp.argmax(axis=1):
            p = inv.get(int(idx), "?")
            if int(idx) == blank:
                continue
            total_nonblank += 1
            counts[p] += 1
    asp_frames = sum(counts[p] for p in ASPIRATED)
    plain_frames = sum(counts[p] for p in PLAIN)
    print(f"non-blank frames across {len(clips)} clips: {total_nonblank}")
    print(f"  frames decoded as an aspirated stop: {asp_frames} "
          f"({100 * asp_frames / max(total_nonblank, 1):.2f}%)")
    print(f"  frames decoded as a plain stop:      {plain_frames} "
          f"({100 * plain_frames / max(total_nonblank, 1):.2f}%)")
    print("  per aspirated symbol:", {p: counts[p] for p in ASPIRATED})

    # 2. When the passage genuinely calls for an aspirated phone, what GOP does
    #    it get, versus the plain phone in the same position?
    print()
    print("=" * 68)
    print("2. GOP of an expected aspirated phone vs its plain counterpart")
    print("   (same audio, same frames - only the hypothesised symbol differs)")
    print("=" * 68)
    pairs_map = dict(zip(ASPIRATED, PLAIN))
    rows = []
    for name, words, wav in clips:
        lp = frame_log_probs(processor, model, wav)
        per_word = [[p for p in pronunciation.expected_phones(w) if p in vocab] for w in words]
        flat = [p for ps in per_word for p in ps]
        ids = [vocab[p] for p in flat]
        spans = gop.forced_align(lp, ids, blank)
        frame_best = lp.max(axis=1)
        for phone, (start, end) in zip(flat, spans):
            if phone not in pairs_map or start < 0:
                continue
            plain = pairs_map[phone]
            if plain not in vocab:
                continue
            g_asp = float((lp[start:end, vocab[phone]] - frame_best[start:end]).mean())
            g_plain = float((lp[start:end, vocab[plain]] - frame_best[start:end]).mean())
            rows.append((phone, plain, g_asp, g_plain))

    if rows:
        by_phone = collections.defaultdict(list)
        for phone, plain, a, b in rows:
            by_phone[phone].append((a, b))
        print(f"{'expected':>9} {'rival':>6} {'n':>4} {'GOP(expected)':>14} {'GOP(plain)':>12} {'plain wins':>11}")
        print("-" * 62)
        for phone in ASPIRATED:
            vals = by_phone.get(phone)
            if not vals:
                continue
            a = np.array([v[0] for v in vals])
            b = np.array([v[1] for v in vals])
            print(f"{phone:>9} {pairs_map[phone]:>6} {len(vals):>4} "
                  f"{a.mean():>14.3f} {b.mean():>12.3f} {100 * float((b > a).mean()):>10.0f}%")
        a = np.array([r[2] for r in rows])
        b = np.array([r[3] for r in rows])
        print("-" * 62)
        print(f"{'ALL':>9} {'':>6} {len(rows):>4} {a.mean():>14.3f} {b.mean():>12.3f} "
              f"{100 * float((b > a).mean()):>10.0f}%")
        print()
        print("Every one of these frames was spoken as the ASPIRATED sound.")
        print("'plain wins' is how often the model still preferred the plain symbol.")


if __name__ == "__main__":
    main()
