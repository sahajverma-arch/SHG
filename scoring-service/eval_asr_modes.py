"""Does any ASR here report what was *said*, or only what was *meant*?

    python eval_asr_modes.py <scratch-dir>
    python eval_asr_modes.py <scratch-dir> --skip-whisper

This is the measurement behind sarvam_asr.py, and it decides whether the
pronunciation tier of the 20-point rubric can ever be diagnostic.

**The question.** `scoring.py` only ever sees a transcript, so a word is scored
`mispronounced` only when the transcript differs from the passage. Whisper is a
language model as much as an acoustic one: hand it a child saying रभीवार where
रविवार belongs and it writes रविवार, because that is the word that fits. The
transcript says correct, the score says correct, and the mistake is invisible.
No rubric design recovers it -- the information was destroyed before scoring
began. Sarvam documents a `verbatim` mode that is not supposed to tidy its
output. This checks whether that is true of *mispronunciations* or only of "um".

**The design.** Each of the 20 minimal pairs in eval_contrastive.py names a
word and a plausible child error one phoneme away (खाना/काना -- aspiration, the
contrast Hindi actually uses). For each pair, Bulbul renders two clips:

    correct        "मेज़ पर खाना रखा है"      <- the real word
    mispronounced  "मेज़ पर काना रखा है"      <- the child's error

Three measurements, and the third is what makes the first two mean anything:

    1. Devanagari modes (transcribe, verbatim) and Whisper. `caught` = the
       rival word reached the transcript. `repaired` = the real word did, i.e.
       the confound. `clean` = correct audio still transcribed correctly, which
       an engine transcribing noise would fail.

    2. Romanised mode (translit). Scored differently and on purpose: there is
       no roman lexicon to check a rival against, so this asks only whether the
       two clips produced *different* tokens at the changed word. That is
       weaker than naming the error -- but it is all scoring needs, since the
       comparison is against expected text either way.

    3. The noise floor. Bulbul runs at temperature 0.6, so the same sentence
       rendered twice is not the same audio. Without this control, measurement
       2 is unreadable: an engine whose output wobbles at random would look
       sensitive. Re-rendering each correct sentence and diffing take 1 against
       take 2 gives the rate of differences that mean nothing.

**Result on 2026-08-26** (see docs/phoneme-scoring.md):

    whisper (local)      5/20 caught   6/20 repaired   17/20 clean
    sarvam transcribe    4/20 caught  15/20 repaired   20/20 clean
    sarvam verbatim      5/20 caught  14/20 repaired   20/20 clean
    sarvam translit     11/20 distinct at the target word, 1/20 noise

`verbatim` did not help -- 5 against 4 is noise at n=20. The interesting line
is the last one, and the reason it differs is the finding: the acoustic model
*did* hear the mispronunciation (janda/jhanda, dol/dhol, khaam/kaam), and it
was the conversion to Devanagari that snapped it back to a real word. The
information is not lost in the microphone. It is lost in the spelling.

**What this cannot tell you.** Bulbul saying काना produces a *clean* काना; a
six-year-old attempting खाना produces something smeared between the two. This
is the easy version, and the asymmetry runs one way only:

    fails here     -> certainly fails on real children
    succeeds here  -> necessary, nowhere near sufficient

Nothing here licenses shipping a pronunciation score off translit. It licenses
the next step, which is recordings of actual children -- the labelled data
docs/phoneme-scoring.md names as the blocker on every approach tried so far.

Costs about Rs.8 the first time and nothing after: clips are cached in the
scratch directory.
"""

import argparse
import difflib
import json
import os
import re
import shutil
import sys
import unicodedata
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("scratch", help="directory for rendered clips and results")
    parser.add_argument(
        "--skip-whisper",
        action="store_true",
        help="Sarvam modes only; Whisper loads a GPU model and is the slow part",
    )
    args = parser.parse_args()

    scratch = Path(args.scratch)
    clips = scratch / "asr-modes"
    clips.mkdir(parents=True, exist_ok=True)
    # Test fixtures go in the scratch tree, never the app's cache: these
    # sentences are deliberately misspelt and must never be served to a child.
    os.environ["SARVAM_TTS_CACHE_DIR"] = str(clips)

    import sarvam_asr
    import sarvam_tts
    from eval_contrastive import PAIRS

    if sarvam_tts.api_key() is None:
        print("No SARVAM_API_KEY. Put it in scoring-service/.env (gitignored).")
        return 1

    engines = build_engines(skip_whisper=args.skip_whisper)
    devanagari, romanised, control = [], [], []

    for index, (sentence, correct, rival) in enumerate(PAIRS):
        if correct not in sentence:
            print(f"  skip {index}: {correct!r} not in {sentence!r}")
            continue
        wrong = sentence.replace(correct, rival)

        right_clip = render(sarvam_tts, sentence)
        wrong_clip = render(sarvam_tts, wrong)
        if right_clip is None or wrong_clip is None:
            print(f"  render FAILED for pair {index}")
            continue

        for name, run in engines.items():
            heard_wrong, heard_right = run(wrong_clip), run(right_clip)
            devanagari.append(
                {
                    "pair": index,
                    "engine": name,
                    "correct": correct,
                    "rival": rival,
                    "caught": says(heard_wrong, rival) and not says(heard_wrong, correct),
                    "repaired": says(heard_wrong, correct),
                    "clean": says(heard_right, correct),
                    "heard_wrong": heard_wrong,
                    "heard_right": heard_right,
                }
            )

        # 2. Romanised: did the changed word come out as a different token?
        wrong_roman = roman(sarvam_asr, wrong_clip)
        right_roman = roman(sarvam_asr, right_clip)
        change = first_change(right_roman, wrong_roman)
        romanised.append(
            {
                "pair": index,
                "correct": correct,
                "rival": rival,
                "distinct": change is not None,
                "was": change[0] if change else "",
                "became": change[1] if change else "",
            }
        )

        # 3. The control: the same sentence, rendered again.
        second = second_take(sarvam_tts, sentence, right_clip)
        if second is not None:
            wobble = first_change(right_roman, roman(sarvam_asr, second))
            control.append(
                {
                    "pair": index,
                    "correct": correct,
                    "differs": wobble is not None,
                    "was": wobble[0] if wobble else "",
                    "became": wobble[1] if wobble else "",
                }
            )

        print(f"  pair {index + 1:>2}/{len(PAIRS)}  {correct} vs {rival}")

    if not devanagari:
        print("Nothing measured.")
        return 1

    report(devanagari, romanised, control, engines)
    out = scratch / "asr-modes.json"
    out.write_text(
        json.dumps(
            {"devanagari": devanagari, "romanised": romanised, "control": control},
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\nfull transcripts: {out}")
    return 0


def render(sarvam_tts, text: str) -> Path | None:
    path = sarvam_tts.cached_path(text, slow=False)
    if path.is_file():
        return path
    return path if sarvam_tts.synthesise(text, slow=False) is not None else None


def second_take(sarvam_tts, text: str, first: Path) -> Path | None:
    """A second render of the same text, kept beside the first.

    Bulbul is stochastic, so this is genuinely different audio for identical
    input -- which is exactly what the noise floor needs. `synthesise` writes to
    the one cache path, so the first take is held aside and put back.
    """
    take2 = first.with_name(first.stem + "-take2.wav")
    if take2.is_file():
        return take2
    keep = first.read_bytes()
    try:
        first.unlink()
        if sarvam_tts.synthesise(text, slow=False) is None:
            return None
        shutil.move(str(first), str(take2))
        return take2
    finally:
        if not first.is_file():
            first.write_bytes(keep)


def build_engines(skip_whisper: bool) -> dict:
    import sarvam_asr

    engines = {}
    if not skip_whisper:
        import main as service

        # The production decode, not a re-tuned one: what /score would have
        # done with this audio is what has to be measured.
        run_whisper = service.get_asr()

        def whisper(path: Path) -> str:
            wav, rate = service.load_audio(path.read_bytes())
            return run_whisper(wav, rate).strip()

        engines["whisper (local)"] = whisper

    for mode in ("transcribe", "verbatim"):
        def sarvam(path: Path, mode=mode) -> str:
            # Deliberately not caught. A failed call swallowed into "" would be
            # scored as "the engine did not catch the error" and "the engine
            # got the correct audio wrong" -- a rate limit would quietly become
            # a worse result, which is how a measurement turns into fiction.
            # The first run of this hit 429s and reported 15/20 clean instead
            # of 20/20 before the retries in sarvam_asr existed.
            return sarvam_asr.transcribe_file(path, mode=mode)

        engines[f"sarvam {mode}"] = sarvam
    return engines


def roman(sarvam_asr, path: Path) -> list[str]:
    text = sarvam_asr.transcribe_file(path, mode="translit")
    return [w for w in re.sub(r"[^\w\s]", " ", text.lower()).split() if w]


def first_change(before: list[str], after: list[str]) -> tuple[str, str] | None:
    """The first token that differs between two romanised readings."""
    if not before or not after:
        return None
    for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(None, before, after).get_opcodes():
        if tag != "equal":
            return (" ".join(before[i1:i2]) or "-", " ".join(after[j1:j2]) or "-")
    return None


def says(transcript: str, word: str) -> bool:
    """Is `word` present in `transcript`, allowing punctuation and spacing?

    Deliberately NOT the fuzzy matcher `scoring.py` uses. The question is
    whether two words one phoneme apart stayed distinct, so a matcher that
    tolerates near-misses would answer it by assumption.
    """
    norm = lambda s: unicodedata.normalize("NFC", s)
    cleaned = re.sub(r"[।॥,.!?;:\"'()\[\]{}‌‍]", " ", norm(transcript))
    return norm(word) in cleaned.split() or norm(word) in cleaned


def report(devanagari, romanised, control, engines) -> None:
    print()
    print("=" * 76)
    print("1. Devanagari output: did the error survive into the transcript?")
    print("=" * 76)
    print(f"{'engine':20} {'caught':>9} {'repaired':>10} {'clean':>8}  verdict")
    print("-" * 76)
    for name in engines:
        mine = [r for r in devanagari if r["engine"] == name]
        if not mine:
            continue
        n = len(mine)
        caught = sum(r["caught"] for r in mine)
        repaired = sum(r["repaired"] for r in mine)
        clean = sum(r["clean"] for r in mine)
        if clean < n * 0.6:
            verdict = "unusable - wrong on correct audio too"
        elif caught >= n * 0.8:
            verdict = "keeps the distinction"
        elif repaired >= n * 0.5:
            verdict = "repairs the error away"
        else:
            verdict = "mixed"
        print(f"{name:20} {caught:>5}/{n:<3} {repaired:>6}/{n:<3} {clean:>4}/{n:<3}  {verdict}")

    print()
    print("=" * 76)
    print("2. Romanised output vs 3. the noise floor")
    print("=" * 76)
    noisy = {r["pair"] for r in control if r["differs"]}
    # A romanised difference is only evidence if the same sentence rendered
    # twice did NOT also differ. Without this subtraction the number is a
    # measure of Bulbul's temperature, not of anything the ASR heard.
    signal = [r for r in romanised if r["distinct"] and r["pair"] not in noisy]
    confounded = [r for r in romanised if r["distinct"] and r["pair"] in noisy]
    n = len(romanised)
    print(f"{'said':>8} {'expected':>9}  {'became':>16} {'was':>16}")
    print("-" * 76)
    for r in romanised:
        if not r["distinct"]:
            continue
        flag = "  <- also wobbled on a repeat render" if r["pair"] in noisy else ""
        print(f"{r['rival']:>8} {r['correct']:>9}  {r['became']:>16} {r['was']:>16}{flag}")
    print("-" * 76)
    print(f"distinct romanised token   : {len(signal) + len(confounded):>2}/{n}")
    print(f"  minus repeat-render noise: {len(confounded):>2}")
    print(f"  net signal               : {len(signal):>2}/{n}")
    if control:
        print(f"noise floor (same text twice): {len(noisy)}/{len(control)} anywhere in the sentence")

    print()
    print("caught   = mispronounced clip transcribed with the rival word")
    print("repaired = mispronounced clip transcribed with the REAL word (the confound)")
    print("clean    = correct clip still transcribed correctly")
    print()
    print("Bulbul saying the rival word is a CLEAN rival; a child attempting the")
    print("real word is smeared between the two. This is the easy version, and")
    print("failing it is conclusive while passing it is not.")


if __name__ == "__main__":
    sys.exit(main())
