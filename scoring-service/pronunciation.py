"""Phoneme-level pronunciation scoring for Hindi.

The transcript scorer in `scoring.py` can only see *which words* a recogniser
decided were said. That is the wrong instrument for pronunciation: an ASR is
built to output real words, so a child who says `काना` gets transcribed as the
real word `खाना` and scores full marks. Blueprint 04 called this out and
settled for it; this module is the part that was missing.

Here the audio goes to a phone recogniser instead of a word recogniser, and
the sounds it heard are compared against the sounds the passage calls for:

    expected   खाना   ->  kʰ aː n aː     (espeak Hindi G2P)
    heard             ->  k  aː n aː     (wav2vec2 phone recogniser)
                            ^ aspiration missing

Both sides speak the same IPA because the recogniser was trained on espeak
phone labels, so the two inventories line up by construction.

This does not replace transcript scoring — that still handles coverage, skipped
words and pace. It replaces the *pronunciation* judgement inside it.
"""

import os
from functools import lru_cache

import numpy as np

import phones

PHONE_MODEL_ID = os.environ.get(
    "PHONE_MODEL_ID", "facebook/wav2vec2-lv-60-espeak-cv-ft"
)
SAMPLE_RATE = 16000

# An expected phone the recogniser never produced is a full error; a spurious
# extra phone is cheaper, because phone recognisers insert freely.
DELETION_COST = 1.0
INSERTION_COST = 0.5

# Share of a word's phones that may be wrong before it counts as mispronounced.
# Below this a word survives ordinary recogniser noise; above it, something was
# genuinely said differently.
WORD_ERROR_THRESHOLD = float(os.environ.get("PHONE_WORD_THRESHOLD", "0.34"))

# Nothing is judged from a word the recogniser barely covered.
MIN_PHONES = 2

_recogniser = None
_g2p = None


# ---------------------------------------------------------------------------
# The two sources of phones
# ---------------------------------------------------------------------------

_espeak_ready = False


def ensure_espeak() -> None:
    """Point phonemizer at the espeak-ng bundled with the Python package.

    Installing espeak system-wide needs administrator rights, which a setup
    script cannot assume; `espeakng-loader` ships the library and its data as
    ordinary package files.

    This has to run before the phone recogniser is loaded, not just before our
    own G2P calls: the Wav2Vec2Phoneme *tokenizer* builds a phonemizer backend
    of its own while initialising, and fails with "espeak not installed" if the
    library has not been registered yet.
    """
    global _espeak_ready
    if _espeak_ready:
        return

    import espeakng_loader
    from phonemizer.backend.espeak.wrapper import EspeakWrapper

    espeakng_loader.make_library_available()
    EspeakWrapper.set_library(espeakng_loader.get_library_path())
    EspeakWrapper.set_data_path(espeakng_loader.get_data_path())
    _espeak_ready = True


def _espeak():
    """espeak-ng Hindi grapheme-to-phoneme."""
    global _g2p
    if _g2p is None:
        ensure_espeak()
        from phonemizer.backend import EspeakBackend

        _g2p = EspeakBackend("hi", preserve_punctuation=False, with_stress=False)
    return _g2p


@lru_cache(maxsize=2048)
def expected_phones(word: str) -> tuple[str, ...]:
    """The sounds a word should be made of."""
    from phonemizer.separator import Separator

    spoken = _espeak().phonemize(
        [word], separator=Separator(phone=" ", word="|"), strip=True
    )
    if not spoken:
        return ()
    return tuple(p for p in spoken[0].replace("|", " ").split() if p)


def _load_recogniser():
    from transformers import AutoModelForCTC, AutoProcessor

    ensure_espeak()
    processor = AutoProcessor.from_pretrained(PHONE_MODEL_ID)
    model = AutoModelForCTC.from_pretrained(PHONE_MODEL_ID)
    model.eval()
    return processor, model


def get_recogniser():
    global _recogniser
    if _recogniser is None:
        _recogniser = _load_recogniser()
    return _recogniser


def heard_phones(wav: np.ndarray, sample_rate: int = SAMPLE_RATE) -> list[str]:
    """The sounds actually present in the audio, as IPA."""
    import torch

    processor, model = get_recogniser()
    inputs = processor(wav, sampling_rate=sample_rate, return_tensors="pt")
    with torch.no_grad():
        logits = model(inputs.input_values).logits
    ids = torch.argmax(logits, dim=-1)
    decoded = processor.batch_decode(ids)[0]
    return [p for p in decoded.split() if p]


# ---------------------------------------------------------------------------
# Alignment
# ---------------------------------------------------------------------------

def align(expected: list[str], heard: list[str]) -> list[tuple[int | None, int | None]]:
    """Least-cost alignment of expected phones to heard phones.

    Returns (expected index, heard index) pairs; `None` on either side marks a
    phone that was missed or inserted.
    """
    n, m = len(expected), len(heard)
    dp = np.zeros((n + 1, m + 1), dtype=np.float64)
    # 0 diagonal, 1 deletion (expected unheard), 2 insertion (extra heard)
    back = np.zeros((n + 1, m + 1), dtype=np.int8)

    for i in range(1, n + 1):
        dp[i][0] = i * DELETION_COST
        back[i][0] = 1
    for j in range(1, m + 1):
        dp[0][j] = j * INSERTION_COST
        back[0][j] = 2

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            diagonal = dp[i - 1][j - 1] + phones.cost(expected[i - 1], heard[j - 1])
            deletion = dp[i - 1][j] + DELETION_COST
            insertion = dp[i][j - 1] + INSERTION_COST
            best = min(diagonal, deletion, insertion)
            dp[i][j] = best
            back[i][j] = 0 if best == diagonal else (1 if best == deletion else 2)

    pairs: list[tuple[int | None, int | None]] = []
    i, j = n, m
    while i > 0 or j > 0:
        move = back[i][j]
        if i > 0 and j > 0 and move == 0:
            pairs.append((i - 1, j - 1))
            i, j = i - 1, j - 1
        elif i > 0 and move == 1:
            pairs.append((i - 1, None))
            i -= 1
        else:
            pairs.append((None, j - 1))
            j -= 1
    pairs.reverse()
    return pairs


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def score_words(passage_words: list[str], wav: np.ndarray, sample_rate: int = SAMPLE_RATE) -> dict:
    """Judge each word of the passage on the sounds actually produced.

    Returns one entry per passage word:
      word, expected, heard, error (0..1), verdict ("ok" | "mispronounced"
      | "unclear"), note (the clearest single difference).
    """
    per_word = [expected_phones(w) for w in passage_words]
    flat: list[str] = []
    owner: list[int] = []
    for index, word_phones in enumerate(per_word):
        for phone in word_phones:
            flat.append(phone)
            owner.append(index)

    if not flat:
        return {"words": [], "heard": [], "phone_error_rate": 0.0}

    heard = heard_phones(wav, sample_rate)
    pairs = align(flat, heard)

    errors = [0.0] * len(passage_words)
    notes: list[list[tuple[float, str]]] = [[] for _ in passage_words]
    matched = [0] * len(passage_words)
    last_word = 0

    for expected_index, heard_index in pairs:
        if expected_index is not None:
            word_index = owner[expected_index]
            last_word = word_index
            expected_phone = flat[expected_index]
            if heard_index is None:
                errors[word_index] += DELETION_COST
                notes[word_index].append(
                    (DELETION_COST, phones.describe(expected_phone, None))
                )
            else:
                c = phones.cost(expected_phone, heard[heard_index])
                errors[word_index] += c
                matched[word_index] += 1
                if c > 0:
                    notes[word_index].append(
                        (c, phones.describe(expected_phone, heard[heard_index]))
                    )
        else:
            # A sound with nothing to attach to: charge it to the word being
            # spoken at the time.
            errors[last_word] += INSERTION_COST

    results = []
    total_error = 0.0
    total_phones = 0
    for index, word in enumerate(passage_words):
        expected = per_word[index]
        count = len(expected)
        total_phones += count
        total_error += errors[index]

        if count < MIN_PHONES:
            verdict = "unclear"
            rate = 0.0
        else:
            rate = min(1.0, errors[index] / count)
            verdict = "ok" if rate <= WORD_ERROR_THRESHOLD else "mispronounced"

        worst = max(notes[index], default=(0.0, ""))[1] if notes[index] else ""
        results.append(
            {
                "word": word,
                "expected": " ".join(expected),
                "error": round(rate, 3),
                "verdict": verdict,
                "note": worst if verdict == "mispronounced" else "",
            }
        )

    return {
        "words": results,
        "heard": heard,
        "phone_error_rate": round(total_error / max(total_phones, 1), 3),
    }
