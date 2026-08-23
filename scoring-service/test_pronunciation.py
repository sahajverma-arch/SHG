"""Tests for the phoneme machinery.

These cover the parts that were verified to work: the phone cost table, CTC
forced alignment, and the GOP calculation. They deliberately do not assert that
mispronunciation *detection* works — measurement showed it does not, at word
granularity, with the models available (see `docs/phoneme-scoring.md`).

Nothing here loads the phone recogniser, so the suite stays fast and offline.
"""

import numpy as np
import pytest

import gop
import phones


# ---------------------------------------------------------------------------
# Phone costs — the contrasts that matter for Hindi
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "expected,heard",
    [
        ("k", "kʰ"),    # क / ख
        ("t", "tʰ"),    # त / थ
        ("p", "pʰ"),    # प / फ
        ("b", "bʰ"),    # ब / भ
        ("ɖ", "ɖʰ"),    # ड / ढ
    ],
)
def test_aspiration_is_a_full_error(expected, heard):
    """Aspiration is phonemic in Hindi — it must never be folded away."""
    assert phones.cost(expected, heard) == 1.0
    assert phones.cost(heard, expected) == 1.0


@pytest.mark.parametrize("dental,retroflex", [("t", "ʈ"), ("d", "ɖ"), ("n", "ɳ")])
def test_dental_and_retroflex_are_a_full_error(dental, retroflex):
    """त/ट and द/ड are separate phonemes, not accents of each other."""
    assert phones.cost(dental, retroflex) == 1.0


@pytest.mark.parametrize(
    "a,b",
    [
        ("t", "t̪"),    # espeak plain vs explicitly dental
        ("d", "d̪"),
        ("s", "s̪"),
        ("ɾ", "r"),     # tap written two ways
        ("ɟ", "dʒ"),    # espeak palatal stop vs the model's affricate
        ("ʋ", "v"),
        ("h", "ɦ"),
        ("ə", "ʌ"),
    ],
)
def test_notation_variants_are_free(a, b):
    """The same sound written differently must cost nothing.

    Without this the scorer reports a mispronunciation on every dental stop,
    because espeak and the recogniser simply spell them differently.
    """
    assert phones.cost(a, b) == 0.0
    assert phones.cost(b, a) == 0.0


def test_vowel_length_is_a_near_miss_not_an_error():
    """Length is phonemic but unreliable from audio, so it is charged partly."""
    assert phones.cost("a", "aː") == pytest.approx(phones.NEAR_COST)
    assert 0 < phones.cost("i", "iː") < 1.0


def test_different_vowels_are_a_full_error():
    assert phones.cost("i", "u") == 1.0


# ---------------------------------------------------------------------------
# CTC forced alignment
# ---------------------------------------------------------------------------

def _log_probs(frames: list[int], vocab_size: int = 5, confidence: float = 0.9):
    """Frames that strongly favour the given token id at each step."""
    probs = np.full((len(frames), vocab_size), (1 - confidence) / (vocab_size - 1))
    for t, token in enumerate(frames):
        probs[t, token] = confidence
    return np.log(probs)


def test_alignment_finds_each_phone_in_order():
    # blank=0; the audio says 1, then 2, with a blank between.
    log_probs = _log_probs([1, 1, 0, 2, 2, 2])
    spans = gop.forced_align(log_probs, [1, 2], blank=0)
    assert len(spans) == 2
    (start_a, end_a), (start_b, end_b) = spans
    assert start_a < end_a <= start_b < end_b
    assert start_a == 0
    assert end_b == 6


def test_alignment_separates_a_repeated_phone_with_a_blank():
    """Two identical phones in a row must not collapse into one."""
    log_probs = _log_probs([1, 0, 1])
    spans = gop.forced_align(log_probs, [1, 1], blank=0)
    assert len(spans) == 2
    assert spans[0][1] <= spans[1][0]


def test_alignment_declines_when_there_are_too_few_frames():
    log_probs = _log_probs([1])
    assert gop.forced_align(log_probs, [1, 2, 3, 4], blank=0) == []


# ---------------------------------------------------------------------------
# GOP
# ---------------------------------------------------------------------------

def test_gop_is_near_zero_when_the_model_agrees():
    """GOP is a ratio against the frame's own best phone, so a confident,
    correct frame scores ~0."""
    log_probs = _log_probs([1, 2], confidence=0.99)
    scores = gop.gop_per_phone(log_probs, [1, 2], blank=0)
    assert all(s > -0.1 for s in scores)


def test_gop_is_strongly_negative_when_the_sound_is_absent():
    """The audio says 1 1; we ask about 3, which is nowhere in it."""
    log_probs = _log_probs([1, 1, 1, 1], confidence=0.99)
    said = gop.gop_per_phone(log_probs, [1], blank=0)[0]
    absent = gop.gop_per_phone(log_probs, [3], blank=0)[0]
    assert absent < said - 2.0


# ---------------------------------------------------------------------------
# Outlier flagging
# ---------------------------------------------------------------------------

def test_flagging_needs_enough_words_to_have_a_distribution():
    flags, _means, _median = gop.flag_words([[-0.1, -0.1], [-9.0, -9.0]])
    assert flags == [False, False], "two words cannot establish a baseline"


def test_a_uniformly_good_reading_flags_nothing():
    """The weakest word of a clean reading is still a clean word."""
    grouped = [[-0.30, -0.32], [-0.28, -0.31], [-0.33, -0.29], [-0.31, -0.30],
               [-0.40, -0.38], [-0.29, -0.30]]
    flags, _means, _median = gop.flag_words(grouped)
    assert not any(flags)


def test_one_clear_outlier_is_flagged():
    grouped = [[-0.30, -0.32], [-0.28, -0.31], [-0.33, -0.29], [-0.31, -0.30],
               [-4.20, -4.60], [-0.29, -0.30]]
    flags, _means, _median = gop.flag_words(grouped)
    assert flags[4]
    assert sum(flags) == 1


def test_short_words_are_not_judged():
    """A one-phone word carries too little evidence to accuse anyone."""
    grouped = [[-0.3, -0.3], [-0.3, -0.3], [-0.3, -0.3], [-0.3, -0.3], [-9.0]]
    flags, means, _median = gop.flag_words(grouped, min_phones=2)
    assert means[4] is None
    assert flags[4] is False
