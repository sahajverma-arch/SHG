"""Tests pinned to a real captured attempt.

The passage/transcript pair below is an actual reading recorded against the
reference implementation, together with the metrics it reported. Three of its
figures are exact rationals (74/77 words, 38s), so they pin our formulas
precisely rather than approximately.

The last test is the one that matters: on this recording the reference
implementation reported zero mispronounced words and full pronunciation marks,
even though four words were clearly misread. We must not.
"""

import math

from scoring import normalize, score_attempt, word_similarity

PASSAGE = (
    "आज रविवार है। मैं अपने दोस्तों के साथ मैदान में खेलने गया। "
    "मैदान बहुत बड़ा और हरा-भरा था। मेरी दोस्त रिया झूले पर बैठी थी। "
    "मेरा दोस्त राज फिसलपट्टी पर खेल रहा था। हम सब मिलकर बहुत मज़े से खेले। "
    "अध्यापिका ने कहा कि हमें बारी-बारी से खेलना चाहिए। "
    "हम सबने मिलकर एक-दूसरे की मदद की। शाम को घर जाते समय हम बहुत थके हुए थे। "
    "मुझे मैदान में खेलना बहुत अच्छा लगता है।"
)

# What the recogniser actually returned. Note रभीवार / जोले / वेटी / ठके.
TRANSCRIPT = (
    "आज रभीवार है, मैं अपने दोस्तों के साथ मैदान में खेलने गया। "
    "मैदान बहुत बड़ा और हरा भरा था। मेरी दोस्त रिया जोले पर वेटी थी। "
    "मेरा दोस्त राज फिसलपट्टी पर खेल रहा था। हम सब मिलकर बहुत मज़े से खेलें। "
    "अध्यापिका ने कहा कि हमें बारी बारी से खेलना चाहिए। "
    "हम सब मिलकर एक दूसरे की मदद की। शाम को घर जाते समय हम बहुत ठके हुए थे। "
    "मुझे मैदान में खेलना बहुत अच्छा लगता है।"
)

DURATION = 38.0
VOCAB = [
    {"word": "मैदान", "meaning_english": "Field"},
    {"word": "झूले", "meaning_english": "Swing"},
    {"word": "फिसलपट्टी", "meaning_english": "Slide"},
    {"word": "खेलने", "meaning_english": "To play"},
    {"word": "अध्यापिका", "meaning_english": "Teacher (female)"},
]

MISREAD = [("रविवार", "रभीवार"), ("झूले", "जोले"), ("बैठी", "वेटी"), ("थके", "ठके")]


def _result():
    return score_attempt(PASSAGE, TRANSCRIPT, DURATION, level="P1-P2", reading_vocabulary=VOCAB)


def test_normalisation_collapses_hyphenated_compounds():
    # हरा-भरा is one passage token; the recogniser split it into two.
    assert "हराभरा" in normalize(PASSAGE)
    assert len(normalize(PASSAGE)) == 74, len(normalize(PASSAGE))


def test_transcript_bloat_ratio_is_exact():
    signals = _result()["fluency_signals"]
    assert signals["transcript_bloat_ratio"] == 77 / 74


def test_duration_slack_pins_the_100wpm_target():
    # 74 words at the P1-P2 target of 100 wpm is 44.4s; 38/44.4 is exact.
    signals = _result()["fluency_signals"]
    assert math.isclose(signals["duration_slack"], 38 / 44.4, rel_tol=1e-12)


def test_hindi_confusions_are_near_matches_not_unrelated_words():
    for expected, said in MISREAD:
        sim = word_similarity(expected, said)
        assert sim >= 0.6, f"{expected}/{said} scored {sim}"
        assert sim < 1.0


def test_misread_words_are_mispronounced_not_correct_or_skipped():
    statuses = {e["word"]: e["status"] for e in _result()["word_analysis"]}
    for expected, _ in MISREAD:
        assert statuses[expected] == "mispronounced", (expected, statuses[expected])


def test_the_bug_we_are_fixing():
    """The reference implementation gave this recording 8/8 and no feedback."""
    result = _result()
    assert result["mispronounced_words"], "mispronunciations must be reported"
    assert result["pronunciation_score"] < 8, result["pronunciation_score"]
    assert result["practice_words"], "learner must get words to practise"


def test_vocabulary_feedback_flags_a_target_word():
    # झूले is a target vocabulary word and was misread as जोले.
    assert "झूले" in _result()["vocab_feedback_words"]


def test_coverage_scaling_stops_a_three_word_read_scoring_full_marks():
    partial = score_attempt(PASSAGE, "आज रविवार है", 2.0, level="P1-P2")
    assert partial["coverage_percent"] < 10
    assert partial["total_score"] <= 2, partial["total_score"]


def test_perfect_read_scores_full_marks():
    perfect = score_attempt(PASSAGE, PASSAGE, 44.4, level="P1-P2")
    assert perfect["coverage_percent"] == 100
    assert perfect["pronunciation_score"] == 8
    assert perfect["total_score"] == 20, perfect


def test_components_always_sum_to_total():
    for transcript, secs in [(TRANSCRIPT, 38.0), (PASSAGE, 44.4), ("आज", 1.0)]:
        r = score_attempt(PASSAGE, transcript, secs, level="P1-P2")
        parts = r["pronunciation_score"] + r["fluency_score"] + r["pace_score"]
        assert parts == r["total_score"], (transcript[:20], parts, r["total_score"])


def test_anusvara_noise_is_not_reported_as_a_misread():
    # खेले / खेलें differ only by an anusvara, which recognisers drop at random.
    assert word_similarity("खेले", "खेलें") == 1.0
    statuses = {e["word"]: e["status"] for e in _result()["word_analysis"]}
    assert statuses["खेले"] == "correct"


def test_merge_does_not_blur_a_wrong_word_into_its_right_neighbour():
    # "सबने मिलकर" heard as "सब मिलकर" is one wrong word, not two.
    statuses = {e["word"]: e["status"] for e in _result()["word_analysis"]}
    assert statuses["सबने"] == "mispronounced"
    assert statuses["मिलकर"] == "correct"


def test_recogniser_splitting_a_compound_still_counts_as_correct():
    # हरा-भरा is one passage token but was transcribed as two words.
    statuses = {e["word"]: e["status"] for e in _result()["word_analysis"]}
    assert statuses["हराभरा"] == "correct"


def test_a_correctly_read_word_is_never_flagged_by_its_wrong_neighbour():
    """Regression: a weak merge used to label an identical word mispronounced.

    Passage ends "...है यह वाक्य..."; the reader stopped after "है". Merging
    "है यह" against the single spoken "है" tarred both with one verdict, so
    "है" came back as mispronounced with heard == "है".
    """
    result = score_attempt(
        "मुझे खेलना अच्छा लगता है यह वाक्य कभी नहीं पढ़ा गया",
        "मुझे खेलना अच्छा लगता है",
        6.0,
    )
    statuses = {e["word"]: e["status"] for e in result["word_analysis"]}
    assert statuses["है"] == "correct"
    assert statuses["यह"] == "skipped"


def test_no_flagged_word_is_identical_to_what_was_heard():
    for entry in _result()["word_analysis"]:
        if entry["status"] == "mispronounced":
            assert entry["word"] != entry["heard"], entry


def test_every_flagged_word_carries_what_was_heard():
    for entry in _result()["word_analysis"]:
        if entry["status"] == "mispronounced":
            assert entry["heard"], entry


def test_silence_is_skipped_not_mispronounced():
    r = score_attempt(PASSAGE, "", 1.0, level="P1-P2")
    assert r["mispronounced_words"] == []
    assert len(r["skipped_words"]) == 74
    assert r["total_score"] == 0
