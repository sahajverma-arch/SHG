import difflib
import re

_DEVANAGARI_PUNCT = re.compile(r"[।,.!?\"'()‌‍]")

# Comfortable read-aloud pace for the passage lengths used in this module.
_WPM_COMFORT_MIN = 90
_WPM_COMFORT_MAX = 160


def normalize(text: str) -> list[str]:
    text = _DEVANAGARI_PUNCT.sub(" ", text)
    return [w for w in text.strip().split() if w]


def score_attempt(expected_text: str, transcript: str, audio_seconds: float) -> dict:
    """Edit-distance + pace scoring.

    This is a proxy, not true phoneme-level pronunciation assessment — see
    blueprint §04. It aligns the transcribed words against the expected
    passage and scores by how much of the passage was read correctly, plus
    how close the pace was to a comfortable reading band.
    """
    expected_words = normalize(expected_text)
    said_words = normalize(transcript)

    matcher = difflib.SequenceMatcher(None, expected_words, said_words)
    word_diff = []
    matched = 0

    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            matched += i2 - i1
            word_diff.extend({"word": w, "status": "match"} for w in expected_words[i1:i2])
        elif tag == "replace":
            word_diff.extend({"word": w, "status": "substitution"} for w in expected_words[i1:i2])
        elif tag == "delete":
            word_diff.extend({"word": w, "status": "missing"} for w in expected_words[i1:i2])
        elif tag == "insert":
            word_diff.extend({"word": w, "status": "extra"} for w in said_words[j1:j2])

    total_expected = max(len(expected_words), 1)
    accuracy_score = round(100 * matched / total_expected, 1)

    minutes = max(audio_seconds / 60, 1e-6)
    wpm = len(said_words) / minutes
    if wpm < _WPM_COMFORT_MIN:
        fluency_score = round(max(0, 100 - (_WPM_COMFORT_MIN - wpm) * 2), 1)
    elif wpm > _WPM_COMFORT_MAX:
        fluency_score = round(max(0, 100 - (wpm - _WPM_COMFORT_MAX) * 2), 1)
    else:
        fluency_score = 100.0

    return {
        "accuracy_score": accuracy_score,
        "fluency_score": fluency_score,
        "words_per_minute": round(wpm, 1),
        "word_diff": word_diff,
    }
