"""Reading-aloud scoring: a 20-point rubric over an ASR transcript.

The scorer aligns what was transcribed against the passage the learner was
asked to read, then reports pronunciation (8), fluency (6) and pace (6),
scaled down by how much of the passage was actually attempted.

Two deliberate design choices, both aimed at the failure mode that makes
transcript-based scoring dishonest:

1. A word that was *said differently* is `mispronounced`, not `correct` and
   not `skipped`. Fuzzy-matching a near miss back to `correct` hands full
   pronunciation marks to a learner who mispronounced half a line; calling it
   `skipped` claims they said nothing at all. Both are wrong, and `skipped`
   should mean silence.
2. Full marks require a clean read. Rounding alone lets a handful of wrong
   words disappear into an 8/8, so any mispronunciation caps pronunciation at
   7 and any missing word caps coverage below 100%.

This remains a proxy, not phoneme-level assessment (blueprint 04): a text
pipeline cannot separate "the learner mispronounced it" from "the recogniser
misheard it". The Hindi-aware confusion folding below narrows that gap for the
errors Hindi learners actually make, but it does not close it.
"""

import difflib
import math
import re
import unicodedata

# ---------------------------------------------------------------------------
# Normalisation
# ---------------------------------------------------------------------------

# Danda, western punctuation, quotes and the joiners Devanagari text picks up.
_PUNCT = re.compile(r"[।॥,.!?;:\"'()\[\]{}‘’“”‌‍]")
# Hyphens join a compound the recogniser usually emits as two words
# (hara-bhara -> harabhara), so they collapse rather than split.
_HYPHENS = re.compile(r"[-‐‑‒–—]")

# Vowel signs, anusvara/visarga/candrabindu, virama and nukta. Stripped only to
# build the comparison skeleton, never from the text shown to the learner.
_MARKS = (
    "ऺऻ़ािीुूृॄॅॆ"
    "ेैॉॊोौ्ॎॏऀँंः"
)
_MARK_TABLE = {ord(c): None for c in _MARKS}

# The consonant confusions Hindi learners actually make. Folding these lets a
# near miss be recognised as a mispronunciation of the expected word rather
# than an unrelated word. Folding can only ever downgrade `skipped` to
# `mispronounced` -- `correct` requires an exact match -- so it cannot
# manufacture a false pass.
_ASPIRATION = {
    "ख": "क",  # kha  -> ka
    "घ": "ग",  # gha  -> ga
    "छ": "च",  # chha -> cha
    "झ": "ज",  # jha  -> ja
    "ठ": "ट",  # ttha -> tta
    "ढ": "ड",  # ddha -> dda
    "थ": "त",  # tha  -> ta
    "ध": "द",  # dha  -> da
    "फ": "प",  # pha  -> pa
    "भ": "ब",  # bha  -> ba
}
_PLACE = {
    "ट": "त",  # tta -> ta   (retroflex -> dental)
    "ड": "द",  # dda -> da
    "ण": "न",  # nna -> na
    "ष": "स",  # ssa -> sa
    "श": "स",  # sha -> sa
    "व": "ब",  # va  -> ba
}

# Recognisers place anusvara/candrabindu/visarga inconsistently in Hindi, so a
# word differing only by these is treated as read correctly. Telling a child
# they mispronounced a word when the recogniser simply dropped a nasal mark is
# worse than missing a genuinely subtle error.
_ASR_NOISE = "ंँः"
_NOISE_TABLE = {ord(c): None for c in _ASR_NOISE}

_NEAR_MATCH_THRESHOLD = 0.6
_SKELETON_MATCH_SCORE = 0.92
_MERGE_WINDOW = 2
# Merging words costs a little, so a merge has to be clearly better than
# aligning them one-to-one. Without this, "sabne milkar" heard as "sab milkar"
# merges into one blurred pair instead of one wrong word and one right one.
_MERGE_COST = 0.1
# Every word in a merged group shares one verdict, so a weak merge can label a
# perfectly-read word as mispronounced purely because its neighbour was wrong.
# Merges are therefore only considered when the joined forms almost match --
# i.e. the reader genuinely ran the words together, or the recogniser split one.
_MERGE_MIN_SIMILARITY = 0.9

# Expected reading pace per level band. A slower target for younger learners.
LEVEL_TARGET_WPM = {"P1-P2": 100, "P3-P4": 120, "P5-P6": 140}
DEFAULT_LEVEL = "P1-P2"

# A read this much faster/slower than the target still scores full pace marks.
_PACE_BAND_LOW = 0.8
_PACE_BAND_HIGH = 1.25


def normalize(text: str) -> list[str]:
    """Split into comparable word tokens, preserving the original glyphs."""
    text = unicodedata.normalize("NFC", text)
    text = _PUNCT.sub(" ", text)
    text = _HYPHENS.sub("", text)
    return [w for w in text.split() if w]


def _fold(word: str) -> str:
    """Collapse the consonant contrasts learners routinely confuse."""
    folded = "".join(_ASPIRATION.get(ch, ch) for ch in word)
    return "".join(_PLACE.get(ch, ch) for ch in folded)


def _skeleton(word: str) -> str:
    """Consonant skeleton: confusions folded and vowel signs removed."""
    return _fold(word).translate(_MARK_TABLE)


def word_similarity(expected: str, said: str) -> float:
    """0..1 similarity. 1.0 only for an exact match."""
    if expected == said:
        return 1.0
    if not expected or not said:
        return 0.0
    if expected.translate(_NOISE_TABLE) == said.translate(_NOISE_TABLE):
        return 1.0
    if _skeleton(expected) == _skeleton(said):
        return _SKELETON_MATCH_SCORE
    return max(
        difflib.SequenceMatcher(None, expected, said).ratio(),
        difflib.SequenceMatcher(None, _fold(expected), _fold(said)).ratio(),
    )


# ---------------------------------------------------------------------------
# Alignment
# ---------------------------------------------------------------------------

_INSERT_PENALTY = 0.15


def _align(expected: list[str], said: list[str]) -> tuple[list[dict], int]:
    """Align transcript words to passage words.

    Returns one entry per expected word plus the number of transcript words
    that matched nothing. Runs a small DP rather than a plain opcode walk so a
    passage word can absorb a recogniser split (one word heard as two) and
    vice versa.
    """
    n, m = len(expected), len(said)
    neg = float("-inf")
    dp = [[neg] * (m + 1) for _ in range(n + 1)]
    back: list[list[tuple | None]] = [[None] * (m + 1) for _ in range(n + 1)]
    dp[0][0] = 0.0

    for i in range(n + 1):
        for j in range(m + 1):
            cur = dp[i][j]
            if cur == neg:
                continue
            if i < n and cur > dp[i + 1][j]:                    # expected not said
                dp[i + 1][j] = cur
                back[i + 1][j] = (i, j, 0, 0, 0.0)
            if j < m and cur - _INSERT_PENALTY > dp[i][j + 1]:  # extra word said
                dp[i][j + 1] = cur - _INSERT_PENALTY
                back[i][j + 1] = (i, j, 0, 1, 0.0)
            for ke in range(1, _MERGE_WINDOW + 1):
                if i + ke > n:
                    break
                for ks in range(1, _MERGE_WINDOW + 1):
                    if j + ks > m:
                        break
                    score = word_similarity(
                        "".join(expected[i:i + ke]), "".join(said[j:j + ks])
                    )
                    if (ke > 1 or ks > 1) and score < _MERGE_MIN_SIMILARITY:
                        continue
                    merge_discount = 1.0 - _MERGE_COST * ((ke - 1) + (ks - 1))
                    cand = cur + score * ke * merge_discount
                    if cand > dp[i + ke][j + ks]:
                        dp[i + ke][j + ks] = cand
                        back[i + ke][j + ks] = (i, j, ke, ks, score)

    entries: list[dict | None] = [None] * n
    extras = 0
    i, j = n, m
    while (i, j) != (0, 0):
        step = back[i][j]
        if step is None:  # unreachable state; fail safe rather than loop
            break
        pi, pj, ke, ks, score = step
        if ke == 0 and ks == 1:
            extras += 1
        elif ke == 0:
            entries[pi] = {"word": expected[pi], "status": "skipped", "heard": None}
        else:
            heard = " ".join(said[pj:pj + ks])
            if score >= 1.0:
                status = "correct"
            elif score >= _NEAR_MATCH_THRESHOLD:
                status = "mispronounced"
            else:
                status = "skipped"
                extras += ks
            for k in range(pi, pi + ke):
                entries[k] = {
                    "word": expected[k],
                    "status": status,
                    "heard": heard if status == "mispronounced" else None,
                }
        i, j = pi, pj

    for k in range(n):
        if entries[k] is None:
            entries[k] = {"word": expected[k], "status": "skipped", "heard": None}
    return entries, extras  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# Sub-scores
# ---------------------------------------------------------------------------

def _pronunciation_score(correct: int, mispronounced: int) -> int:
    """Out of 8, over the words actually attempted."""
    attempted = correct + mispronounced
    if attempted == 0:
        return 0
    score = round(8 * correct / attempted)
    # Full marks mean a clean read. Without this, a few wrong words round away.
    if mispronounced and score >= 8:
        score = 7
    return max(0, min(8, score))


def _fluency_score(bloat_ratio: float, disfluencies: int, passage_len: int) -> int:
    """Out of 6, from repeated/filler words rather than raw speed."""
    bloat_excess = max(0.0, bloat_ratio - 1.05)
    disfluency_rate = disfluencies / max(passage_len, 1)
    penalty = min(1.0, bloat_excess * 2.0 + disfluency_rate * 2.5)
    return max(0, min(6, round(6 * (1 - penalty))))


def _pace_score(slack: float) -> int:
    """Out of 6, from time taken against the level's expected reading time."""
    if slack <= 0:
        return 0
    if _PACE_BAND_LOW <= slack <= _PACE_BAND_HIGH:
        return 6
    drift = _PACE_BAND_LOW - slack if slack < _PACE_BAND_LOW else slack - _PACE_BAND_HIGH
    return max(0, min(6, round(6 * (1 - min(1.0, drift / 0.6)))))


def _apply_coverage(scores: tuple[int, int, int], coverage_percent: float) -> tuple[list[int], int]:
    """Scale raw sub-scores by coverage, keeping the parts summing to the total.

    Scoring the read on its own merits and *then* scaling by how much was
    attempted is what stops three perfectly-read words scoring full marks.
    """
    factor = max(0.0, min(1.0, coverage_percent / 100))
    total = int(math.floor(sum(scores) * factor))
    scaled = [int(round(s * factor)) for s in scores]

    # Reconcile rounding drift; fluency absorbs it first, pronunciation last.
    guard = 0
    while sum(scaled) != total and guard < 64:
        guard += 1
        diff = total - sum(scaled)
        for idx in (1, 2, 0):
            if diff > 0 and scaled[idx] < scores[idx]:
                scaled[idx] += 1
                break
            if diff < 0 and scaled[idx] > 0:
                scaled[idx] -= 1
                break
        else:
            break
    return scaled, total


def _confidence(coverage_percent: float, alignment_quality: float) -> tuple[str, str]:
    if coverage_percent >= 90 and alignment_quality >= 0.6:
        level = "high"
    elif coverage_percent >= 60:
        level = "medium"
    else:
        level = "low"
    if coverage_percent >= 99:
        message = "We heard the whole passage."
    else:
        message = (
            f"We heard {round(coverage_percent)}% of the passage. "
            "Word feedback below covers what you read."
        )
    return level, message


def _vocab_words(reading_vocabulary: list | None) -> set[str]:
    """Accept either ["word"] or [{"word": ..., "meaning_english": ...}]."""
    if not reading_vocabulary:
        return set()
    words: set[str] = set()
    for item in reading_vocabulary:
        raw = item.get("word") if isinstance(item, dict) else item
        if isinstance(raw, str):
            words.update(normalize(raw))
    return words


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def score_attempt(
    expected_text: str,
    transcript: str,
    audio_seconds: float,
    level: str | None = None,
    reading_vocabulary: list | None = None,
) -> dict:
    expected = normalize(expected_text)
    said = normalize(transcript)
    passage_len = len(expected)
    transcript_len = len(said)

    entries, disfluencies = _align(expected, said)

    correct = sum(1 for e in entries if e["status"] == "correct")
    mispronounced = [e["word"] for e in entries if e["status"] == "mispronounced"]
    skipped = [e["word"] for e in entries if e["status"] == "skipped"]
    attempted = correct + len(mispronounced)

    # Coverage is what the learner attempted -- a mispronounced word was read,
    # it was just read wrong; only silence costs coverage.
    coverage_percent = round(100 * attempted / max(passage_len, 1))
    if skipped and coverage_percent >= 100:
        coverage_percent = 99
    alignment_quality = correct / max(passage_len, 1)

    transcript_bloat_ratio = transcript_len / max(passage_len, 1)
    target_wpm = LEVEL_TARGET_WPM.get(level or DEFAULT_LEVEL, LEVEL_TARGET_WPM[DEFAULT_LEVEL])
    expected_seconds = passage_len / target_wpm * 60
    duration_slack = audio_seconds / expected_seconds if expected_seconds else 0.0
    seconds_per_correct_word = audio_seconds / correct if correct else 0.0

    raw = (
        _pronunciation_score(correct, len(mispronounced)),
        _fluency_score(transcript_bloat_ratio, disfluencies, passage_len),
        _pace_score(duration_slack),
    )
    (pron, fluency, pace), total = _apply_coverage(raw, coverage_percent)
    confidence, confidence_message = _confidence(coverage_percent, alignment_quality)

    practice_words = list(dict.fromkeys(mispronounced + skipped))
    vocab_targets = _vocab_words(reading_vocabulary)
    vocab_feedback_words = [w for w in practice_words if w in vocab_targets]

    words_per_minute = round(transcript_len / (audio_seconds / 60), 1) if audio_seconds else 0.0

    return {
        "total_score": total,
        "pronunciation_score": pron,
        "fluency_score": fluency,
        "pace_score": pace,
        "pre_coverage_pronunciation_score": raw[0],
        "pre_coverage_fluency_score": raw[1],
        "pre_coverage_pace_score": raw[2],
        "pre_coverage_total": sum(raw),
        "coverage_percent": coverage_percent,
        "coverage_skill_factor_percent": coverage_percent,
        "alignment_quality": round(alignment_quality, 4),
        "assessment_confidence": confidence,
        "assessment_confidence_message": confidence_message,
        "word_analysis": entries,
        "mispronounced_words": mispronounced,
        "skipped_words": skipped,
        "practice_words": practice_words,
        "vocab_feedback_words": vocab_feedback_words,
        "words_per_minute": words_per_minute,
        "target_wpm": target_wpm,
        "level": level or DEFAULT_LEVEL,
        "fluency_signals": {
            "disfluency_count": disfluencies,
            "seconds_per_correct_word": seconds_per_correct_word,
            "transcript_bloat_ratio": transcript_bloat_ratio,
            "duration_slack": duration_slack,
        },
    }
