"""Phone comparison for Hindi.

Two sources describe the same sounds slightly differently: espeak's Hindi
grapheme-to-phoneme (the expected pronunciation, derived from the passage) and
the wav2vec2 phone recogniser (what was actually said). Comparing them
naively counts notation differences as mispronunciations.

So every comparison runs through a cost:

  0.0  the same sound written differently  — espeak `t`, the model `t̪`
  0.4  a distinction recognisers get wrong — vowel length, nasalisation
  1.0  a real difference

The 1.0 cases are the point of this whole exercise. Hindi's aspirated and
unaspirated stops (क/ख, त/थ, ट/ठ) and its dental/retroflex pairs (त/ट, द/ड)
are separate phonemes that a text pipeline cannot see, because a recogniser
transcribing `काना` will happily write the real word `खाना`. Here they cost
full price and nothing folds them together.
"""

import unicodedata

# Same sound, different transcription convention. Folding these is what stops
# the scorer crying wolf on every dental stop in the passage.
_SAME: dict[str, str] = {
    # espeak writes plain dentals; the recogniser marks them explicitly
    "t̪": "t",
    "d̪": "d",
    "s̪": "s",
    "n̪": "n",
    "n̩": "n",
    "l̩": "l",
    # rhotics: espeak `ɾ`, the model often `r`
    "r": "ɾ",
    "ɹ": "ɾ",
    "r̩": "ɾ",
    # espeak uses the palatal stop for ज/झ, the model the affricate
    "dʒ": "ɟ",
    "dZ": "ɟ",
    "dʒʰ": "ɟʰ",
    "dʑ": "ɟ",
    "tɕ": "tʃ",
    "tS": "tʃ",
    "tɕh": "tʃʰ",
    "tʃh": "tʃʰ",
    # approximants and glottals
    "v": "ʋ",
    "w": "ʋ",
    "ɦ": "h",
    "ɡ": "g",
    # schwa variants
    "ʌ": "ə",
    "ɐ": "ə",
    # aspiration is sometimes written with the ASCII letter
    "kh": "kʰ",
    "th": "tʰ",
    "ph": "pʰ",
    "bh": "bʰ",
    "dh": "dʰ",
    "gh": "ɡʰ",
}

# Vowel qualities a recogniser confuses often enough that treating them as
# outright errors would bury the real ones.
_VOWEL_BASE: dict[str, str] = {
    "ɪ": "i",
    "ʊ": "u",
    "ɛ": "e",
    "ɔ": "o",
    "ᵻ": "i",
    "ə": "ə",
}

_VOWELS = set("aeiouəɪʊɛɔæɑɒʌɐyøœɨʉɯ")

# Distinct phonemes, but close enough acoustically that recognisers slip.
_NEAR_PAIRS = {frozenset({"ə", "a"}), frozenset({"e", "i"}), frozenset({"o", "u"})}

NEAR_COST = 0.4


def canonical(phone: str) -> str:
    """Fold transcription variants onto one spelling."""
    phone = unicodedata.normalize("NFC", phone).strip()
    if not phone:
        return ""
    # Length marks carry meaning in Hindi but are unreliable from audio, so
    # they are handled as a *near* difference rather than folded away here.
    return _SAME.get(phone, phone)


def _vowel_key(phone: str) -> str | None:
    """Base vowel quality, with length and nasalisation removed."""
    stripped = phone.replace("ː", "")
    decomposed = unicodedata.normalize("NFD", stripped)
    bare = "".join(c for c in decomposed if not unicodedata.combining(c))
    if not bare or bare[0] not in _VOWELS:
        return None
    return _VOWEL_BASE.get(bare, bare)


def is_vowel(phone: str) -> bool:
    return _vowel_key(phone) is not None


def cost(expected: str, heard: str) -> float:
    """0.0 identical, 0.4 a near miss, 1.0 a genuine difference."""
    if expected == heard:
        return 0.0

    a = canonical(expected)
    b = canonical(heard)
    if a == b:
        return 0.0

    va = _vowel_key(a)
    vb = _vowel_key(b)
    if va is not None and vb is not None:
        # Same quality, differing only in length or nasalisation.
        if va == vb:
            return NEAR_COST
        if frozenset({va, vb}) in _NEAR_PAIRS:
            return NEAR_COST
        return 1.0

    # One vowel and one consonant, or two different consonants. Aspiration and
    # place contrasts land here deliberately: `k` vs `kʰ` is a full error.
    return 1.0


def describe(expected: str, heard: str | None) -> str:
    """A short, human-readable note about one phone difference."""
    if heard is None:
        return f"missed {expected}"
    return f"{expected} said as {heard}"
