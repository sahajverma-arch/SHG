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


# How the wav2vec2 recogniser actually spells Hindi, which is not how espeak
# writes it. This matters far more than it looks: the model has `kʰ`, `bʰ`,
# `pʰ` and the rest in its vocabulary and never once emits them. Across 545
# non-blank frames of Hindi - real human speech included - it produced an
# aspirated stop zero times. It writes those sounds as an ASCII digraph where
# it has one (`kh`, `th`, `ph`), otherwise as the plain stop followed by a
# separate `h`, and फ usually as the fricative `f`.
#
# Aligning espeak's `kʰ` against this model therefore scored a symbol the model
# never produces, which turned every aspirated phone in every passage into an
# automatic error. Since aspiration is exactly the contrast Hindi reading
# practice cares about (क/ख, त/थ, प/फ, ब/भ, ड/ढ), that one mismatch was
# poisoning the whole measurement.
#
# A phone maps to a tuple of *slots*, each slot a tuple of interchangeable
# symbols to be pooled. Two slots means the model spells the sound with two
# tokens - and that second slot is what a plain rival has to explain as
# silence, which is what makes a dropped aspiration detectable at all.
_MODEL_SPELLING: dict[str, tuple[tuple[str, ...], ...]] = {
    # Aspirated stops. Digraphs exist only for k, t and p.
    "kʰ": (("kʰ", "kh"),),
    "tʰ": (("tʰ", "th"),),
    "ʈʰ": (("ʈʰ", "th", "tʰ"),),
    "pʰ": (("pʰ", "ph", "f"),),
    "cʰ": (("cʰ", "tʃʰ", "tɕh"),),
    "bʰ": (("bʰ", "b"), ("h",)),
    "dʰ": (("dʰ", "dʰː", "d", "t̪"), ("h",)),
    "ɖʰ": (("ɖʰ", "ɖ"), ("h",)),
    "ɡʰ": (("ɡʰ", "ɡ"), ("h",)),
    "ɟʰ": (("ɟʰ", "dʒ", "ɟ"), ("h",)),
    # Plain consonants the model spells with another language's symbol.
    "c": (("tʃ", "tɕ", "c", "ts"),),
    "ɟ": (("dʒ", "dʑ", "ɟ", "dZ"),),
    "t": (("t̪", "t"),),
    "d": (("d", "d["),),
    "ɾ": (("ɾ", "r", "ɹ"),),
    "ʋ": (("ʋ", "v", "w"),),
    "ɳ": (("ɳ", "n"),),
    "ʃ": (("ʃ", "ɕ", "ʂ"),),
    "s": (("s", "s̪"),),
}


def model_slots(phone: str, vocab) -> list[tuple[str, ...]]:
    """The symbols this recogniser would emit for one expected phone.

    Returns a list of slots, each a tuple of interchangeable vocabulary
    symbols. Empty when the model has no way to spell the phone at all.
    """
    slots = []
    for slot in _MODEL_SPELLING.get(phone, ((phone,),)):
        usable = tuple(s for s in slot if s in vocab)
        if usable:
            slots.append(usable)
    return slots


def word_slots(phones, vocab) -> list[tuple[str, ...]]:
    """`model_slots` over a whole word's expected phone sequence."""
    return [slot for phone in phones for slot in model_slots(phone, vocab)]
