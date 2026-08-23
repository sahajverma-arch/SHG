"""Goodness of Pronunciation for Hindi.

Free-decoding phones and comparing the strings does not work here, and that is
measured rather than assumed: on correct native readings it flagged 33-44% of
words as mispronounced with one phone recogniser and 70-77% with another. Two
speakers reading the *same correct text* disagreed on 24-46% of phones, while a
deliberate mispronunciation moved the error by only 11%. The noise is larger
than the signal, so any threshold over it accuses children who read perfectly.

GOP asks a different question. It never lets the model choose what was said.
The phones the passage *calls for* are force-aligned to the audio, and each one
is scored by how confident the model is that it is really there:

    gop(phone) = mean over its frames of  log P(phone | frame)
                                        - log P(best phone | frame)

Zero means the model agrees completely; large negative means the expected sound
is not in the audio. Because the score is a ratio against the frame's own best
phone, a speaker or channel that shifts every phone shifts the reference too,
and much of the systematic offset cancels.

What remains is normalised per reading (see `flag_words`): a word is judged
against the same child's other words, not against an absolute threshold that
would have to hold across every voice, microphone and room.
"""

import numpy as np

NEG_INF = -1e30


def forced_align(log_probs: np.ndarray, targets: list[int], blank: int = 0):
    """Align a known phone sequence to CTC frames.

    `log_probs` is [frames, vocab]; `targets` the expected phone ids. Returns
    one (start, end) frame span per target phone.

    Standard CTC alignment over the blank-interleaved sequence, so a phone may
    occupy many frames and repeated phones stay separated by a blank.
    """
    n_frames = log_probs.shape[0]
    if not targets or n_frames == 0:
        return []

    # blank, y1, blank, y2, ... blank
    extended = [blank]
    for t in targets:
        extended.extend([t, blank])
    length = len(extended)

    if n_frames < length // 2:
        return []

    alpha = np.full((n_frames, length), NEG_INF, dtype=np.float64)
    back = np.zeros((n_frames, length), dtype=np.int8)

    alpha[0, 0] = log_probs[0, extended[0]]
    if length > 1:
        alpha[0, 1] = log_probs[0, extended[1]]

    for t in range(1, n_frames):
        for s in range(length):
            best, choice = alpha[t - 1, s], 0
            if s > 0 and alpha[t - 1, s - 1] > best:
                best, choice = alpha[t - 1, s - 1], 1
            # A blank may be skipped only between two different labels.
            if (
                s > 1
                and extended[s] != blank
                and extended[s] != extended[s - 2]
                and alpha[t - 1, s - 2] > best
            ):
                best, choice = alpha[t - 1, s - 2], 2
            if best > NEG_INF:
                alpha[t, s] = best + log_probs[t, extended[s]]
                back[t, s] = choice

    end = length - 1
    if length > 1 and alpha[n_frames - 1, length - 2] > alpha[n_frames - 1, end]:
        end = length - 2

    path = np.zeros(n_frames, dtype=np.int32)
    s = int(end)
    for t in range(n_frames - 1, -1, -1):
        path[t] = s
        # int() matters: the backpointer array is int8, and NumPy's weak
        # promotion would otherwise narrow `s` to int8 and overflow past 127.
        s -= int(back[t, s])

    spans: list[tuple[int, int]] = []
    for index in range(len(targets)):
        position = 2 * index + 1
        frames = np.nonzero(path == position)[0]
        spans.append((int(frames[0]), int(frames[-1]) + 1) if frames.size else (-1, -1))
    return spans


def gop_per_phone(log_probs: np.ndarray, targets: list[int], blank: int = 0) -> list[float]:
    """How confident the model is in each expected phone, 0 (certain) downwards."""
    spans = forced_align(log_probs, targets, blank)
    if not spans:
        return [float("nan")] * len(targets)

    frame_best = log_probs.max(axis=1)
    scores: list[float] = []
    for phone_id, (start, end) in zip(targets, spans):
        if start < 0:
            # Never given a frame: the sound is simply not there.
            scores.append(float("nan"))
            continue
        window = log_probs[start:end, phone_id] - frame_best[start:end]
        scores.append(float(window.mean()))
    return scores


def flag_words(
    word_phone_scores: list[list[float]],
    min_phones: int = 2,
    z_threshold: float = 1.15,
    floor: float = -1.2,
):
    """Pick out the words pronounced unlike the rest of this reading.

    Judging each word against the same reading's own distribution is what makes
    this usable across voices: a quiet child, a poor microphone or an unusual
    accent moves every word together and cancels out, while a genuinely
    mispronounced word stands apart from its neighbours.

    A word must be both a clear outlier *and* below an absolute floor, so a
    flawless reading does not have its weakest word flagged for being merely
    the weakest.
    """
    means: list[float | None] = []
    for scores in word_phone_scores:
        usable = [s for s in scores if not np.isnan(s)]
        means.append(float(np.mean(usable)) if len(usable) >= min_phones else None)

    present = [m for m in means if m is not None]
    if len(present) < 4:
        return [False] * len(means), means, 0.0

    median = float(np.median(present))
    # Median absolute deviation: robust to the few genuinely bad words we are
    # trying to find, which would inflate a plain standard deviation.
    mad = float(np.median([abs(m - median) for m in present])) or 1e-6
    spread = 1.4826 * mad

    flags = []
    for m in means:
        if m is None:
            flags.append(False)
            continue
        z = (median - m) / spread
        flags.append(bool(z >= z_threshold and m <= floor))
    return flags, means, median
