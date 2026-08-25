"""Where pre-rendered speech lives, and what each clip is called.

Imported by both main.py (which serves the clips) and prerender_tts.py (which
writes them). Those run in *different* virtualenvs -- the renderer needs torch
and IndicF5, the service must not -- so this module stays stdlib-only. Keeping
the naming in one place is the point: if the two sides ever disagreed about a
clip's name the cache would silently miss and every passage would quietly fall
back to edge-tts.
"""

import hashlib
import os
from pathlib import Path

PRERENDER_DIR = Path(
    os.environ.get(
        "TTS_PRERENDER_DIR", str(Path(__file__).resolve().parent / "tts-cache")
    )
)


def prerender_key(text: str) -> str:
    """Name a pre-rendered clip.

    Whitespace-insensitive, so re-indenting a passage does not orphan the audio
    that was rendered for it.
    """
    normalised = " ".join(text.split())
    return hashlib.sha256(normalised.encode("utf-8")).hexdigest()[:16]


def prerendered_path(text: str) -> Path:
    return PRERENDER_DIR / f"{prerender_key(text)}.wav"


def find_prerendered(text: str) -> Path | None:
    path = prerendered_path(text)
    return path if path.is_file() else None
