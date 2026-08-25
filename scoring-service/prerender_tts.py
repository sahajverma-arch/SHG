"""Render passages to speech with IndicF5, ahead of time.

Why ahead of time: IndicF5 (AI4Bharat, MIT) is a 0.4B flow-matching model.
Measured on this project's GTX 1650 it takes roughly half a minute per
sentence, and on CPU it does not finish one in twenty. It cannot sit behind a
button. The passages are a fixed set, so each one is rendered once here and
main.py serves the wav instantly; anything unrendered falls back to edge-tts.

This runs in its OWN virtualenv, not the scoring service's. IndicF5 pins
`numpy<=1.26.4` and `transformers<4.50`, both of which would break the ASR
side. See README.md.

    <indicf5-venv>/Scripts/python prerender_tts.py "पहला वाक्य।" "दूसरा वाक्य।"
    <indicf5-venv>/Scripts/python prerender_tts.py --file passages.txt

`--file` takes one passage per line, UTF-8, blank lines and `#` comments
ignored. Prefer it over arguments: Devanagari through a Windows shell is a
good way to render a subtly mangled passage and not notice.

Already-rendered passages are skipped, so re-running after adding one to the
list only renders the new one.

Access to the weights is gated on Hugging Face (free, instant): accept at
huggingface.co/ai4bharat/IndicF5, then `hf auth login`.
"""

import os
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torchaudio

from tts_cache import PRERENDER_DIR, prerendered_path

HERE = Path(__file__).resolve().parent
PROMPT_DIR = HERE / "tts-prompts"

# IndicF5 clones the speaker from a reference clip, so it needs one plus that
# clip's transcript. This is the pairing the model card itself uses to produce
# Hindi -- the reference is Punjabi, which is the point: the model transfers
# the voice across languages.
REF_NAME = "PAN_F_HAPPY_00001.wav"
REF_URL = f"https://github.com/AI4Bharat/IndicF5/raw/main/prompts/{REF_NAME}"
REF_TEXT = (
    "ਭਹੰਪੀ ਵਿੱਚ ਸਮਾਰਕਾਂ ਦੇ ਭਵਨ ਨਿਰਮਾਣ ਕਲਾ ਦੇ ਵੇਰਵੇ ਗੁੰਝਲਦਾਰ ਅਤੇ ਹੈਰਾਨ "
    "ਕਰਨ ਵਾਲੇ ਹਨ, ਜੋ ਮੈਨੂੰ ਖੁਸ਼ ਕਰਦੇ  ਹਨ।"
)

# The passage the web app falls back to when there is no Supabase project.
DEFAULT_TEXTS = ["सूरज पूरब से निकलता है और पश्चिम में डूब जाता है।"]


def _load_via_soundfile(path, *args, **kwargs):
    """torchaudio 2.13 routes `load` through TorchCodec, which wants FFmpeg's
    shared libraries; the usual Windows ffmpeg is a static build without them.
    The reference prompt is a plain wav, so read it directly and return what
    torchaudio's callers expect: (channels, frames) float32 plus the rate."""
    data, sr = sf.read(str(path), dtype="float32", always_2d=True)
    return torch.from_numpy(data.T.copy()), sr


def _disable_compile():
    """Take torch.compile out of the picture entirely.

    model.py wraps both the vocoder and the transformer in torch.compile. On
    Windows that is a straight loss: there is no Triton, so inductor cannot
    emit GPU kernels, and it spends the effort finding that out -- measured
    here at 528s for one sentence against 31s without it. Routing it through
    dynamo's own "eager" backend is no better; that traces on CPU with the GPU
    sitting idle at 0%.

    The catch is that the published checkpoint was saved FROM a compiled model,
    so its keys carry an `_orig_mod` segment. Dropping the wrapper makes every
    one of them miss, and transformers reports that as a warning rather than an
    error -- leaving a randomly initialised model that still emits confident,
    meaningless audio. `_load_weights` below puts them back.
    """
    torch.compile = lambda module=None, *a, **k: module


def _load_weights(model):
    """Load the checkpoint by hand, with the `_orig_mod` segment stripped."""
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file

    path = hf_hub_download("ai4bharat/IndicF5", filename="model.safetensors")
    state = load_file(path)
    fixed = {key.replace("._orig_mod.", "."): value for key, value in state.items()}

    missing, unexpected = model.load_state_dict(fixed, strict=False)
    # Anything left unexpected means the renaming did not line up, and we would
    # be rendering audio from an untrained model. Refuse rather than ship it.
    if unexpected:
        raise SystemExit(
            f"{len(unexpected)} checkpoint tensors did not match the model, "
            f"e.g. {unexpected[:3]}. Refusing to render from partial weights."
        )
    print(f"weights: loaded {len(fixed) - len(missing)}/{len(fixed)} tensors", flush=True)
    return model


def _read_list(path):
    texts = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            texts.append(line)
    return texts


def main(argv):
    if argv and argv[0] == "--file":
        if len(argv) < 2:
            raise SystemExit("--file needs a path")
        texts = _read_list(argv[1])
        if not texts:
            raise SystemExit(f"no passages found in {argv[1]}")
    else:
        texts = argv or DEFAULT_TEXTS

    PRERENDER_DIR.mkdir(parents=True, exist_ok=True)
    PROMPT_DIR.mkdir(parents=True, exist_ok=True)

    ref_audio = PROMPT_DIR / REF_NAME
    if not ref_audio.is_file():
        print(f"fetching reference prompt {REF_NAME} ...")
        urllib.request.urlretrieve(REF_URL, ref_audio)

    pending = [t for t in texts if not prerendered_path(t).is_file()]
    if not pending:
        print(f"all {len(texts)} passage(s) already rendered in {PRERENDER_DIR}")
        return 0

    torchaudio.load = _load_via_soundfile
    _disable_compile()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("WARNING: no CUDA device. Expect minutes per sentence.", flush=True)
    print(f"device={device}  rendering {len(pending)} of {len(texts)} passage(s)", flush=True)

    from transformers import AutoModel

    t0 = time.perf_counter()
    model = AutoModel.from_pretrained("ai4bharat/IndicF5", trust_remote_code=True)
    _load_weights(model)
    print(f"model loaded in {time.perf_counter() - t0:.1f}s", flush=True)

    for text in pending:
        dest = prerendered_path(text)
        t0 = time.perf_counter()
        audio = model(text, ref_audio_path=str(ref_audio), ref_text=REF_TEXT)
        elapsed = time.perf_counter() - t0

        audio = np.asarray(audio)
        if audio.dtype == np.int16:
            audio = audio.astype(np.float32) / 32768.0
        sf.write(dest, audio.astype(np.float32), samplerate=24000)

        seconds = len(audio) / 24000
        print(f"  {dest.name}  {seconds:.1f}s audio in {elapsed:.0f}s  <- {text[:40]}",
              flush=True)

    print(f"\ndone. {PRERENDER_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
