"""UVR / Roformer backend via the ``audio-separator`` package.

``audio-separator`` wraps the Ultimate Vocal Remover model zoo (BS-Roformer,
Mel-Band Roformer, MDX23C, MDX-Net, VR-Arch). Its API downloads weights on
demand, writes separated files to a directory, and returns their paths. We run
each separation into a temp directory, then read the stems back into the
package's channels-first float32 convention.

GPU is selected automatically by audio-separator when a CUDA-enabled
onnxruntime/torch is present; ``config.device == "cpu"`` forces CPU.
"""

from __future__ import annotations

import contextlib
import logging
import tempfile
from pathlib import Path

import numpy as np

from stems.audio_io import load_audio
from stems.config import RunConfig
from stems.engines.base import BaseSeparator, SeparationResult

# Curated, high-SDR aliases. Keys are friendly names used by presets/CLI; values
# are checkpoint filenames audio-separator fetches on demand (with their configs).
# SDR figures are from audio-separator's published scores (vocals / instrumental).
MODEL_FILES: dict[str, str] = {
    # General 2-stem (both vocals + instrumental from one model).
    "bs_roformer": "model_bs_roformer_ep_368_sdr_12.9628.ckpt",   # 12.10 / 16.31
    "bs_roformer_317": "model_bs_roformer_ep_317_sdr_12.9755.ckpt",  # 11.77 / 16.45
    "mel_roformer": "model_mel_band_roformer_ep_3005_sdr_11.4360.ckpt",  # 10.54 / 15.13
    "mdx23c": "MDX23C-8KFFT-InstVoc_HQ.ckpt",
    # Best dedicated VOCAL models (vocals target).
    "kim_vocals": "vocals_mel_band_roformer.ckpt",            # 12.60 vocals (best)
    "kim_ft": "mel_band_roformer_kim_ft_unwa.ckpt",           # 12.44 vocals
    "vocal_fullness": "mel_band_roformer_vocal_fullness_aname.ckpt",  # fuller, less gated
    # Best dedicated INSTRUMENTAL models (instrumental target / low vocal bleed).
    "inst_v2": "melband_roformer_inst_v2.ckpt",               # 16.06 instrumental
    "inst_bleedless": "mel_band_roformer_instrumental_bleedless_v2_gabox.ckpt",
}

# 2-stem models output vocals + instrumental.
_TWO_STEM = ["vocals", "instrumental"]

# Filename keyword -> canonical stem name (audio-separator tags files like
# "track_(Vocals)_model.wav").
_STEM_KEYWORDS = [
    ("instrumental", "instrumental"),
    ("no_vocals", "instrumental"),
    ("no other", "instrumental"),
    ("vocals", "vocals"),
    ("vocal", "vocals"),
    ("drums", "drums"),
    ("bass", "bass"),
    ("other", "other"),
    ("guitar", "guitar"),
    ("piano", "piano"),
]


def resolve_model_file(model: str) -> str:
    """Map a friendly alias to a checkpoint filename (pass through if unknown)."""
    return MODEL_FILES.get(model, model)


@contextlib.contextmanager
def _quiet_separator():
    """Silence audio-separator's per-chunk noise for the duration of a run.

    The library hard-codes its tqdm progress bars (no ``disable`` flag is
    threaded through), so they spew one line per update whenever stdout isn't a
    TTY — e.g. when output is piped to a file. We force-disable tqdm by patching
    its ``__init__`` (every ``from tqdm import tqdm`` shares this one class), then
    restore it. INFO logging is quieted separately via the ``Separator``'s
    ``log_level``. Our own rich progress bar reports overall job progress.
    """
    from tqdm import std as tqdm_std

    original_init = tqdm_std.tqdm.__init__

    def patched_init(self, *args, **kwargs):
        kwargs["disable"] = True
        original_init(self, *args, **kwargs)

    tqdm_std.tqdm.__init__ = patched_init
    try:
        yield
    finally:
        tqdm_std.tqdm.__init__ = original_init


def _classify_stem(filename: str) -> str:
    low = filename.lower()
    for keyword, canonical in _STEM_KEYWORDS:
        if keyword in low:
            return canonical
    return "other"


class UvrSeparator(BaseSeparator):
    name = "uvr"

    def available_stems(self, model: str) -> list[str]:
        # All curated models here are 2-stem vocal/instrumental separators.
        return list(_TWO_STEM)

    def separate(
        self,
        audio_path: Path,
        model: str,
        config: RunConfig,
        stems: list[str] | None = None,
    ) -> SeparationResult:
        from audio_separator.separator import Separator  # lazy import

        model_file = resolve_model_file(model)
        config.ensure_dirs()

        with tempfile.TemporaryDirectory() as tmp, _quiet_separator():
            separator = Separator(
                log_level=logging.WARNING,  # drop the per-run INFO banner spam
                model_file_dir=str(config.model_dir),
                output_dir=tmp,
                output_format="WAV",
                use_autocast=(config.device == "cuda"),
            )
            separator.load_model(model_filename=model_file)
            produced = separator.separate(str(audio_path))

            wanted = set(stems) if stems else None
            out: dict[str, np.ndarray] = {}
            sr = 44100
            for fname in produced:
                fpath = Path(fname)
                if not fpath.is_absolute():
                    fpath = Path(tmp) / fpath
                stem_name = _classify_stem(fpath.name)
                if wanted is not None and stem_name not in wanted:
                    continue
                audio, sr = load_audio(fpath)
                out[stem_name] = audio

        return SeparationResult(stems=out, sample_rate=sr)
