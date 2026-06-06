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
    # Dedicated GUITAR model (Mel-Band Roformer, target=guitar). Community model;
    # registered via models/download_checks.json. Needs the mlp_expansion_factor
    # shim below to load on older audio-separator builds.
    "guitar": "becruily_guitar.ckpt",
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
    # Check specific instruments before the generic "other": a guitar/piano file
    # tagged like "..._(Guitar)..." must not be swallowed by an "other" match.
    ("guitar", "guitar"),
    ("piano", "piano"),
    ("other", "other"),
]


def resolve_model_file(model: str) -> str:
    """Map a friendly alias to a checkpoint filename (pass through if unknown)."""
    return MODEL_FILES.get(model, model)


def ensure_roformer_mlp_expansion_patch() -> None:
    """Let ``MelBandRoformer`` accept a config ``mlp_expansion_factor``.

    Older ``audio-separator`` builds hardcode the mask-estimator MLP expansion
    (4) and don't accept it as a constructor kwarg, so a Roformer checkpoint
    trained with a different value (e.g. the becruily guitar model, which ships
    ``mlp_expansion_factor: 1`` in its config) fails to load with
    ``unexpected keyword argument 'mlp_expansion_factor'``. We wrap the
    constructor to consume that kwarg and thread it into each ``MaskEstimator``
    (whose own ``__init__`` already supports it) — matching newer
    audio-separator behavior without editing the installed package. Idempotent,
    lazy, and a no-op for configs that omit the key (default 4 = original
    behavior), so it never affects the other models.
    """
    try:  # lazy: only touch the heavy backend when actually separating
        from audio_separator.separator.uvr_lib_v5.roformer import (
            mel_band_roformer as mbr,
        )
    except Exception:
        return  # backend not importable here; nothing to patch

    if getattr(mbr.MelBandRoformer.__init__, "_stems_mlp_patch", False):
        return

    orig_model_init = mbr.MelBandRoformer.__init__
    orig_mask_init = mbr.MaskEstimator.__init__

    def model_init(self, *args, mlp_expansion_factor=4, **kwargs):
        # While the real constructor builds its MaskEstimators (without passing
        # the factor), temporarily inject our value as the MaskEstimator default.
        def mask_init(mask_self, *m_args, **m_kwargs):
            m_kwargs.setdefault("mlp_expansion_factor", mlp_expansion_factor)
            orig_mask_init(mask_self, *m_args, **m_kwargs)

        mbr.MaskEstimator.__init__ = mask_init
        try:
            orig_model_init(self, *args, **kwargs)
        finally:
            mbr.MaskEstimator.__init__ = orig_mask_init

    model_init._stems_mlp_patch = True
    mbr.MelBandRoformer.__init__ = model_init


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

        ensure_roformer_mlp_expansion_patch()
        model_file = resolve_model_file(model)
        config.ensure_dirs()

        with tempfile.TemporaryDirectory() as tmp:
            separator = Separator(
                log_level=logging.WARNING,  # drop the per-run INFO banner spam
                model_file_dir=str(config.model_dir),
                output_dir=tmp,
                output_format="WAV",
                use_autocast=(config.device == "cuda"),
            )
            # load_model is left *outside* _quiet_separator so a first-run weight
            # download keeps its own progress bar; only the per-chunk inference
            # tqdm spam is silenced (our rich bar reports overall job progress).
            separator.load_model(model_filename=model_file)
            with _quiet_separator():
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
