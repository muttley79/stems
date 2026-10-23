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
import re
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
    # Registered for A/B (tools/ab_compare.py) but NOT yet verified end to end -
    # a first smoke test of big_beta4 returned no stems. Confirm before using.
    "big_beta4": "melband_roformer_big_beta4.ckpt",           # 12.52 vocals
    "big_beta5e": "melband_roformer_big_beta5e.ckpt",         # 12.35 vocals
    "vocal_fullness": "mel_band_roformer_vocal_fullness_aname.ckpt",  # fuller, less gated
    # INSTRUMENTAL-scoring models. Only inst_v2 and inst_bleedless actually
    # *target* the instrumental; the rest target vocals and derive instrumental
    # as `mix - vocals` (see `training.target_instrument` in each config), which
    # is why they carry vocal-model artifacts into the instrumental. Kept
    # separate on purpose - an ensemble of three vocals-target models is far less
    # diverse than it looks.
    "inst_v2": "melband_roformer_inst_v2.ckpt",               # 16.06, true target
    "inst_bleedless": "mel_band_roformer_instrumental_bleedless_v2_gabox.ckpt",  # true target
    "inst_gabox": "bs_roformer_vocals_gabox.ckpt",            # 17.21 but vocals-target
    "inst_duality_v2": "melband_roformer_instvox_duality_v2.ckpt",  # 16.08 / 11.01
    # Lead vs backing vocals ("karaoke" models: primary = lead vocal).
    "karaoke": "mel_band_roformer_karaoke_becruily.ckpt",
    "karaoke_gabox": "mel_band_roformer_karaoke_gabox_v2.ckpt",
    "karaoke_viperx": "mel_band_roformer_karaoke_aufr33_viperx_sdr_10.1956.ckpt",
    # Drum kit breakdown (MDX23C, 6 pieces).
    "drumsep": "MDX23C-DrumSep-aufr33-jarredou.ckpt",
    # Dedicated GUITAR model (Mel-Band Roformer, target=guitar). Its filename
    # advertises neither "mel" nor "roformer" and its config sets
    # mlp_expansion_factor: 1, so it needs both ensure_roformer_mlp_expansion_patch()
    # and ensure_roformer_arch_detection_patch() to load.
    "guitar": "becruily_guitar.ckpt",
}

# 2-stem models output vocals + instrumental.
_TWO_STEM = ["vocals", "instrumental"]

# Which stems each model actually produces. Everything not listed is a 2-stem
# vocals/instrumental separator (the common case), so only the exceptions need
# an entry. Mirrors ``demucs_engine._MODEL_STEMS``.
MODEL_STEMS: dict[str, list[str]] = {
    "guitar": ["guitar", "other"],
    "drumsep": ["kick", "snare", "toms", "hh", "ride", "crash"],
    # Karaoke models split an already-isolated vocal into lead and backing.
    "karaoke": ["lead_vocals", "back_vocals"],
    "karaoke_gabox": ["lead_vocals", "back_vocals"],
    "karaoke_viperx": ["lead_vocals", "back_vocals"],
}

# audio-separator names its outputs "{input_basename}_({Stem})_{model_name}.wav",
# so the *filename* contains the user's filename and the model name as well as
# the stem tag. Matching keywords against the whole name is therefore unsafe:
# "inst_bleedless" writes its vocals file as
# "track_(Vocals)_mel_band_roformer_instrumental_bleedless_v2_gabox.wav", where
# "instrumental" appears before "vocals". We pull out the parenthesised tag and
# match on that alone; the whole-name scan stays as a fallback for any model
# that doesn't follow the convention.
_STEM_TAG_RE = re.compile(r"_\(([^)]+)\)")

# Exact parenthesised tag (lowercased) -> canonical stem name. Checked before the
# substring scan below, so short tags like "hh" can't match inside a longer word.
_STEM_TAGS: dict[str, str] = {
    "kick": "kick",
    "snare": "snare",
    "toms": "toms",
    "hh": "hh",
    "ride": "ride",
    "crash": "crash",
}

# Some models emit stems whose tag doesn't describe what they actually are: a
# karaoke model is handed an isolated vocal and splits it, so its "vocals" file
# is the lead and its "instrumental" file is the backing vocals. Keyed by model
# alias, applied after classification.
_STEM_ALIASES: dict[str, dict[str, str]] = {
    alias: {"vocals": "lead_vocals", "instrumental": "back_vocals"}
    for alias in ("karaoke", "karaoke_gabox", "karaoke_viperx")
}

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
    (whose own ``__init__`` already supports it) - matching newer
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


def ensure_float_wav_output_patch() -> None:
    """Make audio-separator hand us float32 WAVs instead of truncated int16.

    Both of the library's writers are unusable for a multi-pass chain:

    - ``write_audio_pydub`` (the default) does
      ``(stem_source * 32767).astype(np.int16)`` - 16-bit, and *truncated*
      rather than rounded or dithered.
    - ``write_audio_soundfile`` (``use_soundfile=True``) calls
      ``sf.write(path, data, sr)`` with no ``subtype``, which libsndfile
      defaults to ``PCM_16`` for WAV anyway. It also never joins
      ``self.output_dir``, so in 0.30.2 it silently writes into the process CWD
      instead of the temp directory we asked for.

    Either way every stem gets quantized to 16 bits *before* being ensembled
    across six models, tail-rescued, and exported at 24 - which makes the
    24-bit export cosmetic. Rather than pick the lesser bug, take over
    ``write_audio`` for WAV targets and write float32 ourselves. This also skips
    the per-stem pydub/ffmpeg round-trip and the normalize call inside the pydub
    writer. Non-WAV targets fall through to the original.

    Same lazy, idempotent shape as :func:`ensure_roformer_mlp_expansion_patch`.
    """
    try:  # lazy: only touch the heavy backend when actually separating
        from audio_separator.separator.common_separator import CommonSeparator
    except Exception:
        return  # backend not importable here; nothing to patch

    if getattr(CommonSeparator.write_audio, "_stems_float_patch", False):
        return

    original = CommonSeparator.write_audio

    def write_audio(self, stem_path: str, stem_source):
        import os

        import soundfile as sf

        if not str(stem_path).lower().endswith(".wav"):
            return original(self, stem_path, stem_source)

        if self.output_dir:
            os.makedirs(self.output_dir, exist_ok=True)
            stem_path = os.path.join(self.output_dir, stem_path)
        # stem_source arrives as (samples, channels) float.
        sf.write(
            stem_path,
            np.ascontiguousarray(stem_source, dtype=np.float32),
            self.sample_rate,
            subtype="FLOAT",
        )

    write_audio._stems_float_patch = True
    CommonSeparator.write_audio = write_audio


def ensure_no_normalize_patch() -> None:
    """Stop audio-separator from peak-limiting stems behind our back.

    ``normalization_threshold`` can only be lowered to 1.0 (the constructor
    rejects anything above), which still rescales any stem that exceeds full
    scale - and separated stems routinely do, since a Roformer's instrumental is
    ``mix - vocals``. Measured on a loud master, the vocals came back 0.84 dB
    quieter and the instrumental 0.91 dB quieter: *different* factors, so
    averaging three instrumental models mixes three different gains, and
    ``mix - instrumental`` (what :func:`stems.ensemble.rescue_vocal_tails` treats
    as vocal tails) picks up a scaled copy of the whole band.

    Neutralizing it restores the invariant `audio_io` documents and CLAUDE.md
    requires: stems sum back to the mix. Stems may then legitimately peak above
    1.0; :func:`stems.audio_io.write_wav` warns when that would clip an integer
    PCM export, and ``--bitdepth 32`` avoids it entirely.
    """
    try:  # lazy: only touch the heavy backend when actually separating
        from audio_separator.separator.uvr_lib_v5 import spec_utils
    except Exception:
        return  # backend not importable here; nothing to patch

    if getattr(spec_utils.normalize, "_stems_noop_patch", False):
        return

    def normalize(wave, max_peak=1.0, min_peak=None):
        return wave

    normalize._stems_noop_patch = True
    spec_utils.normalize = normalize


def apply_chunk_overlap(separator, config: RunConfig) -> None:
    """Give each Roformer model a real overlap-add step. Call after ``load_model``.

    audio-separator's ``mdxc_params["overlap"]`` means two different things
    depending on the branch taken in ``MDXCSeparator.demix``: for MDX23C it is a
    divisor (``hop_size = chunk_size // overlap``), but for Roformer models it is
    a **step in seconds** (``step = int(self.overlap * sample_rate)``). The
    library default of ``8`` is sane as a divisor and badly wrong as a step:

    - ``bs_roformer`` and ``kim_ft`` have ``dim_t: 801, hop_length: 441``, so
      their chunk is ``441 * 800 = 352_800`` samples - exactly 8.000 s. Step ==
      chunk means the Hamming window and its counter cancel, and the output is a
      raw concatenation of independently-inferred 8 s blocks: a hard seam every
      8 seconds. Their own configs ask for ``num_overlap`` 4 and 8.
    - The 11 s-chunk models get only ~27 % where their configs ask for 2-4x.

    So we derive the step from each model's own ``inference.num_overlap`` rather
    than any global guess, with a floor of 2 (a config asking for 1 would
    otherwise be a regression against the accidental 27 %) and a ceiling of
    ``config.uvr_overlap`` so the user can cap the cost - runtime is roughly
    linear in the overlap factor, and ``vocals-max`` is already six passes.
    ``config.uvr_overlap <= 1`` leaves the library default untouched.

    No-op for non-Roformer models, where the value is a divisor and correct.
    """
    if config.uvr_overlap <= 1:
        return

    instance = getattr(separator, "model_instance", None)
    if instance is None or not getattr(instance, "is_roformer", False):
        return
    cfg = getattr(instance, "model_data_cfgdict", None)
    if cfg is None or getattr(instance, "override_model_segment_size", False):
        return

    try:
        sample_rate = int(cfg.audio.sample_rate)
        chunk_seconds = cfg.audio.hop_length * (cfg.inference.dim_t - 1) / sample_rate
        wanted = int(cfg.inference.get("num_overlap", 2) or 2)
    except Exception:  # unfamiliar config layout; leave the default alone
        return

    factor = max(2, min(wanted, config.uvr_overlap))
    instance.overlap = chunk_seconds / factor


def ensure_roformer_arch_detection_patch() -> None:
    """Detect Mel-Band vs BS Roformer from nested config keys, not just the top level.

    audio-separator >= 0.44 picks the Roformer class by looking for
    ``freqs_per_bands`` (BS) or ``num_bands``/``n_mels``/``mel_bands`` (Mel-Band)
    at the *top level* of the config dict - but real checkpoint configs nest
    those under ``model:``, alongside ``audio:``/``training:``/``inference:``.
    When the filename also lacks a "mel...roformer" hint, detection returns None
    and silently defaults to BS. It then applies BS defaults, which injects a
    ``freqs_per_bands`` summing to 126, and construction dies with
    "the number of freqs in the bands must equal 1025 ... but got 126".

    ``becruily_guitar.ckpt`` (Mel-Band, ``num_bands: 60``, no "mel" in its name)
    hits this exactly, taking out the ``6stem-max`` guitar pass. Searching the
    nested dicts fixes it for that model and any other community checkpoint
    whose filename doesn't advertise its architecture.
    """
    try:  # lazy: only touch the heavy backend when actually separating
        from audio_separator.separator.roformer.configuration_normalizer import (
            ConfigurationNormalizer,
        )
    except Exception:
        return  # older build without the new loader; nothing to patch

    if getattr(ConfigurationNormalizer.detect_model_type, "_stems_arch_patch", False):
        return

    original = ConfigurationNormalizer.detect_model_type

    def _scan(node, depth: int = 0) -> str | None:
        if not isinstance(node, dict) or depth > 3:
            return None
        if "freqs_per_bands" in node:
            return "bs_roformer"
        if {"num_bands", "n_mels", "mel_bands"} & set(node):
            return "mel_band_roformer"
        for value in node.values():
            found = _scan(value, depth + 1)
            if found:
                return found
        return None

    def detect_model_type(self, config):
        return original(self, config) or _scan(config)

    detect_model_type._stems_arch_patch = True
    ConfigurationNormalizer.detect_model_type = detect_model_type


@contextlib.contextmanager
def _quiet_separator():
    """Silence audio-separator's per-chunk noise for the duration of a run.

    The library hard-codes its tqdm progress bars (no ``disable`` flag is
    threaded through), so they spew one line per update whenever stdout isn't a
    TTY - e.g. when output is piped to a file. We force-disable tqdm by patching
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
    """Map an audio-separator output filename to a canonical stem name.

    Prefers the parenthesised stem tag (see :data:`_STEM_TAG_RE`) so neither the
    model name nor the user's own filename can shadow it; falls back to scanning
    the whole name when no tag is present.
    """
    match = _STEM_TAG_RE.search(filename)
    if match:
        tag = match.group(1).strip().lower()
        if tag in _STEM_TAGS:
            return _STEM_TAGS[tag]
    candidates = [match.group(1)] if match else []
    candidates.append(filename)
    for candidate in candidates:
        low = candidate.lower()
        for keyword, canonical in _STEM_KEYWORDS:
            if keyword in low:
                return canonical
    return "other"


class UvrSeparator(BaseSeparator):
    name = "uvr"

    def available_stems(self, model: str) -> list[str]:
        # Most curated models are 2-stem vocal/instrumental separators; the
        # exceptions (e.g. the guitar model) are listed in MODEL_STEMS.
        return list(MODEL_STEMS.get(model, _TWO_STEM))

    def separate(
        self,
        audio_path: Path,
        model: str,
        config: RunConfig,
        stems: list[str] | None = None,
    ) -> SeparationResult:
        from audio_separator.separator import Separator  # lazy import

        ensure_roformer_mlp_expansion_patch()
        ensure_roformer_arch_detection_patch()
        ensure_float_wav_output_patch()
        ensure_no_normalize_patch()
        model_file = resolve_model_file(model)
        config.ensure_dirs()

        with tempfile.TemporaryDirectory() as tmp:
            separator = Separator(
                log_level=logging.WARNING,  # drop the per-run INFO banner spam
                model_file_dir=str(config.model_dir),
                output_dir=tmp,
                output_format="WAV",
                use_autocast=(config.device == "cuda"),
                # Keep levels untouched. audio-separator otherwise peak-limits to
                # 0.9 three times per pass (input mix, each stem, and again at
                # write), which fires on any master peaking above -0.9 dBFS. Each
                # model then attenuates by its own factor, so ensembling averages
                # differently-scaled signals and `mix - instrumental` (the
                # residual tail rescue works from) picks up a scaled copy of the
                # whole band. 1.0 is the largest value the validator accepts and
                # makes all three sites no-ops for non-clipping audio.
                # Belt and braces alongside ensure_no_normalize_patch(): 1.0 is
                # the lowest-impact value the constructor will accept.
                normalization_threshold=1.0,
                amplification_threshold=0.0,
                # Leave use_soundfile at its default: ensure_float_wav_output_patch()
                # replaces write_audio outright, so neither built-in writer runs.
            )
            # load_model is left *outside* _quiet_separator so a first-run weight
            # download keeps its own progress bar; only the per-chunk inference
            # tqdm spam is silenced (our rich bar reports overall job progress).
            separator.load_model(model_filename=model_file)
            apply_chunk_overlap(separator, config)
            with _quiet_separator():
                produced = separator.separate(str(audio_path))

            wanted = set(stems) if stems else None
            aliases = _STEM_ALIASES.get(model, {})
            out: dict[str, np.ndarray] = {}
            sr = 44100
            for fname in produced:
                fpath = Path(fname)
                if not fpath.is_absolute():
                    fpath = Path(tmp) / fpath
                stem_name = _classify_stem(fpath.name)
                stem_name = aliases.get(stem_name, stem_name)
                if wanted is not None and stem_name not in wanted:
                    continue
                audio, sr = load_audio(fpath)
                out[stem_name] = audio

        return SeparationResult(stems=out, sample_rate=sr)
