"""Orchestration: turn a preset (or raw model) into exported stem files.

This is the heart of the tool. It resolves which engine(s) to run, executes the
plan (single / ensemble / cascade), optionally filters to requested stems, and
writes each stem to disk via :mod:`stems.audio_io`.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np

from stems.audio_io import export_stem, write_wav
from stems.config import RunConfig
from stems.engines.base import BaseSeparator, SeparationResult
from stems.engines.demucs_engine import DemucsSeparator
from stems.engines.uvr_engine import UvrSeparator
from stems.ensemble import ensemble_results
from stems.presets import Preset, get_preset

# Engines are cheap to construct (heavy work is lazy in .separate).
_ENGINES: dict[str, BaseSeparator] = {
    "demucs": DemucsSeparator(),
    "uvr": UvrSeparator(),
}


def get_engine(name: str) -> BaseSeparator:
    try:
        return _ENGINES[name]
    except KeyError:
        raise KeyError(f"Unknown engine '{name}'. Available: {', '.join(_ENGINES)}")


def _run_single(
    audio_path: Path, engine: str, model: str, config: RunConfig,
    stems: list[str] | None,
) -> SeparationResult:
    return get_engine(engine).separate(audio_path, model, config, stems)


def _run_ensemble(
    audio_path: Path, preset: Preset, config: RunConfig, stems: list[str] | None,
) -> SeparationResult:
    results = [
        get_engine(preset.engine).separate(audio_path, m, config, stems)
        for m in preset.models
    ]
    return ensemble_results(results, method=preset.ensemble_method)


def _ensemble_stem(
    audio_path: Path, models: list[str], stem: str, config: RunConfig, method: str,
) -> SeparationResult:
    """Run each model isolating ``stem``, then merge. Returns a single-stem result."""
    results = [
        get_engine("uvr").separate(audio_path, m, config, stems=[stem])
        for m in models
    ]
    return ensemble_results(results, method=method)


def _run_twostem(
    audio_path: Path, preset: Preset, config: RunConfig, stems: list[str] | None,
) -> SeparationResult:
    """Clean 2-stem: ensemble vocal models for vocals, instrumental models for
    instrumental, each merged independently. Avoids dragging a strong model down
    with a weaker one and uses purpose-built instrumental models to kill bleed.
    """
    want_vocals = stems is None or "vocals" in stems
    want_instrumental = stems is None or "instrumental" in stems

    merged: dict[str, np.ndarray] = {}
    sr = 44100

    if want_vocals:
        res = _ensemble_stem(
            audio_path, preset.vocal_models, "vocals", config, preset.vocal_method
        )
        merged["vocals"] = res.stems["vocals"]
        sr = res.sample_rate

    if want_instrumental:
        res = _ensemble_stem(
            audio_path, preset.instrumental_models, "instrumental", config,
            preset.ensemble_method,
        )
        merged["instrumental"] = res.stems["instrumental"]
        sr = res.sample_rate

    return SeparationResult(stems=merged, sample_rate=sr)


def _run_cascade(
    audio_path: Path, preset: Preset, config: RunConfig, stems: list[str] | None,
) -> SeparationResult:
    """Best 4-stem: build the cleanest instrumental (ensemble of dedicated
    instrumental models), then run Demucs on it for drums/bass/other. Vocals come
    from the vocal-model ensemble and are only computed when actually requested,
    so ``--stems drums,bass,other`` skips the vocal passes entirely.
    """
    want_vocals = stems is None or "vocals" in stems
    drum_stems = ("drums", "bass", "other")
    want_drumset = stems is None or any(s in stems for s in drum_stems)

    merged: dict[str, np.ndarray] = {}
    sr = 44100

    # Clean instrumental via the strong instrumental ensemble (same quality as
    # the 'amazing' vocals-max instrumental).
    inst = _ensemble_stem(
        audio_path, preset.instrumental_models, "instrumental", config,
        preset.ensemble_method,
    )
    instrumental = inst.stems["instrumental"]
    sr = inst.sample_rate

    if want_vocals:
        voc = _ensemble_stem(
            audio_path, preset.vocal_models, "vocals", config, preset.vocal_method
        )
        merged["vocals"] = voc.stems["vocals"]

    if want_drumset:
        # Demucs 4-stem on the clean instrumental -> tight drums/bass/other.
        with tempfile.TemporaryDirectory() as tmp:
            residual_path = Path(tmp) / "residual.wav"
            write_wav(residual_path, instrumental, sr, bitdepth=32)
            demucs_res = get_engine(preset.engine).separate(
                residual_path, preset.models[0], config, stems=None
            )
        for name in drum_stems:
            if name in demucs_res.stems:
                merged[name] = demucs_res.stems[name]

    if stems:
        merged = {k: v for k, v in merged.items() if k in stems}
    return SeparationResult(stems=merged, sample_rate=sr)


def separate_to_result(
    audio_path: Path,
    config: RunConfig,
    preset: str | None = None,
    engine: str | None = None,
    model: str | None = None,
    stems: list[str] | None = None,
) -> SeparationResult:
    """Resolve the plan and produce stems in memory.

    Precedence: an explicit ``model`` (with optional ``engine``) overrides the
    preset. Otherwise the named ``preset`` plan runs.
    """
    if model is not None:
        eng = engine or _infer_engine_for_model(model)
        return _run_single(audio_path, eng, model, config, stems)

    p = get_preset(preset) if preset else get_preset(_default_preset())
    if p.kind == "single":
        return _run_single(audio_path, p.engine, p.models[0], config, stems)
    if p.kind == "ensemble":
        return _run_ensemble(audio_path, p, config, stems)
    if p.kind == "twostem":
        return _run_twostem(audio_path, p, config, stems)
    if p.kind == "cascade":
        return _run_cascade(audio_path, p, config, stems)
    raise ValueError(f"Unsupported preset kind: {p.kind}")


def separate_file(
    audio_path: Path,
    out_dir: Path,
    config: RunConfig,
    preset: str | None = None,
    engine: str | None = None,
    model: str | None = None,
    stems: list[str] | None = None,
    fmt: str = "both",
) -> list[Path]:
    """Separate ``audio_path`` and write stems into ``out_dir``. Returns paths."""
    result = separate_to_result(audio_path, config, preset, engine, model, stems)
    written: list[Path] = []
    for name, audio in result.stems.items():
        written += export_stem(
            out_dir, name, audio, result.sample_rate,
            fmt=fmt, bitdepth=config.bitdepth, mp3_bitrate=config.mp3_bitrate,
        )
    return written


def _infer_engine_for_model(model: str) -> str:
    """Guess the backend for a raw model name."""
    demucs_like = ("htdemucs", "mdx_extra", "demucs")
    if any(model.startswith(p) or model == p for p in demucs_like):
        return "demucs"
    return "uvr"


def _default_preset() -> str:
    from stems.presets import DEFAULT_PRESET

    return DEFAULT_PRESET
