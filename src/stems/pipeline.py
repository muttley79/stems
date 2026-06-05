"""Orchestration: turn a preset (or raw model) into exported stem files.

This is the heart of the tool. It resolves which engine(s) to run, executes the
plan (single / ensemble / cascade), optionally filters to requested stems, and
writes each stem to disk via :mod:`stems.audio_io`.
"""

from __future__ import annotations

import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

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


@dataclass(frozen=True, slots=True)
class Step:
    """One model pass within a plan, reported just before it starts.

    ``index``/``total`` are 1-based ("pass 2 of 6"); ``model`` is the friendly
    model name and ``action`` says what it is doing (e.g. "isolating vocals").
    """

    index: int
    total: int
    model: str
    action: str


# Called before each separation pass so callers (the CLI) can show live progress.
StepCallback = Callable[[Step], None]


class _StepTracker:
    """Counts passes and emits a :class:`Step` to the callback before each one."""

    def __init__(self, callback: StepCallback | None, total: int) -> None:
        self._cb = callback
        self.total = total
        self._i = 0

    def begin(self, model: str, action: str) -> None:
        self._i += 1
        if self._cb is not None:
            self._cb(Step(self._i, self.total, model, action))


def get_engine(name: str) -> BaseSeparator:
    try:
        return _ENGINES[name]
    except KeyError:
        raise KeyError(f"Unknown engine '{name}'. Available: {', '.join(_ENGINES)}")


def _run_single(
    audio_path: Path, engine: str, model: str, config: RunConfig,
    stems: list[str] | None, on_step: StepCallback | None = None,
) -> SeparationResult:
    action = "separating vocals + instrumental" if engine == "uvr" else "splitting stems"
    _StepTracker(on_step, 1).begin(model, action)
    return get_engine(engine).separate(audio_path, model, config, stems)


def _run_ensemble(
    audio_path: Path, preset: Preset, config: RunConfig, stems: list[str] | None,
    on_step: StepCallback | None = None,
) -> SeparationResult:
    tracker = _StepTracker(on_step, len(preset.models))
    results = []
    for m in preset.models:
        tracker.begin(m, "separating")
        results.append(get_engine(preset.engine).separate(audio_path, m, config, stems))
    return ensemble_results(results, method=preset.ensemble_method)


def _ensemble_stem(
    audio_path: Path, models: list[str], stem: str, config: RunConfig, method: str,
    tracker: _StepTracker | None = None,
) -> SeparationResult:
    """Run each model isolating ``stem``, then merge. Returns a single-stem result."""
    results = []
    for m in models:
        if tracker is not None:
            tracker.begin(m, f"isolating {stem}")
        results.append(get_engine("uvr").separate(audio_path, m, config, stems=[stem]))
    return ensemble_results(results, method=method)


def _run_twostem(
    audio_path: Path, preset: Preset, config: RunConfig, stems: list[str] | None,
    on_step: StepCallback | None = None,
) -> SeparationResult:
    """Clean 2-stem: ensemble vocal models for vocals, instrumental models for
    instrumental, each merged independently. Avoids dragging a strong model down
    with a weaker one and uses purpose-built instrumental models to kill bleed.
    """
    want_vocals = stems is None or "vocals" in stems
    want_instrumental = stems is None or "instrumental" in stems

    total = (len(preset.vocal_models) if want_vocals else 0) + (
        len(preset.instrumental_models) if want_instrumental else 0
    )
    tracker = _StepTracker(on_step, total)

    merged: dict[str, np.ndarray] = {}
    sr = 44100

    if want_vocals:
        res = _ensemble_stem(
            audio_path, preset.vocal_models, "vocals", config, preset.vocal_method,
            tracker,
        )
        merged["vocals"] = res.stems["vocals"]
        sr = res.sample_rate

    if want_instrumental:
        res = _ensemble_stem(
            audio_path, preset.instrumental_models, "instrumental", config,
            preset.ensemble_method, tracker,
        )
        merged["instrumental"] = res.stems["instrumental"]
        sr = res.sample_rate

    return SeparationResult(stems=merged, sample_rate=sr)


def _run_cascade(
    audio_path: Path, preset: Preset, config: RunConfig, stems: list[str] | None,
    on_step: StepCallback | None = None,
) -> SeparationResult:
    """Best 4-stem: build the cleanest instrumental (ensemble of dedicated
    instrumental models), then run Demucs on it for drums/bass/other. Vocals come
    from the vocal-model ensemble and are only computed when actually requested,
    so ``--stems drums,bass,other`` skips the vocal passes entirely.
    """
    want_vocals = stems is None or "vocals" in stems
    drum_stems = ("drums", "bass", "other")
    want_drumset = stems is None or any(s in stems for s in drum_stems)

    total = (
        len(preset.instrumental_models)
        + (len(preset.vocal_models) if want_vocals else 0)
        + (1 if want_drumset else 0)
    )
    tracker = _StepTracker(on_step, total)

    merged: dict[str, np.ndarray] = {}
    sr = 44100

    # Clean instrumental via the strong instrumental ensemble (same quality as
    # the 'amazing' vocals-max instrumental).
    inst = _ensemble_stem(
        audio_path, preset.instrumental_models, "instrumental", config,
        preset.ensemble_method, tracker,
    )
    instrumental = inst.stems["instrumental"]
    sr = inst.sample_rate

    if want_vocals:
        voc = _ensemble_stem(
            audio_path, preset.vocal_models, "vocals", config, preset.vocal_method,
            tracker,
        )
        merged["vocals"] = voc.stems["vocals"]

    if want_drumset:
        # Demucs 4-stem on the clean instrumental -> tight drums/bass/other.
        with tempfile.TemporaryDirectory() as tmp:
            residual_path = Path(tmp) / "residual.wav"
            write_wav(residual_path, instrumental, sr, bitdepth=32)
            tracker.begin(preset.models[0], "splitting drums/bass/other")
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
    on_step: StepCallback | None = None,
) -> SeparationResult:
    """Resolve the plan and produce stems in memory.

    Precedence: an explicit ``model`` (with optional ``engine``) overrides the
    preset. Otherwise the named ``preset`` plan runs. ``on_step`` is invoked
    before each model pass for progress reporting.
    """
    if model is not None:
        eng = engine or _infer_engine_for_model(model)
        return _run_single(audio_path, eng, model, config, stems, on_step)

    p = get_preset(preset) if preset else get_preset(_default_preset())
    if p.kind == "single":
        return _run_single(audio_path, p.engine, p.models[0], config, stems, on_step)
    if p.kind == "ensemble":
        return _run_ensemble(audio_path, p, config, stems, on_step)
    if p.kind == "twostem":
        return _run_twostem(audio_path, p, config, stems, on_step)
    if p.kind == "cascade":
        return _run_cascade(audio_path, p, config, stems, on_step)
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
    on_step: StepCallback | None = None,
) -> list[Path]:
    """Separate ``audio_path`` and write stems into ``out_dir``. Returns paths."""
    result = separate_to_result(
        audio_path, config, preset, engine, model, stems, on_step
    )
    written: list[Path] = []
    for name, audio in result.stems.items():
        written += export_stem(
            out_dir, name, audio, result.sample_rate,
            fmt=fmt, bitdepth=config.bitdepth, mp3_bitrate=config.mp3_bitrate,
        )
    return written


def iter_required_models(
    preset: str | None = None,
    engine: str | None = None,
    model: str | None = None,
) -> list[tuple[str, str]]:
    """List the ``(engine, model)`` pairs a run will load, for weight prefetch.

    Mirrors the plan resolution in :func:`separate_to_result` but only collects
    model identities (no separation). Order is the order they'll be used; the
    list is de-duplicated. Used by :func:`prefetch_models` so missing weights can
    be downloaded up front with visible progress, before the batch's live display
    starts. Models filtered out by a ``--stems`` subset are still listed (a small
    over-fetch is cheaper than duplicating the want/skip logic here).
    """
    if model is not None:
        return [(engine or _infer_engine_for_model(model), model)]

    p = get_preset(preset) if preset else get_preset(_default_preset())
    pairs: list[tuple[str, str]] = []
    if p.kind == "single":
        pairs.append((p.engine, p.models[0]))
    elif p.kind == "ensemble":
        pairs += [(p.engine, m) for m in p.models]
    elif p.kind == "twostem":
        pairs += [("uvr", m) for m in p.vocal_models]
        pairs += [("uvr", m) for m in p.instrumental_models]
    elif p.kind == "cascade":
        pairs += [("uvr", m) for m in p.instrumental_models]
        pairs += [("uvr", m) for m in p.vocal_models]
        pairs.append((p.engine, p.models[0]))

    seen: set[tuple[str, str]] = set()
    uniq: list[tuple[str, str]] = []
    for pe in pairs:
        if pe not in seen:
            seen.add(pe)
            uniq.append(pe)
    return uniq


def prefetch_models(
    pairs: list[tuple[str, str]], config: RunConfig, console=None
) -> None:
    """Download any missing model weights, with the backends' own progress bars.

    Runs *before* the batch's live progress display and outside the UVR quiet
    patch, so download bars render cleanly instead of being silenced (UVR's tqdm
    is force-disabled during a run) or garbled by the rich live display (Demucs).
    A failure here is non-fatal: we warn and let the real pass try again, so a
    transient hiccup or an unexpected cache layout never aborts the batch.
    """
    config.ensure_dirs()
    for engine, model in pairs:
        try:
            if engine == "uvr":
                _prefetch_uvr(model, config, console)
            elif engine == "demucs":
                _prefetch_demucs(model, config, console)
        except Exception as exc:  # never let prefetch sink the run
            if console is not None:
                console.print(
                    f"[yellow]Could not prefetch {model}: {exc} "
                    f"(will retry during separation).[/yellow]"
                )


def _say_downloading(console, model: str) -> None:
    msg = f"[cyan]↓ Downloading model[/cyan] [magenta]{model}[/magenta] …"
    if console is not None:
        console.print(msg)
    else:
        print(msg)


def _prefetch_uvr(model: str, config: RunConfig, console) -> None:
    from stems.engines.uvr_engine import resolve_model_file

    ckpt = config.model_dir / resolve_model_file(model)
    if ckpt.exists():
        return
    _say_downloading(console, model)
    import logging
    import tempfile

    from audio_separator.separator import Separator  # lazy: only when missing

    with tempfile.TemporaryDirectory() as tmp:
        sep = Separator(
            log_level=logging.WARNING,
            model_file_dir=str(config.model_dir),
            output_dir=tmp,
        )
        sep.load_model(model_filename=resolve_model_file(model))


def _demucs_cached(model: str) -> bool:
    """Best-effort: are Demucs weights already in the torch hub cache?

    We can't know the exact signature filenames without loading the model, so we
    treat a populated ``<hub>/checkpoints`` dir as cached. First-ever run: dir is
    empty/absent, so the slow download is prefetched with a visible bar; later
    runs skip the prefetch and load normally. (Switching to a not-yet-downloaded
    Demucs model when the cache is already populated falls back to the old
    in-pipeline download — acceptable for this rare case.)
    """
    try:
        import torch

        ckpt_dir = Path(torch.hub.get_dir()) / "checkpoints"
        return ckpt_dir.is_dir() and any(ckpt_dir.glob("*.th"))
    except Exception:
        return False


def _prefetch_demucs(model: str, config: RunConfig, console) -> None:
    if _demucs_cached(model):
        return
    _say_downloading(console, model)
    from demucs.pretrained import get_model  # lazy: only when likely missing

    get_model(name=model)  # triggers torch.hub download with its own progress bar


def _infer_engine_for_model(model: str) -> str:
    """Guess the backend for a raw model name."""
    demucs_like = ("htdemucs", "mdx_extra", "demucs")
    if any(model.startswith(p) or model == p for p in demucs_like):
        return "demucs"
    return "uvr"


def _default_preset() -> str:
    from stems.presets import DEFAULT_PRESET

    return DEFAULT_PRESET
