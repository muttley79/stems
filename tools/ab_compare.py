"""Run two separation configurations over the same input and compare them.

Quality changes here are only worth shipping if they are audible, so this writes
both stem sets side by side (drop them straight into a DAW) *and* prints the
measurements that catch the failure modes we actually hit:

- **peak / RMS per stem** - catches gain drift. audio-separator used to peak-limit
  every stem to 0.9 with a per-stem factor, which silently rescaled ensemble
  members against each other.
- **mix - sum(stems)** - the summability invariant `audio_io` documents and
  CLAUDE.md requires. Should sit at the float32 noise floor (about -160 dB).
  Anything near -15 dB means something is rescaling stems.
- **A-vs-B delta** - where the two runs actually differ, in dB and in time, so a
  change that claims to fix chunk seams can be checked at the seam positions.
- **chunk seam positions** - Roformer models infer in fixed chunks; with too
  little overlap the boundaries are audible. Printed as timestamps so you can
  jump to them.

This is a dev tool: it lives outside ``src/`` and nothing in ``stems`` imports it.

Usage
-----
    # compare inference settings
    python tools/ab_compare.py INPUT --preset vocals-max \
        --a "uvr_overlap=1,shifts=0" --b "uvr_overlap=4,shifts=1"

    # compare model line-ups (same settings)
    python tools/ab_compare.py INPUT --preset vocals-max --a "" \
        --models-a "instrumental=bs_roformer,inst_v2,inst_bleedless" \
        --models-b "instrumental=inst_gabox,inst_v2,inst_bleedless"

Each ``--a`` / ``--b`` is a comma-separated list of ``RunConfig`` field
assignments. Omitting ``--a`` compares the current defaults against themselves,
which is a useful determinism check but not much else.

Note that only ``RunConfig`` knobs can be varied. The engine's correctness fixes
(float32 intermediates, no peak normalization) are unconditional, so both runs
get them - ``uvr_overlap=1`` reverts the chunk-overlap behaviour alone, not the
whole pre-fix pipeline.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import fields, replace
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from stems.audio_io import load_audio, write_wav  # noqa: E402
from stems.config import RunConfig, resolve_device  # noqa: E402
from stems.pipeline import separate_to_result  # noqa: E402
from stems.presets import PRESETS, get_preset  # noqa: E402


def _rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(x.astype(np.float64) ** 2))) if x.size else 0.0


def _db(value: float, reference: float = 1.0) -> float:
    return 20 * np.log10(max(value, 1e-20) / max(reference, 1e-20))


def parse_overrides(spec: str | None) -> dict[str, object]:
    """Turn "uvr_overlap=4,shifts=1" into RunConfig kwargs, typed by the field."""
    if not spec:
        return {}
    types = {f.name: f.type for f in fields(RunConfig)}
    out: dict[str, object] = {}
    for item in spec.split(","):
        item = item.strip()
        if not item:
            continue
        if "=" not in item:
            raise SystemExit(f"override {item!r} is not of the form field=value")
        key, _, raw = item.partition("=")
        key, raw = key.strip(), raw.strip()
        if key not in types:
            raise SystemExit(
                f"unknown RunConfig field {key!r}; choose from {', '.join(sorted(types))}"
            )
        annotation = str(types[key])
        if "int" in annotation and "float" not in annotation:
            out[key] = int(raw)
        elif "float" in annotation:
            out[key] = None if raw.lower() == "none" else float(raw)
        else:
            out[key] = raw
    return out


def preset_variant(base: str, spec: str | None, label: str) -> str:
    """Register a scratch copy of ``base`` with different model lists.

    ``spec`` looks like ``instrumental=inst_gabox,inst_v2;vocal=kim_vocals``.
    Groups are separated by ``;`` because ``,`` separates model names. Returns
    the preset name to run - ``base`` itself when there is nothing to change.
    """
    if not spec:
        return base
    changes: dict[str, list[str]] = {}
    for group in spec.split(";"):
        group = group.strip()
        if not group:
            continue
        key, _, values = group.partition("=")
        models = [m.strip() for m in values.split(",") if m.strip()]
        field = {"vocal": "vocal_models", "instrumental": "instrumental_models",
                 "models": "models"}.get(key.strip())
        if not field:
            raise SystemExit(f"unknown model group {key!r}; use vocal, instrumental or models")
        changes[field] = models
    name = f"{base}-{label}"
    PRESETS[name] = replace(get_preset(base), name=name, **changes)
    return name


def run_one(
    label: str, audio_path: Path, out_root: Path, preset: str, overrides: dict, bitdepth: int
) -> tuple[dict[str, np.ndarray], int]:
    config = RunConfig(device=resolve_device("auto"), bitdepth=bitdepth, **overrides)
    shown = ", ".join(f"{k}={v}" for k, v in overrides.items()) or "defaults"
    print(f"\n=== {label}: {shown} ===", flush=True)

    result = separate_to_result(audio_path, preset=preset, config=config)

    out_dir = out_root / label
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, audio in result.stems.items():
        write_wav(out_dir / f"{name}.wav", audio, result.sample_rate, bitdepth)
    print(f"  wrote {len(result.stems)} stems to {out_dir}")
    return result.stems, result.sample_rate


def report_levels(label: str, stems: dict[str, np.ndarray]) -> None:
    print(f"\n{label} levels")
    for name, audio in sorted(stems.items()):
        peak = float(np.abs(audio).max()) if audio.size else 0.0
        flag = "  <-- above full scale" if peak > 1.0 else ""
        print(f"  {name:14s} peak {peak:7.4f} ({_db(peak):+7.2f} dBFS)   "
              f"rms {_db(_rms(audio)):+7.2f} dBFS{flag}")


def report_summability(label: str, stems: dict[str, np.ndarray], mix: np.ndarray) -> None:
    """mix - sum(stems), relative to the mix. Should be at the float32 floor."""
    arrays = list(stems.values())
    if not arrays:
        return
    n = min([a.shape[-1] for a in arrays] + [mix.shape[-1]])
    channels = max(a.shape[0] for a in arrays)
    total = np.zeros((channels, n), dtype=np.float64)
    for a in arrays:
        chunk = a[..., :n]
        if chunk.shape[0] == 1:
            chunk = np.repeat(chunk, channels, axis=0)
        total += chunk[:channels]
    residual = mix[:channels, :n] - total
    print(f"  {label}: mix - sum(stems) = {_db(_rms(residual), _rms(mix[:channels, :n])):+7.2f} dB "
          f"relative to the mix")


def report_delta(a: dict[str, np.ndarray], b: dict[str, np.ndarray], sr: int) -> None:
    print("\nA vs B")
    for name in sorted(set(a) & set(b)):
        x, y = a[name], b[name]
        n = min(x.shape[-1], y.shape[-1])
        ch = min(x.shape[0], y.shape[0])
        diff = y[:ch, :n] - x[:ch, :n]
        rel = _db(_rms(diff), _rms(x[:ch, :n]))
        # Where they diverge most, in case it lines up with a chunk seam.
        window = max(sr // 4, 1)
        energy = np.abs(diff).mean(axis=0)
        trimmed = energy[: (len(energy) // window) * window]
        loudest = ""
        if trimmed.size:
            blocks = trimmed.reshape(-1, window).mean(axis=1)
            at = int(blocks.argmax()) * window / sr
            loudest = f"   largest at {int(at // 60)}:{at % 60:05.2f}"
        print(f"  {name:14s} difference {rel:+7.2f} dB relative to A{loudest}")


def report_seams(stems: dict[str, np.ndarray], sr: int, chunk_seconds: float) -> None:
    if chunk_seconds <= 0:
        return
    total = max(a.shape[-1] for a in stems.values()) / sr
    marks = [f"{int(t // 60)}:{t % 60:05.2f}"
             for t in np.arange(chunk_seconds, total, chunk_seconds)][:12]
    print(f"\nChunk boundaries every {chunk_seconds:g}s (listen here for seams):")
    print("  " + "  ".join(marks) + ("  ..." if total / chunk_seconds > 12 else ""))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("input", type=Path)
    parser.add_argument("--preset", default="vocals-max")
    parser.add_argument("--a", default="uvr_overlap=1,shifts=0",
                        help="RunConfig overrides for run A (default: pre-fix behaviour)")
    parser.add_argument("--b", default="", help="RunConfig overrides for run B (default: current)")
    parser.add_argument("--models-a", default="",
                        help="model-list overrides for A, e.g. 'instrumental=bs_roformer,inst_v2'")
    parser.add_argument("--models-b", default="",
                        help="model-list overrides for B, e.g. 'instrumental=inst_gabox,inst_v2'")
    parser.add_argument("--out", type=Path, default=Path("_scratch/ab"))
    parser.add_argument("--bitdepth", type=int, default=32,
                        help="32 (float) keeps stems above full scale intact")
    parser.add_argument("--seam-seconds", type=float, default=8.0,
                        help="Roformer chunk length to mark; 0 to skip")
    args = parser.parse_args()

    out_root = args.out / args.input.stem
    mix, mix_sr = load_audio(args.input)

    a_stems, sr = run_one("A", args.input, out_root,
                          preset_variant(args.preset, args.models_a, "a"),
                          parse_overrides(args.a), args.bitdepth)
    b_stems, _ = run_one("B", args.input, out_root,
                         preset_variant(args.preset, args.models_b, "b"),
                         parse_overrides(args.b), args.bitdepth)

    if mix_sr != sr:
        mix, mix_sr = load_audio(args.input, target_sr=sr)

    report_levels("A", a_stems)
    report_levels("B", b_stems)
    print("\nSummability (float32 floor is about -160 dB)")
    report_summability("A", a_stems, mix)
    report_summability("B", b_stems, mix)
    report_delta(a_stems, b_stems, sr)
    report_seams(b_stems, sr, args.seam_seconds)
    print(f"\nBoth stem sets are under {out_root} - listen before trusting any number above.")


if __name__ == "__main__":
    main()
