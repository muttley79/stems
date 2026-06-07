"""Combine outputs of multiple models into a single, higher-quality result.

Two strategies:
- ``average``  - per-sample mean of aligned waveforms. Robust, reduces random
  artifacts, the safe default for the "-max" presets.
- ``max_spec`` - keep, per time-frequency bin, the larger magnitude across
  models, but reconstruct with a single *coherent* phase (the angle of the
  summed spectra) rather than the per-bin winner's phase. Retains the fuller,
  less-gated magnitude of a max combine while avoiding the artifact the naive
  version caused: taking phase from whichever model won each bin makes the phase
  of a held harmonic jump whenever the winner flips between frames, which
  overlap-add turns into an audible amplitude flutter on sustained, multi-voice
  content (e.g. a choir). Falls back to ``average`` if torch is unavailable.

All inputs must share the same set of stem names; lengths are aligned by
trimming to the shortest (models can differ by a few samples at the tail).
"""

from __future__ import annotations

import numpy as np

from stems.engines.base import SeparationResult


def _align_length(arrays: list[np.ndarray]) -> list[np.ndarray]:
    n = min(a.shape[-1] for a in arrays)
    return [a[..., :n] for a in arrays]


def _match_channels(arrays: list[np.ndarray]) -> list[np.ndarray]:
    ch = max(a.shape[0] for a in arrays)
    out = []
    for a in arrays:
        if a.shape[0] == ch:
            out.append(a)
        elif a.shape[0] == 1:
            out.append(np.repeat(a, ch, axis=0))
        else:
            out.append(a[:ch])
    return out


def _average(arrays: list[np.ndarray]) -> np.ndarray:
    arrays = _match_channels(_align_length(arrays))
    return np.mean(np.stack(arrays, axis=0), axis=0).astype(np.float32)


def sum_stems(arrays: list[np.ndarray]) -> np.ndarray:
    """Sum aligned stem waveforms into one (a partial remix). float32, (ch, n).

    Unlike the ensemble combiners this *adds* rather than averages: stems carry
    no normalization, so summing a subset reconstructs exactly that part of the
    mix. Lengths/channels are aligned first (cascade stems can differ by a few
    tail samples).
    """
    arrays = _match_channels(_align_length(arrays))
    return np.sum(np.stack(arrays, axis=0), axis=0).astype(np.float32)


def _max_spec(arrays: list[np.ndarray]) -> np.ndarray:
    try:
        import torch
    except Exception:
        return _average(arrays)

    arrays = _match_channels(_align_length(arrays))
    n_fft, hop = 4096, 1024
    window = torch.hann_window(n_fft)
    specs = []
    for a in arrays:
        t = torch.from_numpy(np.ascontiguousarray(a))
        specs.append(
            torch.stft(t, n_fft=n_fft, hop_length=hop, window=window, return_complex=True)
        )
    stack = torch.stack(specs, dim=0)        # (models, channels, freq, frames)
    max_mag = stack.abs().amax(dim=0)        # per-bin loudest magnitude
    # One coherent phase for every bin (angle of the summed spectra) instead of
    # the per-bin winner's phase: when the loudest model changes between frames
    # the phase no longer jumps, so a sustained harmonic stays smooth.
    phase = torch.angle(stack.sum(dim=0))
    best_spec = torch.polar(max_mag, phase)
    out = torch.istft(
        best_spec, n_fft=n_fft, hop_length=hop, window=window, length=arrays[0].shape[-1]
    )
    return out.numpy().astype(np.float32)


def ensemble_results(
    results: list[SeparationResult], method: str = "average"
) -> SeparationResult:
    """Merge several :class:`SeparationResult` objects stem-by-stem."""
    if not results:
        raise ValueError("ensemble_results requires at least one result")
    if len(results) == 1:
        return results[0]

    combine = _max_spec if method == "max_spec" else _average
    stem_names = set(results[0].stems)
    merged: dict[str, np.ndarray] = {}
    for name in stem_names:
        arrays = [r.stems[name] for r in results if name in r.stems]
        merged[name] = combine(arrays)

    return SeparationResult(stems=merged, sample_rate=results[0].sample_rate)
