"""Combine outputs of multiple models into a single, higher-quality result.

Two strategies:
- ``average``  — per-sample mean of aligned waveforms. Robust, reduces random
  artifacts, the safe default for the "-max" presets.
- ``max_spec`` — keep, per time-frequency bin, the magnitude with the larger
  value across models (phase taken from the contributor), which tends to retain
  detail/transients. Falls back to ``average`` if torch is unavailable.

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


def _max_spec(arrays: list[np.ndarray]) -> np.ndarray:
    try:
        import torch
    except Exception:
        return _average(arrays)

    arrays = _match_channels(_align_length(arrays))
    n_fft, hop = 4096, 1024
    window = torch.hann_window(n_fft)
    best_mag = None
    best_spec = None
    for a in arrays:
        t = torch.from_numpy(np.ascontiguousarray(a))
        spec = torch.stft(t, n_fft=n_fft, hop_length=hop, window=window, return_complex=True)
        mag = spec.abs()
        if best_mag is None:
            best_mag, best_spec = mag, spec
        else:
            take = mag > best_mag
            best_spec = torch.where(take, spec, best_spec)
            best_mag = torch.where(take, mag, best_mag)
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
