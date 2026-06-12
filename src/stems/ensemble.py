"""Combine outputs of multiple models into a single, higher-quality result.

Also home to :func:`rescue_vocal_tails`, the post-step that refills model-gated
reverb/echo tails in a vocal stem from the mix.

Two merge strategies:
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


def rescue_vocal_tails(
    vocals: np.ndarray,
    mix: np.ndarray,
    instrumental: np.ndarray,
    sample_rate: int = 44100,
    cap_db: float = 8.0,
    low_db: float = -35.0,
    high_db: float = -26.0,
    max_freq: float = 3500.0,
) -> np.ndarray:
    """Refill gated reverb/echo tails in a vocal stem from ``mix - instrumental``.

    Every vocal model suppresses low-level vocal content it isn't confident
    about - reverb tails and echo bounces between phrases - which turns a smooth
    decay into audible volume hiccups. That material still exists in the mix, so
    this lifts it back per STFT bin from the residual (``mix - instrumental``),
    under guards that keep instrumental bleed and artifacts out:

    - a bin can be raised at most ``cap_db`` above what the vocal stem already
      has there, so bins where the model heard no voice at all stay silent;
    - the rescue only operates in quiet frames (full effect below ``low_db``
      stem level, none above ``high_db``), so singing passes through unchanged;
    - the residual magnitude is median-filtered over 3 frames (~70 ms), so a
      transient instrumental tick (hi-hat bleed) is ignored while a sustained
      reverb tail survives - lifting raw transients produced audible cracks;
    - nothing above ``max_freq`` is touched (tails carry little HF energy, but
      lifted HF bins were the main source of faint crackle);
    - frames where the residual towers over the stem are skipped (full rescue
      below a 10 dB gap, none above 14 dB): a gated tail leaves the residual
      only a few dB above the stem, while a short vocal gap inside a loud band
      passage leaves a residual that is pure instrumental-ensemble error -
      lifting it audibly amplified bleed/cracks in the gap.

    The frame weight is smoothed over ~150 ms so the blend never snaps, and the
    coherent phase (angle of ``vocal + w*residual``) is applied only where the
    rescue acts - elsewhere the vocal passes through bit-exact. (A global sum
    phase let every residual transient nudge the vocal's phase: faint clicks.)
    No-op (returns ``vocals``) if torch is unavailable.
    """
    try:
        import torch
    except Exception:
        return vocals

    voc, mx, inst = _match_channels(_align_length([vocals, mix, instrumental]))
    voc = voc.astype(np.float32)
    res = (mx - inst).astype(np.float32)
    n = voc.shape[-1]

    n_fft, hop = 4096, 1024
    window = torch.hann_window(n_fft)

    def stft(a: np.ndarray) -> "torch.Tensor":
        t = torch.from_numpy(np.ascontiguousarray(a))
        return torch.stft(
            t, n_fft=n_fft, hop_length=hop, window=window, return_complex=True
        )

    sv, sr_ = stft(voc), stft(res)
    vmag, rmag = sv.abs(), sr_.abs()
    # transient guard: 3-frame temporal median of the residual magnitude
    rpad = torch.nn.functional.pad(rmag, (1, 1), mode="replicate")
    rmag = rpad.unfold(-1, 3, 1).median(dim=-1).values
    cap = 10 ** (cap_db / 20)
    rescued = torch.maximum(vmag, torch.minimum(rmag, vmag * cap))

    # Per-frame stem loudness in the time domain (a 2*hop window centred on each
    # STFT frame), mapped to a 0..1 rescue weight between high_db and low_db.
    n_frames = vmag.shape[-1]

    def frame_rms_db(x: np.ndarray) -> np.ndarray:
        sq = np.pad(x, hop) ** 2
        csum = np.concatenate(([0.0], np.cumsum(sq)))
        starts = np.arange(n_frames) * hop
        ends = np.minimum(starts + 2 * hop, len(sq))
        rms = np.sqrt((csum[ends] - csum[starts]) / np.maximum(ends - starts, 1))
        return 20 * np.log10(rms + 1e-10)

    frame_db = frame_rms_db(voc.mean(axis=0))
    w = np.clip((high_db - frame_db) / (high_db - low_db), 0.0, 1.0)
    # tail-plausibility guard (see docstring): skip bleed-dominated residuals
    gap = frame_rms_db(res.mean(axis=0)) - frame_db
    w *= np.clip((14.0 - gap) / 4.0, 0.0, 1.0)
    kernel = np.hanning(7)
    w = np.convolve(w, kernel / kernel.sum(), mode="same")  # ~150 ms, anti-click
    wt = torch.from_numpy(w.astype(np.float32)).clamp(0, 1).view(1, 1, -1)

    # frequency ceiling with a short cosine taper (no hard spectral edge)
    n_bins = vmag.shape[-2]
    ceil_bin = min(int(max_freq / (sample_rate / 2) * (n_fft // 2)), n_bins)
    allow = np.zeros(n_bins, dtype=np.float32)
    allow[:ceil_bin] = 1.0
    taper = min(8, n_bins - ceil_bin)
    if taper > 0:
        allow[ceil_bin: ceil_bin + taper] = (
            0.5 + 0.5 * np.cos(np.linspace(0, np.pi, taper))
        )
    wt = wt * torch.from_numpy(allow).view(1, -1, 1)

    mag = vmag + wt * (rescued - vmag)
    phase = torch.angle(sv + wt * sr_)
    out = torch.istft(
        torch.polar(mag, phase), n_fft=n_fft, hop_length=hop, window=window,
        length=n,
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
