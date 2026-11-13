"""Audio loading and export.

Conventions used throughout the codebase:
- Audio arrays are float32 with shape ``(channels, samples)`` (channels-first).
- Sample rate is carried alongside the array as an int.

Loading prefers ``soundfile`` (libsndfile) and falls back to decoding via the
system ``ffmpeg`` binary for formats libsndfile can't read (e.g. some MP3/M4A).
Export writes lossless FLAC and/or WAV (24-bit by default) via ``soundfile`` and
a 320 kbps MP3 via ``ffmpeg``; the default set is MP3 + FLAC. Loudness is never altered, so summing all stems
reconstructs (approximately) the original mix.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
import warnings
from pathlib import Path
from typing import Iterable

import numpy as np
import soundfile as sf

from stems.config import DEFAULT_FORMAT

SUPPORTED_INPUT_SUFFIXES = {
    ".wav", ".mp3", ".flac", ".m4a", ".aac", ".ogg", ".opus", ".aiff", ".aif",
    ".wma", ".webm", ".weba", ".mp4", ".mkv", ".mka",
}

_FFMPEG = shutil.which("ffmpeg")
_FFPROBE = shutil.which("ffprobe")


class AudioIOError(RuntimeError):
    """Raised when audio cannot be loaded or written."""


def ffmpeg_available() -> bool:
    return _FFMPEG is not None


def _to_channels_first(data: np.ndarray) -> np.ndarray:
    """soundfile returns (samples,) or (samples, channels); make it (channels, samples)."""
    if data.ndim == 1:
        return data[np.newaxis, :]
    return data.T


def load_audio(path: Path, target_sr: int | None = None) -> tuple[np.ndarray, int]:
    """Load any supported file as float32 ``(channels, samples)`` + sample rate.

    ``target_sr`` forces the result to that rate, re-decoding through ffmpeg's
    resampler when the source differs. Used to line the original mix up with
    backend output, which audio-separator always emits at 44.1 kHz.
    """
    path = Path(path)
    if not path.is_file():
        raise AudioIOError(f"Input file not found: {path}")

    try:
        data, sr = sf.read(str(path), dtype="float32", always_2d=False)
        audio, sr = np.ascontiguousarray(_to_channels_first(data)), int(sr)
    except Exception:
        # Fall back to ffmpeg for formats libsndfile can't decode.
        if not ffmpeg_available():
            raise AudioIOError(
                f"Could not read {path} with libsndfile and ffmpeg is not installed."
            )
        audio, sr = _load_via_ffmpeg(path)

    if target_sr is not None and sr != int(target_sr):
        if not ffmpeg_available():
            raise AudioIOError(
                f"Resampling {path} to {target_sr} Hz needs ffmpeg, which is not installed."
            )
        audio, sr = _load_via_ffmpeg(path, target_sr=int(target_sr))
    return audio, sr


def _load_via_ffmpeg(path: Path, target_sr: int | None = None) -> tuple[np.ndarray, int]:
    """Decode to a temporary WAV with ffmpeg, then read it back."""
    sr = target_sr or _probe_sample_rate(path) or 44100
    with tempfile.TemporaryDirectory() as tmp:
        wav = Path(tmp) / "decoded.wav"
        cmd = [
            _FFMPEG, "-hide_banner", "-loglevel", "error", "-y",
            "-i", str(path),
            "-ar", str(sr), "-c:a", "pcm_f32le", str(wav),
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0:
            raise AudioIOError(f"ffmpeg failed to decode {path}: {proc.stderr.strip()}")
        data, out_sr = sf.read(str(wav), dtype="float32", always_2d=False)
        return np.ascontiguousarray(_to_channels_first(data)), int(out_sr)


def _probe_sample_rate(path: Path) -> int | None:
    if not _FFPROBE:
        return None
    cmd = [
        _FFPROBE, "-v", "error", "-select_streams", "a:0",
        "-show_entries", "stream=sample_rate", "-of", "json", str(path),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        return None
    try:
        info = json.loads(proc.stdout)
        return int(info["streams"][0]["sample_rate"])
    except (KeyError, IndexError, ValueError, json.JSONDecodeError):
        return None


def _warn_if_clipping(path: Path, audio: np.ndarray, bitdepth: int) -> None:
    peak = float(np.abs(audio).max()) if audio.size else 0.0
    if peak > 1.0:
        hint = (
            "use --bitdepth 32 with WAV to keep it intact"
            if path.suffix.lower() == ".wav"
            else "write WAV with --bitdepth 32 to keep it intact"
        )
        warnings.warn(
            f"{path.name} peaks at {20 * np.log10(peak):+.2f} dBFS and will clip "
            f"at {bitdepth}-bit; {hint}.",
            RuntimeWarning,
            stacklevel=3,
        )


def write_wav(path: Path, audio: np.ndarray, sr: int, bitdepth: int = 24) -> Path:
    """Write float32 ``(channels, samples)`` audio to a PCM WAV.

    Stems carry no normalization, so one can legitimately peak above full scale
    (an instrumental is ``mix - vocals``, which can exceed the mix). Integer PCM
    can't represent that and libsndfile saturates silently, so warn instead -
    ``--bitdepth 32`` writes float and preserves it exactly.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    subtype = {16: "PCM_16", 24: "PCM_24", 32: "FLOAT"}.get(bitdepth, "PCM_24")
    if subtype != "FLOAT":
        _warn_if_clipping(path, audio, bitdepth)
    # soundfile expects (samples, channels).
    sf.write(str(path), np.ascontiguousarray(audio.T), sr, subtype=subtype)
    return path


def write_flac(path: Path, audio: np.ndarray, sr: int, bitdepth: int = 24) -> Path:
    """Write float32 ``(channels, samples)`` audio to a lossless FLAC.

    FLAC is integer-only, so ``bitdepth`` is capped at 24 (32 -> 24); peaks above
    full scale clip, and :func:`write_wav` at 32-bit is the way to keep them.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    bits = 16 if bitdepth == 16 else 24
    _warn_if_clipping(path, audio, bits)
    sf.write(
        str(path), np.ascontiguousarray(audio.T), sr,
        format="FLAC", subtype=f"PCM_{bits}",
    )
    return path


def write_mp3(path: Path, audio: np.ndarray, sr: int, bitrate: str = "320k") -> Path:
    """Encode float32 ``(channels, samples)`` audio to MP3 via ffmpeg.

    Routes through a temporary WAV so we don't depend on libsndfile MP3 support.
    """
    if not ffmpeg_available():
        raise AudioIOError("ffmpeg is required for MP3 export but was not found on PATH.")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        wav = Path(tmp) / "stem.wav"
        write_wav(wav, audio, sr, bitdepth=32)
        cmd = [
            _FFMPEG, "-hide_banner", "-loglevel", "error", "-y",
            "-i", str(wav),
            "-c:a", "libmp3lame", "-b:a", bitrate, str(path),
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0:
            raise AudioIOError(f"ffmpeg failed to encode MP3 {path}: {proc.stderr.strip()}")
    return path


# Output formats, in write order. ``--format`` takes a comma-separated subset
# (e.g. ``mp3,flac,wav``); these legacy single-word values are still accepted.
OUTPUT_FORMATS = ("flac", "wav", "mp3")
_FORMAT_ALIASES = {"both": ("wav", "mp3"), "all": OUTPUT_FORMATS}


def parse_formats(fmt: str | Iterable[str]) -> tuple[str, ...]:
    """Normalise a format spec (``"mp3,flac"``, ``"both"``, ``["wav"]``) to a
    de-duplicated tuple in :data:`OUTPUT_FORMATS` order. Raises ``ValueError``
    on an empty spec or an unknown format."""
    parts = fmt.split(",") if isinstance(fmt, str) else list(fmt)
    wanted: set[str] = set()
    for part in parts:
        part = part.strip().lower()
        if not part:
            continue
        if part in _FORMAT_ALIASES:
            wanted.update(_FORMAT_ALIASES[part])
        elif part in OUTPUT_FORMATS:
            wanted.add(part)
        else:
            raise ValueError(
                f"unknown output format {part!r} (choose from {', '.join(OUTPUT_FORMATS)})"
            )
    if not wanted:
        raise ValueError("no output format given")
    return tuple(f for f in OUTPUT_FORMATS if f in wanted)


def export_stem(
    out_dir: Path,
    stem_name: str,
    audio: np.ndarray,
    sr: int,
    fmt: str | Iterable[str] = DEFAULT_FORMAT,
    bitdepth: int = 24,
    mp3_bitrate: str = "320k",
) -> list[Path]:
    """Write a single stem in the requested format(s). Returns written paths."""
    out_dir = Path(out_dir)
    written: list[Path] = []
    for f in parse_formats(fmt):
        if f == "flac":
            written.append(write_flac(out_dir / f"{stem_name}.flac", audio, sr, bitdepth))
        elif f == "wav":
            written.append(write_wav(out_dir / f"{stem_name}.wav", audio, sr, bitdepth))
        elif f == "mp3":
            written.append(write_mp3(out_dir / f"{stem_name}.mp3", audio, sr, mp3_bitrate))
    return written
