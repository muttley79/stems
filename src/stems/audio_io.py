"""Audio loading and export.

Conventions used throughout the codebase:
- Audio arrays are float32 with shape ``(channels, samples)`` (channels-first).
- Sample rate is carried alongside the array as an int.

Loading prefers ``soundfile`` (libsndfile) and falls back to decoding via the
system ``ffmpeg`` binary for formats libsndfile can't read (e.g. some MP3/M4A).
Export writes lossless 24-bit (or 16-bit) WAV via ``soundfile`` and, optionally,
a 320 kbps MP3 via ``ffmpeg``. Loudness is never altered, so summing all stems
reconstructs (approximately) the original mix.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf

SUPPORTED_INPUT_SUFFIXES = {
    ".wav", ".mp3", ".flac", ".m4a", ".aac", ".ogg", ".opus", ".aiff", ".aif", ".wma",
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


def load_audio(path: Path) -> tuple[np.ndarray, int]:
    """Load any supported file as float32 ``(channels, samples)`` + sample rate."""
    path = Path(path)
    if not path.is_file():
        raise AudioIOError(f"Input file not found: {path}")

    try:
        data, sr = sf.read(str(path), dtype="float32", always_2d=False)
        return np.ascontiguousarray(_to_channels_first(data)), int(sr)
    except Exception:
        # Fall back to ffmpeg for formats libsndfile can't decode.
        if not ffmpeg_available():
            raise AudioIOError(
                f"Could not read {path} with libsndfile and ffmpeg is not installed."
            )
        return _load_via_ffmpeg(path)


def _load_via_ffmpeg(path: Path) -> tuple[np.ndarray, int]:
    """Decode to a temporary WAV with ffmpeg, then read it back."""
    sr = _probe_sample_rate(path) or 44100
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


def write_wav(path: Path, audio: np.ndarray, sr: int, bitdepth: int = 24) -> Path:
    """Write float32 ``(channels, samples)`` audio to a PCM WAV."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    subtype = {16: "PCM_16", 24: "PCM_24", 32: "FLOAT"}.get(bitdepth, "PCM_24")
    # soundfile expects (samples, channels).
    sf.write(str(path), np.ascontiguousarray(audio.T), sr, subtype=subtype)
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


def export_stem(
    out_dir: Path,
    stem_name: str,
    audio: np.ndarray,
    sr: int,
    fmt: str = "both",
    bitdepth: int = 24,
    mp3_bitrate: str = "320k",
) -> list[Path]:
    """Write a single stem in the requested format(s). Returns written paths."""
    out_dir = Path(out_dir)
    written: list[Path] = []
    if fmt in ("wav", "both"):
        written.append(write_wav(out_dir / f"{stem_name}.wav", audio, sr, bitdepth))
    if fmt in ("mp3", "both"):
        written.append(write_mp3(out_dir / f"{stem_name}.mp3", audio, sr, mp3_bitrate))
    return written
