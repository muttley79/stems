"""Round-trip tests for audio I/O on a synthetic signal (no models needed)."""

import numpy as np
import pytest

from stems import audio_io


def _sine(sr=44100, seconds=1.0, freq=440.0, channels=2):
    t = np.linspace(0, seconds, int(sr * seconds), endpoint=False)
    mono = 0.2 * np.sin(2 * np.pi * freq * t).astype(np.float32)
    return np.stack([mono] * channels, axis=0)  # (channels, samples)


def test_wav_roundtrip_shape_and_sr(tmp_path):
    sr = 44100
    audio = _sine(sr=sr)
    out = audio_io.write_wav(tmp_path / "tone.wav", audio, sr, bitdepth=24)
    assert out.is_file()

    loaded, loaded_sr = audio_io.load_audio(out)
    assert loaded_sr == sr
    assert loaded.shape[0] == 2  # channels-first
    assert loaded.shape[1] == audio.shape[1]
    # 24-bit PCM round-trip should be near-lossless for a -14 dBFS tone.
    np.testing.assert_allclose(loaded, audio, atol=1e-3)


def test_export_stem_both_formats(tmp_path):
    if not audio_io.ffmpeg_available():
        pytest.skip("ffmpeg not available for MP3 export")
    sr = 44100
    audio = _sine(sr=sr)
    written = audio_io.export_stem(tmp_path, "vocals", audio, sr, fmt="both")
    names = {p.name for p in written}
    assert names == {"vocals.wav", "vocals.mp3"}
    assert all(p.is_file() for p in written)


def test_mono_input_becomes_channels_first(tmp_path):
    sr = 22050
    mono = _sine(sr=sr, channels=1)  # (1, samples)
    out = audio_io.write_wav(tmp_path / "mono.wav", mono, sr)
    loaded, _ = audio_io.load_audio(out)
    assert loaded.ndim == 2
    assert loaded.shape[0] == 1
