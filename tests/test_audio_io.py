"""Round-trip tests for audio I/O on a synthetic signal (no models needed)."""

import warnings

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


def test_write_wav_warns_when_integer_pcm_would_clip(tmp_path):
    """Stems carry no normalization, so one can exceed full scale; don't clip silently."""
    hot = _sine() * 6.0  # peaks at 1.2
    with pytest.warns(RuntimeWarning, match="clip"):
        audio_io.write_wav(tmp_path / "hot.wav", hot, 44100, bitdepth=24)


def test_write_wav_float_keeps_above_full_scale_without_warning(tmp_path):
    hot = _sine() * 6.0
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # any warning here fails the test
        out = audio_io.write_wav(tmp_path / "hot32.wav", hot, 44100, bitdepth=32)
    loaded, _ = audio_io.load_audio(out)
    assert np.abs(loaded).max() > 1.0


def test_load_audio_resamples_to_target_sr(tmp_path):
    """Tail rescue needs the mix at the stem rate; UVR output is always 44.1 kHz."""
    if not audio_io.ffmpeg_available():
        pytest.skip("ffmpeg not available")
    src = audio_io.write_wav(tmp_path / "48k.wav", _sine(sr=48000), 48000, bitdepth=32)

    same, sr = audio_io.load_audio(src)
    assert sr == 48000

    resampled, sr = audio_io.load_audio(src, target_sr=44100)
    assert sr == 44100
    assert resampled.shape[0] == same.shape[0]
    # One second of audio, whatever the rate.
    assert resampled.shape[1] == pytest.approx(44100, rel=0.01)


def test_export_stem_both_formats(tmp_path):
    if not audio_io.ffmpeg_available():
        pytest.skip("ffmpeg not available for MP3 export")
    sr = 44100
    audio = _sine(sr=sr)
    written = audio_io.export_stem(tmp_path, "vocals", audio, sr, fmt="both")
    names = {p.name for p in written}
    assert names == {"vocals.wav", "vocals.mp3"}
    assert all(p.is_file() for p in written)


def test_parse_formats():
    assert audio_io.parse_formats("mp3,flac") == ("flac", "mp3")
    assert audio_io.parse_formats(" WAV , mp3,wav") == ("wav", "mp3")
    assert audio_io.parse_formats("both") == ("wav", "mp3")
    assert audio_io.parse_formats(["flac"]) == ("flac",)
    with pytest.raises(ValueError):
        audio_io.parse_formats("ogg")
    with pytest.raises(ValueError):
        audio_io.parse_formats(" , ")


def test_flac_roundtrip_is_lossless_at_24bit(tmp_path):
    sr = 44100
    audio = _sine(sr=sr) * 0.5
    written = audio_io.export_stem(tmp_path, "drums", audio, sr, fmt="flac")
    assert [p.name for p in written] == ["drums.flac"]
    loaded, out_sr = audio_io.load_audio(written[0])
    assert out_sr == sr
    assert np.max(np.abs(loaded - audio)) < 2.0 / 2**23


def test_flac_32bit_request_caps_at_24(tmp_path):
    import soundfile as sf
    out = audio_io.write_flac(tmp_path / "x.flac", _sine(), 44100, bitdepth=32)
    assert sf.info(str(out)).subtype == "PCM_24"


def test_mono_input_becomes_channels_first(tmp_path):
    sr = 22050
    mono = _sine(sr=sr, channels=1)  # (1, samples)
    out = audio_io.write_wav(tmp_path / "mono.wav", mono, sr)
    loaded, _ = audio_io.load_audio(out)
    assert loaded.ndim == 2
    assert loaded.shape[0] == 1
