"""Pipeline wiring tests using a stub engine (no torch/models required).

We monkeypatch the engine registry with a fake separator so we can exercise the
single / ensemble / cascade plans and the file-export path quickly on CPU.
"""

import numpy as np
import pytest

from stems import pipeline
from stems.config import RunConfig
from stems.engines.base import BaseSeparator, SeparationResult
from stems.ensemble import ensemble_results


SR = 44100


def _const(value, n=SR, channels=2):
    return np.full((channels, n), value, dtype=np.float32)


class FakeEngine(BaseSeparator):
    """Returns deterministic stems based on the requested model/engine name."""

    def __init__(self, name, stem_map):
        self.name = name
        self._stem_map = stem_map  # model -> {stem: value}

    def available_stems(self, model):
        return list(self._stem_map[model])

    def separate(self, audio_path, model, config, stems=None):
        produced = {k: _const(v) for k, v in self._stem_map[model].items()}
        if stems:
            produced = {k: v for k, v in produced.items() if k in stems}
        return SeparationResult(stems=produced, sample_rate=SR)


@pytest.fixture
def stub_engines(monkeypatch):
    uvr = FakeEngine("uvr", {
        "bs_roformer": {"vocals": 0.5, "instrumental": 0.1},
        "mel_roformer": {"vocals": 0.7, "instrumental": 0.3},
        "kim_vocals": {"vocals": 0.4, "other": 0.9},
        "kim_ft": {"vocals": 0.6, "other": 0.9},
        "vocal_fullness": {"vocals": 0.5, "other": 0.9},
        "inst_v2": {"instrumental": 0.3, "vocals": 0.9},
        "inst_bleedless": {"instrumental": 0.2, "vocals": 0.9},
        # Dedicated guitar model: distinct value so an override is provable.
        "guitar": {"guitar": 0.7, "other": 0.1},
    })
    demucs = FakeEngine("demucs", {
        "htdemucs_ft": {"vocals": 0.0, "drums": 0.2, "bass": 0.4, "other": 0.6},
        "htdemucs_6s": {"vocals": 0.0, "drums": 0.2, "bass": 0.4,
                        "other": 0.6, "guitar": 0.8, "piano": 0.9},
    })
    monkeypatch.setitem(pipeline._ENGINES, "uvr", uvr)
    monkeypatch.setitem(pipeline._ENGINES, "demucs", demucs)


def test_ensemble_average():
    a = SeparationResult({"vocals": _const(0.4)}, SR)
    b = SeparationResult({"vocals": _const(0.6)}, SR)
    merged = ensemble_results([a, b], method="average")
    assert np.allclose(merged.stems["vocals"], 0.5)


def test_single_preset(stub_engines, tmp_path):
    cfg = RunConfig(device="cpu")
    res = pipeline.separate_to_result(tmp_path / "x.wav", cfg, preset="4stem")
    assert set(res.stems) == {"vocals", "drums", "bass", "other"}


def test_twostem_preset_merges_independently(stub_engines, tmp_path):
    cfg = RunConfig(device="cpu")
    res = pipeline.separate_to_result(tmp_path / "x.wav", cfg, preset="vocals-max")
    assert set(res.stems) == {"vocals", "instrumental"}
    # instrumental = mean of bs_roformer(0.1) + inst_v2(0.3) = 0.2 (average method)
    assert np.allclose(res.stems["instrumental"], 0.2, atol=1e-6)
    # vocals use the max_spec ensemble of the three vocal models; just assert it
    # produced a real, non-empty signal (exact value isn't meaningful for DC).
    assert res.stems["vocals"].shape == res.stems["instrumental"].shape
    assert np.isfinite(res.stems["vocals"]).all()


def test_cascade_preset_uses_roformer_vocals(stub_engines, tmp_path):
    cfg = RunConfig(device="cpu")
    res = pipeline.separate_to_result(tmp_path / "x.wav", cfg, preset="4stem-max")
    assert set(res.stems) == {"vocals", "drums", "bass", "other"}
    # drums/bass/other come from Demucs on the clean instrumental
    assert np.allclose(res.stems["drums"], 0.2)
    # vocals come from the vocal ensemble (not Demucs's empty 0.0 vocals)
    assert np.isfinite(res.stems["vocals"]).all()
    assert np.any(res.stems["vocals"] != 0.0)


def test_6stem_max_overrides_guitar(stub_engines, tmp_path):
    cfg = RunConfig(device="cpu")
    res = pipeline.separate_to_result(
        tmp_path / "x.wav", cfg, preset="6stem-max", guitar_source="instrumental"
    )
    assert set(res.stems) == {"vocals", "drums", "bass", "other", "guitar", "piano"}
    # guitar comes from the dedicated guitar model (0.7), not htdemucs_6s (0.8)
    assert np.allclose(res.stems["guitar"], 0.7)
    # piano still comes from the htdemucs_6s pass
    assert np.allclose(res.stems["piano"], 0.9)
    # vocals from the ensemble (not Demucs's empty 0.0 vocals)
    assert np.any(res.stems["vocals"] != 0.0)


def test_6stem_max_requires_guitar_source(stub_engines, tmp_path):
    cfg = RunConfig(device="cpu")
    with pytest.raises(ValueError):
        pipeline.separate_to_result(tmp_path / "x.wav", cfg, preset="6stem-max")


def test_6stem_max_no_drums_source(stub_engines, tmp_path):
    # The 'no-drums' bed is built from instrumental minus demucs drums/bass and
    # written to a temp wav; the guitar still resolves from the guitar model.
    cfg = RunConfig(device="cpu")
    res = pipeline.separate_to_result(
        tmp_path / "x.wav", cfg, preset="6stem-max", guitar_source="no-drums"
    )
    assert np.allclose(res.stems["guitar"], 0.7)


def test_stems_filter(stub_engines, tmp_path):
    cfg = RunConfig(device="cpu")
    res = pipeline.separate_to_result(
        tmp_path / "x.wav", cfg, preset="6stem", stems=["vocals", "piano"]
    )
    assert set(res.stems) == {"vocals", "piano"}


def test_separate_file_writes_outputs(stub_engines, tmp_path):
    cfg = RunConfig(device="cpu")
    out = tmp_path / "out"
    written = pipeline.separate_file(
        tmp_path / "x.wav", out, cfg, preset="4stem", fmt="wav"
    )
    assert len(written) == 4
    assert all(p.suffix == ".wav" and p.is_file() for p in written)


def test_sum_stems_aligns_and_adds():
    from stems.ensemble import sum_stems

    a = _const(0.3, n=SR)
    b = _const(0.4, n=SR - 5)  # a few samples shorter at the tail
    out = sum_stems([a, b])
    assert out.shape == (2, SR - 5)  # aligned to the shorter length
    assert np.allclose(out, 0.7)


def test_separate_file_combine_writes_single_mix(stub_engines, tmp_path):
    import soundfile as sf

    cfg = RunConfig(device="cpu", bitdepth=32)  # FLOAT WAV → exact readback
    out = tmp_path / "out"
    written = pipeline.separate_file(
        tmp_path / "x.wav", out, cfg, preset="6stem", fmt="wav",
        combine=["drums", "bass"],
    )
    assert len(written) == 1
    assert written[0].name == "drums+bass.wav"
    # Only the combined mix is written - no per-stem files.
    assert {f.name for f in out.iterdir()} == {"drums+bass.wav"}
    audio, sr = sf.read(str(written[0]))
    assert sr == SR
    assert np.allclose(audio, 0.6, atol=1e-6)  # drums(0.2) + bass(0.4)


def test_cli_mix_and_stems_conflict():
    from typer.testing import CliRunner

    from stems.cli import app

    result = CliRunner().invoke(app, [
        "separate", "x.wav", "out", "-p", "6stem",
        "--mix", "vocals,drums", "--stems", "vocals",
    ])
    assert result.exit_code != 0
    assert "cannot be used together" in result.output


def test_cli_mix_rejects_unknown_stem():
    from typer.testing import CliRunner

    from stems.cli import app

    result = CliRunner().invoke(app, [
        "separate", "x.wav", "out", "-p", "6stem", "--mix", "vocals,bogus",
    ])
    assert result.exit_code != 0
    assert "bogus" in result.output
