"""Preset registry sanity checks (no heavy deps required)."""

import pytest

from stems.presets import DEFAULT_PRESET, PRESETS, get_preset


def test_default_preset_exists():
    assert DEFAULT_PRESET in PRESETS


def test_all_presets_have_stems_and_models():
    for name, p in PRESETS.items():
        assert p.name == name
        assert p.output_stems, f"{name} has no output stems"
        assert p.engine in ("demucs", "uvr")
        if p.kind == "twostem":
            assert p.vocal_models and p.instrumental_models, \
                f"{name} twostem preset needs vocal+instrumental models"
        else:
            assert p.models, f"{name} has no models"


def test_twostem_preset_uses_strong_models():
    p = get_preset("vocals-max")
    assert p.kind == "twostem"
    # the weak mel_roformer must not be in the vocal ensemble
    assert "mel_roformer" not in p.vocal_models


def test_get_preset_unknown_raises():
    with pytest.raises(KeyError):
        get_preset("does-not-exist")


def test_cascade_preset_has_vocal_model():
    p = get_preset("4stem-max")
    assert p.kind == "cascade"
    assert p.vocal_engine == "uvr"
    assert p.vocal_model
