"""UVR engine logic that runs without the heavy backend or any model weights."""

import types

import pytest

from stems.config import RunConfig
from stems.engines import uvr_engine
from stems.engines.uvr_engine import (
    MODEL_FILES,
    MODEL_STEMS,
    UvrSeparator,
    _classify_stem,
    apply_chunk_overlap,
    resolve_model_file,
)


def test_resolve_model_file_passes_unknown_through():
    assert resolve_model_file("bs_roformer") == MODEL_FILES["bs_roformer"]
    assert resolve_model_file("some_custom.ckpt") == "some_custom.ckpt"


def test_classify_stem_prefers_the_parenthesised_tag():
    """The model name is in the filename too, and can contain a stem word.

    inst_bleedless writes its vocals file with "instrumental" in the model name;
    matching the whole filename would label it `instrumental`.
    """
    name = "track_(Vocals)_mel_band_roformer_instrumental_bleedless_v2_gabox.wav"
    assert _classify_stem(name) == "vocals"


def test_classify_stem_ignores_the_users_filename():
    assert _classify_stem("my_instrumental_demo_(Vocals)_model.wav") == "vocals"
    assert _classify_stem("piano_practice_(Drums)_htdemucs.wav") == "drums"


def test_classify_stem_exact_tags_beat_substrings():
    # "hh" is short enough to appear inside other words; exact tags run first.
    assert _classify_stem("song_(HH)_MDX23C-DrumSep.wav") == "hh"
    assert _classify_stem("song_(Kick)_MDX23C-DrumSep.wav") == "kick"


def test_classify_stem_falls_back_to_keyword_scan():
    assert _classify_stem("song_vocals_model.wav") == "vocals"
    assert _classify_stem("something_unrecognized.wav") == "other"


def test_available_stems_knows_multi_stem_models():
    engine = UvrSeparator()
    assert engine.available_stems("bs_roformer") == ["vocals", "instrumental"]
    assert engine.available_stems("drumsep") == MODEL_STEMS["drumsep"]


def _fake_separator(*, dim_t, num_overlap, is_roformer=True, overlap=8):
    """Stand-in for a loaded audio-separator Separator, config shape and all."""
    cfg = types.SimpleNamespace(
        audio=types.SimpleNamespace(hop_length=441, sample_rate=44100),
        # ml_collections ConfigDict supports both attribute and .get access.
        inference=types.SimpleNamespace(
            dim_t=dim_t,
            get=lambda key, default=None: {"num_overlap": num_overlap}.get(key, default),
        ),
    )
    instance = types.SimpleNamespace(
        is_roformer=is_roformer,
        model_data_cfgdict=cfg,
        override_model_segment_size=False,
        overlap=overlap,
    )
    return types.SimpleNamespace(model_instance=instance)


@pytest.mark.parametrize(
    "dim_t, num_overlap, uvr_overlap, expected_step",
    [
        # 8 s chunk asking for 4x: the library default of 8 s gave it none at all.
        (801, 4, 4, 2.0),
        # 8 s chunk asking for 8x, capped by the user's ceiling of 4.
        (801, 8, 4, 2.0),
        # 11 s chunk asking for 1x: floored at 2x so we never regress below the
        # ~27% the 8 s default happened to give it.
        (1101, 1, 4, 5.5),
        (1101, 4, 4, 2.75),
        # A lower ceiling wins.
        (801, 4, 2, 4.0),
    ],
)
def test_apply_chunk_overlap_derives_step_from_model_config(
    dim_t, num_overlap, uvr_overlap, expected_step
):
    separator = _fake_separator(dim_t=dim_t, num_overlap=num_overlap)
    apply_chunk_overlap(separator, RunConfig(uvr_overlap=uvr_overlap))
    assert separator.model_instance.overlap == pytest.approx(expected_step)


def test_apply_chunk_overlap_is_a_noop_when_disabled():
    separator = _fake_separator(dim_t=801, num_overlap=4)
    apply_chunk_overlap(separator, RunConfig(uvr_overlap=1))
    assert separator.model_instance.overlap == 8


def test_apply_chunk_overlap_skips_non_roformer_models():
    """For MDX23C `overlap` is a divisor, not a step - a seconds value would break it."""
    separator = _fake_separator(dim_t=801, num_overlap=4, is_roformer=False)
    apply_chunk_overlap(separator, RunConfig(uvr_overlap=4))
    assert separator.model_instance.overlap == 8


def test_karaoke_models_relabel_their_stems():
    """Karaoke models tag lead/backing vocals as Vocals/Instrumental."""
    aliases = uvr_engine._STEM_ALIASES["karaoke"]
    assert aliases[_classify_stem("x_(Vocals)_karaoke.wav")] == "lead_vocals"
    assert aliases[_classify_stem("x_(Instrumental)_karaoke.wav")] == "back_vocals"
