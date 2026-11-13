"""Live-preview tests: the writer, the engine-side partial reconstruction, and
the pipeline wiring. No models; the Roformer loop is simulated with torch."""

from types import SimpleNamespace

import numpy as np
import pytest
import soundfile as sf

from stems import pipeline
from stems.config import RunConfig
from stems.engines import uvr_engine
from stems.engines.base import BaseSeparator, SeparationResult
from stems.jobs import outputs_exist
from stems.preview import PREVIEW_PREFIX, PreviewWriter

SR = 44100


def _noise(n, seed, channels=2):
    return np.random.default_rng(seed).standard_normal((channels, n)).astype(np.float32) * 0.1


def test_writer_writes_filters_and_cleans_up(tmp_path):
    seen = []
    w = PreviewWriter(tmp_path, interval=0.0, wanted=["vocals"], on_first_write=seen.append)
    assert w.due()
    w.update({"vocals": _noise(1000, 1), "drums": _noise(1000, 2)}, SR)
    w.update({"vocals": _noise(2000, 3)}, SR)  # grows; first-write fires once

    path = tmp_path / f"{PREVIEW_PREFIX}vocals.wav"
    assert seen == [[path]]
    assert sf.info(str(path)).frames == 2000
    assert not (tmp_path / f"{PREVIEW_PREFIX}drums.wav").exists()
    assert not list(tmp_path.glob("*.partial"))

    w.pass_done()
    assert not w.due()
    w.update({"vocals": _noise(10, 4)}, SR)  # ignored once the pass is done
    assert sf.info(str(path)).frames == 2000

    w.cleanup()
    assert not path.exists()


def test_writer_throttles_by_interval(tmp_path):
    w = PreviewWriter(tmp_path, interval=3600.0)
    assert not w.due()


def test_outputs_exist_ignores_preview_files(tmp_path):
    (tmp_path / f"{PREVIEW_PREFIX}vocals.wav").write_bytes(b"")
    assert not outputs_exist(tmp_path)
    (tmp_path / "vocals.flac").write_bytes(b"")
    assert outputs_exist(tmp_path)


class _Sink:
    active = True

    def __init__(self):
        self.calls = []

    def due(self):
        return True

    def update(self, stems, sr):
        self.calls.append(({k: v.copy() for k, v in stems.items()}, sr))


def _simulate_roformer(state, mix, vocals, chunk, step):
    """Run the chunk loop of MDXCSeparator.demix (Roformer branch) through the
    state's hooks, with a "model" that returns the true vocals per chunk."""
    import torch

    n = mix.shape[1]
    window = torch.tensor(np.hamming(chunk), dtype=torch.float32)
    result = torch.zeros((2,) + mix.shape)  # instruments: vocals, instrumental
    voc_t = torch.tensor(vocals)
    for i in range(0, n, step):
        start, length = i, min(chunk, n - i)
        if i + chunk > n:
            start, length = n - chunk, chunk
        x = torch.stack([voc_t[:, start:start + chunk], torch.zeros(2, chunk)])
        result[..., start:start + length] += x[..., :length] * window[:length]
        state.chunk(result, window, start, length)


def test_partial_prefix_matches_truth():
    pytest.importorskip("torch")
    n, chunk, step = 20_000, 4_000, 1_000
    vocals = _noise(n, 5)
    mix = vocals + _noise(n, 6)

    sink = _Sink()
    state = uvr_engine._PreviewState(sink, aliases={})
    inst = SimpleNamespace(
        is_roformer=True, pitch_shift=0, sample_rate=SR,
        model_data_cfgdict=SimpleNamespace(training=SimpleNamespace(
            instruments=["Vocals", "Instrumental"], target_instrument="Vocals",
        )),
    )
    state.begin(inst, mix)
    _simulate_roformer(state, mix, vocals, chunk, step)

    assert sink.calls, "no partial previews were emitted"
    lengths = [stems["vocals"].shape[1] for stems, _ in sink.calls]
    assert lengths == sorted(lengths) and lengths[-1] < n  # grows, never final
    for stems, sr in sink.calls:
        pos = stems["vocals"].shape[1]
        assert sr == SR
        np.testing.assert_allclose(stems["vocals"], vocals[:, :pos], atol=1e-5)
        np.testing.assert_allclose(stems["instrumental"], mix[:, :pos] - vocals[:, :pos], atol=1e-5)


def test_karaoke_aliases_apply_to_previews():
    state = uvr_engine._PreviewState(_Sink(), uvr_engine._STEM_ALIASES["karaoke"])
    assert state._canonical("Vocals") == "lead_vocals"
    assert state._canonical("Instrumental") == "back_vocals"


class _PreviewingEngine(BaseSeparator):
    """Fake UVR engine that pushes one preview through the active sink."""

    name = "uvr"

    def available_stems(self, model):
        return ["vocals", "instrumental"]

    def separate(self, audio_path, model, config, stems=None):
        sink = uvr_engine._preview_sink.get()
        if sink is not None and sink.active:
            sink.update({"vocals": _noise(100, 7)}, SR)
            self.saw_file = (sink.out_dir / f"{PREVIEW_PREFIX}vocals.wav").exists()
            sink.pass_done()
        return SeparationResult(
            stems={"vocals": _noise(SR, 8), "instrumental": _noise(SR, 9)},
            sample_rate=SR,
        )


def test_separate_file_preview_lifecycle(monkeypatch, tmp_path):
    eng = _PreviewingEngine()
    monkeypatch.setitem(pipeline._ENGINES, "uvr", eng)
    seen = []
    out = tmp_path / "out"
    written = pipeline.separate_file(
        tmp_path / "x.wav", out, RunConfig(device="cpu"),
        model="bs_roformer", engine="uvr", fmt="flac",
        preview=True, on_preview=seen.append,
    )
    assert eng.saw_file
    assert [p.name for p in seen[0]] == [f"{PREVIEW_PREFIX}vocals.wav"]
    assert {p.name for p in written} == {"vocals.flac", "instrumental.flac"}
    assert not list(out.glob(f"{PREVIEW_PREFIX}*"))  # cleaned up after export
