"""GUI worker tests — exercise the background runner without torch or a window.

The worker reuses the same pipeline as the CLI, so we stub the engine registry
exactly like ``test_pipeline`` does and assert that ``run_job`` turns a real
separation into the expected stream of queue events. ``customtkinter`` is never
imported here (the window lives in ``stems.gui.app``, which we don't touch).
"""

import queue
import threading

import numpy as np
import pytest

from stems import pipeline
from stems.engines.base import BaseSeparator, SeparationResult
from stems.gui.worker import JobParams, run_job

SR = 44100


def _const(value, n=SR, channels=2):
    return np.full((channels, n), value, dtype=np.float32)


class FakeEngine(BaseSeparator):
    def __init__(self, stem_map):
        self._stem_map = stem_map

    def available_stems(self, model):
        return list(self._stem_map[model])

    def separate(self, audio_path, model, config, stems=None):
        produced = {k: _const(v) for k, v in self._stem_map[model].items()}
        if stems:
            produced = {k: v for k, v in produced.items() if k in stems}
        return SeparationResult(stems=produced, sample_rate=SR)


@pytest.fixture
def stub_engines(monkeypatch):
    demucs = FakeEngine({
        "htdemucs_ft": {"vocals": 0.0, "drums": 0.2, "bass": 0.4, "other": 0.6},
    })
    monkeypatch.setitem(pipeline._ENGINES, "demucs", demucs)
    # Prefetch must never reach the network in tests.
    monkeypatch.setattr(pipeline, "prefetch_models", lambda *a, **k: None)


def _drain(q):
    events = []
    while True:
        try:
            events.append(q.get_nowait())
        except queue.Empty:
            return events


def test_run_job_emits_expected_events(stub_engines, tmp_path):
    src = tmp_path / "song.wav"
    src.write_bytes(b"")  # discover_inputs only checks suffix/existence

    params = JobParams(
        input_path=src, output_root=tmp_path / "out", preset="4stem", fmt="wav",
    )
    q: "queue.Queue" = queue.Queue()
    run_job(params, q, threading.Event())

    events = _drain(q)
    kinds = [e.kind for e in events]
    assert "file_start" in kinds
    assert "step" in kinds
    assert kinds[-1] == "batch_done"          # always finishes with batch_done
    assert events[-1].data["done"] == 1

    done = next(e for e in events if e.kind == "file_done")
    assert len(done.data["written"]) == 4     # 4 stems, wav only


def test_run_job_reports_missing_input(tmp_path):
    params = JobParams(input_path=tmp_path / "nope.wav", output_root=tmp_path)
    q: "queue.Queue" = queue.Queue()
    run_job(params, q, threading.Event())

    events = _drain(q)
    assert events[-1].kind == "batch_done"
    # A nonexistent file yields no inputs (a clean "nothing to do"), not a crash.
    assert events[-1].data["failed"] == 0


def test_cancel_before_start_skips_work(stub_engines, tmp_path):
    src = tmp_path / "song.wav"
    src.write_bytes(b"")
    params = JobParams(input_path=src, output_root=tmp_path / "out", preset="4stem")

    cancel = threading.Event()
    cancel.set()
    q: "queue.Queue" = queue.Queue()
    run_job(params, q, cancel)

    events = _drain(q)
    assert events[-1].kind == "batch_done"
    assert events[-1].data["cancelled"] is True
    assert events[-1].data["done"] == 0
