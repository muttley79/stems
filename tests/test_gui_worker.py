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
from stems.gui import worker
from stems.gui.worker import JobParams, run_job, run_jobs

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
    # A nonexistent path is a real error (reported, not a crash) → counts as failed.
    assert events[-1].data["failed"] == 1
    assert any(e.kind == "file_error" for e in events)


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


def _make_job(tmp_path, name):
    src = tmp_path / name
    src.write_bytes(b"")
    return JobParams(
        input_path=src, output_root=tmp_path / "out", preset="4stem", fmt="wav",
    )


def test_run_jobs_processes_queue_and_aggregates(stub_engines, tmp_path):
    job_q: "queue.Queue" = queue.Queue()
    job_q.put((1, _make_job(tmp_path, "a.wav")))
    job_q.put((2, _make_job(tmp_path, "b.wav")))
    events_q: "queue.Queue" = queue.Queue()

    run_jobs(job_q, events_q, threading.Event(), threading.Event())

    events = _drain(events_q)
    kinds = [e.kind for e in events]
    assert kinds.count("job_start") == 2
    assert kinds.count("job_done") == 2
    assert kinds[-1] == "queue_done"
    assert events[-1].data["done"] == 2          # one stub file per job
    assert events[-1].data["jobs"] == 2


def test_run_jobs_always_prefixes_and_continues_numbering(stub_engines, tmp_path):
    from pathlib import Path

    # Every run is prefixed, and numbering resumes from what's already on disk:
    # a pre-existing 01_song from a "previous session" pushes new runs to 02/03.
    src = tmp_path / "song.wav"
    src.write_bytes(b"")
    out = tmp_path / "out"
    (out / "01_song").mkdir(parents=True)  # stand-in for an earlier session

    job_q: "queue.Queue" = queue.Queue()
    job_q.put((1, JobParams(input_path=src, output_root=out, preset="4stem", fmt="wav")))
    job_q.put((2, JobParams(input_path=src, output_root=out, preset="4stem", fmt="wav")))
    events_q: "queue.Queue" = queue.Queue()

    run_jobs(job_q, events_q, threading.Event(), threading.Event())

    done_dirs = [e.data["out_dir"] for e in _drain(events_q) if e.kind == "file_done"]
    assert [Path(d).name for d in done_dirs] == ["02_song", "03_song"]
    assert all(Path(d).exists() for d in done_dirs)


def test_unique_output_dir_continues_from_disk(tmp_path):
    from stems.jobs import unique_output_dir

    # No NN_trk folders yet → first is 01_trk.
    assert unique_output_dir(tmp_path / "trk").name == "01_trk"
    # Once 01_trk exists on disk, the next is 02_trk (pure existence check).
    (tmp_path / "01_trk").mkdir()
    assert unique_output_dir(tmp_path / "trk").name == "02_trk"


def test_run_jobs_picks_up_job_added_after_start(stub_engines, tmp_path):
    # Seed one job; a second is "appended" before the worker drains the first.
    job_q: "queue.Queue" = queue.Queue()
    job_q.put((1, _make_job(tmp_path, "a.wav")))
    job_q.put((2, _make_job(tmp_path, "b.wav")))  # stand-in for a live append
    events_q: "queue.Queue" = queue.Queue()

    run_jobs(job_q, events_q, threading.Event(), threading.Event())

    starts = [e for e in _drain(events_q) if e.kind == "job_start"]
    assert [e.data["id"] for e in starts] == [1, 2]


def test_run_jobs_skip_current_advances_to_next(stub_engines, tmp_path, monkeypatch):
    # Job 1 is a folder of two files; "cancel task" is simulated by tripping
    # skip_current after the first file, so the second file is abandoned and the
    # queue moves on to job 2. (run_jobs clears skip at each job start, so the
    # flag must be set *during* the job, not before.)
    folder = tmp_path / "album"
    folder.mkdir()
    (folder / "t1.wav").write_bytes(b"")
    (folder / "t2.wav").write_bytes(b"")
    job1 = JobParams(
        input_path=folder, output_root=tmp_path / "out", preset="4stem", fmt="wav",
    )

    skip = threading.Event()
    real_sep = worker.separate_file
    calls = {"n": 0}

    def wrapped(*args, **kwargs):
        calls["n"] += 1
        out = real_sep(*args, **kwargs)
        if calls["n"] == 1:  # after job 1's first file, request a task cancel
            skip.set()
        return out

    monkeypatch.setattr(worker, "separate_file", wrapped)

    job_q: "queue.Queue" = queue.Queue()
    job_q.put((1, job1))
    job_q.put((2, _make_job(tmp_path, "b.wav")))
    events_q: "queue.Queue" = queue.Queue()

    run_jobs(job_q, events_q, threading.Event(), skip)

    events = _drain(events_q)
    kinds = [e.kind for e in events]
    assert "job_cancelled" in kinds          # job 1 stopped early
    assert kinds.count("job_start") == 2     # job 2 still ran
    assert kinds[-1] == "queue_done"
