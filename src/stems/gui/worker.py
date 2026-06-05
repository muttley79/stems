"""Background batch runner for the GUI.

This mirrors what :func:`stems.jobs.run_batch` does, but instead of writing to a
rich console it reports progress as structured :class:`Event` objects on a
``queue.Queue``. The worker runs on a background thread so the window never
freezes; it is the *only* writer to the queue, and it never touches a widget.

Threading contract (see the GUI module docstring):

    worker thread                         main (UI) thread
    ─────────────                         ────────────────
    separate_file(on_step=cb)             root.after(.., _drain_events)
      └─ cb(Step) -> events.put(..)  ───►   ev = queue.get_nowait(); apply(ev)

Heavy backends (torch/demucs/audio-separator) stay lazy-imported inside the
engines' ``.separate()``; importing this module is light.
"""

from __future__ import annotations

import queue
import threading
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from stems.config import (
    DEFAULT_MP3_BITRATE, DEFAULT_OVERLAP, DEFAULT_SEGMENT, RunConfig,
    resolve_device,
)
from stems.jobs import discover_inputs, output_dir_for, outputs_exist
from stems.pipeline import (
    Step, iter_required_models, prefetch_models, separate_file,
)


@dataclass(slots=True)
class JobParams:
    """User-chosen settings collected from the GUI form."""

    input_path: Path
    output_root: Path
    preset: str | None = None
    model: str | None = None
    engine: str | None = None
    stems: list[str] | None = None
    fmt: str = "both"
    bitdepth: int = 24
    device: str = "auto"
    segment: float | None = DEFAULT_SEGMENT
    overlap: float = DEFAULT_OVERLAP
    recursive: bool = False
    skip_existing: bool = False


@dataclass(slots=True)
class Event:
    """A progress event from the worker to the UI.

    ``kind`` is one of: ``prefetch``, ``file_start``, ``step``, ``file_skipped``,
    ``file_done``, ``file_error``, ``batch_done``, ``log``. ``data`` carries the
    kind-specific payload (see :func:`run_job` for what each kind sends).
    """

    kind: str
    data: dict[str, Any] = field(default_factory=dict)


class _PrefetchConsole:
    """Shim passed to :func:`prefetch_models` as its ``console``.

    ``prefetch_models`` only calls ``console.print("↓ Downloading <model> …")``
    when a model is genuinely missing, so each such call means a real download is
    starting. We forward those as ``prefetch`` events (and everything else as a
    plain ``log`` line) so the UI can show an animated bar only while fetching.
    """

    def __init__(self, events: "queue.Queue[Event]") -> None:
        self._events = events

    def print(self, message: str = "", *args: Any, **kwargs: Any) -> None:
        text = _strip_markup(str(message))
        if "Downloading model" in text:
            # Pull the model name out of "↓ Downloading model <name> …".
            model = text.split("Downloading model", 1)[1].strip(" .…")
            self._events.put(Event("prefetch", {"model": model or "model"}))
        else:
            self._events.put(Event("log", {"message": text}))


def run_job(
    params: JobParams,
    events: "queue.Queue[Event]",
    cancel: threading.Event,
) -> None:
    """Process all inputs, emitting :class:`Event`s. Runs on a worker thread.

    Always finishes by emitting a ``batch_done`` event (even on early failure or
    cancellation) so the UI can re-enable its controls in one place.
    """
    summary = {"done": 0, "skipped": 0, "failed": 0, "cancelled": False}
    try:
        config = RunConfig(
            device=resolve_device(params.device),
            segment=params.segment,
            overlap=params.overlap,
            bitdepth=params.bitdepth,
            mp3_bitrate=DEFAULT_MP3_BITRATE,
        )
        events.put(Event("log", {"message": f"Device: {config.device}"}))

        inputs = discover_inputs(params.input_path, recursive=params.recursive)
        if not inputs:
            events.put(Event("log", {"message": "No supported audio files found."}))
            return

        config.ensure_dirs()

        # Fetch any missing weights up front (same as run_batch). The shim turns
        # the backends' "Downloading model X" notices into prefetch events.
        prefetch_models(
            iter_required_models(
                preset=params.preset, engine=params.engine, model=params.model
            ),
            config,
            _PrefetchConsole(events),
        )

        total = len(inputs)
        for idx, f in enumerate(inputs, start=1):
            if cancel.is_set():
                summary["cancelled"] = True
                events.put(Event("log", {"message": "Cancelled."}))
                break

            out_dir = output_dir_for(f, params.input_path, params.output_root)
            events.put(Event(
                "file_start",
                {"name": f.name, "path": str(f), "index": idx, "total": total},
            ))

            if params.skip_existing and outputs_exist(out_dir):
                summary["skipped"] += 1
                events.put(Event("file_skipped", {"name": f.name, "index": idx}))
                continue

            def on_step(s: Step) -> None:
                events.put(Event("step", {
                    "index": s.index, "total": s.total,
                    "model": s.model, "action": s.action,
                }))

            try:
                written = separate_file(
                    f, out_dir, config,
                    preset=params.preset, engine=params.engine,
                    model=params.model, stems=params.stems, fmt=params.fmt,
                    on_step=on_step,
                )
                summary["done"] += 1
                events.put(Event("file_done", {
                    "name": f.name, "index": idx,
                    "out_dir": str(out_dir),
                    "written": [str(p) for p in written],
                }))
            except Exception as exc:  # keep batch going on per-file failure
                summary["failed"] += 1
                events.put(Event("file_error", {
                    "name": f.name, "index": idx, "message": str(exc),
                }))
    except Exception as exc:  # discovery/config-level failure: report and stop
        events.put(Event("file_error", {
            "name": str(params.input_path), "message": str(exc),
            "traceback": traceback.format_exc(),
        }))
    finally:
        events.put(Event("batch_done", summary))


def _strip_markup(text: str) -> str:
    """Drop rich-style ``[tag]`` markup so plain log lines read cleanly."""
    out = []
    depth = 0
    for ch in text:
        if ch == "[":
            depth += 1
        elif ch == "]" and depth:
            depth -= 1
        elif depth == 0:
            out.append(ch)
    return "".join(out).strip()
