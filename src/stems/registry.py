"""Shared job registry for the GUI and LAN web server.

A single :class:`JobRegistry` owns the worker thread and fans out progress
events to all subscribers (the GUI event queue, web SSE clients, etc.) via
:class:`BroadcastQueue`.  This avoids two workers fighting over the GPU when
both the desktop UI and the web server are active at the same time.
"""

from __future__ import annotations

import queue
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class JobInfo:
    id: int
    source: str          # "gui" | "web"
    label: str
    status: str = "queued"  # queued / running / done / failed / cancelled
    output_dir: Path | None = None
    created: float = field(default_factory=time.time)
    started: float | None = None
    elapsed: float | None = None
    error: str | None = None


class BroadcastQueue:
    """Drop-in replacement for a write-only queue.Queue that fans out every
    :meth:`put` to an arbitrary number of subscriber queues."""

    def __init__(self) -> None:
        self._subs: list[queue.Queue] = []
        self._lock = threading.Lock()

    def subscribe(self, q: queue.Queue) -> None:
        with self._lock:
            if q not in self._subs:
                self._subs.append(q)

    def unsubscribe(self, q: queue.Queue) -> None:
        with self._lock:
            self._subs = [s for s in self._subs if s is not q]

    def put(self, item: Any) -> None:
        with self._lock:
            subs = list(self._subs)
        for q in subs:
            q.put(item)

    # Alias so this works as a drop-in for queue.Queue in _process_one.
    def put_nowait(self, item: Any) -> None:
        self.put(item)


class _RelayQueue:
    """Wraps BroadcastQueue, intercepting file_done to capture out_dir."""

    def __init__(self, broadcast: BroadcastQueue) -> None:
        self._broadcast = broadcast
        self.last_out_dir: str | None = None

    def put(self, item: Any) -> None:
        if getattr(item, "kind", None) == "file_done":
            self.last_out_dir = item.data.get("out_dir")
        self._broadcast.put(item)

    def put_nowait(self, item: Any) -> None:
        self.put(item)


class JobRegistry:
    """Persistent shared worker + event fan-out for GUI and web clients.

    Both the GUI and the web server submit :class:`~stems.gui.worker.JobParams`
    here; a single daemon worker thread processes them sequentially so there is
    never more than one model running on the GPU at a time.
    """

    def __init__(self, output_root: Path) -> None:
        self.output_root = Path(output_root)
        self._jobs: list[JobInfo] = []
        self._jobs_by_id: dict[int, JobInfo] = {}
        self._next_id = 0
        self._lock = threading.Lock()
        self._worker_q: queue.Queue[tuple[int, Any]] = queue.Queue()
        self._broadcast = BroadcastQueue()
        self._cancel = threading.Event()
        self._skip = threading.Event()
        self._stop = threading.Event()
        self._worker: threading.Thread | None = None
        self._job_index = 0

    # ---------------------------------------------------------------- public API

    def subscribe(self, q: queue.Queue) -> None:
        self._broadcast.subscribe(q)

    def unsubscribe(self, q: queue.Queue) -> None:
        self._broadcast.unsubscribe(q)

    def allocate_id(self) -> int:
        """Reserve a job ID without submitting the job yet."""
        with self._lock:
            job_id = self._next_id
            self._next_id += 1
        return job_id

    def submit_job(
        self,
        params: Any,
        label: str,
        source: str = "gui",
        job_id: int | None = None,
    ) -> int:
        """Queue ``params`` for separation. Returns the assigned job ID."""
        from stems.gui.worker import Event

        with self._lock:
            if job_id is None:
                job_id = self._next_id
                self._next_id += 1
            # (pre-allocated IDs have already incremented _next_id via allocate_id)
            info = JobInfo(id=job_id, source=source, label=label)
            self._jobs.append(info)
            self._jobs_by_id[job_id] = info

        self._worker_q.put((job_id, params))
        self._broadcast.put(Event("job_queued", {
            "id": job_id, "label": label, "source": source,
        }))
        self._ensure_worker()
        return job_id

    def cancel_job(self, job_id: int) -> None:
        """Cancel a queued job, or skip the currently-running one."""
        from stems.gui.worker import Event

        with self._lock:
            info = self._jobs_by_id.get(job_id)
            if info is None:
                return
            if info.status == "queued":
                info.status = "cancelled"
                self._broadcast.put(Event("job_cancelled", {"id": job_id}))
                return
            if info.status == "running":
                self._skip.set()

    def cancel_all(self) -> None:
        self._cancel.set()

    def get_jobs(self) -> list[JobInfo]:
        with self._lock:
            return list(self._jobs)

    def get_job(self, job_id: int) -> JobInfo | None:
        with self._lock:
            return self._jobs_by_id.get(job_id)

    def start_worker(self) -> None:
        self._ensure_worker()

    def stop(self) -> None:
        self._stop.set()
        if self._worker is not None:
            self._worker.join(timeout=3)

    # -------------------------------------------------------------- internals

    def _ensure_worker(self) -> None:
        if self._worker is None or not self._worker.is_alive():
            self._stop.clear()
            self._worker = threading.Thread(
                target=self._run_loop, daemon=True, name="stems-registry-worker"
            )
            self._worker.start()

    def _run_loop(self) -> None:
        from stems.gui.worker import Event, _process_one

        while not self._stop.is_set():
            try:
                job_id, params = self._worker_q.get(timeout=0.5)
            except queue.Empty:
                continue

            # Skip jobs that were cancelled while waiting in the queue.
            with self._lock:
                info = self._jobs_by_id.get(job_id)
                if info and info.status == "cancelled":
                    continue

            self._cancel.clear()
            self._skip.clear()

            with self._lock:
                self._job_index += 1
                idx = self._job_index
                if info:
                    info.status = "running"
                    info.started = time.time()

            self._broadcast.put(Event("job_start", {
                "id": job_id, "index": idx,
                "name": params.input_path.name,
            }))

            summary: dict[str, Any] = {
                "done": 0, "skipped": 0, "failed": 0, "cancelled": False
            }
            relay = _RelayQueue(self._broadcast)
            _process_one(
                params, relay, self._cancel, self._skip, summary,
                number_outputs=True,
            )

            t_started = info.started if info else time.time()
            elapsed = time.time() - t_started

            if self._cancel.is_set():
                with self._lock:
                    if info:
                        info.status = "cancelled"
                        info.elapsed = elapsed
                self._broadcast.put(Event("job_cancelled", {"id": job_id}))
                # Drain remaining queued items and mark them cancelled too.
                self._drain_on_cancel()
            elif self._skip.is_set():
                with self._lock:
                    if info:
                        info.status = "cancelled"
                        info.elapsed = elapsed
                self._broadcast.put(Event("job_cancelled", {"id": job_id}))
            elif summary.get("failed", 0) > 0 and summary.get("done", 0) == 0:
                with self._lock:
                    if info:
                        info.status = "failed"
                        info.elapsed = elapsed
                        info.error = "see log"
                self._broadcast.put(Event("job_error", {
                    "id": job_id, "message": "see log for details",
                }))
            else:
                out_dir = relay.last_out_dir
                with self._lock:
                    if info:
                        info.status = "done"
                        info.elapsed = elapsed
                        if out_dir:
                            info.output_dir = Path(out_dir)
                self._broadcast.put(Event("job_done", {
                    "id": job_id,
                    "out_dir": out_dir,
                }))

    def _drain_on_cancel(self) -> None:
        """After cancel_all, purge remaining queued jobs and emit cancelled events."""
        from stems.gui.worker import Event

        while True:
            try:
                rem_id, _ = self._worker_q.get_nowait()
            except queue.Empty:
                break
            with self._lock:
                rem_info = self._jobs_by_id.get(rem_id)
                if rem_info and rem_info.status == "queued":
                    rem_info.status = "cancelled"
            self._broadcast.put(Event("job_cancelled", {"id": rem_id}))
        self._cancel.clear()
