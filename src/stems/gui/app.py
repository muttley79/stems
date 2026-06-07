"""CustomTkinter window for the stems separator.

A thin desktop front-end over the same pipeline the CLI uses. The actual work
runs on a background thread (:mod:`stems.gui.worker`); this module only builds
widgets and drains the worker's event queue on the main thread via
``self.after`` so every widget update is main-thread-safe.
"""

from __future__ import annotations

import os
import queue
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from tkinter import filedialog
from typing import Any

import customtkinter as ctk

from stems import __version__
from stems.audio_io import SUPPORTED_INPUT_SUFFIXES
from stems.config import DEFAULT_OVERLAP, DEFAULT_SEGMENT
from stems.gui.settings import load_settings, save_settings
from stems.gui.worker import Event, JobParams, run_jobs
from stems.pipeline import GUITAR_SOURCES
from stems.presets import DEFAULT_PRESET, PRESETS

# Optional native file drag-and-drop. tkinterdnd2 wraps the tkdnd Tcl extension;
# if it isn't installed the window still works (Browse buttons), drops are just
# not wired up. See _enable_dnd / _register_drop below.
try:
    from tkinterdnd2 import DND_FILES, TkinterDnD
    _DND_AVAILABLE = True
except Exception:  # pragma: no cover - optional dependency
    DND_FILES = None
    TkinterDnD = None
    _DND_AVAILABLE = False

_FORMATS = ["both", "wav", "mp3"]
_DEVICES = ["auto", "cuda", "cpu"]
_BITDEPTHS = ["16", "24", "32"]
# Vocal-ensemble merge for the "-max" presets (label shown → pipeline value).
# Max-spec keeps a fuller, less-gated vocal; Average is smoother (no chance of
# the merge flutter). Insertion order = button order; first is the default.
_VOCAL_BLENDS = {"Max-spec (full)": "max_spec", "Average (smooth)": "average"}
_DEFAULT_VOCAL_BLEND = next(iter(_VOCAL_BLENDS))
_POLL_MS = 100
_SHOW_POLL_MS = 300  # how often the UI checks for a "come to front" ping
_TICK_MS = 1000      # elapsed-time refresh interval

# Status → (badge text, colour) for a queued job's row.
_STATUS_STYLE = {
    "queued": ("queued", "gray60"),
    "running": ("running", "#3b8ed0"),
    "done": ("done", "#2faa5d"),
    "failed": ("failed", "#d0492b"),
    "skipped": ("skipped", "gray60"),
    "cancelled": ("cancelled", "#d98a29"),
}


def _fmt_mmss(seconds: float) -> str:
    seconds = max(0, int(seconds))
    return f"{seconds // 60:02d}:{seconds % 60:02d}"


@dataclass
class QueuedJob:
    """One queued separation job plus its row widgets (UI-thread state only)."""

    id: int
    params: JobParams
    label: str
    status: str = "queued"
    started: float | None = None      # monotonic time the job began
    elapsed: float | None = None      # final duration once finished
    out_dir: Path | None = None       # last written output folder (for reveal)
    widgets: dict[str, Any] = field(default_factory=dict)

# Mix the tkinterdnd2 wrapper into the window only when available, so drop-target
# registration works on this root; otherwise StemsApp is a plain CTk window.
_APP_BASES = (ctk.CTk, TkinterDnD.DnDWrapper) if _DND_AVAILABLE else (ctk.CTk,)


class StemsApp(*_APP_BASES):
    """Main application window."""

    def __init__(self, single_instance=None) -> None:
        super().__init__()
        self._dnd_ready = self._enable_dnd()
        self.title(f"stems · audio separator  (v{__version__})")
        self.geometry("1300x788")
        self.minsize(1300, 712)
        # Two columns. The left (options) absorbs any extra width; the right
        # column has no weight, so it shrinks to its natural content width -
        # the action-button row - and the window's right edge hugs the buttons.
        self.grid_columnconfigure(0, weight=1, minsize=760)
        self.grid_columnconfigure(1, weight=0)

        # Worker plumbing: a fresh queue + cancel flag are created per run.
        self._events: "queue.Queue[Event]" | None = None
        self._cancel = threading.Event()
        self._skip_current = threading.Event()
        self._thread: threading.Thread | None = None
        self._last_out_dir: Path | None = None
        self._downloading = False

        # Job queue state (all UI-thread owned).
        self._queue: list[QueuedJob] = []
        self._jobs_by_id: dict[int, QueuedJob] = {}
        self._next_job_id = 0
        self._job_q: "queue.Queue[tuple[int, JobParams]]" | None = None
        self._running = False
        self._current_job_id: int | None = None
        self._run_started: float | None = None
        self._ticking = False

        # Persisted prefs (e.g. last-browsed folders for the file dialogs).
        self._settings = load_settings()

        self._build_inputs()
        self._build_plan_controls()
        self._build_advanced()
        self._build_run_area()

        # When another launch pings the single-instance server, raise this window
        # to the front. The server flag is set from a socket thread; we poll it
        # here on the main thread so the actual raise stays main-thread-safe.
        self._single_instance = single_instance
        if single_instance is not None:
            single_instance.start()
            self.after(_SHOW_POLL_MS, self._poll_show_request)

    # ----------------------------------------------------------------- layout

    def _section(self, title: str, row: int) -> ctk.CTkFrame:
        frame = ctk.CTkFrame(self)
        frame.grid(row=row, column=0, padx=16, pady=(10, 0), sticky="ew")
        frame.grid_columnconfigure(1, weight=1)
        ctk.CTkLabel(
            frame, text=title, font=ctk.CTkFont(size=13, weight="bold")
        ).grid(row=0, column=0, columnspan=3, padx=12, pady=(8, 4), sticky="w")
        return frame

    def _build_inputs(self) -> None:
        frame = self._section("Input / Output", row=0)

        self.input_var = ctk.StringVar()
        ctk.CTkLabel(frame, text="Input").grid(
            row=1, column=0, padx=12, pady=4, sticky="w"
        )
        hint = "drop a file/folder here…" if self._dnd_ready else "choose a file/folder →"
        self.input_entry = ctk.CTkEntry(
            frame, textvariable=self.input_var, placeholder_text=hint
        )
        self.input_entry.grid(row=1, column=1, padx=8, pady=4, sticky="ew")
        self._register_drop(self.input_entry, self.input_var)
        btns = ctk.CTkFrame(frame, fg_color="transparent")
        btns.grid(row=1, column=2, padx=8, pady=4)
        ctk.CTkButton(btns, text="File…", width=70, command=self._pick_file).grid(
            row=0, column=0, padx=(0, 4)
        )
        ctk.CTkButton(
            btns, text="Folder…", width=80, command=self._pick_folder
        ).grid(row=0, column=1)

        self.output_var = ctk.StringVar(value=str(Path("output").resolve()))
        ctk.CTkLabel(frame, text="Output").grid(
            row=2, column=0, padx=12, pady=(4, 10), sticky="w"
        )
        self.output_entry = ctk.CTkEntry(frame, textvariable=self.output_var)
        self.output_entry.grid(row=2, column=1, padx=8, pady=(4, 10), sticky="ew")
        self._register_drop(self.output_entry, self.output_var, dirs_only=True)
        ctk.CTkButton(
            frame, text="Browse…", width=80, command=self._pick_output
        ).grid(row=2, column=2, padx=8, pady=(4, 10))

    def _build_plan_controls(self) -> None:
        frame = self._section("Plan", row=1)

        ctk.CTkLabel(frame, text="Preset").grid(
            row=1, column=0, padx=12, pady=4, sticky="w"
        )
        self.preset_var = ctk.StringVar(value=DEFAULT_PRESET)
        self.preset_menu = ctk.CTkOptionMenu(
            frame, values=list(PRESETS), variable=self.preset_var,
            command=self._on_preset_change,
        )
        self.preset_menu.grid(row=1, column=1, padx=8, pady=4, sticky="w")

        self.preset_desc = ctk.CTkLabel(
            frame, text="", wraplength=420, justify="left", text_color="gray60"
        )
        self.preset_desc.grid(row=1, column=2, padx=8, pady=4, sticky="w")

        ctk.CTkLabel(frame, text="Model override").grid(
            row=2, column=0, padx=12, pady=4, sticky="w"
        )
        self.model_var = ctk.StringVar()
        ctk.CTkEntry(
            frame, textvariable=self.model_var,
            placeholder_text="(blank = use preset; e.g. htdemucs_6s)",
        ).grid(row=2, column=1, columnspan=2, padx=8, pady=4, sticky="ew")

        opts = ctk.CTkFrame(frame, fg_color="transparent")
        opts.grid(row=3, column=0, columnspan=3, padx=12, pady=(4, 10), sticky="w")

        ctk.CTkLabel(opts, text="Format").grid(row=0, column=0, padx=(0, 6))
        self.format_var = ctk.StringVar(value="both")
        ctk.CTkSegmentedButton(
            opts, values=_FORMATS, variable=self.format_var
        ).grid(row=0, column=1, padx=(0, 16))

        ctk.CTkLabel(opts, text="Device").grid(row=0, column=2, padx=(0, 6))
        self.device_var = ctk.StringVar(value="auto")
        ctk.CTkOptionMenu(
            opts, values=_DEVICES, variable=self.device_var, width=90
        ).grid(row=0, column=3, padx=(0, 16))

        ctk.CTkLabel(opts, text="WAV bits").grid(row=0, column=4, padx=(0, 6))
        self.bitdepth_var = ctk.StringVar(value="24")
        ctk.CTkOptionMenu(
            opts, values=_BITDEPTHS, variable=self.bitdepth_var, width=70
        ).grid(row=0, column=5)

        flags = ctk.CTkFrame(frame, fg_color="transparent")
        flags.grid(row=4, column=0, columnspan=3, padx=12, pady=(0, 10), sticky="w")
        self.recursive_var = ctk.BooleanVar(value=False)
        ctk.CTkCheckBox(
            flags, text="Recurse into subfolders", variable=self.recursive_var
        ).grid(row=0, column=0, padx=(0, 20))
        # No "skip existing" control: every run writes a fresh numbered folder
        # (NN_<track>/), so there is never a pre-existing output to skip.

        # Guitar-source picker - only meaningful for guitar-bearing presets
        # (6stem-max); shown/hidden by _on_preset_change.
        self.guitar_row = ctk.CTkFrame(frame, fg_color="transparent")
        self.guitar_row.grid(
            row=5, column=0, columnspan=3, padx=12, pady=(0, 10), sticky="w"
        )
        ctk.CTkLabel(self.guitar_row, text="Guitar source").grid(
            row=0, column=0, padx=(0, 6)
        )
        self.guitar_source_var = ctk.StringVar(value="instrumental")
        ctk.CTkSegmentedButton(
            self.guitar_row, values=list(GUITAR_SOURCES),
            variable=self.guitar_source_var,
        ).grid(row=0, column=1)

        # Vocal-blend picker - only meaningful for presets that ensemble vocals
        # (the "-max" presets); shown/hidden by _on_preset_change.
        self.vocal_blend_row = ctk.CTkFrame(frame, fg_color="transparent")
        self.vocal_blend_row.grid(
            row=6, column=0, columnspan=3, padx=12, pady=(0, 10), sticky="w"
        )
        ctk.CTkLabel(self.vocal_blend_row, text="Vocal blend").grid(
            row=0, column=0, padx=(0, 6)
        )
        self.vocal_blend_var = ctk.StringVar(value=_DEFAULT_VOCAL_BLEND)
        ctk.CTkSegmentedButton(
            self.vocal_blend_row, values=list(_VOCAL_BLENDS),
            variable=self.vocal_blend_var,
        ).grid(row=0, column=1)

        # Selective output - sum the checked stems into one combined file.
        # Checkboxes are rebuilt per preset (its output_stems) and the row is
        # shown only for multi-stem presets; both handled by _on_preset_change.
        self.mix_row = ctk.CTkFrame(frame, fg_color="transparent")
        self.mix_row.grid(
            row=7, column=0, columnspan=3, padx=12, pady=(0, 10), sticky="w"
        )
        ctk.CTkLabel(self.mix_row, text="Combine into one file").grid(
            row=0, column=0, padx=(0, 6), sticky="w"
        )
        self.mix_checks = ctk.CTkFrame(self.mix_row, fg_color="transparent")
        self.mix_checks.grid(row=0, column=1, sticky="w")
        self.mix_vars: dict[str, ctk.BooleanVar] = {}

        self._on_preset_change(DEFAULT_PRESET)

    def _build_advanced(self) -> None:
        frame = self._section("Advanced (Demucs)", row=2)

        ctk.CTkLabel(frame, text="Segment (s)").grid(
            row=1, column=0, padx=12, pady=(4, 10), sticky="w"
        )
        self.segment_var = ctk.StringVar(
            value="" if DEFAULT_SEGMENT is None else str(DEFAULT_SEGMENT)
        )
        ctk.CTkEntry(frame, textvariable=self.segment_var, width=100).grid(
            row=1, column=1, padx=8, pady=(4, 10), sticky="w"
        )
        ctk.CTkLabel(frame, text="Overlap (0–1)").grid(
            row=1, column=2, padx=12, pady=(4, 10), sticky="e"
        )
        self.overlap_var = ctk.StringVar(value=str(DEFAULT_OVERLAP))
        ctk.CTkEntry(frame, textvariable=self.overlap_var, width=100).grid(
            row=1, column=3, padx=(0, 12), pady=(4, 10), sticky="w"
        )

    def _build_run_area(self) -> None:
        frame = ctk.CTkFrame(self)
        # Right column: the run controls + queue, alongside the option sections.
        frame.grid(row=0, column=1, rowspan=3, padx=(0, 16), pady=(10, 0),
                   sticky="nsew")
        frame.grid_columnconfigure(0, weight=1)
        frame.grid_rowconfigure(5, weight=1)  # queue panel grows
        # Root row 3 holds the full-width Log; give it the spare vertical space
        # so the option sections (rows 0-2) stay packed at their natural height.
        # minsize keeps a usable log even with the tallest preset (6stem-max).
        self.grid_rowconfigure(3, weight=1, minsize=120)

        buttons = ctk.CTkFrame(frame, fg_color="transparent")
        buttons.grid(row=0, column=0, padx=12, pady=8, sticky="ew")
        self.add_btn = ctk.CTkButton(
            buttons, text="Add to queue", command=self._on_add_to_queue, width=120,
        )
        self.add_btn.grid(row=0, column=0, padx=(0, 8))
        self.run_btn = ctk.CTkButton(
            buttons, text="Run", command=self._on_run, width=90
        )
        self.run_btn.grid(row=0, column=1, padx=(0, 8))
        self.cancel_btn = ctk.CTkButton(
            buttons, text="Cancel all", command=self._on_cancel, width=100,
            state="disabled", fg_color="gray40",
        )
        self.cancel_btn.grid(row=0, column=2, padx=(0, 8))
        self.open_btn = ctk.CTkButton(
            buttons, text="Open output folder", command=self._open_output,
            width=160, state="disabled",
        )
        self.open_btn.grid(row=0, column=3)

        status_row = ctk.CTkFrame(frame, fg_color="transparent")
        status_row.grid(row=1, column=0, padx=12, pady=(2, 0), sticky="ew")
        status_row.grid_columnconfigure(0, weight=1)
        self.status_var = ctk.StringVar(value="Ready.")
        ctk.CTkLabel(status_row, textvariable=self.status_var, anchor="w").grid(
            row=0, column=0, sticky="ew"
        )
        self.overall_var = ctk.StringVar(value="")
        ctk.CTkLabel(
            status_row, textvariable=self.overall_var, anchor="e",
            text_color="gray60",
        ).grid(row=0, column=1, sticky="e")

        self.step_bar = ctk.CTkProgressBar(frame)
        self.step_bar.set(0)
        self.step_bar.grid(row=2, column=0, padx=12, pady=(4, 2), sticky="ew")

        self.file_bar = ctk.CTkProgressBar(frame, height=8)
        self.file_bar.set(0)
        self.file_bar.grid(row=3, column=0, padx=12, pady=(0, 8), sticky="ew")

        queue_header = ctk.CTkFrame(frame, fg_color="transparent")
        queue_header.grid(row=4, column=0, padx=12, pady=(0, 2), sticky="ew")
        queue_header.grid_columnconfigure(0, weight=1)
        ctk.CTkLabel(
            queue_header, text="Queue", font=ctk.CTkFont(size=13, weight="bold"),
            anchor="w",
        ).grid(row=0, column=0, sticky="w")
        self.clear_btn = ctk.CTkButton(
            queue_header, text="Clear", command=self._on_clear_queue, width=70,
            fg_color="gray40",
        )
        self.clear_btn.grid(row=0, column=1, sticky="e")

        self.queue_frame = ctk.CTkScrollableFrame(frame, height=150)
        self.queue_frame.grid(row=5, column=0, padx=12, pady=(0, 8), sticky="nsew")
        self.queue_frame.grid_columnconfigure(0, weight=1)
        self.queue_empty = ctk.CTkLabel(
            self.queue_frame,
            text="No jobs queued. Add to queue to enable Run.",
            text_color="gray50",
        )
        self.queue_empty.grid(row=0, column=0, padx=8, pady=8, sticky="w")

        # Log spans the full window width along the bottom, under both columns,
        # filling what would otherwise be empty space below the option sections.
        log_frame = ctk.CTkFrame(self)
        log_frame.grid(
            row=3, column=0, columnspan=2, padx=16, pady=(10, 16), sticky="nsew"
        )
        log_frame.grid_columnconfigure(0, weight=1)
        log_frame.grid_rowconfigure(1, weight=1)
        ctk.CTkLabel(
            log_frame, text="Log", anchor="w", text_color="gray60"
        ).grid(row=0, column=0, padx=12, pady=(8, 0), sticky="w")
        self.log = ctk.CTkTextbox(log_frame, wrap="word", height=120)
        self.log.grid(row=1, column=0, padx=12, pady=(4, 12), sticky="nsew")
        self.log.configure(state="disabled")

        self._update_run_button()  # start disabled until a job is queued

    # ----------------------------------------------------------- drag-and-drop

    def _enable_dnd(self) -> bool:
        """Bootstrap tkinterdnd2 on this window. Returns True if drops will work."""
        if not _DND_AVAILABLE:
            return False
        try:
            self.TkdndVersion = TkinterDnD._require(self)
            return True
        except Exception:  # tkdnd Tcl package missing/unloadable - skip silently
            return False

    def _register_drop(self, widget, var, dirs_only: bool = False) -> None:
        """Make ``widget`` accept dropped files/folders, filling ``var``."""
        if not self._dnd_ready:
            return
        try:
            widget.drop_target_register(DND_FILES)
            widget.dnd_bind(
                "<<Drop>>",
                lambda e, v=var, d=dirs_only: self._on_drop(e, v, d),
            )
        except Exception:  # registration unsupported for this widget - ignore
            pass

    def _on_drop(self, event, var, dirs_only: bool) -> None:
        # event.data is a Tcl list; splitlist handles {braced paths with spaces}.
        try:
            paths = list(self.tk.splitlist(event.data))
        except Exception:
            paths = [event.data]
        if not paths:
            return
        chosen = paths[0]
        if dirs_only and not Path(chosen).is_dir():
            chosen = str(Path(chosen).parent)
        var.set(chosen)
        # Keep the dialogs' memory in step with drops too.
        key = "output_dir" if var is self.output_var else "input_dir"
        directory = chosen if Path(chosen).is_dir() else str(Path(chosen).parent)
        self._remember(key, directory)

    # ------------------------------------------------------ single-instance

    def _poll_show_request(self) -> None:
        """Bring this window forward if another launch asked us to; re-arm."""
        si = self._single_instance
        if si is None:
            return
        if si.consume_show_request():
            self._bring_to_front()
        self.after(_SHOW_POLL_MS, self._poll_show_request)

    def _bring_to_front(self) -> None:
        """Restore, raise, and briefly pin the window above others."""
        try:
            self.deiconify()
            self.lift()
            self.focus_force()
            # A momentary topmost toggle is the reliable way to steal foreground
            # on Windows without leaving the window permanently on top.
            self.attributes("-topmost", True)
            self.after(300, lambda: self.attributes("-topmost", False))
        except Exception:
            pass

    # --------------------------------------------------------------- handlers

    def _on_preset_change(self, name: str) -> None:
        preset = PRESETS.get(name)
        if preset is None:
            self.preset_desc.configure(text="")
            return
        self.preset_desc.configure(
            text=f"{preset.description}\n→ {', '.join(preset.output_stems)}"
        )
        if getattr(preset, "guitar_model", ""):
            self.guitar_row.grid()
        else:
            self.guitar_row.grid_remove()
        # Vocal blend only applies when the preset merges several vocal models.
        if getattr(preset, "vocal_models", []):
            self.vocal_blend_row.grid()
        else:
            self.vocal_blend_row.grid_remove()
        # Rebuild the selective-output checkboxes for this preset's stems. Only
        # meaningful for multi-stem presets (combining vocals+instrumental is a
        # no-op), so the row is hidden for 2-stem plans.
        for child in self.mix_checks.winfo_children():
            child.destroy()
        self.mix_vars = {}
        if len(preset.output_stems) > 2:
            for i, stem in enumerate(preset.output_stems):
                var = ctk.BooleanVar(value=False)
                self.mix_vars[stem] = var
                ctk.CTkCheckBox(
                    self.mix_checks, text=stem, variable=var, width=70,
                ).grid(row=0, column=i, padx=(0, 8))
            self.mix_row.grid()
        else:
            self.mix_row.grid_remove()

    def _initial_dir(self, key: str, fallback: str | None = None) -> str | None:
        """Remembered dir for a dialog if it still exists, else fallback/None."""
        remembered = self._settings.get(key)
        if remembered and Path(remembered).is_dir():
            return remembered
        if fallback and Path(fallback).is_dir():
            return fallback
        return None  # None → the dialog uses its own default location

    def _remember(self, key: str, directory) -> None:
        """Persist a last-used directory for future dialogs."""
        directory = str(directory)
        if self._settings.get(key) == directory:
            return
        self._settings[key] = directory
        save_settings(self._settings)

    def _pick_file(self) -> None:
        exts = " ".join(f"*{s}" for s in sorted(SUPPORTED_INPUT_SUFFIXES))
        path = filedialog.askopenfilename(
            title="Choose an audio file",
            filetypes=[("Audio", exts), ("All files", "*.*")],
            initialdir=self._initial_dir("input_dir"),
        )
        if path:
            self.input_var.set(path)
            self._remember("input_dir", Path(path).parent)

    def _pick_folder(self) -> None:
        path = filedialog.askdirectory(
            title="Choose a folder of audio",
            initialdir=self._initial_dir("input_dir"),
        )
        if path:
            self.input_var.set(path)
            self._remember("input_dir", Path(path).parent)

    def _pick_output(self) -> None:
        path = filedialog.askdirectory(
            title="Choose output folder",
            initialdir=self._initial_dir("output_dir", self.output_var.get()),
        )
        if path:
            self.output_var.set(path)
            self._remember("output_dir", path)

    def _reveal_folder(self, target: Path | None) -> None:
        """Open ``target`` in the OS file manager (used by the Open button and the
        clickable "done" badge)."""
        if target is None:
            return
        target = Path(target)
        if not target.exists():
            self._append_log(f"Output folder does not exist yet: {target}")
            return
        try:
            if sys.platform.startswith("win"):
                os.startfile(target)  # noqa: S606 - intended shell open
            elif sys.platform == "darwin":
                import subprocess
                subprocess.Popen(["open", str(target)])
            else:
                import subprocess
                subprocess.Popen(["xdg-open", str(target)])
        except Exception as exc:
            self._append_log(f"Could not open folder: {exc}")

    def _open_output(self) -> None:
        self._reveal_folder(self._last_out_dir or Path(self.output_var.get()))

    # ------------------------------------------------------------------- queue

    def _job_label(self, params: JobParams) -> str:
        plan = params.model or params.preset or "default"
        return f"{params.input_path.name}  ·  {plan}"

    def _on_add_to_queue(self) -> None:
        self._enqueue_current()

    def _enqueue_current(self) -> bool:
        """Snapshot the current form as a queued job. Returns False if invalid."""
        params = self._collect_params()
        if params is None:
            return False
        self._next_job_id += 1
        job = QueuedJob(
            id=self._next_job_id, params=params, label=self._job_label(params)
        )
        self._queue.append(job)
        self._jobs_by_id[job.id] = job
        self._render_job_row(job)
        # If a run is already in flight, hand the new job to the live worker queue.
        if self._running and self._job_q is not None:
            self._job_q.put((job.id, job.params))
        self._update_run_button()
        return True

    def _render_job_row(self, job: QueuedJob) -> None:
        self.queue_empty.grid_remove()
        # Grid at the job's (monotonic) id so rows never collide; removed rows
        # leave a zero-height gap, which the grid collapses visually.
        row = ctk.CTkFrame(self.queue_frame)
        row.grid(row=job.id, column=0, sticky="ew", padx=2, pady=2)
        row.grid_columnconfigure(0, weight=1)
        name = ctk.CTkLabel(row, text=job.label, anchor="w")
        name.grid(row=0, column=0, padx=(8, 6), pady=4, sticky="ew")
        time_lbl = ctk.CTkLabel(row, text="", width=52, text_color="gray60")
        time_lbl.grid(row=0, column=1, padx=4)
        badge = ctk.CTkLabel(row, text="", width=72)
        badge.grid(row=0, column=2, padx=4)
        # Clicking a finished job's "done" badge opens its output folder. Bound
        # once here; the handler is a no-op unless the job is done.
        badge.bind("<Button-1>", lambda _e, j=job: self._on_badge_click(j))
        action = ctk.CTkButton(row, text="✕", width=30, fg_color="gray40")
        action.grid(row=0, column=3, padx=(4, 8))
        job.widgets = {
            "row": row, "name": name, "time": time_lbl, "badge": badge,
            "action": action,
        }
        self._set_row_state(job)

    def _set_row_state(self, job: QueuedJob) -> None:
        text, colour = _STATUS_STYLE.get(job.status, ("queued", "gray60"))
        badge = job.widgets["badge"]
        badge.configure(text=text, text_color=colour)
        # A finished job's green "done" badge is clickable (bound once in
        # _render_job_row); show a hand cursor to advertise it. CTkLabel.configure
        # doesn't take `cursor`, so set it on the inner tk widgets directly.
        clickable = job.status == "done" and job.out_dir is not None
        cursor = "hand2" if clickable else ""
        try:
            badge._label.configure(cursor=cursor)
            badge._canvas.configure(cursor=cursor)
        except Exception:  # internal attrs are best-effort cosmetics only
            pass
        action = job.widgets["action"]
        if job.status == "running":
            action.configure(
                text="Cancel task", width=90, fg_color="#a23",
                command=lambda j=job: self._on_cancel_task(j), state="normal",
            )
        elif job.status == "queued":
            action.configure(
                text="✕", width=30, fg_color="gray40",
                command=lambda j=job: self._on_remove_job(j),
                state="disabled" if self._running else "normal",
            )
        else:  # done / failed / skipped / cancelled
            action.configure(text="✓" if job.status == "done" else "-",
                             width=30, fg_color="gray30", state="disabled")
        self._update_run_button()

    def _on_badge_click(self, job: QueuedJob) -> None:
        """Open the job's output folder when its "done" badge is clicked."""
        if job.status == "done" and job.out_dir is not None:
            self._reveal_folder(job.out_dir)

    def _on_remove_job(self, job: QueuedJob) -> None:
        if self._running or job.status != "queued":
            return
        job.widgets["row"].destroy()
        self._queue.remove(job)
        self._jobs_by_id.pop(job.id, None)
        if not self._queue:
            self.queue_empty.grid()
        self._update_run_button()

    def _on_clear_queue(self) -> None:
        if self._running:
            return
        for job in list(self._queue):
            job.widgets["row"].destroy()
        self._queue.clear()
        self._jobs_by_id.clear()
        self.queue_empty.grid()
        self._update_run_button()

    def _on_cancel_task(self, job: QueuedJob) -> None:
        """Cancel just the running job; the queue advances to the next one."""
        if job.id != self._current_job_id:
            return
        self._skip_current.set()
        self.status_var.set("Cancelling this task…")
        job.widgets["action"].configure(state="disabled")

    # --------------------------------------------------------------------- run

    def _on_run(self) -> None:
        # Run is disabled unless the queue has pending jobs (see
        # _update_run_button), so there is always something to start here.
        pending = [j for j in self._queue if j.status == "queued"]
        if not pending:
            return
        self._start_queue_run(pending)

    def _prepare_run(self) -> None:
        self._cancel = threading.Event()
        self._skip_current = threading.Event()
        self._events = queue.Queue()
        self._last_out_dir = None
        self._current_job_id = None
        self._run_started = time.monotonic()
        self._set_running(True)
        self._clear_log()
        self.step_bar.set(0)
        self.file_bar.set(0)
        self._arm_ticker()
        self.after(_POLL_MS, self._drain_events)

    def _start_queue_run(self, pending: list[QueuedJob]) -> None:
        self._prepare_run()
        self._job_q = queue.Queue()
        for job in pending:
            self._job_q.put((job.id, job.params))
        plural = "job" if len(pending) == 1 else "jobs"
        self.status_var.set(f"Starting {len(pending)} {plural}…")
        self._thread = threading.Thread(
            target=run_jobs,
            args=(self._job_q, self._events, self._cancel, self._skip_current),
            daemon=True,
        )
        self._thread.start()

    def _on_cancel(self) -> None:
        self._cancel.set()
        self.status_var.set("Cancelling after the current file…")
        self.cancel_btn.configure(state="disabled")

    def _collect_params(self) -> JobParams | None:
        raw_input = self.input_var.get().strip()
        if not raw_input:
            self._append_log("Please choose an input file or folder.")
            return None
        input_path = Path(raw_input)
        if not input_path.exists():
            self._append_log(f"Input path does not exist: {input_path}")
            return None

        output_root = Path(self.output_var.get().strip() or "output")
        model = self.model_var.get().strip() or None
        preset = None if model else self.preset_var.get()
        # Only forward a guitar source for presets that actually use one.
        guitar_source = None
        if preset and getattr(PRESETS.get(preset), "guitar_model", ""):
            guitar_source = self.guitar_source_var.get()
        # Only forward a vocal blend for presets that ensemble vocal models.
        vocal_method = None
        if preset and getattr(PRESETS.get(preset), "vocal_models", []):
            vocal_method = _VOCAL_BLENDS.get(self.vocal_blend_var.get())
        # Selective output: checked stems are summed into one combined file.
        selected = [s for s, v in self.mix_vars.items() if v.get()]
        combine = selected if (preset and selected) else None

        segment_text = self.segment_var.get().strip()
        try:
            segment = float(segment_text) if segment_text else None
            overlap = float(self.overlap_var.get().strip() or DEFAULT_OVERLAP)
        except ValueError:
            self._append_log("Segment and overlap must be numbers.")
            return None

        return JobParams(
            input_path=input_path,
            output_root=output_root,
            preset=preset,
            model=model,
            engine=None,
            stems=None,
            fmt=self.format_var.get(),
            bitdepth=int(self.bitdepth_var.get()),
            device=self.device_var.get(),
            segment=segment,
            overlap=overlap,
            recursive=self.recursive_var.get(),
            skip_existing=False,  # GUI always writes a fresh numbered folder
            guitar_source=guitar_source,
            vocal_method=vocal_method,
            combine=combine,
        )

    # -------------------------------------------------------------- event pump

    def _drain_events(self) -> None:
        """Apply all queued worker events on the main thread, then re-arm."""
        if self._events is None:
            return
        finished = False
        try:
            while True:
                ev = self._events.get_nowait()
                if self._apply_event(ev):
                    finished = True
        except queue.Empty:
            pass

        if finished:
            self._set_running(False)
        else:
            self.after(_POLL_MS, self._drain_events)

    def _apply_event(self, ev: Event) -> bool:
        """Render one event. Returns True when the run is finished."""
        kind, data = ev.kind, ev.data
        if kind == "job_start":
            self._on_job_start(data)
        elif kind == "job_done":
            self._finish_job(data["id"], "done")
        elif kind == "job_error":
            self._finish_job(data["id"], "failed")
        elif kind == "job_cancelled":
            self._finish_job(data["id"], "cancelled")
        elif kind == "queue_done":
            self._end_download()
            self._finish(data)
            return True
        elif kind == "prefetch":
            self._begin_download(data["model"])
        elif kind == "file_start":
            self._end_download()
            total = data["total"]
            self.file_bar.set((data["index"] - 1) / total if total else 0)
            self.status_var.set(
                f"[{data['index']}/{total}] {data['name']}"
            )
            self._append_log(f"▶ {data['name']}")
        elif kind == "step":
            self._end_download()
            total = data["total"] or 1
            self.step_bar.set(data["index"] / total)
            self.status_var.set(
                f"[{data['index']}/{total}] {data['model']} · {data['action']}"
            )
        elif kind == "file_skipped":
            self._append_log(f"  skipped (output exists): {data['name']}")
        elif kind == "file_done":
            self.step_bar.set(1)
            out_dir = Path(data["out_dir"])
            self._last_out_dir = out_dir
            job = self._jobs_by_id.get(self._current_job_id)
            if job is not None:
                job.out_dir = out_dir  # remember it for the clickable "done" badge
            self.open_btn.configure(state="normal")
            self._append_log(
                f"  ✓ {data['name']} → {len(data['written'])} files"
            )
        elif kind == "file_error":
            self._append_log(f"  ✗ {data['name']}: {data['message']}")
        elif kind == "log":
            if data.get("message"):
                self._append_log(data["message"])
        elif kind == "batch_done":
            self._end_download()
            self._finish(data)
            return True
        return False

    def _on_job_start(self, data: dict) -> None:
        job = self._jobs_by_id.get(data["id"])
        self._current_job_id = data["id"]
        self.step_bar.set(0)
        self.file_bar.set(0)
        if job is not None:
            job.status = "running"
            job.started = time.monotonic()
            job.elapsed = None
            self._set_row_state(job)
        self._append_log(f"▶ Job {data['index']}: {data['name']}")

    def _finish_job(self, job_id: int, status: str) -> None:
        job = self._jobs_by_id.get(job_id)
        if job is None:
            return
        job.status = status
        if job.started is not None:
            job.elapsed = time.monotonic() - job.started
            job.widgets["time"].configure(text=_fmt_mmss(job.elapsed))
        self._set_row_state(job)
        if self._current_job_id == job_id:
            self._current_job_id = None

    def _finish(self, summary: dict) -> None:
        self.file_bar.set(1)
        self.step_bar.set(1 if summary.get("done") else 0)
        if "jobs" in summary:
            msg = (
                f"Queue done: {summary.get('jobs', 0)} job(s) - "
                f"{summary.get('done', 0)} files separated, "
                f"{summary.get('skipped', 0)} skipped, "
                f"{summary.get('failed', 0)} failed."
            )
        else:
            msg = (
                f"Done: {summary.get('done', 0)} separated, "
                f"{summary.get('skipped', 0)} skipped, "
                f"{summary.get('failed', 0)} failed."
            )
        if summary.get("cancelled"):
            msg = "Cancelled. " + msg
        self.status_var.set(msg)
        self._append_log(msg)

    # ---------------------------------------------------------- download bar

    def _begin_download(self, model: str) -> None:
        self.status_var.set(f"↓ Downloading {model}…")
        self._append_log(f"↓ Downloading model {model}…")
        if not self._downloading:
            self._downloading = True
            self.step_bar.configure(mode="indeterminate")
            self.step_bar.start()

    def _end_download(self) -> None:
        if self._downloading:
            self._downloading = False
            self.step_bar.stop()
            self.step_bar.configure(mode="determinate")
            self.step_bar.set(0)

    # ------------------------------------------------------- elapsed-time ticker

    def _arm_ticker(self) -> None:
        if not self._ticking:
            self._ticking = True
            self._tick()

    def _tick(self) -> None:
        if not self._ticking:
            return
        now = time.monotonic()
        if self._run_started is not None:
            self.overall_var.set(f"⏱ {_fmt_mmss(now - self._run_started)}")
        job = (
            self._jobs_by_id.get(self._current_job_id)
            if self._current_job_id is not None else None
        )
        if job is not None and job.started is not None and job.status == "running":
            job.widgets["time"].configure(text=_fmt_mmss(now - job.started))
        self.after(_TICK_MS, self._tick)

    def _stop_ticker(self) -> None:
        self._ticking = False

    # --------------------------------------------------------------- ui state

    def _update_run_button(self) -> None:
        """Run is enabled only when at least one pending (queued) job exists and
        no run is already in flight."""
        has_pending = any(j.status == "queued" for j in self._queue)
        self.run_btn.configure(
            state="normal" if (has_pending and not self._running) else "disabled"
        )

    def _set_running(self, running: bool) -> None:
        self._running = running
        self.cancel_btn.configure(state="normal" if running else "disabled")
        # Add stays enabled while running (live append); Clear is idle-only.
        self.clear_btn.configure(state="disabled" if running else "normal")
        for job in self._queue:  # refresh per-row remove/cancel availability
            self._set_row_state(job)
        if not running:
            self._current_job_id = None
            self._stop_ticker()
            self._end_download()
        self._update_run_button()

    def _clear_log(self) -> None:
        self.log.configure(state="normal")
        self.log.delete("1.0", "end")
        self.log.configure(state="disabled")

    def _append_log(self, text: str) -> None:
        self.log.configure(state="normal")
        self.log.insert("end", text + "\n")
        self.log.see("end")
        self.log.configure(state="disabled")
