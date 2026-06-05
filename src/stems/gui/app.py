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
from pathlib import Path
from tkinter import filedialog

import customtkinter as ctk

from stems import __version__
from stems.audio_io import SUPPORTED_INPUT_SUFFIXES
from stems.config import DEFAULT_OVERLAP, DEFAULT_SEGMENT
from stems.gui.worker import Event, JobParams, run_job
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
_POLL_MS = 100

# Mix the tkinterdnd2 wrapper into the window only when available, so drop-target
# registration works on this root; otherwise StemsApp is a plain CTk window.
_APP_BASES = (ctk.CTk, TkinterDnD.DnDWrapper) if _DND_AVAILABLE else (ctk.CTk,)


class StemsApp(*_APP_BASES):
    """Main application window."""

    def __init__(self) -> None:
        super().__init__()
        self._dnd_ready = self._enable_dnd()
        self.title(f"stems · audio separator  (v{__version__})")
        self.geometry("760x720")
        self.minsize(680, 640)
        self.grid_columnconfigure(0, weight=1)

        # Worker plumbing: a fresh queue + cancel flag are created per run.
        self._events: "queue.Queue[Event]" | None = None
        self._cancel = threading.Event()
        self._thread: threading.Thread | None = None
        self._last_out_dir: Path | None = None
        self._downloading = False

        self._build_inputs()
        self._build_plan_controls()
        self._build_advanced()
        self._build_run_area()

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
        self.skip_var = ctk.BooleanVar(value=False)
        ctk.CTkCheckBox(
            flags, text="Skip files with existing output", variable=self.skip_var
        ).grid(row=0, column=1)

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
        frame.grid(row=3, column=0, padx=16, pady=(10, 16), sticky="nsew")
        frame.grid_columnconfigure(0, weight=1)
        frame.grid_rowconfigure(4, weight=1)
        self.grid_rowconfigure(3, weight=1)

        buttons = ctk.CTkFrame(frame, fg_color="transparent")
        buttons.grid(row=0, column=0, padx=12, pady=8, sticky="ew")
        self.run_btn = ctk.CTkButton(
            buttons, text="Separate", command=self._on_run, width=120
        )
        self.run_btn.grid(row=0, column=0, padx=(0, 8))
        self.cancel_btn = ctk.CTkButton(
            buttons, text="Cancel", command=self._on_cancel, width=100,
            state="disabled", fg_color="gray40",
        )
        self.cancel_btn.grid(row=0, column=1, padx=(0, 8))
        self.open_btn = ctk.CTkButton(
            buttons, text="Open output folder", command=self._open_output,
            width=160, state="disabled",
        )
        self.open_btn.grid(row=0, column=2)

        self.status_var = ctk.StringVar(value="Ready.")
        ctk.CTkLabel(
            frame, textvariable=self.status_var, anchor="w",
        ).grid(row=1, column=0, padx=12, pady=(2, 0), sticky="ew")

        self.step_bar = ctk.CTkProgressBar(frame)
        self.step_bar.set(0)
        self.step_bar.grid(row=2, column=0, padx=12, pady=(4, 2), sticky="ew")

        self.file_bar = ctk.CTkProgressBar(frame, height=8)
        self.file_bar.set(0)
        self.file_bar.grid(row=3, column=0, padx=12, pady=(0, 8), sticky="ew")

        self.log = ctk.CTkTextbox(frame, wrap="word")
        self.log.grid(row=4, column=0, padx=12, pady=(0, 12), sticky="nsew")
        self.log.configure(state="disabled")

    # ----------------------------------------------------------- drag-and-drop

    def _enable_dnd(self) -> bool:
        """Bootstrap tkinterdnd2 on this window. Returns True if drops will work."""
        if not _DND_AVAILABLE:
            return False
        try:
            self.TkdndVersion = TkinterDnD._require(self)
            return True
        except Exception:  # tkdnd Tcl package missing/unloadable — skip silently
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
        except Exception:  # registration unsupported for this widget — ignore
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

    # --------------------------------------------------------------- handlers

    def _on_preset_change(self, name: str) -> None:
        preset = PRESETS.get(name)
        if preset is None:
            self.preset_desc.configure(text="")
            return
        self.preset_desc.configure(
            text=f"{preset.description}\n→ {', '.join(preset.output_stems)}"
        )

    def _pick_file(self) -> None:
        exts = " ".join(f"*{s}" for s in sorted(SUPPORTED_INPUT_SUFFIXES))
        path = filedialog.askopenfilename(
            title="Choose an audio file",
            filetypes=[("Audio", exts), ("All files", "*.*")],
        )
        if path:
            self.input_var.set(path)

    def _pick_folder(self) -> None:
        path = filedialog.askdirectory(title="Choose a folder of audio")
        if path:
            self.input_var.set(path)

    def _pick_output(self) -> None:
        path = filedialog.askdirectory(title="Choose output folder")
        if path:
            self.output_var.set(path)

    def _open_output(self) -> None:
        target = self._last_out_dir or Path(self.output_var.get())
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

    def _on_run(self) -> None:
        params = self._collect_params()
        if params is None:
            return

        self._cancel = threading.Event()
        self._events = queue.Queue()
        self._last_out_dir = None
        self._set_running(True)
        self._clear_log()
        self.status_var.set("Starting…")
        self.step_bar.set(0)
        self.file_bar.set(0)

        self._thread = threading.Thread(
            target=run_job, args=(params, self._events, self._cancel), daemon=True
        )
        self._thread.start()
        self.after(_POLL_MS, self._drain_events)

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
            skip_existing=self.skip_var.get(),
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
        """Render one event. Returns True when the batch is finished."""
        kind, data = ev.kind, ev.data
        if kind == "prefetch":
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
            self._last_out_dir = Path(data["out_dir"])
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

    def _finish(self, summary: dict) -> None:
        self.file_bar.set(1)
        self.step_bar.set(1 if summary.get("done") else 0)
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

    # --------------------------------------------------------------- ui state

    def _set_running(self, running: bool) -> None:
        self.run_btn.configure(state="disabled" if running else "normal")
        self.cancel_btn.configure(state="normal" if running else "disabled")
        if not running:
            self._end_download()

    def _clear_log(self) -> None:
        self.log.configure(state="normal")
        self.log.delete("1.0", "end")
        self.log.configure(state="disabled")

    def _append_log(self, text: str) -> None:
        self.log.configure(state="normal")
        self.log.insert("end", text + "\n")
        self.log.see("end")
        self.log.configure(state="disabled")
