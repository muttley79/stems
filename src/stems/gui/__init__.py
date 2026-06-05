"""Desktop GUI for stems — a second thin front-end over the same pipeline.

The GUI mirrors the CLI: it collects a plan (preset or raw model) plus output
settings and runs the exact same core (:mod:`stems.pipeline`) on a background
thread, streaming progress back to the window. The CLI is unaffected.

``customtkinter`` is an optional dependency (the ``gui`` extra); it is imported
lazily inside :func:`main` so the rest of the package — and the unit tests —
import without it.
"""

from __future__ import annotations


def main() -> None:
    """Launch the stems desktop GUI (single instance only)."""
    from stems.gui.single_instance import acquire_single_instance_lock

    # Only one window at a time: a second launch just tells the user and exits.
    if not acquire_single_instance_lock():
        _warn_already_running()
        return

    try:
        import customtkinter as ctk
    except ImportError as exc:  # pragma: no cover - depends on optional extra
        raise SystemExit(
            "The stems GUI needs the 'customtkinter' package.\n"
            "Install it with:  pip install -e .[gui]\n"
            "             or:  pip install customtkinter"
        ) from exc

    from stems.gui.app import StemsApp

    ctk.set_appearance_mode("system")
    ctk.set_default_color_theme("blue")
    app = StemsApp()
    app.mainloop()


def _warn_already_running() -> None:  # pragma: no cover - UI-only path
    """Tell the user a window is already open. Best-effort; never raises."""
    try:
        import tkinter as tk
        from tkinter import messagebox

        root = tk.Tk()
        root.withdraw()
        messagebox.showinfo("stems", "stems GUI is already running.")
        root.destroy()
    except Exception:
        pass


if __name__ == "__main__":  # pragma: no cover
    main()
