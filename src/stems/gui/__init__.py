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
    """Launch the stems desktop GUI (single instance only).

    If a window is already open, this signals it to come to the front and exits
    instead of opening a second one.
    """
    from stems.gui.single_instance import acquire_or_signal

    server = acquire_or_signal()
    if server is None:
        return  # an existing window was raised to the front; nothing else to do

    try:
        import customtkinter as ctk
    except ImportError as exc:  # pragma: no cover - depends on optional extra
        server.close()
        raise SystemExit(
            "The stems GUI needs the 'customtkinter' package.\n"
            "Install it with:  pip install -e .[gui]\n"
            "             or:  pip install customtkinter"
        ) from exc

    from stems.gui.app import StemsApp

    ctk.set_appearance_mode("system")
    ctk.set_default_color_theme("blue")
    app = StemsApp(single_instance=server)
    try:
        app.mainloop()
    finally:
        server.close()


if __name__ == "__main__":  # pragma: no cover
    main()
