"""Hide child-process console windows when the GUI runs windowless.

A ``pythonw`` (no-console) GUI process spawns every console child - ffmpeg for
decode/MP3 export, plus anything demucs / audio-separator shell out to - in its
*own* console window unless told otherwise, which looks like stray black boxes
flashing during a run. We install a process-wide default so every
``subprocess.Popen`` created in this process gets ``CREATE_NO_WINDOW`` and a
hidden ``STARTUPINFO``. No-op off Windows; safe to call more than once.

This patches the class once at startup (rather than touching each call site) so
it also covers subprocesses created deep inside third-party libraries.
"""

from __future__ import annotations

import sys

_installed = False


def suppress_child_consoles() -> None:
    """Make all child processes in this process windowless (Windows only)."""
    global _installed
    if _installed or not sys.platform.startswith("win"):
        return

    import subprocess

    create_no_window = getattr(subprocess, "CREATE_NO_WINDOW", 0x08000000)
    original_init = subprocess.Popen.__init__

    def _init(self, *args, **kwargs):  # type: ignore[no-untyped-def]
        kwargs["creationflags"] = (kwargs.get("creationflags") or 0) | create_no_window
        startupinfo = kwargs.get("startupinfo") or subprocess.STARTUPINFO()
        startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
        startupinfo.wShowWindow = subprocess.SW_HIDE
        kwargs["startupinfo"] = startupinfo
        original_init(self, *args, **kwargs)

    subprocess.Popen.__init__ = _init  # type: ignore[method-assign]
    _installed = True
