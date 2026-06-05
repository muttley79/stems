"""Single-instance guard for the desktop GUI.

Stops a second ``stems-gui`` window from running at the same time, using an OS
*advisory* lock on a small lock file (``msvcrt`` on Windows, ``fcntl`` elsewhere).
The lock is held by the process for its whole lifetime and the OS releases it
automatically on exit — including a crash — so there is never a stale lock to
clean up, and no extra dependency is needed.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path
from typing import IO

# Keep the locked file handle alive for the whole process; dropping/closing it
# releases the lock. Held at module scope so callers needn't keep a reference.
_lock_handle: IO | None = None


def acquire_single_instance_lock(
    name: str = "stems-gui", lock_dir: Path | None = None
) -> bool:
    """Try to become the sole running instance.

    Returns ``True`` if this process acquired the lock, ``False`` if another
    instance already holds it. On any unexpected error it *fails open* (returns
    ``True``) so a locking quirk never prevents the app from starting.
    """
    global _lock_handle
    directory = Path(lock_dir) if lock_dir else Path(tempfile.gettempdir())
    lock_path = directory / f"{name}.lock"
    try:
        fh = open(lock_path, "a+")
    except OSError:
        return True  # can't even create the lock file — don't block startup

    if _try_lock(fh):
        _lock_handle = fh
        return True
    fh.close()
    return False


def _try_lock(fh: IO) -> bool:
    """Take a non-blocking exclusive lock on the first byte. False if held."""
    try:
        fh.seek(0)
        if sys.platform.startswith("win"):
            import msvcrt

            msvcrt.locking(fh.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl

            fcntl.flock(fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        return True
    except OSError:
        return False


def release_single_instance_lock() -> None:
    """Release the lock early. Optional — the OS also releases it on exit."""
    global _lock_handle
    if _lock_handle is None:
        return
    try:
        _lock_handle.seek(0)
        if sys.platform.startswith("win"):
            import msvcrt

            msvcrt.locking(_lock_handle.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl

            fcntl.flock(_lock_handle.fileno(), fcntl.LOCK_UN)
    except OSError:
        pass
    finally:
        try:
            _lock_handle.close()
        except OSError:
            pass
        _lock_handle = None
