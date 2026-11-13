"""Live in-progress previews: listen to a separation while it is still running.

While the first UVR (Roformer) pass of a track runs, :class:`PreviewWriter`
keeps ``_in_progress_<stem>.wav`` files in the output folder up to date with
everything separated so far (they grow from the start of the track). At the end
of that pass the files are rewritten to full length, and once the real stems
are exported they are deleted.

The preview is the output of that *one* model. Ensemble presets merge several
models and apply tail rescue afterwards, so the final stems sound somewhat
different (usually fuller). Demucs-only plans produce no preview.

The engine side lives in :mod:`stems.engines.uvr_engine`
(``ensure_preview_patch`` + :func:`preview_sink`); this module only decides
when and where to write. Writes are atomic (temp file + ``os.replace``) so a
player never opens a half-written file. On Windows a player that holds the file
open blocks the replace; that update is skipped, and the next one retries.
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Callable

import numpy as np

PREVIEW_PREFIX = "_in_progress_"
DEFAULT_PREVIEW_INTERVAL = 15.0  # seconds between snapshot rewrites


def is_preview_file(path: Path) -> bool:
    return Path(path).name.startswith(PREVIEW_PREFIX)


class PreviewWriter:
    """Receives partial stems from the engine and rewrites the preview files.

    Protocol used by the engine: ``active`` (still accepting a pass?),
    ``due()`` (time for a snapshot?), ``update(stems, sr)``, ``pass_done()``.
    Only the first pass is previewed; later passes restart from the top of the
    track, and switching to them would make the preview shrink.
    """

    def __init__(
        self,
        out_dir: Path,
        interval: float = DEFAULT_PREVIEW_INTERVAL,
        wanted: list[str] | None = None,
        on_first_write: Callable[[list[Path]], None] | None = None,
    ) -> None:
        self.out_dir = Path(out_dir)
        self.interval = interval
        self.wanted = set(wanted) if wanted else None
        self.active = True
        self._last = time.monotonic()
        self._on_first_write = on_first_write
        self._written: set[Path] = set()

    def due(self) -> bool:
        return self.active and time.monotonic() - self._last >= self.interval

    def update(self, stems: dict[str, np.ndarray], sr: int) -> None:
        if not self.active:
            return
        self._last = time.monotonic()
        fresh: list[Path] = []
        for name, audio in stems.items():
            if self.wanted is not None and name not in self.wanted:
                continue
            path = self._write(name, np.asarray(audio, dtype=np.float32), sr)
            if path is not None and path not in self._written:
                self._written.add(path)
                fresh.append(path)
        if fresh and self._on_first_write is not None:
            try:
                self._on_first_write(sorted(fresh))
            except Exception:
                pass  # a reporting hiccup must never break the separation

    def pass_done(self) -> None:
        self.active = False

    def cleanup(self) -> None:
        """Delete the preview files (call after the real stems are exported)."""
        self.active = False
        for path in list(self._written):
            try:
                path.unlink(missing_ok=True)
            except OSError:
                pass  # still open in a player; harmless leftover
        self._written.clear()

    def _write(self, name: str, audio: np.ndarray, sr: int) -> Path | None:
        target = self.out_dir / f"{PREVIEW_PREFIX}{name}.wav"
        tmp = self.out_dir / f"{PREVIEW_PREFIX}{name}.partial"
        try:
            # 32-bit float: stems can exceed full scale, and integer PCM would
            # both clip and warn on every snapshot. soundfile infers the format
            # from the extension, so name the format explicitly for ".partial".
            self.out_dir.mkdir(parents=True, exist_ok=True)
            import soundfile as sf

            sf.write(
                str(tmp), np.ascontiguousarray(audio.T), sr,
                format="WAV", subtype="FLOAT",
            )
            os.replace(tmp, target)
            return target
        except OSError:
            try:
                tmp.unlink(missing_ok=True)
            except OSError:
                pass
            return target if target.exists() else None


__all__ = [
    "DEFAULT_PREVIEW_INTERVAL", "PREVIEW_PREFIX", "PreviewWriter",
    "is_preview_file",
]
