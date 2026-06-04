"""Runtime configuration: device detection, cache paths, and quality defaults.

Centralizes the few environment-dependent choices so the rest of the code can
stay pure. Tuned for an 8 GB-class GPU (e.g. RTX 3060 Ti) by default.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

# Root of the repo (…/src/stems/config.py -> repo root is parents[2]).
PACKAGE_ROOT = Path(__file__).resolve().parent
REPO_ROOT = PACKAGE_ROOT.parents[1]

# Where model weights are downloaded/cached. Override with STEMS_MODEL_DIR.
MODEL_DIR = Path(os.environ.get("STEMS_MODEL_DIR", REPO_ROOT / "models"))

# Demucs inference defaults tuned for ~8 GB VRAM. Smaller `segment` lowers peak
# memory; higher `overlap` improves quality at the cost of speed.
DEFAULT_SEGMENT: float | None = 7.8  # seconds per chunk (None = model default)
DEFAULT_OVERLAP: float = 0.25

# Export defaults.
DEFAULT_BITDEPTH = 24
DEFAULT_MP3_BITRATE = "320k"


def resolve_device(requested: str = "auto") -> str:
    """Resolve a device string to a concrete torch device.

    `auto` picks CUDA when available, otherwise CPU. An explicit `cuda` request
    falls back to CPU (with no error) if CUDA is unavailable, so the tool still
    runs on machines without a GPU.
    """
    requested = (requested or "auto").lower()
    try:
        import torch

        cuda_ok = torch.cuda.is_available()
    except Exception:  # torch not importable yet / broken install
        cuda_ok = False

    if requested == "cpu":
        return "cpu"
    if requested == "cuda":
        return "cuda" if cuda_ok else "cpu"
    # auto
    return "cuda" if cuda_ok else "cpu"


@dataclass(slots=True)
class RunConfig:
    """Resolved per-run settings passed through the pipeline."""

    device: str = "cpu"
    segment: float | None = DEFAULT_SEGMENT
    overlap: float = DEFAULT_OVERLAP
    bitdepth: int = DEFAULT_BITDEPTH
    mp3_bitrate: str = DEFAULT_MP3_BITRATE
    model_dir: Path = MODEL_DIR

    def ensure_dirs(self) -> None:
        self.model_dir.mkdir(parents=True, exist_ok=True)
