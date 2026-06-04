"""Engine abstraction.

Every backend (Demucs, UVR/Roformer, …) implements :class:`BaseSeparator`.
A separation returns a :class:`SeparationResult`: a dict of named stems plus the
sample rate. Stems are float32 ``(channels, samples)`` arrays (channels-first),
matching :mod:`stems.audio_io`.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from stems.config import RunConfig


@dataclass(slots=True)
class SeparationResult:
    """Output of a single model run."""

    stems: dict[str, np.ndarray]  # stem name -> (channels, samples) float32
    sample_rate: int

    @property
    def names(self) -> list[str]:
        return list(self.stems)


class BaseSeparator(ABC):
    """Common interface for all separation backends."""

    #: Short backend identifier, e.g. "demucs" or "uvr".
    name: str = "base"

    @abstractmethod
    def available_stems(self, model: str) -> list[str]:
        """Stem names a given model produces (e.g. ``["vocals", "instrumental"]``)."""

    @abstractmethod
    def separate(
        self,
        audio_path: Path,
        model: str,
        config: RunConfig,
        stems: list[str] | None = None,
    ) -> SeparationResult:
        """Separate ``audio_path`` with ``model``.

        ``stems`` optionally restricts which stems to return; ``None`` means all.
        Implementations should honor ``config.device``/segment settings.
        """
