"""Demucs v4 backend.

Uses the stable lower-level Demucs API (``pretrained.get_model`` +
``apply.apply_model``) rather than ``demucs.api``, which is not present in all
4.x builds. Demucs returns a tensor shaped ``(sources, channels, samples)``; we
convert to the channels-first float32 numpy convention used across the package.

The mean/std normalize-then-denormalize step mirrors Demucs' own ``separate.py``
recipe so output levels match the reference CLI.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from stems.config import RunConfig
from stems.engines.base import BaseSeparator, SeparationResult

# Stem sets per model (Demucs' native source order).
_MODEL_STEMS: dict[str, list[str]] = {
    "htdemucs": ["drums", "bass", "other", "vocals"],
    "htdemucs_ft": ["drums", "bass", "other", "vocals"],
    "htdemucs_6s": ["drums", "bass", "other", "vocals", "guitar", "piano"],
    "mdx_extra": ["drums", "bass", "other", "vocals"],
    "mdx_extra_q": ["drums", "bass", "other", "vocals"],
}

DEFAULT_MODEL = "htdemucs_ft"


class DemucsSeparator(BaseSeparator):
    name = "demucs"

    def available_stems(self, model: str) -> list[str]:
        return list(_MODEL_STEMS.get(model, _MODEL_STEMS["htdemucs"]))

    def separate(
        self,
        audio_path: Path,
        model: str,
        config: RunConfig,
        stems: list[str] | None = None,
    ) -> SeparationResult:
        import torch
        from demucs.apply import apply_model
        from demucs.audio import AudioFile
        from demucs.pretrained import get_model

        net = get_model(name=model)
        net.to(config.device)
        net.eval()
        sr = int(net.samplerate)

        # Load at the model's sample rate / channel count (ffmpeg-backed).
        # AudioFile.read returns (streams, channels, samples); take the stream.
        wav = AudioFile(Path(audio_path)).read(
            samplerate=sr, channels=net.audio_channels
        )
        if wav.dim() == 3:
            wav = wav[0]  # -> (channels, samples)

        ref = wav.mean(0)
        wav = (wav - ref.mean()) / (ref.std() + 1e-8)

        with torch.no_grad():
            out = apply_model(
                net,
                wav[None],
                device=config.device,
                segment=config.segment,
                overlap=config.overlap,
                progress=False,
            )[0]
        out = out * ref.std() + ref.mean()

        wanted = set(stems) if stems else None
        result: dict[str, np.ndarray] = {}
        for name, source in zip(net.sources, out):
            if wanted is not None and name not in wanted:
                continue
            result[name] = source.detach().cpu().numpy().astype(np.float32, copy=False)

        return SeparationResult(stems=result, sample_rate=sr)
