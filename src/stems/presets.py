"""Named presets mapping friendly names to a concrete separation plan.

A :class:`Preset` describes *how* to produce a stem set:

- ``single``   — one engine + one model.
- ``ensemble`` — run several models (same engine), merge per-stem (see
  :mod:`stems.ensemble`). Used for the highest-fidelity 2-stem result.
- ``cascade``  — extract vocals with the best vocal model, subtract to get the
  backing track, then run Demucs on the residual for drums/bass/other. Yields
  the best practical 4-stem split.

The CLI ``--model`` flag bypasses presets entirely (see :mod:`stems.pipeline`).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

PlanKind = Literal["single", "ensemble", "cascade", "twostem"]


@dataclass(frozen=True, slots=True)
class Preset:
    name: str
    description: str
    kind: PlanKind
    engine: str  # "demucs" or "uvr"; for cascade this is the residual engine
    models: list[str] = field(default_factory=list)
    output_stems: list[str] = field(default_factory=list)
    ensemble_method: str = "average"  # used for the instrumental ensemble
    vocal_method: str = "average"     # used for the vocal ensemble (may differ)
    # cascade-only: which model isolates vocals before the residual pass.
    vocal_engine: str = "uvr"
    vocal_model: str = "bs_roformer"
    # twostem-only: separate ensembles for each stem. Vocals are taken from the
    # vocal models, instrumental from the instrumental models, then each is
    # merged independently — the pro approach to a clean 2-stem split.
    vocal_models: list[str] = field(default_factory=list)
    instrumental_models: list[str] = field(default_factory=list)


PRESETS: dict[str, Preset] = {
    "vocals": Preset(
        name="vocals",
        description="Vocals + instrumental (BS-Roformer ep368, single fast pass).",
        kind="single",
        engine="uvr",
        models=["bs_roformer"],
        output_stems=["vocals", "instrumental"],
    ),
    "vocals-max": Preset(
        name="vocals-max",
        description=(
            "Best 2-stem: max-spec ensemble of Kim + Kim FT + Fullness for fuller, "
            "less-gated vocals; averaged ensemble of BS-Roformer + Inst-V2 + "
            "Bleedless for a clean, low-residue instrumental. 6 passes."
        ),
        kind="twostem",
        engine="uvr",
        output_stems=["vocals", "instrumental"],
        ensemble_method="average",          # instrumental: average (kept clean)
        vocal_method="max_spec",            # vocals: max-spec (fuller, less gated)
        vocal_models=["kim_vocals", "kim_ft", "vocal_fullness"],
        instrumental_models=["bs_roformer", "inst_v2", "inst_bleedless"],
    ),
    "4stem": Preset(
        name="4stem",
        description="Vocals, drums, bass, other (Demucs htdemucs_ft).",
        kind="single",
        engine="demucs",
        models=["htdemucs_ft"],
        output_stems=["vocals", "drums", "bass", "other"],
    ),
    "4stem-max": Preset(
        name="4stem-max",
        description=(
            "Best 4-stem: ensemble instrumental split, then Demucs htdemucs_ft on "
            "the clean instrumental for tight drums/bass/other (+ ensemble vocals)."
        ),
        kind="cascade",
        engine="demucs",
        models=["htdemucs_ft"],
        output_stems=["vocals", "drums", "bass", "other"],
        ensemble_method="average",
        vocal_method="max_spec",
        vocal_models=["kim_vocals", "kim_ft", "vocal_fullness"],
        instrumental_models=["bs_roformer", "inst_v2", "inst_bleedless"],
    ),
    "6stem": Preset(
        name="6stem",
        description="Vocals, drums, bass, guitar, piano, other (Demucs htdemucs_6s).",
        kind="single",
        engine="demucs",
        models=["htdemucs_6s"],
        output_stems=["vocals", "drums", "bass", "guitar", "piano", "other"],
    ),
}

# Default preset when none is supplied: best 2-stem (vocals + instrumental).
DEFAULT_PRESET = "vocals-max"


def get_preset(name: str) -> Preset:
    try:
        return PRESETS[name]
    except KeyError:
        valid = ", ".join(PRESETS)
        raise KeyError(f"Unknown preset '{name}'. Available: {valid}")
