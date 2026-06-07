"""Typer-based command line interface.

Commands:
    stems separate INPUT [OUTPUT_DIR]   separate a file or folder into stems
    stems presets                       list presets and their stem outputs
    stems models                        list available models per backend

Run ``stems --help`` or ``stems separate --help`` for full option docs.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

# Windows consoles often default to cp1252, which can't encode the box-drawing
# and arrow glyphs rich emits. Force UTF-8 (replacing anything unmappable) so
# output never crashes regardless of the active code page.
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        try:
            _stream.reconfigure(encoding="utf-8", errors="replace")
        except (ValueError, OSError):
            pass

from stems import __version__
from stems.config import (
    DEFAULT_BITDEPTH, DEFAULT_MP3_BITRATE, DEFAULT_OVERLAP, DEFAULT_SEGMENT,
    RunConfig, resolve_device,
)
from stems.engines.demucs_engine import _MODEL_STEMS as DEMUCS_MODELS
from stems.engines.uvr_engine import MODEL_FILES as UVR_MODELS
from stems.jobs import run_batch
from stems.pipeline import GUITAR_SOURCES
from stems.presets import DEFAULT_PRESET, PRESETS, get_preset

app = typer.Typer(
    add_completion=False,
    help="Professional audio stem separator (Demucs + UVR/Roformer).",
    no_args_is_help=True,
)
console = Console()


def _version_callback(value: bool) -> None:
    if value:
        console.print(f"stems {__version__}")
        raise typer.Exit()


@app.callback()
def _main(
    version: bool = typer.Option(
        False, "--version", callback=_version_callback, is_eager=True,
        help="Show version and exit.",
    ),
) -> None:
    """stems - split audio into vocals/instrumental, 4-stem, or 6-stem."""


@app.command()
def separate(
    input: Path = typer.Argument(..., help="Input audio file or folder."),
    output_dir: Path = typer.Argument(
        Path("output"), help="Output directory (default: ./output)."
    ),
    preset: Optional[str] = typer.Option(
        None, "--preset", "-p",
        help=f"Preset plan ({', '.join(PRESETS)}). Default: {DEFAULT_PRESET}.",
    ),
    model: Optional[str] = typer.Option(
        None, "--model", "-m",
        help="Raw model override (e.g. htdemucs_6s, bs_roformer). Bypasses preset.",
    ),
    engine: Optional[str] = typer.Option(
        None, "--engine", "-e", help="Engine for --model: demucs | uvr (auto-detected)."
    ),
    stems: Optional[str] = typer.Option(
        None, "--stems", help="Comma-separated stems to keep (e.g. vocals,drums)."
    ),
    mix: Optional[str] = typer.Option(
        None, "--mix",
        help=(
            "Combine these stems into ONE file (e.g. vocals,drums → "
            "vocals+drums.wav). Only the mix is written. Conflicts with --stems."
        ),
    ),
    vocal_method: Optional[str] = typer.Option(
        None, "--vocal-method",
        help=(
            "Vocal-ensemble merge for -max presets: max_spec (fuller, default) | "
            "average (smoother). Ignored by single-model presets."
        ),
    ),
    guitar_source: Optional[str] = typer.Option(
        None, "--guitar-source",
        help=(
            "For 6stem-max: audio fed to the guitar model - instrumental "
            "(faint/acoustic) | no-drums (prominent/electric) | mix. Required for "
            "that preset."
        ),
    ),
    fmt: str = typer.Option(
        "both", "--format", "-f", help="Output format: wav | mp3 | both."
    ),
    bitdepth: int = typer.Option(
        DEFAULT_BITDEPTH, "--bitdepth", help="WAV bit depth: 16 | 24 | 32."
    ),
    device: str = typer.Option(
        "auto", "--device", help="Compute device: auto | cuda | cpu."
    ),
    recursive: bool = typer.Option(
        False, "--recursive", "-r", help="Recurse into subfolders (folder input)."
    ),
    skip_existing: bool = typer.Option(
        False, "--skip-existing", help="Skip files whose output already exists."
    ),
    segment: Optional[float] = typer.Option(
        DEFAULT_SEGMENT, "--segment", help="Demucs chunk size (s); lower = less VRAM."
    ),
    overlap: float = typer.Option(
        DEFAULT_OVERLAP, "--overlap", help="Demucs chunk overlap (0-1); higher = better."
    ),
) -> None:
    """Separate INPUT into stems written under OUTPUT_DIR."""
    if fmt not in ("wav", "mp3", "both"):
        raise typer.BadParameter("--format must be wav, mp3, or both")
    if vocal_method is not None and vocal_method not in ("max_spec", "average"):
        raise typer.BadParameter("--vocal-method must be max_spec or average")
    if model is None and preset is None:
        preset = DEFAULT_PRESET

    if guitar_source is not None and guitar_source not in GUITAR_SOURCES:
        raise typer.BadParameter(
            f"--guitar-source must be one of {', '.join(GUITAR_SOURCES)}"
        )
    # A guitar-bearing preset needs an explicit source (no sensible default).
    if model is None and preset and get_preset(preset).guitar_model \
            and guitar_source is None:
        raise typer.BadParameter(
            f"preset '{preset}' requires --guitar-source "
            f"({' | '.join(GUITAR_SOURCES)})."
        )

    if mix is not None and stems is not None:
        raise typer.BadParameter("--mix and --stems cannot be used together.")

    resolved_device = resolve_device(device)
    stem_list = [s.strip() for s in stems.split(",")] if stems else None
    mix_list = [s.strip() for s in mix.split(",")] if mix else None
    # Validate mix stems against the preset's outputs (skip for raw --model).
    if mix_list and model is None and preset:
        valid = set(get_preset(preset).output_stems)
        unknown = [s for s in mix_list if s not in valid]
        if unknown:
            raise typer.BadParameter(
                f"--mix stems {unknown} are not produced by preset '{preset}' "
                f"(available: {', '.join(get_preset(preset).output_stems)})."
            )

    config = RunConfig(
        device=resolved_device,
        segment=segment,
        overlap=overlap,
        bitdepth=bitdepth,
        mp3_bitrate=DEFAULT_MP3_BITRATE,
    )

    plan_desc = model or preset
    console.print(
        f"[bold]stems[/bold] → plan=[cyan]{plan_desc}[/cyan] "
        f"device=[cyan]{resolved_device}[/cyan] format=[cyan]{fmt}[/cyan]"
    )

    run_batch(
        input_path=input,
        output_root=output_dir,
        config=config,
        preset=preset,
        engine=engine,
        model=model,
        stems=stem_list,
        fmt=fmt,
        recursive=recursive,
        skip_existing=skip_existing,
        guitar_source=guitar_source,
        vocal_method=vocal_method,
        combine=mix_list,
    )


@app.command()
def presets() -> None:
    """List available presets and the stems they produce."""
    table = Table(title="Presets")
    table.add_column("Name", style="cyan", no_wrap=True)
    table.add_column("Kind")
    table.add_column("Output stems")
    table.add_column("Description")
    for p in PRESETS.values():
        marker = " (default)" if p.name == DEFAULT_PRESET else ""
        table.add_row(
            p.name + marker, p.kind, ", ".join(p.output_stems), p.description
        )
    console.print(table)


@app.command()
def models() -> None:
    """List models known per backend."""
    demucs_table = Table(title="Demucs models")
    demucs_table.add_column("Model", style="cyan")
    demucs_table.add_column("Stems")
    for name, stem_set in DEMUCS_MODELS.items():
        demucs_table.add_row(name, ", ".join(stem_set))
    console.print(demucs_table)

    uvr_table = Table(title="UVR / Roformer models (2-stem)")
    uvr_table.add_column("Alias", style="cyan")
    uvr_table.add_column("Checkpoint")
    for alias, ckpt in UVR_MODELS.items():
        uvr_table.add_row(alias, ckpt)
    console.print(uvr_table)
    console.print(
        "[dim]Tip: run `audio-separator --list_models` for the full UVR zoo.[/dim]"
    )


if __name__ == "__main__":
    app()
