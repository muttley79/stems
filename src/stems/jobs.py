"""Job discovery and batch execution.

Resolves a single file or a (optionally recursive) folder into a list of input
files, and runs the pipeline over them with progress reporting, skip-existing
support, and a final summary. Output mirrors the input folder structure:

    OUTPUT_DIR/<relative-subdir>/<track-name>/<stem>.{wav,mp3}
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from rich.console import Console
from rich.progress import (
    BarColumn, Progress, SpinnerColumn, TextColumn, TimeElapsedColumn,
)

from stems.audio_io import SUPPORTED_INPUT_SUFFIXES
from stems.config import RunConfig
from stems.pipeline import (
    Step, iter_required_models, prefetch_models, separate_file,
)

console = Console()


@dataclass(slots=True)
class JobResult:
    input_path: Path
    output_dir: Path
    written: list[Path] = field(default_factory=list)
    skipped: bool = False
    error: str | None = None


def discover_inputs(path: Path, recursive: bool = False) -> list[Path]:
    """Return the list of audio files to process for a file or folder input."""
    path = Path(path)
    if path.is_file():
        return [path]
    if not path.is_dir():
        raise FileNotFoundError(f"Input path does not exist: {path}")

    globber = path.rglob if recursive else path.glob
    files = sorted(
        p for p in globber("*")
        if p.is_file() and p.suffix.lower() in SUPPORTED_INPUT_SUFFIXES
    )
    return files


def _output_dir_for(
    input_file: Path, input_root: Path, output_root: Path
) -> Path:
    """Compute the per-track output directory, preserving subfolders in batch."""
    if input_root.is_file():
        return output_root / input_file.stem
    rel = input_file.relative_to(input_root).parent
    return output_root / rel / input_file.stem


def _outputs_exist(out_dir: Path) -> bool:
    if not out_dir.is_dir():
        return False
    return any(out_dir.glob("*.wav")) or any(out_dir.glob("*.mp3"))


def run_batch(
    input_path: Path,
    output_root: Path,
    config: RunConfig,
    preset: str | None = None,
    engine: str | None = None,
    model: str | None = None,
    stems: list[str] | None = None,
    fmt: str = "both",
    recursive: bool = False,
    skip_existing: bool = False,
) -> list[JobResult]:
    """Process all discovered inputs; returns one :class:`JobResult` per file."""
    input_path = Path(input_path)
    output_root = Path(output_root)
    inputs = discover_inputs(input_path, recursive=recursive)
    if not inputs:
        console.print("[yellow]No supported audio files found.[/yellow]")
        return []

    config.ensure_dirs()

    # Fetch any missing weights up front, outside the live display below, so the
    # backends' own download bars show cleanly instead of hanging silently (UVR)
    # or colliding with the rich progress display (Demucs).
    prefetch_models(
        iter_required_models(preset=preset, engine=engine, model=model),
        config, console,
    )

    results: list[JobResult] = []

    multi = len(inputs) > 1
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("{task.completed}/{task.total}"),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        # Outer bar tracks files (only worth showing for a batch); the inner bar
        # tracks the model passes within the current file's plan.
        file_task = progress.add_task(
            "[bold]Files[/bold]", total=len(inputs), visible=multi
        )
        step_task = progress.add_task("[dim]preparing…[/dim]", total=1)
        for f in inputs:
            out_dir = _output_dir_for(f, input_path, output_root)
            progress.reset(
                step_task, total=1, description=f"[cyan]{f.name}[/cyan]"
            )

            if skip_existing and _outputs_exist(out_dir):
                results.append(JobResult(f, out_dir, skipped=True))
                progress.update(
                    step_task, completed=1,
                    description=f"[cyan]{f.name}[/cyan] · [dim]skipped[/dim]",
                )
                progress.advance(file_task)
                continue

            # Reflect each model pass on the inner bar as it starts.
            plan_total = {"n": 1}

            def on_step(s: Step, _name: str = f.name) -> None:
                plan_total["n"] = s.total
                progress.update(
                    step_task, total=s.total, completed=s.index,
                    description=(
                        f"[cyan]{_name}[/cyan] · [{s.index}/{s.total}] "
                        f"[magenta]{s.model}[/magenta] · {s.action}"
                    ),
                )

            try:
                written = separate_file(
                    f, out_dir, config,
                    preset=preset, engine=engine, model=model,
                    stems=stems, fmt=fmt, on_step=on_step,
                )
                results.append(JobResult(f, out_dir, written=written))
                progress.update(
                    step_task, completed=plan_total["n"],
                    description=f"[cyan]{f.name}[/cyan] · [green]done[/green]",
                )
            except Exception as exc:  # keep batch going on per-file failure
                results.append(JobResult(f, out_dir, error=str(exc)))
                progress.update(
                    step_task,
                    description=f"[cyan]{f.name}[/cyan] · [red]failed[/red]",
                )
                console.print(f"[red]Failed:[/red] {f.name} — {exc}")
            progress.advance(file_task)

    _print_summary(results)
    return results


def _print_summary(results: list[JobResult]) -> None:
    done = sum(1 for r in results if r.written and not r.error)
    skipped = sum(1 for r in results if r.skipped)
    failed = sum(1 for r in results if r.error)
    console.print(
        f"\n[bold]Summary:[/bold] {done} separated, "
        f"{skipped} skipped, {failed} failed."
    )
