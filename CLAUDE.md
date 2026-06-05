# CLAUDE.md

Guidance for Claude Code (and humans) working in this repository.

## What this project is

`stems` is a **CLI tool that separates audio into stems** (vocals/instrumental,
4-stem, or 6-stem) at studio quality. It is a thin CLI over a reusable core engine
that fronts two separation backends:

- **Demucs v4** (`demucs.api.Separator`) — 4/6-stem.
- **`audio-separator`** (UVR/Roformer model zoo) — best vocals/instrumental.

Output is lossless 24-bit WAV + 320 kbps MP3. Single files and recursive folder
batches are supported.

## Environment

- Target **Python 3.10–3.12** (3.10 recommended). On this machine `py -3.10` /
  `python3` is 3.10.11; the bare `python` is 3.13 — don't build the venv with it.
- **ffmpeg** must be on `PATH` (decoding + MP3 encoding).
- GPU is auto-detected via `torch.cuda.is_available()`. Dev box: RTX 3060 Ti, 8 GB
  — inference is chunked (`config.DEFAULT_SEGMENT`) to fit.
- Heavy deps (`torch`, `demucs`, `audio-separator`) are **imported lazily** inside
  engine `.separate()` methods so the CLI, presets/models listings, and unit tests
  import and run without them.

## Module map

```
src/stems/
├── __init__.py        # __version__
├── __main__.py        # `python -m stems`
├── cli.py             # Typer app: separate / presets / models commands
├── config.py          # resolve_device(), RunConfig, cache + quality defaults
├── audio_io.py        # load_audio(), write_wav/write_mp3, export_stem()
├── presets.py         # Preset dataclass + PRESETS registry + DEFAULT_PRESET
├── engines/
│   ├── base.py        # BaseSeparator ABC, SeparationResult
│   ├── demucs_engine.py  # DemucsSeparator (+ _MODEL_STEMS table)
│   └── uvr_engine.py     # UvrSeparator (+ MODEL_FILES alias table)
├── ensemble.py        # ensemble_results(): average / max_spec merge
├── pipeline.py        # separate_to_result()/separate_file(): the orchestrator
├── jobs.py            # discover_inputs(), run_batch(): batch + progress + summary
└── gui/               # optional CustomTkinter desktop front-end (gui extra)
    ├── __init__.py    # main(): lazy-imports customtkinter, launches the window
    ├── app.py         # StemsApp(ctk.CTk): widgets + queue-drained event pump
    └── worker.py      # run_job(): background batch runner emitting queue events
tests/                 # pytest; backends are stubbed, no models/GPU needed
```

## Core conventions

- **Audio arrays:** float32, shape `(channels, samples)` (channels-first), with the
  sample rate carried alongside. `audio_io` handles the `(samples, channels)` ↔
  channels-first conversion at the libsndfile boundary.
- **No loudness normalization** anywhere — stems must remain summable back to the mix.
- A separation returns a `SeparationResult` (`{stem_name: ndarray}` + `sample_rate`).
- Stem names are canonical lowercase: `vocals, instrumental, drums, bass, other,
  guitar, piano`.

## Separation plans (presets)

`presets.py` maps a friendly name to a `Preset` with one of three `kind`s, executed
by `pipeline.py`:

- **`single`** — one engine + one model.
- **`ensemble`** — several models (same engine) merged by `ensemble.ensemble_results`.
- **`cascade`** — isolate vocals with `vocal_engine`/`vocal_model` (UVR Roformer),
  write the **instrumental residual** to a temp WAV, then run Demucs on that residual
  for drums/bass/other. Keeps vocals out of the instrumental stems. This is
  `DEFAULT_PRESET = "4stem-max"`.

`--model` on the CLI bypasses presets (`pipeline.separate_to_result` takes
`model`/`engine` directly; engine is inferred by `_infer_engine_for_model` if omitted).

## Adding things

- **New raw model:** add to `_MODEL_STEMS` (Demucs) or `MODEL_FILES` (UVR) so it
  shows in `stems models` and gets correct stem names / checkpoint resolution.
- **New preset:** add a `Preset` to `PRESETS`. Reuse an existing `kind`; only add a
  new orchestration branch in `pipeline.py` if a genuinely new strategy is needed.
- **New backend:** implement `BaseSeparator` in `engines/`, register it in
  `pipeline._ENGINES`, keep heavy imports inside `.separate()`.
- **"Fine elements" (deferred):** lead/backing vocals, kick/snare split, de-reverb,
  de-noise are intended to be added as new presets/cascade steps — no architectural
  change required.

## Testing & running

```powershell
pytest                              # fast; stubs backends, no downloads
stems separate song.mp3 out\ -p 6stem
python -m stems presets
```

When adding code, prefer extending `pipeline.py`/`presets.py` over touching the CLI.
Keep new heavy dependencies lazy-imported. Match the existing channels-first float32
audio convention and avoid introducing normalization.

## Gotchas

- Don't add MP3 support expectations to libsndfile — MP3 always goes through ffmpeg.
- `uvr_engine` classifies stems by **filename keywords** from audio-separator's
  output (`_classify_stem`); if a new model names files differently, extend
  `_STEM_KEYWORDS`.
- `--device cuda` silently falls back to CPU when CUDA is unavailable (by design).
- First run downloads weights to `./models/` (or `STEMS_MODEL_DIR`); expect a delay.
