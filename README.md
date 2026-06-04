# stems

**Professional CLI audio stem separator.** Splits MP3/WAV (and most common audio
formats) into stems at studio quality, using a multi-backend engine:

- **Demucs v4** (Meta) — best all-round 4-stem and 6-stem separation.
- **`audio-separator`** — the Ultimate Vocal Remover (UVR) model zoo: **BS-Roformer**,
  **Mel-Band Roformer**, **MDX23C**, MDX-Net, VR-Arch — best-in-class
  vocals/instrumental isolation.

Output is lossless **24-bit WAV** plus a **320 kbps MP3** copy of every stem.
Single files and whole folders (recursive) are supported, with skip-existing for
resumable batches.

**Input formats:** WAV, MP3, FLAC, OGG/Opus, M4A/AAC, WMA, AIFF, and WebM/MP4/MKV
(audio track) — anything libsndfile or ffmpeg can decode. Output sample rate
follows the source (e.g. a 48 kHz WebM yields 48 kHz stems).

---

## Capabilities

| Preset | Output stems | Strategy |
|--------|--------------|----------|
| `vocals` | vocals, instrumental | BS-Roformer ep368 (single fast pass) |
| `vocals-max` | vocals, instrumental | **Vocals:** max-spectrogram ensemble of Kim Jensen + Kim FT + Fullness (fuller, less "gated"). **Instrumental:** averaged ensemble of BS-Roformer + Inst-V2. 5 passes — the cleanest, most natural 2-stem. |
| `4stem` | vocals, drums, bass, other | Demucs `htdemucs_ft` (single pass) |
| `4stem-max` *(default)* | vocals, drums, bass, other | Cascade: ensemble instrumental → Demucs `htdemucs_ft` on the clean instrumental (+ ensemble vocals) |
| `6stem` | vocals, drums, bass, guitar, piano, other | Demucs `htdemucs_6s` |

- **Ensemble** — runs several models and merges per-stem (waveform average or
  max-spectrogram) for the cleanest possible result.
- **Cascade** (`4stem-max`) — builds the cleanest instrumental via the dedicated
  instrumental ensemble, then runs Demucs on *that* (not the raw mix) so vocals
  never bleed into drums/bass/other. Vocals come from the vocal ensemble and are
  only computed when requested — `--stems drums,bass,other` skips the vocal
  passes for speed.
- **Raw model override** — `--model htdemucs_6s` or `--model bs_roformer` bypasses
  presets for full control.

> **On guitar/piano:** only `6stem` (`htdemucs_6s`) outputs them, and they are
> weak — it's the single open model that attempts a 6-way split and it sacrifices
> quality on *every* stem to do so. For usable drums/bass/other, prefer
> `4stem-max`. Clean guitar/piano isolation is still an unsolved problem in
> open-source source separation.

---

## Requirements

- **Python 3.10–3.12** (3.10 recommended for the widest ML wheel compatibility).
- **ffmpeg** on `PATH` (used for decoding arbitrary formats and MP3 export).
- **GPU (optional but recommended):** an NVIDIA CUDA GPU dramatically speeds up
  separation. An 8 GB card (e.g. RTX 3060 Ti) runs every preset here; chunked
  inference keeps memory in check. CPU works but is much slower.

---

## Installation

```powershell
# From the project root (D:\dev\stems)
py -3.10 -m venv .venv
.\.venv\Scripts\Activate.ps1

# 1) Install the CUDA build of PyTorch FIRST (pick the CUDA version for your driver):
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu124

# 2) Install stems and the rest of the dependencies:
pip install -e .

# 3) GPU runtime for the UVR/Roformer backend (audio-separator imports onnxruntime):
pip install onnxruntime-gpu
```

CPU-only machine? Skip step 1, run `pip install -e .`, and install plain
`onnxruntime` instead of `onnxruntime-gpu` in step 3 (PyTorch CPU build is pulled
in automatically). Model weights download on first use into `./models/` for UVR
models and `~/.cache/torch/hub` for Demucs (override the former with the
`STEMS_MODEL_DIR` environment variable).

---

## Usage

```powershell
# Minimal: vocals + instrumental, default output dir → ./output/<track>/
stems separate song.mp3 -p vocals-max

# Absolute minimal: default preset (4stem-max cascade) → ./output/<track>/
stems separate song.mp3

# 6 stems (vocals, drums, bass, guitar, piano, other)
stems separate song.wav out\ --preset 6stem

# Cleanest vocals/instrumental via model ensemble (explicit output dir)
stems separate song.mp3 out\ -p vocals-max

# Only keep certain stems, WAV only
stems separate song.wav out\ -p 6stem --stems vocals,drums --format wav

# Batch an entire library recursively, resume-friendly
stems separate .\album\ out\ -r --skip-existing -p 4stem-max

# Use a specific raw model
stems separate song.mp3 out\ --model htdemucs_6s

# Force CPU / tune VRAM
stems separate song.mp3 out\ --device cpu
stems separate song.mp3 out\ --segment 5 --overlap 0.1   # less VRAM
```

Discover what's available:

```powershell
stems presets     # list presets and their output stems
stems models      # list known Demucs + UVR models
stems --help
stems separate --help
```

### CLI options (`stems separate`)

| Option | Default | Description |
|--------|---------|-------------|
| `INPUT` | — | File or folder to process. |
| `OUTPUT_DIR` | `output` | Where stems are written. |
| `-p, --preset` | `4stem-max` | Named plan (`vocals`, `vocals-max`, `4stem`, `4stem-max`, `6stem`). |
| `-m, --model` | — | Raw model override; bypasses preset. |
| `-e, --engine` | auto | Engine for `--model`: `demucs` or `uvr`. |
| `--stems` | all | Comma list of stems to keep (e.g. `vocals,drums`). |
| `-f, --format` | `both` | `wav`, `mp3`, or `both`. |
| `--bitdepth` | `24` | WAV bit depth: `16`, `24`, or `32`. |
| `--device` | `auto` | `auto`, `cuda`, or `cpu`. |
| `-r, --recursive` | off | Recurse into subfolders for folder input. |
| `--skip-existing` | off | Skip files whose output already exists. |
| `--segment` | `7.8` | Demucs chunk size (s); lower = less VRAM. |
| `--overlap` | `0.25` | Demucs chunk overlap (0–1); higher = better/slower. |

---

## Output layout

If you omit the output directory, it defaults to **`./output`** (relative to where
you run the command), with a subfolder named after the input file:

```
output/                       # default OUTPUT_DIR (override by passing one)
└── <track-name>/             # derived from the input filename
    ├── vocals.wav   vocals.mp3
    ├── drums.wav    drums.mp3
    ├── bass.wav     bass.mp3
    └── other.wav    other.mp3
```

So `stems separate song.mp3 -p vocals-max` writes to `./output/song/vocals.wav`
(+ `.mp3`) and `./output/song/instrumental.wav` (+ `.mp3`).

In batch mode the input folder structure is mirrored under `OUTPUT_DIR`. Stems are
**not** loudness-normalized, so summing all stems of a Demucs run reconstructs
(approximately) the original mix.

---

## Architecture

See [CLAUDE.md](CLAUDE.md) for the full module map and design notes. In short:

```
src/stems/
├── cli.py            # Typer CLI (separate / presets / models)
├── config.py         # device detection, cache paths, quality defaults
├── presets.py        # named plans → engine/model/strategy
├── audio_io.py       # load + 24-bit WAV / 320k MP3 export
├── engines/
│   ├── base.py       # BaseSeparator interface + SeparationResult
│   ├── demucs_engine.py
│   └── uvr_engine.py
├── ensemble.py       # average / max-spectrogram model merging
├── pipeline.py       # single / ensemble / cascade orchestration + export
└── jobs.py           # batch discovery, progress, skip-existing, summary
```

---

## Development

```powershell
pip install -e ".[dev]"
pytest                     # unit tests (no models/GPU needed)
```

The unit tests stub out the heavy backends, so they run fast on CPU without
downloading any model weights.

---

## Troubleshooting

- **`ffmpeg not found`** — install ffmpeg and ensure it's on `PATH`
  (`ffmpeg -version`). Required for non-WAV input and MP3 output.
- **CUDA out of memory** — lower `--segment` (e.g. `5`) and/or `--overlap`, or
  run `--device cpu`.
- **First run is slow** — model weights download once into `./models/`.
- **Wrong stems from `--model`** — pass `--engine demucs|uvr` to disambiguate.

## License

MIT.
