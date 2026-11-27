# Sample Vault — Local Audio Sample Search & Auto-Organizer

Offline-first local Audio ML tool for music producers (designed around Bitwig Studio + macOS Apple Silicon workflows). Index your entire sample library once, then search by plain English, drop in a reference sound to find sonic twins, filter by BPM/key/type, and auto-organize into clean symlinked folder trees.

## Stack
- **Embeddings & Zero-Shot Tagging**: LAION CLAP (`laion/clap-htsat-unfused`) running locally on Apple Silicon **MPS** (or CPU fallback), cached offline after first download.
- **Audio MIR**: `librosa` + `soundfile` + macOS native `afconvert` fallback for BPM detection, harmonic CQT musical key + Camelot Wheel detection, one-shot vs. loop classification, and precomputed 120-bin waveforms.
- **Vector Index**: Local SQLite + `sqlite-vec` (`vec0` cosine distance) in `data/sample_vault.db`.
- **UI & API**: FastAPI + zero-build modern web UI with instant auditioning, interactive waveforms, drag-and-drop reference search, Finder reveal (for dragging directly into Bitwig Studio), and non-destructive Auto-Organizer with undo.

## Quick Start

```bash
git clone https://github.com/10brink/sample-vault.git
cd sample-vault

# Create a Python 3.12+ virtual environment and install dependencies
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt

# Launch the local web UI (opens http://127.0.0.1:8742)
./run.sh --open
```

Then open **Folders & Index** in the UI to add your Bitwig Studio library, OneKey, Splice, or custom sample folders (or generate & index the built-in synthetic demo pack with `.venv/bin/python cli.py demo`).

## CLI Usage

```bash
# Index one or more folders
.venv/bin/python cli.py index ~/Documents/Bitwig\ Studio/Library ~/Samples

# Search in plain English
.venv/bin/python cli.py search "punchy acoustic kick"
.venv/bin/python cli.py search "warm dusty Rhodes chord" --type one-shot

# Find closest matches to a reference audio file
.venv/bin/python cli.py similar /path/to/reference_kick.wav

# Start the web UI server (default http://127.0.0.1:8742)
./run.sh --open
```

## Keyboard Shortcuts (Web UI)
- `/` — Focus semantic search bar
- `↑` / `↓` — Step through samples and auto-audition
- `Space` — Play / Pause
- `L` — Toggle loop auditioning
- `F` — Favorite active sample
- **Drag & drop** any audio file anywhere on the window to run audio-to-audio similarity search
