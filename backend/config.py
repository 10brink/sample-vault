"""Configuration and constants for Sample Vault."""

from __future__ import annotations

import os
from pathlib import Path

APP_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = APP_DIR / "data"
DB_PATH = DATA_DIR / "sample_vault.db"
UNDO_DIR = DATA_DIR / "undo_manifests"
FRONTEND_DIR = APP_DIR / "frontend"
DEMO_SAMPLES_DIR = APP_DIR / "demo_samples"

DATA_DIR.mkdir(parents=True, exist_ok=True)
UNDO_DIR.mkdir(parents=True, exist_ok=True)

# LAION CLAP checkpoint (HTSAT-unfused, 512-dim embeddings, public/no API key, verified on MPS)
CLAP_MODEL_ID = os.environ.get(
    "SAMPLE_VAULT_CLAP_MODEL",
    "laion/clap-htsat-unfused",
)
EMBEDDING_DIM = 512
CLAP_SAMPLE_RATE = 48000
ANALYSIS_SAMPLE_RATE = 22050
MAX_ANALYSIS_SECONDS = 30.0

SUPPORTED_EXTENSIONS = {
    ".wav",
    ".aif",
    ".aiff",
    ".flac",
    ".ogg",
    ".mp3",
    ".m4a",
    ".opus",
    ".caf",
}

DEFAULT_HOST = os.environ.get("SAMPLE_VAULT_HOST", "127.0.0.1")
DEFAULT_PORT = int(os.environ.get("SAMPLE_VAULT_PORT", "8742"))


def detect_common_sample_dirs() -> list[dict[str, str]]:
    """Return existing common macOS / Bitwig Studio / sample directories on this machine."""
    home = Path.home()
    candidates = [
        ("Bitwig Studio Library", home / "Documents" / "Bitwig Studio" / "Library"),
        ("Bitwig Installed Packages", home / "Library" / "Application Support" / "Bitwig" / "Bitwig Studio" / "installed-packages"),
        ("Bitwig Projects", home / "Documents" / "Bitwig Studio" / "Projects"),
        ("Samples", home / "Samples"),
        ("Music / Samples", home / "Music" / "Samples"),
        ("Splice Sounds", home / "Splice" / "sounds"),
        ("Music", home / "Music"),
        ("Demo Pack (Built-in)", DEMO_SAMPLES_DIR),
    ]
    found: list[dict[str, str]] = []
    for label, path in candidates:
        try:
            if path.exists() and path.is_dir():
                found.append({"label": label, "path": str(path)})
        except PermissionError:
            continue
    return found
