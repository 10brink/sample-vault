"""Incremental background audio library indexer with live progress streaming."""

from __future__ import annotations

import logging
import os
import queue
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterator, Optional

from .analyzer import analyze_audio_file
from .config import SUPPORTED_EXTENSIONS
from .embedder import ClapEmbedder, embedder as default_embedder
from .store import SampleStore, store as default_store

logger = logging.getLogger(__name__)


@dataclass
class IndexProgress:
    running: bool = False
    phase: str = "idle"  # idle, loading_model, scanning, indexing, done, error, cancelled
    total_files: int = 0
    processed_files: int = 0
    indexed_files: int = 0
    skipped_files: int = 0
    removed_files: int = 0
    failed_files: int = 0
    current_file: str = ""
    message: str = "Idle"
    started_at: Optional[float] = None
    finished_at: Optional[float] = None


class LibraryIndexer:
    """Runs incremental indexing across watched folders in a background worker thread."""

    def __init__(
        self,
        sample_store: SampleStore = default_store,
        clap_embedder: ClapEmbedder = default_embedder,
    ) -> None:
        self.store = sample_store
        self.embedder = clap_embedder
        self._lock = threading.Lock()
        self._cancel_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._progress = IndexProgress()
        self._subscribers: list[queue.Queue[dict[str, Any]]] = []

    def get_progress(self) -> dict[str, Any]:
        with self._lock:
            return asdict(self._progress)

    def subscribe(self) -> Iterator[dict[str, Any]]:
        q: queue.Queue[dict[str, Any]] = queue.Queue(maxsize=128)
        with self._lock:
            self._subscribers.append(q)
            initial = asdict(self._progress)
        yield initial
        try:
            while True:
                try:
                    event = q.get(timeout=15.0)
                    yield event
                except queue.Empty:
                    yield {"heartbeat": True, **self.get_progress()}
        finally:
            with self._lock:
                if q in self._subscribers:
                    self._subscribers.remove(q)

    def _emit(self) -> None:
        payload = asdict(self._progress)
        dead: list[queue.Queue[dict[str, Any]]] = []
        for q in self._subscribers:
            try:
                q.put_nowait(payload)
            except queue.Full:
                try:
                    q.get_nowait()
                    q.put_nowait(payload)
                except Exception:
                    dead.append(q)
        for q in dead:
            self._subscribers.remove(q)

    def _update(self, **kwargs: Any) -> None:
        with self._lock:
            for k, v in kwargs.items():
                setattr(self._progress, k, v)
            self._emit()

    def start(self, folder_paths: Optional[list[str]] = None, force: bool = False) -> bool:
        with self._lock:
            if self._progress.running:
                return False
            self._cancel_event.clear()
            self._progress = IndexProgress(
                running=True,
                phase="scanning",
                started_at=time.time(),
                message="Starting library scan...",
            )
            self._emit()

        self._thread = threading.Thread(
            target=self._run,
            args=(folder_paths, force),
            daemon=True,
            name="sample-vault-indexer",
        )
        self._thread.start()
        return True

    def cancel(self) -> None:
        self._cancel_event.set()

    @staticmethod
    def _scan_folder(root: Path) -> list[Path]:
        audio_files: list[Path] = []
        for dirpath, dirnames, filenames in os.walk(root):
            # Skip hidden directories and macOS bundle metadata
            dirnames[:] = [
                d for d in dirnames if not d.startswith(".") and d not in ("__MACOSX", "node_modules")
            ]
            for fn in filenames:
                if fn.startswith(".") or fn.startswith("._"):
                    continue
                ext = Path(fn).suffix.lower()
                if ext in SUPPORTED_EXTENSIONS:
                    audio_files.append(Path(dirpath) / fn)
        audio_files.sort()
        return audio_files

    def _run(self, folder_paths: Optional[list[str]], force: bool) -> None:
        try:
            if folder_paths:
                roots = [str(Path(p).expanduser().resolve()) for p in folder_paths]
            else:
                roots = [f["path"] for f in self.store.list_folders()]

            if not roots:
                self._update(
                    running=False,
                    phase="done",
                    finished_at=time.time(),
                    message="No folders configured to index.",
                )
                return

            # 1. Scan all folders and figure out incremental work list
            self._update(phase="scanning", message="Scanning folders for audio files...")
            work_items: list[tuple[str, Path, float, int]] = []
            skipped = 0
            removed = 0

            for root_str in roots:
                root_path = Path(root_str)
                if not root_path.exists():
                    continue
                existing_map = self.store.get_indexed_file_map(root_str)
                found_files = self._scan_folder(root_path)
                found_set: set[str] = set()

                for fpath in found_files:
                    fstr = str(fpath)
                    found_set.add(fstr)
                    try:
                        st = fpath.stat()
                    except OSError:
                        continue
                    prev = existing_map.get(fstr)
                    if not force and prev is not None:
                        _, prev_mtime, prev_size = prev
                        if abs(prev_mtime - st.st_mtime) < 1e-3 and prev_size == st.st_size:
                            skipped += 1
                            continue
                    work_items.append((root_str, fpath, st.st_mtime, st.st_size))

                # Prune deleted files from index
                stale_ids = [sid for p, (sid, _, _) in existing_map.items() if p not in found_set]
                if stale_ids:
                    self.store.delete_samples(stale_ids)
                    removed += len(stale_ids)

            total = len(work_items)
            self._update(
                total_files=total,
                skipped_files=skipped,
                removed_files=removed,
                message=(
                    f"Found {total} new/changed samples ({skipped} up to date)"
                    if total > 0
                    else f"All {skipped} samples are already up to date."
                ),
            )

            if total == 0:
                for root_str in roots:
                    self.store.mark_folder_indexed(root_str)
                self._update(
                    running=False,
                    phase="done",
                    finished_at=time.time(),
                    message=f"Library up to date ({skipped} indexed, {removed} pruned).",
                )
                return

            # 2. Ensure CLAP model is loaded
            if not self.embedder.is_loaded:
                self._update(
                    phase="loading_model",
                    message=f"Loading CLAP model ({self.embedder.model_id}) on {self.embedder.device.upper()}...",
                )
                self.embedder.ensure_loaded()

            # 3. Process files
            self._update(phase="indexing", message=f"Indexing {total} samples...")
            indexed = 0
            failed = 0

            for idx, (root_str, fpath, mtime, size_bytes) in enumerate(work_items, start=1):
                if self._cancel_event.is_set():
                    self._update(
                        running=False,
                        phase="cancelled",
                        finished_at=time.time(),
                        message=f"Indexing cancelled after {indexed} samples.",
                    )
                    return

                self._update(
                    processed_files=idx - 1,
                    current_file=fpath.name,
                    message=f"Analyzing [{idx}/{total}] {fpath.name}",
                )

                try:
                    analysis = analyze_audio_file(fpath)
                    emb = self.embedder.embed_audio(analysis.audio_48k)
                    auto_category, auto_tags = self.embedder.classify_tags(
                        emb,
                        sample_type=analysis.sample_type,
                        duration_sec=analysis.duration_sec,
                    )

                    try:
                        rel_dir = str(fpath.parent.relative_to(Path(root_str)))
                        if rel_dir == ".":
                            rel_dir = ""
                    except ValueError:
                        rel_dir = fpath.parent.name

                    record = {
                        "path": str(fpath),
                        "filename": fpath.name,
                        "folder_root": root_str,
                        "rel_dir": rel_dir,
                        "mtime": mtime,
                        "size_bytes": size_bytes,
                        "duration_sec": analysis.duration_sec,
                        "sample_rate": analysis.sample_rate,
                        "channels": analysis.channels,
                        "bpm": analysis.bpm,
                        "key_name": analysis.key_name,
                        "camelot": analysis.camelot,
                        "sample_type": analysis.sample_type,
                        "type_confidence": analysis.type_confidence,
                        "auto_category": auto_category,
                        "auto_tags": auto_tags,
                        "energy": analysis.energy,
                        "brightness": analysis.brightness,
                        "waveform_peaks": analysis.waveform_peaks,
                    }
                    self.store.upsert_sample(record, emb)
                    indexed += 1
                except Exception as exc:
                    failed += 1
                    logger.warning("Failed to index %s: %s", fpath, exc)

                self._update(
                    processed_files=idx,
                    indexed_files=indexed,
                    failed_files=failed,
                )

            for root_str in roots:
                self.store.mark_folder_indexed(root_str)

            self._update(
                running=False,
                phase="done",
                current_file="",
                finished_at=time.time(),
                message=f"Done! Indexed {indexed} samples ({skipped} unchanged, {failed} errors).",
            )
        except Exception as exc:
            logger.exception("Indexer error")
            self._update(
                running=False,
                phase="error",
                finished_at=time.time(),
                message=f"Indexing failed: {exc}",
            )

    def index_sync(self, folder_paths: Optional[list[str]] = None, force: bool = False) -> dict[str, Any]:
        """Run indexing synchronously (used by CLI and tests)."""
        self._cancel_event.clear()
        self._progress = IndexProgress(running=True, phase="scanning", started_at=time.time())
        self._run(folder_paths, force)
        return self.get_progress()


indexer = LibraryIndexer()
