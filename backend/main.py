"""FastAPI backend server for Sample Vault."""

from __future__ import annotations

import asyncio
import json
import logging
import mimetypes
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Optional

from fastapi import FastAPI, File, Form, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from .analyzer import analyze_audio_file
from .config import (
    CLAP_MODEL_ID,
    DEMO_SAMPLES_DIR,
    FRONTEND_DIR,
    SUPPORTED_EXTENSIONS,
    detect_common_sample_dirs,
)
from .embedder import embedder
from .indexer import indexer
from .organizer import apply_organization, preview_organization, undo_organization
from .store import store

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("sample-vault")

app = FastAPI(title="Sample Vault", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class FolderCreateRequest(BaseModel):
    path: str
    label: Optional[str] = None
    auto_index: bool = True


class IndexStartRequest(BaseModel):
    folder_paths: Optional[list[str]] = None
    force: bool = False


class SampleUpdateRequest(BaseModel):
    favorite: Optional[bool] = None
    custom_tags: Optional[list[str]] = None
    bpm: Optional[float] = None
    key_name: Optional[str] = None
    camelot: Optional[str] = None
    sample_type: Optional[str] = None
    auto_category: Optional[str] = None


class OrganizeRequest(BaseModel):
    target_dir: str
    template: str = "type_category_key"
    mode: str = "symlink"
    folder_root: Optional[str] = None


# ── Status & Library Info ────────────────────────────────────────────────────


@app.get("/api/status")
def get_status() -> dict[str, Any]:
    return {
        "model_id": CLAP_MODEL_ID,
        "model_loaded": embedder.is_loaded,
        "model_loading": embedder.is_loading,
        "model_error": embedder.load_error,
        "device": embedder.device,
        "stats": store.get_stats(),
        "folders": store.list_folders(),
        "indexing": indexer.get_progress(),
        "demo_available": DEMO_SAMPLES_DIR.exists(),
    }


# ── Folder Management ────────────────────────────────────────────────────────


@app.get("/api/folders")
def list_folders() -> list[dict[str, Any]]:
    return store.list_folders()


@app.get("/api/folders/common")
def get_common_folders() -> list[dict[str, str]]:
    return detect_common_sample_dirs()


@app.post("/api/folders/pick")
def pick_folder_native() -> dict[str, Optional[str]]:
    """Open native macOS Finder folder chooser dialog."""
    if sys.platform != "darwin":
        raise HTTPException(status_code=501, detail="Native folder picker is macOS-only.")
    try:
        proc = subprocess.run(
            [
                "osascript",
                "-e",
                'POSIX path of (choose folder with prompt "Select a Sample Library Folder")',
            ],
            capture_output=True,
            text=True,
            timeout=120,
        )
        if proc.returncode != 0:
            return {"path": None}
        chosen = proc.stdout.strip().rstrip("/")
        return {"path": chosen or None}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Folder picker error: {exc}") from exc


@app.post("/api/folders")
def add_folder(req: FolderCreateRequest) -> dict[str, Any]:
    target = Path(req.path).expanduser().resolve()
    if not target.exists() or not target.is_dir():
        raise HTTPException(status_code=400, detail=f"Directory not found: {target}")
    folder = store.add_folder(str(target), label=req.label)
    if req.auto_index:
        indexer.start([folder["path"]])
    return folder


@app.delete("/api/folders/{folder_id}")
def delete_folder(folder_id: int, remove_samples: bool = True) -> dict[str, Any]:
    ok = store.remove_folder(folder_id, remove_samples=remove_samples)
    if not ok:
        raise HTTPException(status_code=404, detail="Folder not found")
    return {"deleted": True, "id": folder_id}


# ── Indexing ─────────────────────────────────────────────────────────────────


@app.post("/api/index/start")
def start_indexing(req: IndexStartRequest) -> dict[str, Any]:
    started = indexer.start(folder_paths=req.folder_paths, force=req.force)
    return {"started": started, "progress": indexer.get_progress()}


@app.post("/api/index/cancel")
def cancel_indexing() -> dict[str, Any]:
    indexer.cancel()
    return {"cancelled": True}


@app.get("/api/index/progress")
def get_index_progress() -> dict[str, Any]:
    return indexer.get_progress()


@app.get("/api/index/events")
async def stream_index_events() -> StreamingResponse:
    async def event_gen():
        loop = asyncio.get_running_loop()
        gen = indexer.subscribe()

        def next_event():
            return next(gen)

        try:
            while True:
                data = await loop.run_in_executor(None, next_event)
                yield f"data: {json.dumps(data)}\n\n"
        except (StopIteration, asyncio.CancelledError):
            return

    return StreamingResponse(event_gen(), media_type="text/event-stream")


# ── Search & Discovery ───────────────────────────────────────────────────────


@app.get("/api/search")
def search_samples(
    q: Optional[str] = Query(default=None, description="Plain-English semantic search query"),
    sample_type: Optional[str] = Query(default=None),
    bpm_min: Optional[float] = Query(default=None),
    bpm_max: Optional[float] = Query(default=None),
    key: Optional[str] = Query(default=None),
    category: Optional[str] = Query(default=None),
    folder: Optional[str] = Query(default=None),
    favorites_only: bool = Query(default=False),
    sort: str = Query(default="recent"),
    limit: int = Query(default=80, ge=1, le=500),
    offset: int = Query(default=0, ge=0),
) -> dict[str, Any]:
    query_clean = (q or "").strip()
    if query_clean:
        emb = embedder.embed_text(query_clean)
        results = store.search_by_embedding(
            emb,
            limit=limit,
            sample_type=sample_type,
            bpm_min=bpm_min,
            bpm_max=bpm_max,
            key_name=key,
            category=category,
            folder_root=folder,
            favorites_only=favorites_only,
            text_boost_query=query_clean,
        )
        return {"mode": "semantic_text", "query": query_clean, "count": len(results), "results": results}

    results = store.list_samples(
        limit=limit,
        offset=offset,
        sample_type=sample_type,
        bpm_min=bpm_min,
        bpm_max=bpm_max,
        key_name=key,
        category=category,
        folder_root=folder,
        favorites_only=favorites_only,
        sort=sort,
    )
    return {"mode": "browse", "query": "", "count": len(results), "results": results}


@app.post("/api/search/audio")
async def search_by_reference_audio(
    file: UploadFile = File(...),
    sample_type: Optional[str] = Form(default=None),
    bpm_min: Optional[float] = Form(default=None),
    bpm_max: Optional[float] = Form(default=None),
    key: Optional[str] = Form(default=None),
    category: Optional[str] = Form(default=None),
    favorites_only: bool = Form(default=False),
    limit: int = Form(default=40),
) -> dict[str, Any]:
    suffix = Path(file.filename or "reference.wav").suffix.lower()
    if suffix not in SUPPORTED_EXTENSIONS:
        suffix = ".wav"

    raw = await file.read()
    if not raw:
        raise HTTPException(status_code=400, detail="Empty audio upload.")

    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(raw)
        tmp_path = Path(tmp.name)

    try:
        analysis = analyze_audio_file(tmp_path)
        emb = embedder.embed_audio(analysis.audio_48k)
        auto_cat, auto_tags = embedder.classify_tags(
            emb,
            sample_type=analysis.sample_type,
            duration_sec=analysis.duration_sec,
        )
        results = store.search_by_embedding(
            emb,
            limit=limit,
            sample_type=sample_type,
            bpm_min=bpm_min,
            bpm_max=bpm_max,
            key_name=key,
            category=category,
            favorites_only=favorites_only,
        )
        return {
            "mode": "reference_audio",
            "reference": {
                "filename": file.filename or "reference.wav",
                "duration_sec": analysis.duration_sec,
                "bpm": analysis.bpm,
                "key_name": analysis.key_name,
                "camelot": analysis.camelot,
                "sample_type": analysis.sample_type,
                "auto_category": auto_cat,
                "auto_tags": auto_tags,
                "waveform_peaks": analysis.waveform_peaks,
            },
            "count": len(results),
            "results": results,
        }
    except Exception as exc:
        logger.exception("Reference audio search failed")
        raise HTTPException(status_code=400, detail=f"Could not analyze audio: {exc}") from exc
    finally:
        tmp_path.unlink(missing_ok=True)


@app.get("/api/samples/{sample_id}/similar")
def find_similar_samples(
    sample_id: int,
    limit: int = Query(default=30, ge=1, le=200),
    sample_type: Optional[str] = Query(default=None),
) -> dict[str, Any]:
    sample = store.get_sample(sample_id)
    if not sample:
        raise HTTPException(status_code=404, detail="Sample not found")
    emb = store.get_sample_embedding(sample_id)
    if emb is None:
        raise HTTPException(status_code=404, detail="Embedding not found")
    results = store.search_by_embedding(
        emb,
        limit=limit,
        sample_type=sample_type,
        exclude_id=sample_id,
    )
    return {"mode": "similar_sample", "source_sample": sample, "count": len(results), "results": results}


# ── Single Sample Operations & Audio Streaming ──────────────────────────────


@app.get("/api/samples/{sample_id}")
def get_sample(sample_id: int) -> dict[str, Any]:
    sample = store.get_sample(sample_id)
    if not sample:
        raise HTTPException(status_code=404, detail="Sample not found")
    return sample


@app.patch("/api/samples/{sample_id}")
def patch_sample(sample_id: int, req: SampleUpdateRequest) -> dict[str, Any]:
    updates = {k: v for k, v in req.model_dump(exclude_unset=True).items()}
    updated = store.update_sample(sample_id, updates)
    if not updated:
        raise HTTPException(status_code=404, detail="Sample not found")
    return updated


@app.get("/api/audio/{sample_id}")
def stream_audio(sample_id: int) -> FileResponse:
    sample = store.get_sample(sample_id)
    if not sample:
        raise HTTPException(status_code=404, detail="Sample not found")
    audio_path = Path(sample["path"])
    if not audio_path.exists():
        raise HTTPException(status_code=404, detail="Audio file no longer exists on disk")
    media_type, _ = mimetypes.guess_type(str(audio_path))
    if not media_type:
        media_type = "audio/wav"
    return FileResponse(str(audio_path), media_type=media_type, filename=audio_path.name)


@app.post("/api/samples/{sample_id}/reveal")
def reveal_in_finder(sample_id: int) -> dict[str, Any]:
    """Reveal the sample file in macOS Finder (ready to drag into Bitwig Studio)."""
    sample = store.get_sample(sample_id)
    if not sample:
        raise HTTPException(status_code=404, detail="Sample not found")
    audio_path = Path(sample["path"])
    if not audio_path.exists():
        raise HTTPException(status_code=404, detail="File not found on disk")
    if sys.platform == "darwin":
        subprocess.Popen(["open", "-R", str(audio_path)])
        return {"revealed": True, "path": str(audio_path)}
    return {"revealed": False, "path": str(audio_path)}


# ── Auto-Organizer ───────────────────────────────────────────────────────────


@app.post("/api/organize/preview")
def api_organize_preview(req: OrganizeRequest) -> dict[str, Any]:
    return preview_organization(
        req.target_dir,
        template=req.template,
        mode=req.mode,
        folder_root=req.folder_root,
    )


@app.post("/api/organize/apply")
def api_organize_apply(req: OrganizeRequest) -> dict[str, Any]:
    try:
        return apply_organization(
            req.target_dir,
            template=req.template,
            mode=req.mode,
            folder_root=req.folder_root,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/organize/undo/{manifest_id}")
def api_organize_undo(manifest_id: str) -> dict[str, Any]:
    try:
        return undo_organization(manifest_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


# ── Frontend Static Assets ───────────────────────────────────────────────────


if FRONTEND_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")


@app.get("/")
def serve_index() -> FileResponse:
    index_file = FRONTEND_DIR / "index.html"
    if not index_file.exists():
        return JSONResponse({"status": "Sample Vault API running; frontend not found."})  # type: ignore[return-value]
    return FileResponse(str(index_file))
