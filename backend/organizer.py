"""Safe Auto-Organizer: preview and materialize structured Bitwig-ready sample trees."""

from __future__ import annotations

import json
import os
import re
import shutil
import time
from pathlib import Path
from typing import Any, Optional

from .config import UNDO_DIR
from .store import SampleStore, store as default_store

SAFE_NAME_RE = re.compile(r"[^A-Za-z0-9._\-\s]+")


def _clean_segment(text: str) -> str:
    cleaned = SAFE_NAME_RE.sub("", text).strip()
    return cleaned or "Uncategorized"


def _bpm_bucket(bpm: Optional[float]) -> str:
    if bpm is None:
        return "No_BPM"
    low = int((bpm // 10) * 10)
    return f"{low}-{low + 9}_BPM"


def _build_relative_target(sample: dict[str, Any], template: str) -> Path:
    """Compute clean relative output path based on chosen layout template."""
    type_dir = "Loops" if sample["sample_type"] == "loop" else "One-Shots"
    cat_dir = _clean_segment(sample["auto_category"])
    key_dir = _clean_segment(sample["key_name"] or "Unpitched")
    bpm_dir = _bpm_bucket(sample["bpm"])

    # Optionally enrich filename with BPM/Key prefix for fast Bitwig browser scanning
    orig_name = Path(sample["filename"]).name

    if template == "type_category_key":
        if sample["sample_type"] == "loop":
            return Path(type_dir) / cat_dir / bpm_dir / orig_name
        return Path(type_dir) / cat_dir / orig_name
    elif template == "category_type":
        return Path(cat_dir) / type_dir / orig_name
    elif template == "key_category":
        return Path(key_dir) / cat_dir / orig_name
    elif template == "bpm_category":
        if sample["sample_type"] == "loop":
            return Path("Loops") / bpm_dir / cat_dir / orig_name
        return Path("One-Shots") / cat_dir / orig_name
    else:
        return Path(type_dir) / cat_dir / orig_name


def preview_organization(
    target_dir: str,
    *,
    template: str = "type_category_key",
    mode: str = "symlink",
    folder_root: Optional[str] = None,
    sample_store: SampleStore = default_store,
) -> dict[str, Any]:
    """Generate a dry-run preview of how samples will be organized."""
    dest_root = Path(target_dir).expanduser().resolve()
    samples = sample_store.get_all_samples_for_organizer(folder_root=folder_root)

    seen_dests: dict[Path, int] = {}
    operations: list[dict[str, Any]] = []
    folder_summary: dict[str, int] = {}

    for s in samples:
        src = Path(s["path"])
        if not src.exists():
            continue
        rel = _build_relative_target(s, template)
        dest = dest_root / rel

        # Resolve filename collisions cleanly
        if dest in seen_dests:
            seen_dests[dest] += 1
            dest = dest.with_name(f"{dest.stem}_{seen_dests[dest]}{dest.suffix}")
        else:
            seen_dests[dest] = 1

        parent_rel = str(dest.parent.relative_to(dest_root))
        folder_summary[parent_rel] = folder_summary.get(parent_rel, 0) + 1
        operations.append(
            {
                "sample_id": s["id"],
                "source": str(src),
                "destination": str(dest),
                "rel_destination": str(dest.relative_to(dest_root)),
                "category": s["auto_category"],
                "sample_type": s["sample_type"],
                "bpm": s["bpm"],
                "key_name": s["key_name"],
            }
        )

    tree_summary = [
        {"folder": k, "count": v}
        for k, v in sorted(folder_summary.items(), key=lambda x: x[0])
    ]

    return {
        "target_dir": str(dest_root),
        "template": template,
        "mode": mode,
        "total_operations": len(operations),
        "folder_summary": tree_summary,
        "preview_items": operations[:200],
    }


def apply_organization(
    target_dir: str,
    *,
    template: str = "type_category_key",
    mode: str = "symlink",
    folder_root: Optional[str] = None,
    sample_store: SampleStore = default_store,
) -> dict[str, Any]:
    """Execute a safe auto-organization plan (symlink or copy) and save an undo manifest."""
    if mode not in ("symlink", "copy"):
        raise ValueError("Mode must be 'symlink' or 'copy' to protect original files.")

    plan = preview_organization(
        target_dir,
        template=template,
        mode=mode,
        folder_root=folder_root,
        sample_store=sample_store,
    )
    dest_root = Path(plan["target_dir"])
    dest_root.mkdir(parents=True, exist_ok=True)

    # Recompute full operations (not truncated to 200)
    samples = sample_store.get_all_samples_for_organizer(folder_root=folder_root)
    seen_dests: dict[Path, int] = {}
    created: list[dict[str, str]] = []
    skipped = 0
    errors: list[str] = []

    for s in samples:
        src = Path(s["path"])
        if not src.exists():
            continue
        rel = _build_relative_target(s, template)
        dest = dest_root / rel
        if dest in seen_dests:
            seen_dests[dest] += 1
            dest = dest.with_name(f"{dest.stem}_{seen_dests[dest]}{dest.suffix}")
        else:
            seen_dests[dest] = 1

        dest.parent.mkdir(parents=True, exist_ok=True)
        try:
            if dest.exists() or dest.is_symlink():
                skipped += 1
                continue
            if mode == "symlink":
                os.symlink(src, dest)
            else:
                shutil.copy2(src, dest)
            created.append({"source": str(src), "destination": str(dest), "mode": mode})
        except Exception as exc:
            errors.append(f"{src.name}: {exc}")

    manifest_id = f"org_{int(time.time())}"
    manifest_path = UNDO_DIR / f"{manifest_id}.json"
    manifest_data = {
        "id": manifest_id,
        "created_at": time.time(),
        "target_dir": str(dest_root),
        "template": template,
        "mode": mode,
        "created_files": created,
    }
    manifest_path.write_text(json.dumps(manifest_data, indent=2))

    return {
        "manifest_id": manifest_id,
        "target_dir": str(dest_root),
        "mode": mode,
        "created_count": len(created),
        "skipped_count": skipped,
        "error_count": len(errors),
        "errors": errors[:20],
    }


def undo_organization(manifest_id: str) -> dict[str, Any]:
    """Remove files/symlinks created by a previous organization run."""
    manifest_path = UNDO_DIR / f"{manifest_id}.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Undo manifest '{manifest_id}' not found.")

    data = json.loads(manifest_path.read_text())
    removed = 0
    for item in data.get("created_files", []):
        dest = Path(item["destination"])
        if dest.is_symlink() or dest.exists():
            try:
                dest.unlink()
                removed += 1
            except OSError:
                pass

    # Prune empty directories inside target_dir
    target_root = Path(data["target_dir"])
    if target_root.exists():
        for dirpath, _, _ in os.walk(target_root, topdown=False):
            p = Path(dirpath)
            try:
                if p != target_root and not any(p.iterdir()):
                    p.rmdir()
            except OSError:
                pass

    manifest_path.unlink(missing_ok=True)
    return {"manifest_id": manifest_id, "removed_count": removed}
