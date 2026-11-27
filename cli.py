#!/usr/bin/env python3
"""Sample Vault CLI: serve the web UI, index folders, search via CLAP, or auto-organize."""

from __future__ import annotations

import argparse
import sys
import webbrowser
from pathlib import Path

from backend.config import DEFAULT_HOST, DEFAULT_PORT, DEMO_SAMPLES_DIR


def cmd_serve(args: argparse.Namespace) -> None:
    import uvicorn

    url = f"http://{args.host}:{args.port}"
    print(f"Starting Sample Vault at {url}")
    if args.open:
        webbrowser.open(url)
    uvicorn.run(
        "backend.main:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level="info",
    )


def cmd_index(args: argparse.Namespace) -> None:
    from backend.indexer import indexer
    from backend.store import store

    roots: list[str] = []
    for folder in args.folders:
        p = Path(folder).expanduser().resolve()
        if not p.exists() or not p.is_dir():
            print(f"Error: folder not found: {p}", file=sys.stderr)
            sys.exit(1)
        f = store.add_folder(str(p))
        roots.append(f["path"])

    result = indexer.index_sync(roots or None, force=args.force)
    print(result["message"])


def cmd_search(args: argparse.Namespace) -> None:
    from backend.embedder import embedder
    from backend.store import store

    query = " ".join(args.query).strip()
    emb = embedder.embed_text(query)
    results = store.search_by_embedding(
        emb,
        limit=args.limit,
        sample_type=args.type,
        bpm_min=args.bpm_min,
        bpm_max=args.bpm_max,
        key_name=args.key,
        text_boost_query=query,
    )
    if not results:
        print("No matching samples found. Run `python cli.py index <folder>` first.")
        return

    print(f'\nTop {len(results)} matches for "{query}":\n')
    for i, s in enumerate(results, start=1):
        sim_pct = int(round((s.get("similarity", 0.0)) * 100))
        bpm_str = f"{s['bpm']:.0f} BPM" if s["bpm"] else "— BPM"
        key_str = s["key_name"] or "—"
        print(
            f"  {i:2d}. [{sim_pct:3d}%] {s['filename']:<36} | {s['sample_type']:<8} | "
            f"{s['auto_category']:<16} | {bpm_str:<7} | {key_str:<7} | {', '.join(s['auto_tags'])}"
        )


def cmd_similar(args: argparse.Namespace) -> None:
    from backend.analyzer import analyze_audio_file
    from backend.embedder import embedder
    from backend.store import store

    ref = Path(args.audio_file).expanduser().resolve()
    if not ref.exists():
        print(f"Error: file not found: {ref}", file=sys.stderr)
        sys.exit(1)

    analysis = analyze_audio_file(ref)
    emb = embedder.embed_audio(analysis.audio_48k)
    results = store.search_by_embedding(emb, limit=args.limit)
    print(f"\nClosest matches to {ref.name} ({analysis.sample_type}, {analysis.bpm or '—'} BPM, {analysis.key_name or '—'}):\n")
    for i, s in enumerate(results, start=1):
        sim_pct = int(round((s.get("similarity", 0.0)) * 100))
        print(
            f"  {i:2d}. [{sim_pct:3d}%] {s['filename']:<36} | {s['sample_type']:<8} | "
            f"{s['auto_category']:<16} | {s['path']}"
        )


def cmd_demo(args: argparse.Namespace) -> None:
    from scripts.generate_demo_samples import generate_pack

    out_dir = Path(args.output).expanduser().resolve() if args.output else DEMO_SAMPLES_DIR
    files = generate_pack(out_dir)
    print(f"Generated {len(files)} demo audio samples in {out_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="sample-vault",
        description="Local Audio Sample Search & Auto-Organizer powered by CLAP + librosa + sqlite-vec",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_serve = sub.add_parser("serve", help="Start the local Sample Vault web UI & API")
    p_serve.add_argument("--host", default=DEFAULT_HOST)
    p_serve.add_argument("--port", type=int, default=DEFAULT_PORT)
    p_serve.add_argument("--open", action="store_true", help="Open browser automatically")
    p_serve.add_argument("--reload", action="store_true")
    p_serve.set_defaults(func=cmd_serve)

    p_index = sub.add_parser("index", help="Index one or more sample folders")
    p_index.add_argument("folders", nargs="*", help="Folder paths to add & index")
    p_index.add_argument("--force", action="store_true", help="Re-index unchanged files")
    p_index.set_defaults(func=cmd_index)

    p_search = sub.add_parser("search", help="Search samples by plain English")
    p_search.add_argument("query", nargs="+", help="Natural language description")
    p_search.add_argument("--type", choices=["one-shot", "loop"])
    p_search.add_argument("--bpm-min", type=float)
    p_search.add_argument("--bpm-max", type=float)
    p_search.add_argument("--key")
    p_search.add_argument("--limit", type=int, default=12)
    p_search.set_defaults(func=cmd_search)

    p_sim = sub.add_parser("similar", help="Find closest matches to a reference audio file")
    p_sim.add_argument("audio_file")
    p_sim.add_argument("--limit", type=int, default=12)
    p_sim.set_defaults(func=cmd_similar)

    p_demo = sub.add_parser("demo", help="Generate a synthetic demo sample pack for testing")
    p_demo.add_argument("--output", default=None)
    p_demo.set_defaults(func=cmd_demo)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
