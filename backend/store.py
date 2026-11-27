"""Local SQLite + sqlite-vec metadata and vector similarity store."""

from __future__ import annotations

import json
import logging
import sqlite3
import struct
import threading
import time
from pathlib import Path
from typing import Any, Optional

import numpy as np

from .config import DB_PATH, EMBEDDING_DIM

logger = logging.getLogger(__name__)


def _serialize_f32(vec: np.ndarray) -> bytes:
    arr = np.asarray(vec, dtype=np.float32).reshape(-1)
    return struct.pack(f"{len(arr)}f", *arr.tolist())


def _deserialize_f32(blob: bytes) -> np.ndarray:
    return np.frombuffer(blob, dtype=np.float32)


class SampleStore:
    """Thread-safe local SQLite + sqlite-vec database for indexed audio samples."""

    def __init__(self, db_path: Path = DB_PATH) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._vec_enabled = False
        self._conn = self._open_connection()
        self._init_schema()

    def _open_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA synchronous=NORMAL;")
        conn.execute("PRAGMA foreign_keys=ON;")
        try:
            import sqlite_vec

            conn.enable_load_extension(True)
            sqlite_vec.load(conn)
            conn.enable_load_extension(False)
            self._vec_enabled = True
            logger.info("sqlite-vec extension loaded successfully.")
        except Exception as exc:
            self._vec_enabled = False
            logger.warning("sqlite-vec unavailable (%s); using NumPy cosine fallback.", exc)
        return conn

    @property
    def vec_backend(self) -> str:
        return "sqlite-vec" if self._vec_enabled else "numpy-cosine"

    def _init_schema(self) -> None:
        with self._lock:
            cur = self._conn.cursor()
            cur.executescript(
                """
                CREATE TABLE IF NOT EXISTS watched_folders (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    path TEXT NOT NULL UNIQUE,
                    label TEXT NOT NULL,
                    added_at REAL NOT NULL,
                    last_indexed_at REAL
                );

                CREATE TABLE IF NOT EXISTS samples (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    path TEXT NOT NULL UNIQUE,
                    filename TEXT NOT NULL,
                    folder_root TEXT NOT NULL,
                    rel_dir TEXT NOT NULL,
                    mtime REAL NOT NULL,
                    size_bytes INTEGER NOT NULL,
                    duration_sec REAL NOT NULL,
                    sample_rate INTEGER NOT NULL,
                    channels INTEGER NOT NULL,
                    bpm REAL,
                    key_name TEXT,
                    camelot TEXT,
                    sample_type TEXT NOT NULL,
                    type_confidence REAL NOT NULL,
                    auto_category TEXT NOT NULL,
                    auto_tags TEXT NOT NULL,
                    custom_tags TEXT NOT NULL DEFAULT '[]',
                    energy REAL NOT NULL,
                    brightness REAL NOT NULL,
                    waveform_peaks TEXT NOT NULL,
                    embedding BLOB NOT NULL,
                    favorite INTEGER NOT NULL DEFAULT 0,
                    indexed_at REAL NOT NULL
                );

                CREATE INDEX IF NOT EXISTS idx_samples_folder ON samples(folder_root);
                CREATE INDEX IF NOT EXISTS idx_samples_type ON samples(sample_type);
                CREATE INDEX IF NOT EXISTS idx_samples_category ON samples(auto_category);
                CREATE INDEX IF NOT EXISTS idx_samples_bpm ON samples(bpm);
                CREATE INDEX IF NOT EXISTS idx_samples_key ON samples(key_name);
                CREATE INDEX IF NOT EXISTS idx_samples_fav ON samples(favorite);
                """
            )
            if self._vec_enabled:
                cur.execute(
                    f"""
                    CREATE VIRTUAL TABLE IF NOT EXISTS vec_samples USING vec0(
                        sample_id INTEGER PRIMARY KEY,
                        embedding float[{EMBEDDING_DIM}] distance_metric=cosine
                    );
                    """
                )
            self._conn.commit()

    # ── Watched Folders ──────────────────────────────────────────────────────

    def list_folders(self) -> list[dict[str, Any]]:
        with self._lock:
            rows = self._conn.execute(
                """
                SELECT f.*, COUNT(s.id) AS sample_count
                FROM watched_folders f
                LEFT JOIN samples s ON s.folder_root = f.path
                GROUP BY f.id
                ORDER BY f.added_at DESC
                """
            ).fetchall()
            return [dict(r) for r in rows]

    def add_folder(self, path: str, label: Optional[str] = None) -> dict[str, Any]:
        resolved = str(Path(path).expanduser().resolve())
        lbl = label or Path(resolved).name or resolved
        now = time.time()
        with self._lock:
            self._conn.execute(
                """
                INSERT INTO watched_folders (path, label, added_at)
                VALUES (?, ?, ?)
                ON CONFLICT(path) DO UPDATE SET label = excluded.label
                """,
                (resolved, lbl, now),
            )
            self._conn.commit()
            row = self._conn.execute(
                "SELECT * FROM watched_folders WHERE path = ?",
                (resolved,),
            ).fetchone()
            return dict(row)

    def remove_folder(self, folder_id: int, remove_samples: bool = True) -> bool:
        with self._lock:
            row = self._conn.execute(
                "SELECT path FROM watched_folders WHERE id = ?",
                (folder_id,),
            ).fetchone()
            if not row:
                return False
            folder_path = row["path"]
            if remove_samples:
                sample_ids = [
                    r["id"]
                    for r in self._conn.execute(
                        "SELECT id FROM samples WHERE folder_root = ?",
                        (folder_path,),
                    ).fetchall()
                ]
                if sample_ids:
                    self._delete_samples_unlocked(sample_ids)
            self._conn.execute("DELETE FROM watched_folders WHERE id = ?", (folder_id,))
            self._conn.commit()
            return True

    def mark_folder_indexed(self, folder_path: str) -> None:
        with self._lock:
            self._conn.execute(
                "UPDATE watched_folders SET last_indexed_at = ? WHERE path = ?",
                (time.time(), folder_path),
            )
            self._conn.commit()

    # ── Samples CRUD ─────────────────────────────────────────────────────────

    def get_indexed_file_map(self, folder_root: str) -> dict[str, tuple[int, float, int]]:
        """Return {path: (id, mtime, size_bytes)} for incremental indexing."""
        with self._lock:
            rows = self._conn.execute(
                "SELECT id, path, mtime, size_bytes FROM samples WHERE folder_root = ?",
                (folder_root,),
            ).fetchall()
            return {r["path"]: (r["id"], r["mtime"], r["size_bytes"]) for r in rows}

    def upsert_sample(self, record: dict[str, Any], embedding: np.ndarray) -> int:
        emb_blob = _serialize_f32(embedding)
        now = time.time()
        with self._lock:
            existing = self._conn.execute(
                "SELECT id, favorite, custom_tags FROM samples WHERE path = ?",
                (record["path"],),
            ).fetchone()

            favorite = existing["favorite"] if existing else 0
            custom_tags = existing["custom_tags"] if existing else "[]"

            if existing:
                sample_id = int(existing["id"])
                self._conn.execute(
                    """
                    UPDATE samples SET
                        filename=?, folder_root=?, rel_dir=?, mtime=?, size_bytes=?,
                        duration_sec=?, sample_rate=?, channels=?, bpm=?, key_name=?, camelot=?,
                        sample_type=?, type_confidence=?, auto_category=?, auto_tags=?,
                        custom_tags=?, energy=?, brightness=?, waveform_peaks=?, embedding=?,
                        favorite=?, indexed_at=?
                    WHERE id=?
                    """,
                    (
                        record["filename"],
                        record["folder_root"],
                        record["rel_dir"],
                        record["mtime"],
                        record["size_bytes"],
                        record["duration_sec"],
                        record["sample_rate"],
                        record["channels"],
                        record["bpm"],
                        record["key_name"],
                        record["camelot"],
                        record["sample_type"],
                        record["type_confidence"],
                        record["auto_category"],
                        json.dumps(record["auto_tags"]),
                        custom_tags,
                        record["energy"],
                        record["brightness"],
                        json.dumps(record["waveform_peaks"]),
                        emb_blob,
                        favorite,
                        now,
                        sample_id,
                    ),
                )
                if self._vec_enabled:
                    self._conn.execute("DELETE FROM vec_samples WHERE sample_id = ?", (sample_id,))
                    self._conn.execute(
                        "INSERT INTO vec_samples (sample_id, embedding) VALUES (?, ?)",
                        (sample_id, emb_blob),
                    )
            else:
                cur = self._conn.execute(
                    """
                    INSERT INTO samples (
                        path, filename, folder_root, rel_dir, mtime, size_bytes,
                        duration_sec, sample_rate, channels, bpm, key_name, camelot,
                        sample_type, type_confidence, auto_category, auto_tags,
                        custom_tags, energy, brightness, waveform_peaks, embedding,
                        favorite, indexed_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        record["path"],
                        record["filename"],
                        record["folder_root"],
                        record["rel_dir"],
                        record["mtime"],
                        record["size_bytes"],
                        record["duration_sec"],
                        record["sample_rate"],
                        record["channels"],
                        record["bpm"],
                        record["key_name"],
                        record["camelot"],
                        record["sample_type"],
                        record["type_confidence"],
                        record["auto_category"],
                        json.dumps(record["auto_tags"]),
                        custom_tags,
                        record["energy"],
                        record["brightness"],
                        json.dumps(record["waveform_peaks"]),
                        emb_blob,
                        favorite,
                        now,
                    ),
                )
                sample_id = int(cur.lastrowid)
                if self._vec_enabled:
                    self._conn.execute(
                        "INSERT INTO vec_samples (sample_id, embedding) VALUES (?, ?)",
                        (sample_id, emb_blob),
                    )
            self._conn.commit()
            return sample_id

    def delete_samples(self, sample_ids: list[int]) -> None:
        if not sample_ids:
            return
        with self._lock:
            self._delete_samples_unlocked(sample_ids)
            self._conn.commit()

    def _delete_samples_unlocked(self, sample_ids: list[int]) -> None:
        placeholders = ",".join("?" for _ in sample_ids)
        if self._vec_enabled:
            self._conn.execute(
                f"DELETE FROM vec_samples WHERE sample_id IN ({placeholders})",
                sample_ids,
            )
        self._conn.execute(
            f"DELETE FROM samples WHERE id IN ({placeholders})",
            sample_ids,
        )

    def _row_to_dict(self, row: sqlite3.Row, score: Optional[float] = None) -> dict[str, Any]:
        d = {
            "id": row["id"],
            "path": row["path"],
            "filename": row["filename"],
            "folder_root": row["folder_root"],
            "rel_dir": row["rel_dir"],
            "size_bytes": row["size_bytes"],
            "duration_sec": row["duration_sec"],
            "sample_rate": row["sample_rate"],
            "channels": row["channels"],
            "bpm": row["bpm"],
            "key_name": row["key_name"],
            "camelot": row["camelot"],
            "sample_type": row["sample_type"],
            "type_confidence": row["type_confidence"],
            "auto_category": row["auto_category"],
            "auto_tags": json.loads(row["auto_tags"] or "[]"),
            "custom_tags": json.loads(row["custom_tags"] or "[]"),
            "energy": row["energy"],
            "brightness": row["brightness"],
            "waveform_peaks": json.loads(row["waveform_peaks"] or "[]"),
            "favorite": bool(row["favorite"]),
            "indexed_at": row["indexed_at"],
        }
        if score is not None:
            d["similarity"] = round(float(score), 4)
        return d

    def get_sample(self, sample_id: int) -> Optional[dict[str, Any]]:
        with self._lock:
            row = self._conn.execute("SELECT * FROM samples WHERE id = ?", (sample_id,)).fetchone()
            return self._row_to_dict(row) if row else None

    def get_sample_embedding(self, sample_id: int) -> Optional[np.ndarray]:
        with self._lock:
            row = self._conn.execute(
                "SELECT embedding FROM samples WHERE id = ?",
                (sample_id,),
            ).fetchone()
            if not row or not row["embedding"]:
                return None
            return _deserialize_f32(row["embedding"]).copy()

    def update_sample(self, sample_id: int, updates: dict[str, Any]) -> Optional[dict[str, Any]]:
        allowed = {
            "favorite",
            "custom_tags",
            "bpm",
            "key_name",
            "camelot",
            "sample_type",
            "auto_category",
        }
        fields: list[str] = []
        values: list[Any] = []
        for k, v in updates.items():
            if k not in allowed:
                continue
            if k == "favorite":
                values.append(1 if v else 0)
            elif k == "custom_tags":
                values.append(json.dumps(list(v)))
            else:
                values.append(v)
            fields.append(f"{k} = ?")

        if not fields:
            return self.get_sample(sample_id)

        values.append(sample_id)
        with self._lock:
            self._conn.execute(
                f"UPDATE samples SET {', '.join(fields)} WHERE id = ?",
                values,
            )
            self._conn.commit()
            return self.get_sample(sample_id)

    def get_stats(self) -> dict[str, Any]:
        with self._lock:
            total = self._conn.execute("SELECT COUNT(*) FROM samples").fetchone()[0]
            oneshots = self._conn.execute(
                "SELECT COUNT(*) FROM samples WHERE sample_type = 'one-shot'"
            ).fetchone()[0]
            loops = self._conn.execute(
                "SELECT COUNT(*) FROM samples WHERE sample_type = 'loop'"
            ).fetchone()[0]
            favorites = self._conn.execute(
                "SELECT COUNT(*) FROM samples WHERE favorite = 1"
            ).fetchone()[0]
            cat_rows = self._conn.execute(
                """
                SELECT auto_category, COUNT(*) AS cnt
                FROM samples
                GROUP BY auto_category
                ORDER BY cnt DESC
                """
            ).fetchall()
            key_rows = self._conn.execute(
                """
                SELECT key_name, COUNT(*) AS cnt
                FROM samples
                WHERE key_name IS NOT NULL
                GROUP BY key_name
                ORDER BY cnt DESC
                """
            ).fetchall()
            return {
                "total_samples": total,
                "one_shots": oneshots,
                "loops": loops,
                "favorites": favorites,
                "categories": [{"name": r["auto_category"], "count": r["cnt"]} for r in cat_rows],
                "keys": [{"name": r["key_name"], "count": r["cnt"]} for r in key_rows],
                "vec_backend": self.vec_backend,
            }

    # ── Search & Vector Similarity ───────────────────────────────────────────

    def _build_filter_clause(
        self,
        sample_type: Optional[str] = None,
        bpm_min: Optional[float] = None,
        bpm_max: Optional[float] = None,
        key_name: Optional[str] = None,
        category: Optional[str] = None,
        folder_root: Optional[str] = None,
        favorites_only: bool = False,
        exclude_id: Optional[int] = None,
        alias: str = "",
    ) -> tuple[str, list[Any]]:
        prefix = f"{alias}." if alias else ""
        clauses: list[str] = []
        params: list[Any] = []

        if sample_type in ("one-shot", "loop"):
            clauses.append(f"{prefix}sample_type = ?")
            params.append(sample_type)
        if bpm_min is not None:
            clauses.append(f"{prefix}bpm IS NOT NULL AND {prefix}bpm >= ?")
            params.append(float(bpm_min))
        if bpm_max is not None:
            clauses.append(f"{prefix}bpm IS NOT NULL AND {prefix}bpm <= ?")
            params.append(float(bpm_max))
        if key_name:
            clauses.append(f"({prefix}key_name = ? OR {prefix}camelot = ?)")
            params.extend([key_name, key_name])
        if category:
            clauses.append(f"{prefix}auto_category = ?")
            params.append(category)
        if folder_root:
            clauses.append(f"{prefix}folder_root = ?")
            params.append(folder_root)
        if favorites_only:
            clauses.append(f"{prefix}favorite = 1")
        if exclude_id is not None:
            clauses.append(f"{prefix}id != ?")
            params.append(int(exclude_id))

        where_sql = (" WHERE " + " AND ".join(clauses)) if clauses else ""
        return where_sql, params

    def search_by_embedding(
        self,
        query_emb: np.ndarray,
        *,
        limit: int = 60,
        sample_type: Optional[str] = None,
        bpm_min: Optional[float] = None,
        bpm_max: Optional[float] = None,
        key_name: Optional[str] = None,
        category: Optional[str] = None,
        folder_root: Optional[str] = None,
        favorites_only: bool = False,
        exclude_id: Optional[int] = None,
        text_boost_query: Optional[str] = None,
    ) -> list[dict[str, Any]]:
        """Cosine vector search via sqlite-vec (with filter support & optional lexical boost)."""
        query_emb = np.asarray(query_emb, dtype=np.float32).reshape(-1)
        norm = float(np.linalg.norm(query_emb))
        if norm > 1e-8:
            query_emb = query_emb / norm

        with self._lock:
            total = self._conn.execute("SELECT COUNT(*) FROM samples").fetchone()[0]
            if total == 0:
                return []

            has_filters = any(
                [
                    sample_type in ("one-shot", "loop"),
                    bpm_min is not None,
                    bpm_max is not None,
                    bool(key_name),
                    bool(category),
                    bool(folder_root),
                    favorites_only,
                    exclude_id is not None,
                ]
            )

            candidates: list[tuple[sqlite3.Row, float]] = []

            if self._vec_enabled and not has_filters:
                k = min(total, max(limit, 80))
                vec_rows = self._conn.execute(
                    """
                    SELECT sample_id, distance
                    FROM vec_samples
                    WHERE embedding MATCH ?
                    AND k = ?
                    ORDER BY distance
                    """,
                    (_serialize_f32(query_emb), k),
                ).fetchall()
                id_to_sim = {int(r["sample_id"]): max(0.0, 1.0 - float(r["distance"])) for r in vec_rows}
                if id_to_sim:
                    placeholders = ",".join("?" for _ in id_to_sim)
                    rows = self._conn.execute(
                        f"SELECT * FROM samples WHERE id IN ({placeholders})",
                        list(id_to_sim.keys()),
                    ).fetchall()
                    candidates = [(r, id_to_sim[int(r["id"])]) for r in rows]
            else:
                # Filtered or fallback search: fetch filtered subset and score via vectorized BLAS dot product
                where_sql, params = self._build_filter_clause(
                    sample_type=sample_type,
                    bpm_min=bpm_min,
                    bpm_max=bpm_max,
                    key_name=key_name,
                    category=category,
                    folder_root=folder_root,
                    favorites_only=favorites_only,
                    exclude_id=exclude_id,
                )
                rows = self._conn.execute(f"SELECT * FROM samples{where_sql}", params).fetchall()
                if rows:
                    mat = np.vstack([_deserialize_f32(r["embedding"]) for r in rows])
                    sims = mat @ query_emb
                    candidates = [(rows[i], float(sims[i])) for i in range(len(rows))]

            # Optional light lexical tie-boost when user searches by plain English
            tokens: list[str] = []
            if text_boost_query:
                tokens = [t.lower() for t in text_boost_query.split() if len(t) >= 3]

            scored: list[tuple[float, float, sqlite3.Row]] = []
            for row, sim in candidates:
                rank_score = sim
                if tokens:
                    haystack = " ".join(
                        [
                            row["filename"].lower(),
                            row["rel_dir"].lower(),
                            row["auto_category"].lower(),
                            row["auto_tags"].lower(),
                            row["custom_tags"].lower(),
                        ]
                    )
                    hits = sum(1 for tok in tokens if tok in haystack)
                    if hits > 0:
                        rank_score += min(0.08, 0.025 * hits)
                scored.append((rank_score, sim, row))

            scored.sort(key=lambda item: item[0], reverse=True)
            return [self._row_to_dict(row, score=sim) for _, sim, row in scored[:limit]]

    def list_samples(
        self,
        *,
        limit: int = 120,
        offset: int = 0,
        sample_type: Optional[str] = None,
        bpm_min: Optional[float] = None,
        bpm_max: Optional[float] = None,
        key_name: Optional[str] = None,
        category: Optional[str] = None,
        folder_root: Optional[str] = None,
        favorites_only: bool = False,
        sort: str = "recent",
    ) -> list[dict[str, Any]]:
        where_sql, params = self._build_filter_clause(
            sample_type=sample_type,
            bpm_min=bpm_min,
            bpm_max=bpm_max,
            key_name=key_name,
            category=category,
            folder_root=folder_root,
            favorites_only=favorites_only,
        )
        order_map = {
            "recent": "indexed_at DESC",
            "name": "filename COLLATE NOCASE ASC",
            "bpm_asc": "CASE WHEN bpm IS NULL THEN 1 ELSE 0 END, bpm ASC",
            "bpm_desc": "CASE WHEN bpm IS NULL THEN 1 ELSE 0 END, bpm DESC",
            "duration_asc": "duration_sec ASC",
            "duration_desc": "duration_sec DESC",
            "energy_desc": "energy DESC",
        }
        order_by = order_map.get(sort, "indexed_at DESC")
        params.extend([limit, offset])
        with self._lock:
            rows = self._conn.execute(
                f"SELECT * FROM samples{where_sql} ORDER BY {order_by} LIMIT ? OFFSET ?",
                params,
            ).fetchall()
            return [self._row_to_dict(r) for r in rows]

    def get_all_samples_for_organizer(self, folder_root: Optional[str] = None) -> list[dict[str, Any]]:
        where_sql, params = self._build_filter_clause(folder_root=folder_root)
        with self._lock:
            rows = self._conn.execute(
                f"SELECT * FROM samples{where_sql} ORDER BY auto_category, filename",
                params,
            ).fetchall()
            return [self._row_to_dict(r) for r in rows]


store = SampleStore()
