import sqlite3
import time
from contextlib import contextmanager

import numpy as np

import config


@contextmanager
def _db():
    c = sqlite3.connect(config.DB_PATH, timeout=30)
    c.row_factory = sqlite3.Row
    c.execute("PRAGMA journal_mode=WAL")
    try:
        yield c
        c.commit()
    finally:
        c.close()


def init():
    with _db() as c:
        c.executescript(
            """
            CREATE TABLE IF NOT EXISTS chunks(
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                source TEXT, chunk_index INTEGER, text TEXT, embedding BLOB);
            CREATE INDEX IF NOT EXISTS ix_chunks_source ON chunks(source);
            CREATE TABLE IF NOT EXISTS sessions(
                id TEXT PRIMARY KEY, created REAL, updated REAL,
                pkl_path TEXT, message_count INTEGER);
            """
        )


def replace_source(source, texts, embs):
    """Delete old chunks of this source and insert the new ones (re-upload safe)."""
    with _db() as c:
        c.execute("DELETE FROM chunks WHERE source=?", (source,))
        c.executemany(
            "INSERT INTO chunks(source, chunk_index, text, embedding) VALUES (?,?,?,?)",
            [(source, i, t, embs[i].astype(np.float32).tobytes()) for i, t in enumerate(texts)],
        )


def load_all():
    with _db() as c:
        rows = c.execute("SELECT source, text, embedding FROM chunks ORDER BY id").fetchall()
    if not rows:
        return [], [], np.zeros((0, config.EMBED_DIM), dtype=np.float32)
    sources = [r["source"] for r in rows]
    texts = [r["text"] for r in rows]
    mat = np.stack([np.frombuffer(r["embedding"], dtype=np.float32) for r in rows])
    return sources, texts, mat


def upsert_session(sid, pkl_path, count):
    now = time.time()
    with _db() as c:
        c.execute(
            """INSERT INTO sessions(id, created, updated, pkl_path, message_count)
               VALUES (?,?,?,?,?)
               ON CONFLICT(id) DO UPDATE SET updated=excluded.updated,
               message_count=excluded.message_count""",
            (sid, now, now, str(pkl_path), count),
        )


def list_sessions():
    with _db() as c:
        return [dict(r) for r in c.execute("SELECT * FROM sessions ORDER BY updated DESC")]


def stats():
    with _db() as c:
        return [dict(r) for r in c.execute(
            "SELECT source, COUNT(*) AS chunks FROM chunks GROUP BY source ORDER BY source")]
