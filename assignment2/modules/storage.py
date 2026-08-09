"""
storage.py
----------
Persists the corpus as TWO separate tables (Section B requirement: "Store
extracted metadata separately from document contents"):

  * `metadata` -- id, url, source_type, category, title, depth, parent_url,
                  discovered_at, content_hash, is_exact_duplicate,
                  is_near_duplicate, duplicate_of, word_count
  * `content`  -- id, raw_text, clean_text

Backed by SQLite (Python stdlib `sqlite3`, no extra dependency) so the split
is real and inspectable outside of Streamlit too, not just two DataFrames
living in the same process. The DB file (data/ir_store.db) is one of the
"supporting files" submitted with the assignment.
"""
from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Optional

import pandas as pd

DB_PATH = Path(__file__).resolve().parent.parent / "data" / "ir_store.db"

METADATA_COLUMNS = [
    "id", "url", "source_type", "category", "title", "depth", "parent_url",
    "discovered_at", "content_hash", "is_exact_duplicate", "is_near_duplicate",
    "duplicate_of", "word_count", "doc_id",
]
CONTENT_COLUMNS = ["id", "raw_text", "clean_text"]


def get_connection(db_path: Optional[Path] = None) -> sqlite3.Connection:
    path = db_path or DB_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    return sqlite3.connect(str(path))


def init_db(conn: sqlite3.Connection) -> None:
    conn.execute("""
        CREATE TABLE IF NOT EXISTS metadata (
            id TEXT PRIMARY KEY, url TEXT, source_type TEXT, category TEXT,
            title TEXT, depth INTEGER, parent_url TEXT, discovered_at TEXT,
            content_hash TEXT, is_exact_duplicate INTEGER, is_near_duplicate INTEGER,
            duplicate_of TEXT, word_count INTEGER, doc_id TEXT
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS content (
            id TEXT PRIMARY KEY, raw_text TEXT, clean_text TEXT
        )
    """)
    conn.commit()


def save_corpus(metadata_df: pd.DataFrame, content_df: pd.DataFrame,
                 db_path: Optional[Path] = None) -> None:
    """Replaces the persisted corpus with the given metadata/content frames."""
    conn = get_connection(db_path)
    try:
        init_db(conn)
        metadata_df[METADATA_COLUMNS].to_sql("metadata", conn, if_exists="replace", index=False)
        content_df[CONTENT_COLUMNS].to_sql("content", conn, if_exists="replace", index=False)
        conn.commit()
    finally:
        conn.close()


def load_corpus(db_path: Optional[Path] = None):
    """Returns (metadata_df, content_df) or (None, None) if nothing persisted yet."""
    path = db_path or DB_PATH
    if not path.exists():
        return None, None
    conn = get_connection(db_path)
    try:
        tables = pd.read_sql("SELECT name FROM sqlite_master WHERE type='table'", conn)["name"].tolist()
        if "metadata" not in tables or "content" not in tables:
            return None, None
        metadata_df = pd.read_sql("SELECT * FROM metadata", conn)
        content_df = pd.read_sql("SELECT * FROM content", conn)
        return metadata_df, content_df
    finally:
        conn.close()


def db_file_size_kb(db_path: Optional[Path] = None) -> float:
    path = db_path or DB_PATH
    return round(path.stat().st_size / 1024, 1) if path.exists() else 0.0


__all__ = ["save_corpus", "load_corpus", "init_db", "get_connection", "db_file_size_kb",
           "METADATA_COLUMNS", "CONTENT_COLUMNS", "DB_PATH"]
