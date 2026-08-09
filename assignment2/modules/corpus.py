"""
corpus.py
---------
Assembles the working document collection from the heterogeneous sources
required by Section B: a publicly-available dataset (BBC-style news CSV)
combined with a simulated API feed (mock_api.py), using the SAME unified
metadata/content schema the crawler (crawler.py) produces -- so all three
sources can be merged into one corpus and everything downstream (mining,
indexing, search, recommendation, evaluation) is source-agnostic.
"""
from __future__ import annotations

from pathlib import Path
from typing import Tuple

import pandas as pd

from modules.mock_api import fetch_latest_articles
from modules.mock_web import _content_hash
from modules.storage import CONTENT_COLUMNS, METADATA_COLUMNS

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
DATASET_PATH = DATA_DIR / "news_corpus.csv"


def load_dataset(path: Path = DATASET_PATH) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["category"] = df["category"].astype(str)
    return df


def _row_to_metadata(doc_id: str, url: str, source_type: str, category: str, title: str,
                      word_count: int, content_hash: str) -> dict:
    return {
        "id": doc_id, "url": url, "source_type": source_type, "category": category,
        "title": title, "depth": 0, "parent_url": None, "discovered_at": None,
        "content_hash": content_hash, "is_exact_duplicate": 0, "is_near_duplicate": 0,
        "duplicate_of": None, "word_count": word_count, "doc_id": doc_id,
    }


def build_base_corpus(dataset_df: pd.DataFrame, include_api: bool = True
                       ) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Builds the unified (metadata_df, content_df) pair for the dataset
    source plus (optionally) the mock API source. depth=0 / parent_url=None
    for both, since neither was discovered via hyperlink traversal."""
    metadata_rows, content_rows = [], []

    for row in dataset_df.itertuples():
        chash = _content_hash(row.text)
        metadata_rows.append(_row_to_metadata(
            row.doc_id, f"dataset://bbc/{row.doc_id}", "dataset", row.category,
            row.title, len(str(row.text).split()), chash,
        ))
        content_rows.append({"id": row.doc_id, "raw_text": row.text, "clean_text": None})

    if include_api:
        for rec in fetch_latest_articles():
            chash = _content_hash(rec["text"])
            metadata_rows.append(_row_to_metadata(
                rec["id"], f"api://mocknews/{rec['id']}", "api", rec["category"],
                rec["title"], len(rec["text"].split()), chash,
            ))
            content_rows.append({"id": rec["id"], "raw_text": rec["text"], "clean_text": None})

    metadata_df = pd.DataFrame(metadata_rows, columns=METADATA_COLUMNS)
    content_df = pd.DataFrame(content_rows, columns=CONTENT_COLUMNS)
    return metadata_df, content_df


def merge_corpus(base_metadata: pd.DataFrame, base_content: pd.DataFrame,
                  new_metadata: pd.DataFrame, new_content: pd.DataFrame
                  ) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Merges crawl results into the working corpus, de-duplicating by id
    (first occurrence wins) -- keeps the corpus consistent across repeated
    crawls without growing unbounded on re-runs."""
    metadata_df = pd.concat([base_metadata, new_metadata], ignore_index=True).drop_duplicates(
        subset="id", keep="first")
    content_df = pd.concat([base_content, new_content], ignore_index=True).drop_duplicates(
        subset="id", keep="first")
    return metadata_df.reset_index(drop=True), content_df.reset_index(drop=True)


__all__ = ["load_dataset", "build_base_corpus", "merge_corpus", "DATASET_PATH"]
