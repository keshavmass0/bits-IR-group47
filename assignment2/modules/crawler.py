"""
crawler.py
----------
A breadth-first crawler satisfying Section B:
  * configurable crawling depth
  * multiple seed sources
  * duplicate URL handling
  * duplicate document (content) handling
  * metadata stored separately from content (see storage.py)

The crawler is generic graph-traversal code: `fetch_fn(url)` is the only
coupling point to *how* a page is retrieved, so it works identically whether
pages come from the offline mock_web graph or a live HTTP fetch
(modules/mock_web.try_live_fetch).
"""
from __future__ import annotations

import re
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Callable, Dict, List, Optional, Set, Tuple

import pandas as pd

from modules.mock_web import Page, _content_hash


def _tokens(text: str) -> Set[str]:
    return set(re.findall(r"[a-z0-9]+", text.lower()))


def _jaccard(a: Set[str], b: Set[str]) -> float:
    if not a or not b:
        return 0.0
    inter = len(a & b)
    union = len(a | b)
    return inter / union if union else 0.0


@dataclass
class CrawlStats:
    pages_fetched: int = 0
    pages_skipped_url_dup: int = 0
    pages_skipped_content_dup: int = 0
    near_duplicates_flagged: int = 0
    max_depth_reached: int = 0


def crawl(seeds: List[str], max_depth: int, max_pages: int,
          fetch_fn: Callable[[str], Optional[Page]],
          near_dup_threshold: float = 0.85,
          skip_near_duplicates: bool = False,
          started_at: Optional[datetime] = None
          ) -> Tuple[pd.DataFrame, pd.DataFrame, List[Tuple[str, str]], CrawlStats]:
    """BFS crawl from `seeds` up to `max_depth`, capped at `max_pages`.

    Returns (metadata_df, content_df, edges, stats).
    """
    started_at = started_at or datetime.now()
    visited_urls: Set[str] = set()
    seen_hashes: Dict[str, str] = {}          # content_hash -> first url that had it
    seen_token_sets: List[Tuple[str, Set[str]]] = []  # (url, tokens) for near-dup checks

    metadata_rows: List[dict] = []
    content_rows: List[dict] = []
    edges: List[Tuple[str, str]] = []
    stats = CrawlStats()

    queue = deque((s, 0, None) for s in seeds)
    tick = 0

    while queue and stats.pages_fetched < max_pages:
        url, depth, parent = queue.popleft()

        # --- Duplicate URL handling ---
        if url in visited_urls:
            stats.pages_skipped_url_dup += 1
            continue
        if depth > max_depth:
            continue
        visited_urls.add(url)

        page = fetch_fn(url)
        if page is None:
            continue

        stats.pages_fetched += 1
        stats.max_depth_reached = max(stats.max_depth_reached, depth)
        tick += 1
        discovered_at = started_at + timedelta(seconds=tick)

        chash = _content_hash(page.text)
        is_exact_dup = page.is_exact_duplicate or chash in seen_hashes
        duplicate_of = page.duplicate_of or seen_hashes.get(chash)

        is_near_dup = page.is_near_duplicate
        if not is_exact_dup and not is_near_dup:
            toks = _tokens(page.text)
            for other_url, other_toks in seen_token_sets:
                if _jaccard(toks, other_toks) >= near_dup_threshold:
                    is_near_dup = True
                    duplicate_of = duplicate_of or other_url
                    break
            seen_token_sets.append((url, toks))

        if is_exact_dup:
            stats.pages_skipped_content_dup += 1
        if is_near_dup:
            stats.near_duplicates_flagged += 1

        if chash not in seen_hashes:
            seen_hashes[chash] = url

        # Skip storing content for exact duplicates (nothing new to index);
        # near-duplicates are kept but flagged, so later stages (indexing,
        # ranking, recommendation, evaluation) can be studied WITH and
        # WITHOUT them, per Section G Q2.
        keep_content = not is_exact_dup and not (skip_near_duplicates and is_near_dup)

        metadata_rows.append({
            "id": url, "url": url, "source_type": page.source_type,
            "category": page.category, "title": page.title, "depth": depth,
            "parent_url": parent, "discovered_at": discovered_at.isoformat(),
            "content_hash": chash, "is_exact_duplicate": int(is_exact_dup),
            "is_near_duplicate": int(is_near_dup), "duplicate_of": duplicate_of,
            "word_count": len(page.text.split()), "doc_id": page.doc_id,
        })
        if keep_content:
            content_rows.append({"id": url, "raw_text": page.text, "clean_text": None})

        for link in page.outlinks:
            edges.append((url, link))
            if depth + 1 <= max_depth:
                queue.append((link, depth + 1, url))

    metadata_df = pd.DataFrame(metadata_rows)
    content_df = pd.DataFrame(content_rows)
    return metadata_df, content_df, edges, stats


__all__ = ["crawl", "CrawlStats"]
