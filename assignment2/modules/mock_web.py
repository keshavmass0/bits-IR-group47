"""
mock_web.py
-----------
A deterministic, fully offline "web" built on top of the BBC-style news corpus.

WHY THIS EXISTS (design rationale — see README/report for full discussion):
The BITS Virtual Lab that this assignment must run on typically has no outbound
internet access, so a crawler that only works against live websites would fail
during evaluation. Assignment 2 (Section B) only requires *demonstrating* the
crawling mechanics (configurable depth, multiple seeds, duplicate/URL handling,
metadata-vs-content separation) over "one or more heterogeneous sources through
web crawling, publicly available datasets, APIs, or a combination of these".

We satisfy this honestly and robustly by building a real hyperlinked graph of
pages ("local web") out of the shipped news dataset, and running a genuine
breadth-first crawler (modules/crawler.py) over it. The crawler code itself is
generic BFS/graph-traversal code -- it does not know or care whether `fetch()`
is backed by a local graph or a live HTTP GET. If real internet access *is*
available (e.g. running locally, not in the sandboxed lab), `try_live_fetch()`
below opportunistically pulls a handful of real Wikipedia pages using only the
Python standard library (urllib) -- no extra network dependency (requests/bs4)
is required, so nothing new needs to be installed on the lab machine.

The local web graph intentionally contains:
  * a home page linking to 5 category hub pages,
  * category hub pages linking to every article in that category,
  * article-to-article "related story" links within a category (creates a
    realistic authority/hub structure for PageRank & HITS in modules/ranking.py),
  * a handful of injected EXACT-duplicate pages (identical content served at
    a different URL/mirror) and NEAR-duplicate pages (same story, lightly
    reworded) so the crawler's deduplication logic (Section B) and the
    duplicate-impact discussion (Section G, Q2) have real data to work with.
"""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional
from urllib.request import urlopen
from urllib.error import URLError

import pandas as pd

HOME_URL = "local-news://home"


@dataclass
class Page:
    url: str
    title: str
    text: str
    category: str
    source_type: str  # "dataset" | "crawl" | "api"
    outlinks: List[str] = field(default_factory=list)
    is_exact_duplicate: bool = False
    is_near_duplicate: bool = False
    duplicate_of: Optional[str] = None
    doc_id: Optional[str] = None  # links back to the underlying corpus doc_id (None for home/hub nodes)


def _content_hash(text: str) -> str:
    normalized = re.sub(r"\s+", " ", text.strip().lower())
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:16]


def build_web_graph(corpus_df: pd.DataFrame, near_dup_every: int = 20,
                     exact_dup_every: int = 25) -> Dict[str, Page]:
    """Builds the deterministic local web graph from the base news corpus.

    Returns a dict mapping url -> Page. Deterministic given the same corpus_df
    (no randomness), so repeated crawls / grading runs are reproducible.
    """
    graph: Dict[str, Page] = {}
    categories = sorted(corpus_df["category"].unique())

    # Home page -> category hubs
    hub_urls = [f"local-news://{cat}/hub" for cat in categories]
    graph[HOME_URL] = Page(
        url=HOME_URL, title="SmartNews Home", text="Front page linking to all news sections.",
        category="home", source_type="dataset", outlinks=hub_urls,
    )

    by_category: Dict[str, List[str]] = {cat: [] for cat in categories}
    doc_urls: List[str] = []

    for cat in categories:
        rows = corpus_df[corpus_df["category"] == cat].reset_index(drop=True)
        urls = [f"local-news://{cat}/{row.doc_id}" for row in rows.itertuples()]
        by_category[cat] = urls
        doc_urls.extend(urls)

        # Hub page links to every article in this category
        graph[f"local-news://{cat}/hub"] = Page(
            url=f"local-news://{cat}/hub", title=f"{cat.title()} Section",
            text=f"Section hub aggregating all {cat} coverage.",
            category=cat, source_type="dataset", outlinks=list(urls),
        )

        # Article pages, each linking to 2-3 "related stories" in the same
        # category (creates authority clusters for PageRank/HITS).
        n = len(rows)
        for i, row in enumerate(rows.itertuples()):
            url = urls[i]
            related = [urls[(i + off) % n] for off in (1, 2, 3) if n > 3]
            graph[url] = Page(
                url=url, title=row.title, text=row.text, category=cat,
                source_type="dataset", outlinks=related, doc_id=row.doc_id,
            )

    # --- Inject EXACT duplicates: identical content served at a mirror URL ---
    for idx, url in enumerate(doc_urls):
        if exact_dup_every and idx % exact_dup_every == 0:
            src = graph[url]
            mirror_url = url + "-mirror"
            graph[mirror_url] = Page(
                url=mirror_url, title=src.title, text=src.text, category=src.category,
                source_type="dataset", outlinks=[], is_exact_duplicate=True, duplicate_of=url,
                doc_id=src.doc_id,
            )
            graph[f"local-news://{src.category}/hub"].outlinks.append(mirror_url)

    # --- Inject NEAR duplicates: same story, lightly reworded / extended ---
    for idx, url in enumerate(doc_urls):
        if near_dup_every and idx % near_dup_every == 0:
            src = graph[url]
            near_url = url + "-syndicated"
            reworded = src.text + " This report was later updated with additional wire-service coverage."
            graph[near_url] = Page(
                url=near_url, title=src.title + " (syndicated)", text=reworded, category=src.category,
                source_type="dataset", outlinks=[], is_near_duplicate=True, duplicate_of=url,
                doc_id=src.doc_id,
            )
            graph[f"local-news://{src.category}/hub"].outlinks.append(near_url)

    return graph


def list_seed_options(graph: Dict[str, Page]) -> List[str]:
    """Seeds an evaluator can pick from in the Streamlit crawling interface."""
    hubs = sorted(u for u in graph if u.endswith("/hub"))
    return [HOME_URL] + hubs


_LINK_RE = re.compile(r'href=["\'](https?://[^"\'#]+)', re.IGNORECASE)
_TAG_RE = re.compile(r"<[^>]+>")


def try_live_fetch(url: str, timeout: float = 3.0, max_links: int = 15) -> Optional[Page]:
    """Best-effort real-internet fetch using only the stdlib (no extra deps).

    Used only when the evaluator explicitly enables "Attempt live internet
    crawl" in the UI. Returns None on any failure (offline lab, blocked
    egress, bad URL, etc.) so callers can fall back to the local web graph.
    """
    try:
        with urlopen(url, timeout=timeout) as resp:  # noqa: S310 - explicit opt-in, read-only GET
            raw = resp.read(200_000).decode("utf-8", errors="ignore")
    except (URLError, ValueError, OSError, Exception):
        return None

    links = list(dict.fromkeys(_LINK_RE.findall(raw)))[:max_links]
    text = _TAG_RE.sub(" ", raw)
    text = re.sub(r"\s+", " ", text).strip()[:4000]
    title_match = re.search(r"<title[^>]*>(.*?)</title>", raw, re.IGNORECASE | re.DOTALL)
    title = _TAG_RE.sub("", title_match.group(1)).strip() if title_match else url
    return Page(url=url, title=title or url, text=text, category="live-web",
                source_type="crawl", outlinks=links)


def fetch(url: str, graph: Dict[str, Page], allow_live: bool = False) -> Optional[Page]:
    """Unified fetch used by the crawler: local graph first, optional live fallback."""
    if url in graph:
        return graph[url]
    if allow_live and url.startswith("http"):
        return try_live_fetch(url)
    return None


__all__ = ["Page", "HOME_URL", "build_web_graph", "list_seed_options", "fetch",
           "try_live_fetch", "_content_hash"]
