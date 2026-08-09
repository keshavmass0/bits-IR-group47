"""
ranking.py
----------
Manual (numpy-only, no networkx dependency) implementations of PageRank and
HITS over the link graph produced by the crawler -- Section D: "Use any of
the algorithm (Page Rank/HITS) and display how ranking is important".
"""
from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np


def _build_adjacency(edges: List[Tuple[str, str]]) -> Tuple[List[str], Dict[str, int], np.ndarray]:
    nodes = sorted({u for e in edges for u in e})
    idx = {n: i for i, n in enumerate(nodes)}
    n = len(nodes)
    adj = np.zeros((n, n))
    for src, dst in edges:
        if src in idx and dst in idx:
            adj[idx[src], idx[dst]] = 1.0
    return nodes, idx, adj


def pagerank(edges: List[Tuple[str, str]], damping: float = 0.85, max_iter: int = 100,
             tol: float = 1e-8) -> Dict[str, float]:
    """Classic power-iteration PageRank with dangling-node redistribution."""
    nodes, idx, adj = _build_adjacency(edges)
    n = len(nodes)
    if n == 0:
        return {}

    out_degree = adj.sum(axis=1)
    dangling = (out_degree == 0).astype(float)
    # Row-normalized transition matrix (0 rows for dangling nodes, handled separately)
    safe_out_degree = np.where(out_degree == 0, 1.0, out_degree)
    transition = adj / safe_out_degree[:, None]   # transition[i, j] = P(i -> j)

    scores = np.full(n, 1.0 / n)
    for _ in range(max_iter):
        dangling_mass = float(scores @ dangling) / n   # redistribute dangling mass evenly
        new_scores = (1 - damping) / n + damping * (scores @ transition + dangling_mass)
        if np.abs(new_scores - scores).sum() < tol:
            scores = new_scores
            break
        scores = new_scores

    return {nodes[i]: float(scores[i]) for i in range(n)}


def hits(edges: List[Tuple[str, str]], max_iter: int = 100, tol: float = 1e-8
          ) -> Tuple[Dict[str, float], Dict[str, float]]:
    """HITS: returns (hub_scores, authority_scores), both L2-normalized."""
    nodes, idx, adj = _build_adjacency(edges)
    n = len(nodes)
    if n == 0:
        return {}, {}

    hub = np.ones(n)
    auth = np.ones(n)
    for _ in range(max_iter):
        new_auth = adj.T @ hub
        new_auth = new_auth / (np.linalg.norm(new_auth) or 1.0)
        new_hub = adj @ new_auth
        new_hub = new_hub / (np.linalg.norm(new_hub) or 1.0)
        if np.abs(new_hub - hub).sum() + np.abs(new_auth - auth).sum() < tol:
            hub, auth = new_hub, new_auth
            break
        hub, auth = new_hub, new_auth

    return ({nodes[i]: float(hub[i]) for i in range(n)},
            {nodes[i]: float(auth[i]) for i in range(n)})


def normalize_scores(scores: Dict[str, float]) -> Dict[str, float]:
    if not scores:
        return {}
    values = np.array(list(scores.values()))
    lo, hi = values.min(), values.max()
    if hi - lo < 1e-12:
        return {k: 0.0 for k in scores}
    return {k: (v - lo) / (hi - lo) for k, v in scores.items()}


def blend_scores(relevance: Dict[str, float], link_score: Dict[str, float],
                  alpha: float = 0.7) -> Dict[str, float]:
    """final = alpha * relevance + (1 - alpha) * link_importance, both normalized to [0,1]
    so link-based scores never overwhelm or get overwhelmed by relevance scale."""
    rel_n = normalize_scores(relevance)
    link_n = normalize_scores(link_score)
    keys = set(rel_n) | set(link_n)
    return {k: alpha * rel_n.get(k, 0.0) + (1 - alpha) * link_n.get(k, 0.0) for k in keys}


__all__ = ["pagerank", "hits", "normalize_scores", "blend_scores"]
