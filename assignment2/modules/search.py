"""
search.py
---------
Section D: "intelligent web search module that supports diverse query
processing, ranked document retrieval, and efficient search over indexed
collections", including query optimization (synonym expansion + spelling
correction) and a demonstration of why ranking (PageRank/HITS) matters.
"""
from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

from modules.indexing import SearchIndex, candidate_positions_for_terms
from modules.preprocessing import preprocess, stem
from modules.ranking import blend_scores

# A small built-in synonym/abbreviation map -- "advanced query optimization
# techniques" (Section D) without needing WordNet/internet downloads.
QUERY_SYNONYMS: Dict[str, List[str]] = {
    "ai": ["artificial", "intelligence"],
    "govt": ["government"],
    "econ": ["economic", "economy"],
    "tech": ["technology", "technological"],
    "pm": ["prime", "minister"],
    "cup": ["tournament", "championship"],
    "movie": ["film"],
    "film": ["movie"],
    "job": ["employment"],
    "match": ["game", "fixture"],
}


def _edit_distance(a: str, b: str) -> int:
    if a == b:
        return 0
    dp = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        prev, dp[0] = dp[0], i
        for j, cb in enumerate(b, 1):
            cur = dp[j]
            dp[j] = prev if ca == cb else 1 + min(prev, dp[j], dp[j - 1])
            prev = cur
    return dp[-1]


def correct_spelling(term: str, vocabulary: List[str], max_distance: int = 2) -> Optional[str]:
    """Suggests the closest in-vocabulary term for an out-of-vocabulary query word."""
    if term in vocabulary:
        return term
    best, best_dist = None, max_distance + 1
    for v in vocabulary:
        if abs(len(v) - len(term)) > max_distance:
            continue
        d = _edit_distance(term, v)
        if d < best_dist:
            best, best_dist = v, d
    return best if best_dist <= max_distance else None


def process_query(query: str, index: SearchIndex, normalize: Optional[str] = None,
                   expand_synonyms: bool = True, correct_typos: bool = True) -> Dict:
    """Normalizes, expands and spell-corrects a raw query string.

    `normalize` defaults to the SAME normalization the index was built with
    (index.normalize) so query terms and indexed terms live in the same
    vocabulary space -- e.g. if the index was built with stemming, query
    terms are stemmed too before lookup.
    """
    normalize = index.normalize if normalize is None else normalize
    raw_tokens = preprocess(query, use_stopwords=True, normalize="none")
    vocabulary = list(index.tfidf_vectorizer.vocabulary_.keys())

    expanded: List[str] = []
    corrections: Dict[str, str] = {}
    for tok in raw_tokens:
        candidates = [tok]
        if expand_synonyms and tok in QUERY_SYNONYMS:
            candidates.extend(QUERY_SYNONYMS[tok])
        for c in candidates:
            term = stem(c) if normalize == "stem" else c
            if correct_typos and term not in vocabulary:
                fixed = correct_spelling(term, vocabulary)
                if fixed and fixed != term:
                    corrections[term] = fixed
                    term = fixed
            expanded.append(term)

    return {"raw_tokens": raw_tokens, "expanded_terms": list(dict.fromkeys(expanded)),
            "corrections": corrections}


def search(query: str, index: SearchIndex, normalize: Optional[str] = None,
           top_k: int = 10, expand_synonyms: bool = True, correct_typos: bool = True
           ) -> pd.DataFrame:
    """Two-stage retrieval: (1) inverted-index boolean OR pre-filter for
    efficiency, (2) TF-IDF cosine-similarity ranking over the candidates."""
    parsed = process_query(query, index, normalize, expand_synonyms, correct_typos)
    terms = parsed["expanded_terms"]
    if not terms:
        return pd.DataFrame(columns=["doc_id", "relevance_score"])

    candidate_positions = candidate_positions_for_terms(index, terms)
    if not candidate_positions:
        return pd.DataFrame(columns=["doc_id", "relevance_score"])

    query_vec = index.tfidf_vectorizer.transform([" ".join(terms)])
    positions = sorted(candidate_positions)
    sims = cosine_similarity(query_vec, index.tfidf_matrix[positions]).ravel()

    ranked = sorted(zip(positions, sims), key=lambda x: x[1], reverse=True)[:top_k]
    rows = [{"doc_id": index.doc_ids[pos], "relevance_score": round(float(score), 4)}
            for pos, score in ranked if score > 0]
    return pd.DataFrame(rows)


def rerank_with_link_scores(base_results: pd.DataFrame, link_scores: Dict[str, float],
                             alpha: float = 0.7) -> pd.DataFrame:
    """Blends baseline relevance with a link-importance score (PageRank or HITS
    authority) to demonstrate why ranking matters -- Section D requirement.
    """
    if base_results.empty:
        return base_results
    relevance = dict(zip(base_results["doc_id"], base_results["relevance_score"]))
    blended = blend_scores(relevance, link_scores, alpha=alpha)

    before = base_results.sort_values("relevance_score", ascending=False).reset_index(drop=True)
    old_rank = {doc_id: i + 1 for i, doc_id in enumerate(before["doc_id"])}

    out = base_results.copy()
    out["link_score_raw"] = out["doc_id"].map(link_scores).fillna(0.0)
    out["blended_score"] = out["doc_id"].map(blended).fillna(out["relevance_score"])
    out = out.sort_values("blended_score", ascending=False).reset_index(drop=True)
    out["old_rank"] = out["doc_id"].map(old_rank)
    out["new_rank"] = out.index + 1
    out["rank_change"] = out["old_rank"] - out["new_rank"]   # positive = moved up
    return out


__all__ = ["process_query", "search", "rerank_with_link_scores", "correct_spelling", "QUERY_SYNONYMS"]
