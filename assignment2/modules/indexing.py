"""
indexing.py
-----------
Index management (Streamlit "Index management" panel): builds and reports
on an inverted index (for fast boolean pre-filtering) and a TF-IDF matrix
(for ranked cosine-similarity retrieval). Kept as a small dataclass so the
Streamlit app can build it once per corpus version and cache it.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Set

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

from modules.preprocessing import preprocess


@dataclass
class SearchIndex:
    doc_ids: List[str]
    tfidf_vectorizer: TfidfVectorizer
    tfidf_matrix: any
    inverted_index: Dict[str, Set[int]] = field(default_factory=dict)  # term -> set of row positions
    postings_count: Dict[str, int] = field(default_factory=dict)
    doc_id_to_pos: Dict[str, int] = field(default_factory=dict)
    normalize: str = "none"
    use_stopwords: bool = True

    def stats(self) -> Dict:
        vocab = self.tfidf_vectorizer.get_feature_names_out()
        total_postings = sum(len(v) for v in self.inverted_index.values())
        nnz = self.tfidf_matrix.nnz
        density = round(nnz / (self.tfidf_matrix.shape[0] * self.tfidf_matrix.shape[1]) * 100, 4)
        return {
            "documents": len(self.doc_ids),
            "vocabulary_size": len(vocab),
            "total_postings": total_postings,
            "avg_postings_per_term": round(total_postings / max(len(vocab), 1), 2),
            "tfidf_nonzero_entries": int(nnz),
            "tfidf_matrix_density_pct": density,
        }

    def top_terms_by_document_frequency(self, top_n: int = 15) -> pd.DataFrame:
        rows = [(term, len(docs)) for term, docs in self.inverted_index.items()]
        rows.sort(key=lambda r: r[1], reverse=True)
        return pd.DataFrame(rows[:top_n], columns=["term", "document_frequency"])


def build_index(doc_ids: List[str], texts: List[str], use_stopwords: bool = True,
                 normalize: str = "none", max_features: int = 8000) -> SearchIndex:
    """Builds the inverted index + TF-IDF matrix over already-cleaned text.

    `normalize` controls stemming/lemmatization applied before indexing, so
    the Index Management panel can demonstrate its effect on vocabulary size.
    """
    processed_docs = [preprocess(t, use_stopwords=use_stopwords, normalize=normalize) for t in texts]
    processed_strings = [" ".join(toks) for toks in processed_docs]

    vec = TfidfVectorizer(max_features=max_features)
    matrix = vec.fit_transform(processed_strings)

    inverted: Dict[str, Set[int]] = defaultdict(set)
    for pos, toks in enumerate(processed_docs):
        for tok in set(toks):
            inverted[tok].add(pos)

    return SearchIndex(
        doc_ids=list(doc_ids),
        tfidf_vectorizer=vec,
        tfidf_matrix=matrix,
        inverted_index=dict(inverted),
        postings_count={term: len(docs) for term, docs in inverted.items()},
        doc_id_to_pos={doc_id: i for i, doc_id in enumerate(doc_ids)},
        normalize=normalize,
        use_stopwords=use_stopwords,
    )


def candidate_positions_for_terms(index: SearchIndex, terms: List[str]) -> Set[int]:
    """Boolean OR pre-filter over the inverted index -- the fast first stage
    of the two-stage retrieval pipeline used in search.py."""
    positions: Set[int] = set()
    for term in terms:
        positions |= index.inverted_index.get(term, set())
    return positions


__all__ = ["SearchIndex", "build_index", "candidate_positions_for_terms"]
