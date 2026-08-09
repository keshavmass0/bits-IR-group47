"""
evaluation.py
-------------
Section F: Precision, Recall, F1, Precision@K, Recall@K, MAP, MRR, NDCG,
plus a small ground-truth query set (built deterministically from the
corpus's category labels) so the metrics have real relevance judgments to
compute against, and a helper to compare multiple ranking methods.
"""
from __future__ import annotations

import math
from typing import Callable, Dict, List

import pandas as pd

# A representative, category-anchored query set. Relevance judgments are
# derived deterministically at runtime (see build_eval_queries): a document
# is judged RELEVANT to a query if it belongs to the query's target category
# AND contains at least one query term -- a standard, reproducible way to
# manufacture ground truth for a categorized news corpus without manual
# per-document annotation.
DEFAULT_QUERY_SEEDS = [
    ("stock market economic growth", "business"),
    ("company profits investment", "business"),
    ("government policy reform", "politics"),
    ("prime minister election", "politics"),
    ("football team tournament match", "sport"),
    ("player championship victory", "sport"),
    ("artificial intelligence technology", "tech"),
    ("software cloud computing data", "tech"),
    ("film actor award", "entertainment"),
    ("music concert celebrity", "entertainment"),
]


def build_eval_queries(metadata_df: pd.DataFrame, content_df: pd.DataFrame,
                        min_relevant: int = 3) -> List[Dict]:
    merged = metadata_df.merge(content_df, on="id", how="inner")
    queries = []
    for query_text, category in DEFAULT_QUERY_SEEDS:
        terms = query_text.lower().split()
        in_cat = merged[merged["category"] == category]
        mask = in_cat["raw_text"].str.lower().apply(lambda t: any(term in t for term in terms))
        relevant_ids = in_cat.loc[mask, "id"].tolist()
        if len(relevant_ids) >= min_relevant:
            queries.append({"query": query_text, "category": category, "relevant_ids": set(relevant_ids)})
    return queries


# ------------------------------------------------------------------
# Core metrics
# ------------------------------------------------------------------
def precision(retrieved: List[str], relevant: set) -> float:
    if not retrieved:
        return 0.0
    return len(set(retrieved) & relevant) / len(retrieved)


def recall(retrieved: List[str], relevant: set) -> float:
    if not relevant:
        return 0.0
    return len(set(retrieved) & relevant) / len(relevant)


def f1_score_(retrieved: List[str], relevant: set) -> float:
    p, r = precision(retrieved, relevant), recall(retrieved, relevant)
    return 2 * p * r / (p + r) if (p + r) else 0.0


def precision_at_k(ranked: List[str], relevant: set, k: int) -> float:
    return precision(ranked[:k], relevant)


def recall_at_k(ranked: List[str], relevant: set, k: int) -> float:
    return recall(ranked[:k], relevant)


def average_precision(ranked: List[str], relevant: set) -> float:
    """AP for one query; mean over queries = MAP."""
    if not relevant:
        return 0.0
    hits, score = 0, 0.0
    for i, doc_id in enumerate(ranked, 1):
        if doc_id in relevant:
            hits += 1
            score += hits / i
    return score / len(relevant) if relevant else 0.0


def reciprocal_rank(ranked: List[str], relevant: set) -> float:
    for i, doc_id in enumerate(ranked, 1):
        if doc_id in relevant:
            return 1.0 / i
    return 0.0


def ndcg_at_k(ranked: List[str], relevant: set, k: int) -> float:
    """Binary-relevance NDCG@K."""
    dcg = sum((1.0 if doc_id in relevant else 0.0) / math.log2(i + 1)
              for i, doc_id in enumerate(ranked[:k], 1))
    ideal_hits = min(len(relevant), k)
    idcg = sum(1.0 / math.log2(i + 1) for i in range(1, ideal_hits + 1))
    return dcg / idcg if idcg > 0 else 0.0


def evaluate_ranked_list(ranked: List[str], relevant: set, k: int = 10) -> Dict[str, float]:
    return {
        "precision": round(precision(ranked, relevant), 4),
        "recall": round(recall(ranked, relevant), 4),
        "f1": round(f1_score_(ranked, relevant), 4),
        f"precision@{k}": round(precision_at_k(ranked, relevant, k), 4),
        f"recall@{k}": round(recall_at_k(ranked, relevant, k), 4),
        "average_precision": round(average_precision(ranked, relevant), 4),
        "reciprocal_rank": round(reciprocal_rank(ranked, relevant), 4),
        f"ndcg@{k}": round(ndcg_at_k(ranked, relevant, k), 4),
    }


def evaluate_system(queries: List[Dict], search_fn: Callable[[str], List[str]], k: int = 10
                     ) -> pd.DataFrame:
    """search_fn(query_text) -> ranked list of doc_ids. Returns a per-query
    metrics table plus a final 'MEAN' row (this is MAP/MRR/etc. aggregated)."""
    rows = []
    for q in queries:
        ranked = search_fn(q["query"])
        metrics = evaluate_ranked_list(ranked, q["relevant_ids"], k=k)
        rows.append({"query": q["query"], "category": q["category"], **metrics})
    df = pd.DataFrame(rows)
    if not df.empty:
        mean_row = {"query": "MEAN", "category": "-"}
        mean_row.update(df.drop(columns=["query", "category"]).mean(numeric_only=True).round(4).to_dict())
        df = pd.concat([df, pd.DataFrame([mean_row])], ignore_index=True)
    return df


def compare_methods(queries: List[Dict], method_search_fns: Dict[str, Callable[[str], List[str]]],
                     k: int = 10) -> pd.DataFrame:
    """Runs evaluate_system for each named ranking method and returns a single
    comparison table of MEAN metrics -- Section F: "comparative analysis
    using tables and visualizations"."""
    rows = []
    for name, fn in method_search_fns.items():
        df = evaluate_system(queries, fn, k=k)
        mean_row = df[df["query"] == "MEAN"].iloc[0].to_dict()
        mean_row["method"] = name
        mean_row.pop("query", None)
        mean_row.pop("category", None)
        rows.append(mean_row)
    cols = ["method"] + [c for c in rows[0] if c != "method"] if rows else []
    return pd.DataFrame(rows, columns=cols if rows else None)


__all__ = ["build_eval_queries", "precision", "recall", "f1_score_", "precision_at_k",
           "recall_at_k", "average_precision", "reciprocal_rank", "ndcg_at_k",
           "evaluate_ranked_list", "evaluate_system", "compare_methods", "DEFAULT_QUERY_SEEDS"]
