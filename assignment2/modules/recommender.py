"""
recommender.py
--------------
Section E: content-based, collaborative and hybrid recommendation, each
returning Top-K items with similarity/predicted scores.

There is no real user-interaction log available for this corpus (it is a
static news dataset), so -- exactly as with the mock API in mock_api.py --
we generate a reproducible SYNTHETIC user-item rating matrix to demonstrate
collaborative filtering mechanics. This is disclosed clearly in the README
and report; it is standard practice for coursework recommenders when no
real implicit/explicit feedback log exists. Content-based recommendation
uses only the real corpus (TF-IDF vectors), no synthetic data involved.
"""
from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity


# ------------------------------------------------------------------
# Content-based recommendation
# ------------------------------------------------------------------
def content_based_recommend(doc_id: str, doc_ids: List[str], tfidf_matrix, top_k: int = 5
                             ) -> pd.DataFrame:
    if doc_id not in doc_ids:
        return pd.DataFrame(columns=["doc_id", "similarity_score"])
    idx = doc_ids.index(doc_id)
    sims = cosine_similarity(tfidf_matrix[idx], tfidf_matrix).ravel()
    order = np.argsort(sims)[::-1]
    rows = [{"doc_id": doc_ids[i], "similarity_score": round(float(sims[i]), 4)}
            for i in order if doc_ids[i] != doc_id][:top_k]
    return pd.DataFrame(rows)


# ------------------------------------------------------------------
# Synthetic user-item ratings (for collaborative filtering demo)
# ------------------------------------------------------------------
def generate_synthetic_ratings(metadata_df: pd.DataFrame, n_users: int = 60,
                                ratings_per_user: int = 15, seed: int = 47) -> pd.DataFrame:
    """Each synthetic user has 1-2 favourite categories and rates a mix of
    docs from those categories (4-5 stars) and a few random docs (1-3 stars),
    mimicking realistic preference-biased implicit/explicit feedback."""
    rng = np.random.RandomState(seed)
    categories = sorted(metadata_df["category"].unique())
    doc_ids_by_cat = {c: metadata_df.loc[metadata_df["category"] == c, "id"].tolist() for c in categories}

    rows = []
    for u in range(n_users):
        user_id = f"U{u + 1:03d}"
        favourite_cats = rng.choice(categories, size=min(2, len(categories)), replace=False)
        n_fav = int(ratings_per_user * 0.7)
        n_rand = ratings_per_user - n_fav

        fav_pool = [d for c in favourite_cats for d in doc_ids_by_cat[c]]
        if fav_pool:
            fav_docs = rng.choice(fav_pool, size=min(n_fav, len(fav_pool)), replace=False)
            for d in fav_docs:
                rows.append({"user_id": user_id, "doc_id": d, "rating": int(rng.choice([4, 5]))})

        rand_pool = metadata_df["id"].tolist()
        rand_docs = rng.choice(rand_pool, size=min(n_rand, len(rand_pool)), replace=False)
        for d in rand_docs:
            rows.append({"user_id": user_id, "doc_id": d, "rating": int(rng.choice([1, 2, 3]))})

    df = pd.DataFrame(rows).drop_duplicates(subset=["user_id", "doc_id"])
    return df.reset_index(drop=True)


def _build_rating_matrix(ratings_df: pd.DataFrame):
    users = sorted(ratings_df["user_id"].unique())
    docs = sorted(ratings_df["doc_id"].unique())
    u_idx = {u: i for i, u in enumerate(users)}
    d_idx = {d: i for i, d in enumerate(docs)}
    matrix = np.zeros((len(users), len(docs)))
    for row in ratings_df.itertuples():
        matrix[u_idx[row.user_id], d_idx[row.doc_id]] = row.rating
    return matrix, users, docs, u_idx, d_idx


# ------------------------------------------------------------------
# Collaborative filtering (item-based kNN + latent-factor SVD)
# ------------------------------------------------------------------
def collaborative_recommend(user_id: str, ratings_df: pd.DataFrame, top_k: int = 5,
                             n_components: int = 20) -> pd.DataFrame:
    """Latent-factor collaborative filtering: TruncatedSVD on the user-item
    matrix, predicted rating = reconstructed matrix entry. Recommends the
    highest-predicted UNRATED documents for the user."""
    matrix, users, docs, u_idx, d_idx = _build_rating_matrix(ratings_df)
    if user_id not in u_idx or matrix.shape[1] < 2:
        return pd.DataFrame(columns=["doc_id", "predicted_rating"])

    k = max(1, min(n_components, min(matrix.shape) - 1))
    svd = TruncatedSVD(n_components=k, random_state=47)
    user_factors = svd.fit_transform(matrix)
    reconstructed = user_factors @ svd.components_

    uidx = u_idx[user_id]
    already_rated = set(ratings_df.loc[ratings_df["user_id"] == user_id, "doc_id"])
    scores = reconstructed[uidx]

    order = np.argsort(scores)[::-1]
    rows = []
    for i in order:
        doc_id = docs[i]
        if doc_id in already_rated:
            continue
        rows.append({"doc_id": doc_id, "predicted_rating": round(float(scores[i]), 3)})
        if len(rows) >= top_k:
            break
    return pd.DataFrame(rows)


# ------------------------------------------------------------------
# Hybrid recommendation
# ------------------------------------------------------------------
def hybrid_recommend(user_id: str, seed_doc_id: Optional[str], ratings_df: pd.DataFrame,
                      doc_ids: List[str], tfidf_matrix, top_k: int = 5, alpha: float = 0.5
                      ) -> pd.DataFrame:
    """Combines a normalized content-based similarity score (from a seed
    document the user is currently viewing) with a normalized collaborative
    predicted-rating score. alpha weighs content vs collaborative."""
    cb = content_based_recommend(seed_doc_id, doc_ids, tfidf_matrix, top_k=len(doc_ids)) \
        if seed_doc_id else pd.DataFrame(columns=["doc_id", "similarity_score"])
    cf = collaborative_recommend(user_id, ratings_df, top_k=len(doc_ids))

    def _minmax(series: pd.Series) -> pd.Series:
        if series.empty or series.max() == series.min():
            return series * 0.0
        return (series - series.min()) / (series.max() - series.min())

    cb_scores = dict(zip(cb["doc_id"], _minmax(cb["similarity_score"]))) if not cb.empty else {}
    cf_scores = dict(zip(cf["doc_id"], _minmax(cf["predicted_rating"]))) if not cf.empty else {}

    already_rated = set(ratings_df.loc[ratings_df["user_id"] == user_id, "doc_id"])
    candidates = (set(cb_scores) | set(cf_scores)) - already_rated - {seed_doc_id}

    rows = []
    for doc_id in candidates:
        score = alpha * cb_scores.get(doc_id, 0.0) + (1 - alpha) * cf_scores.get(doc_id, 0.0)
        rows.append({
            "doc_id": doc_id,
            "content_score": round(cb_scores.get(doc_id, 0.0), 3),
            "collaborative_score": round(cf_scores.get(doc_id, 0.0), 3),
            "hybrid_score": round(score, 3),
        })
    out = pd.DataFrame(rows).sort_values("hybrid_score", ascending=False).head(top_k).reset_index(drop=True)
    return out


__all__ = ["content_based_recommend", "generate_synthetic_ratings", "collaborative_recommend",
           "hybrid_recommend"]
