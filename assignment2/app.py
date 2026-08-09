"""
app.py -- SmartIR-2: End-to-End Information Retrieval System
==============================================================
BITS WILP Information Retrieval (S2-25) -- Assignment 2, Team 47.

A single Streamlit front end covering every required panel (Section A):
Dashboard, Crawling interface, Text Mining, Index management, Search &
Ranking visualization, Recommendation panel, Evaluation dashboard,
Performance analytics, and the compulsory Inference & Discussion section.

The ENTIRE workflow (crawl -> preprocess/mine -> index -> search/rank ->
recommend -> evaluate -> infer) runs from this front end only, as required.
See README.md for install/run instructions and design rationale (in
particular: why crawling is demonstrated over a deterministic local web
graph rather than live internet, so grading on the BITS virtual lab is
reliable with no network access required).
"""
from __future__ import annotations

import time
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st

from modules import corpus as corpus_mod
from modules import crawler as crawler_mod
from modules import evaluation as eval_mod
from modules import indexing as indexing_mod
from modules import mock_web
from modules import ranking as ranking_mod
from modules import recommender as rec_mod
from modules import search as search_mod
from modules import storage as storage_mod
from modules import text_mining as mining_mod
from modules.preprocessing import preprocess_to_string

st.set_page_config(page_title="SmartIR-2 | End-to-End IR System", page_icon="🧭", layout="wide")


# ----------------------------------------------------------------------
# Performance instrumentation (feeds the Performance Analytics tab)
# ----------------------------------------------------------------------
class timed:
    def __init__(self, action: str):
        self.action = action

    def __enter__(self):
        self.t0 = time.time()
        return self

    def __exit__(self, *exc):
        dt = time.time() - self.t0
        st.session_state.perf_log.append({
            "action": self.action, "duration_seconds": round(dt, 4),
            "timestamp": datetime.now().strftime("%H:%M:%S"),
        })


# ----------------------------------------------------------------------
# Cached loaders / one-time builders
# ----------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def _load_dataset_df():
    return corpus_mod.load_dataset()


@st.cache_resource(show_spinner=False)
def _build_web_graph(dataset_df: pd.DataFrame):
    return mock_web.build_web_graph(dataset_df)


def _init_session_state():
    if "perf_log" not in st.session_state:
        st.session_state.perf_log = []

    if "metadata_df" not in st.session_state:
        persisted_meta, persisted_content = storage_mod.load_corpus()
        if persisted_meta is not None and not persisted_meta.empty:
            st.session_state.metadata_df = persisted_meta
            st.session_state.content_df = persisted_content
        else:
            dataset_df = _load_dataset_df()
            with timed("build_base_corpus"):
                meta, content = corpus_mod.build_base_corpus(dataset_df, include_api=True)
            st.session_state.metadata_df = meta
            st.session_state.content_df = content
            storage_mod.save_corpus(meta, content)

    if "web_graph" not in st.session_state:
        st.session_state.web_graph = _build_web_graph(_load_dataset_df())

    st.session_state.setdefault("link_edges", [])
    st.session_state.setdefault("crawl_history", [])
    st.session_state.setdefault("index", None)
    st.session_state.setdefault("ratings_df", None)
    st.session_state.setdefault("eval_queries", None)
    st.session_state.setdefault("pagerank_scores", {})
    st.session_state.setdefault("hits_scores", ({}, {}))
    st.session_state.setdefault("corpus_size_history", [])


def get_index(force_rebuild: bool = False, use_stopwords: bool = True, normalize: str = "none",
              max_features: int = 8000) -> indexing_mod.SearchIndex:
    idx = st.session_state.index
    if idx is not None and not force_rebuild:
        return idx
    content_df = st.session_state.content_df.dropna(subset=["raw_text"])
    with timed("build_index"):
        idx = indexing_mod.build_index(
            content_df["id"].tolist(), content_df["raw_text"].astype(str).tolist(),
            use_stopwords=use_stopwords, normalize=normalize, max_features=max_features,
        )
    st.session_state.index = idx
    return idx


def get_ratings_df() -> pd.DataFrame:
    if st.session_state.ratings_df is None:
        with timed("generate_synthetic_ratings"):
            ratings = rec_mod.generate_synthetic_ratings(st.session_state.metadata_df)
        st.session_state.ratings_df = ratings
        ratings.to_csv(corpus_mod.DATA_DIR / "synthetic_ratings.csv", index=False)
    return st.session_state.ratings_df


def get_eval_queries():
    if st.session_state.eval_queries is None:
        st.session_state.eval_queries = eval_mod.build_eval_queries(
            st.session_state.metadata_df, st.session_state.content_df)
    return st.session_state.eval_queries


def doc_id_link_scores(url_scores: dict) -> dict:
    """Aggregates URL-keyed link-importance scores (PageRank/HITS) up to the
    underlying corpus doc_id, summing across mirrors/near-duplicates -- this
    is what makes duplicate pages visibly inflate a document's authority
    (Section G, Q2 evidence)."""
    meta = st.session_state.metadata_df
    url_to_doc = dict(zip(meta.get("url", meta["id"]), meta.get("doc_id", meta["id"])))
    doc_scores: dict = {}
    for url, score in url_scores.items():
        doc_id = url_to_doc.get(url)
        if doc_id:
            doc_scores[doc_id] = doc_scores.get(doc_id, 0.0) + score
    return doc_scores


_init_session_state()

# ========================================================================
st.title("🧭 SmartIR-2: End-to-End Information Retrieval System")
st.caption("BITS WILP Information Retrieval (S2-25) · Assignment 2 · Team 47 · "
           "Crawling → Text Mining → Indexing → Search & Ranking → Recommendation → Evaluation")

TABS = st.tabs([
    "🏠 Dashboard", "🕷️ Crawling", "🧪 Text Mining", "🗂️ Index Management",
    "🔍 Search & Ranking", "⭐ Recommendations", "📊 Evaluation",
    "📈 Performance Analytics", "🧠 Inference & Discussion",
])

# ------------------------------------------------------------------
# TAB 1: DASHBOARD
# ------------------------------------------------------------------
with TABS[0]:
    st.subheader("System Overview")
    meta = st.session_state.metadata_df

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Documents", len(meta))
    c2.metric("Categories", meta["category"].nunique())
    c3.metric("Sources", meta["source_type"].nunique())
    c4.metric("Exact duplicates flagged", int(meta["is_exact_duplicate"].sum()))
    c5.metric("Near-duplicates flagged", int(meta["is_near_duplicate"].sum()))

    left, right = st.columns(2)
    with left:
        st.markdown("**Documents by category**")
        st.bar_chart(meta["category"].value_counts())
    with right:
        st.markdown("**Documents by source type**")
        st.bar_chart(meta["source_type"].value_counts())

    st.markdown("---")
    st.markdown("""
**Pipeline covered end-to-end in this app** (use the tabs above, left to right):

`🕷️ Crawling` (Section B: heterogeneous sources, depth/seed control, dedup, metadata⧸content
separation) → `🧪 Text Mining` (Section C: preprocessing, keyword extraction, profiling,
classification, comparative analysis) → `🗂️ Index Management` (inverted index + TF-IDF) →
`🔍 Search & Ranking` (Section D: query optimization, ranked retrieval, PageRank/HITS) →
`⭐ Recommendations` (Section E: content-based, collaborative, hybrid) →
`📊 Evaluation` (Section F: Precision/Recall/F1/P@K/R@K/MAP/MRR/NDCG) →
`📈 Performance Analytics` (timing/scalability) → `🧠 Inference & Discussion` (Section G).
""")

    st.markdown("**Sample of the working corpus (metadata only -- content lives in a separate table, see Crawling tab)**")
    st.dataframe(meta[["id", "source_type", "category", "title", "depth", "word_count"]].head(10),
                 use_container_width=True)

# ------------------------------------------------------------------
# TAB 2: CRAWLING
# ------------------------------------------------------------------
with TABS[1]:
    st.subheader("Crawling Interface")
    st.info(
        "🌐 **Design note:** the BITS virtual lab has no outbound internet access, so this "
        "crawler runs a genuine breadth-first traversal over a deterministic **local web "
        "graph** built from the news corpus (home → category hubs → articles → related "
        "stories), with injected duplicate/near-duplicate mirror pages. The crawl logic "
        "(BFS, depth limits, dedup) is identical to what would run against live URLs -- see "
        "`modules/mock_web.py` / `modules/crawler.py`. An optional **live internet** toggle "
        "below lets you also crawl real URLs with the Python standard library when internet "
        "*is* available (e.g. running locally).",
        icon="🌐",
    )

    graph = st.session_state.web_graph
    seed_options = mock_web.list_seed_options(graph)

    col1, col2, col3 = st.columns(3)
    with col1:
        seeds = st.multiselect("Seed source(s)", seed_options, default=[mock_web.HOME_URL])
        max_depth = st.slider("Max crawl depth", 0, 4, 2)
    with col2:
        max_pages = st.slider("Max pages to fetch", 10, 1200, 200, step=10)
        near_dup_threshold = st.slider("Near-duplicate similarity threshold (Jaccard)", 0.5, 0.99, 0.85)
    with col3:
        skip_near = st.checkbox("Drop near-duplicates from content store", value=False)
        allow_live = st.checkbox("Attempt live internet crawl (optional, needs real URLs below)", value=False)

    live_urls: list = []
    if allow_live:
        raw = st.text_area("Real seed URL(s), one per line (only used if internet is reachable)",
                            placeholder="https://en.wikipedia.org/wiki/Information_retrieval")
        live_urls = [u.strip() for u in raw.splitlines() if u.strip()]

    if st.button("🚀 Run Crawl", type="primary"):
        all_seeds = list(seeds) + live_urls

        def fetch_fn(url: str):
            return mock_web.fetch(url, graph, allow_live=allow_live)

        with timed("crawl"):
            crawl_meta, crawl_content, edges, stats = crawler_mod.crawl(
                all_seeds, max_depth, max_pages, fetch_fn,
                near_dup_threshold=near_dup_threshold, skip_near_duplicates=skip_near,
            )

        st.session_state.metadata_df, st.session_state.content_df = corpus_mod.merge_corpus(
            st.session_state.metadata_df, st.session_state.content_df, crawl_meta, crawl_content)
        st.session_state.link_edges = list({*st.session_state.link_edges, *edges})
        st.session_state.crawl_history.append({
            "time": datetime.now().strftime("%H:%M:%S"), "seeds": ", ".join(all_seeds),
            "max_depth": max_depth, **stats.__dict__,
        })
        st.session_state.corpus_size_history.append({
            "time": datetime.now().strftime("%H:%M:%S"), "documents": len(st.session_state.metadata_df)})
        storage_mod.save_corpus(st.session_state.metadata_df, st.session_state.content_df)

        st.success(f"Crawl complete: {stats.pages_fetched} pages fetched.")
        s1, s2, s3, s4 = st.columns(4)
        s1.metric("Pages fetched", stats.pages_fetched)
        s2.metric("Skipped (URL dup)", stats.pages_skipped_url_dup)
        s3.metric("Skipped (content dup)", stats.pages_skipped_content_dup)
        s4.metric("Near-duplicates flagged", stats.near_duplicates_flagged)

        st.markdown("**Newly discovered pages (metadata)**")
        st.dataframe(crawl_meta[["id", "source_type", "category", "depth", "parent_url",
                                  "is_exact_duplicate", "is_near_duplicate", "word_count"]],
                     use_container_width=True, height=280)

    if st.session_state.crawl_history:
        st.markdown("---")
        st.markdown("**Crawl run history**")
        st.dataframe(pd.DataFrame(st.session_state.crawl_history), use_container_width=True)

    dup_rows = st.session_state.metadata_df[
        (st.session_state.metadata_df["is_exact_duplicate"] == 1) |
        (st.session_state.metadata_df["is_near_duplicate"] == 1)]
    if not dup_rows.empty:
        st.markdown("---")
        st.markdown("**Duplicate / near-duplicate inspector** (Section G, Q2 evidence)")
        st.dataframe(dup_rows[["id", "duplicate_of", "is_exact_duplicate", "is_near_duplicate"]],
                     use_container_width=True, height=200)

    st.caption(f"Persisted store: `data/ir_store.db` ({storage_mod.db_file_size_kb()} KB) -- "
               "metadata and content are two separate SQLite tables.")

# ------------------------------------------------------------------
# TAB 3: TEXT MINING
# ------------------------------------------------------------------
with TABS[2]:
    st.subheader("Text Preprocessing & Mining")
    working = st.session_state.metadata_df.merge(st.session_state.content_df, on="id", how="inner")
    working = working.dropna(subset=["raw_text"])

    left, right = st.columns(2)
    with left:
        st.markdown("**Document length distribution (words)**")
        bins = pd.cut(working["raw_text"].str.split().apply(len), bins=10)
        length_counts = bins.value_counts().sort_index()
        length_counts.index = length_counts.index.astype(str)  # Interval -> str for chart serialization
        st.bar_chart(length_counts)
    with right:
        st.markdown("**Category distribution in working corpus**")
        st.bar_chart(working["category"].value_counts())

    st.markdown("---")
    st.markdown("### Keyword extraction & document profiling")
    tfidf_vec, tfidf_matrix = mining_mod.build_tfidf(working["raw_text"].astype(str).tolist())
    doc_choice = st.selectbox("Choose a document", working["id"].tolist(),
                               format_func=lambda i: f"{i} — {working.set_index('id').loc[i, 'title']}")
    doc_idx = working.reset_index(drop=True).index[working.reset_index(drop=True)["id"] == doc_choice][0]
    profile = mining_mod.document_profile(
        doc_choice, working.set_index("id").loc[doc_choice, "raw_text"],
        working.set_index("id").loc[doc_choice, "category"], tfidf_vec, tfidf_matrix, doc_idx)

    p1, p2, p3, p4 = st.columns(4)
    p1.metric("Word count", profile["word_count"])
    p2.metric("Unique words", profile["unique_words"])
    p3.metric("Lexical diversity", profile["lexical_diversity"])
    p4.metric("Readability (Flesch)", profile["readability_flesch"])

    kc1, kc2 = st.columns(2)
    with kc1:
        st.markdown("**Top keywords — TF-IDF**")
        st.dataframe(pd.DataFrame(profile["top_keywords_tfidf"], columns=["term", "score"]),
                     use_container_width=True)
    with kc2:
        st.markdown("**Top keywords — RAKE**")
        st.dataframe(pd.DataFrame(profile["top_keywords_rake"], columns=["phrase", "score"]),
                     use_container_width=True)

    st.markdown("---")
    st.markdown("### Document classification")
    model_type = st.radio("Classifier", ["nb", "logreg"], horizontal=True,
                           format_func=lambda x: "Multinomial Naive Bayes" if x == "nb" else "Logistic Regression")
    feat_type = st.radio("Feature type", ["tfidf", "bow"], horizontal=True,
                          format_func=lambda x: "TF-IDF" if x == "tfidf" else "Bag-of-Words")
    if st.button("Train classifier"):
        with timed("train_classifier"):
            result = mining_mod.classify_documents(
                working["raw_text"].astype(str).tolist(), working["category"].astype(str).tolist(),
                feature_type=feat_type, model_type=model_type)
        m1, m2, m3 = st.columns(3)
        m1.metric("Accuracy", result["accuracy"])
        m2.metric("F1 (macro)", result["f1_macro"])
        m3.metric("Test set size", result["n_test"])
        st.markdown("**Confusion matrix**")
        st.dataframe(result["confusion_matrix"], use_container_width=True)

    st.markdown("---")
    st.markdown("### Comparative analysis: preprocessing × feature-extraction strategies")
    if st.button("Run comparative analysis (BoW/TF-IDF × raw/stem/lemma)"):
        with timed("comparative_analysis"):
            comp_df = mining_mod.comparative_analysis(working, text_col="raw_text", label_col="category")
        st.dataframe(comp_df, use_container_width=True)
        st.markdown("**Accuracy by strategy**")
        st.bar_chart(comp_df.set_index(comp_df["normalization"] + " / " + comp_df["feature_type"])["accuracy"])

# ------------------------------------------------------------------
# TAB 4: INDEX MANAGEMENT
# ------------------------------------------------------------------
with TABS[3]:
    st.subheader("Index Management")
    st.markdown("Builds an **inverted index** (fast boolean pre-filtering) and a **TF-IDF matrix** "
                "(ranked cosine-similarity retrieval) over the current working corpus.")

    ic1, ic2, ic3 = st.columns(3)
    with ic1:
        idx_normalize = st.selectbox("Normalization", ["none", "stem", "lemma"], key="idx_norm")
    with ic2:
        idx_stopwords = st.checkbox("Remove stopwords", value=True, key="idx_stop")
    with ic3:
        idx_max_features = st.slider("Max vocabulary size", 1000, 20000, 8000, step=1000)

    if st.button("🔧 Build / Rebuild Index", type="primary"):
        idx = get_index(force_rebuild=True, use_stopwords=idx_stopwords,
                         normalize=idx_normalize, max_features=idx_max_features)
        st.success("Index rebuilt.")
    else:
        idx = get_index()

    stats = idx.stats()
    s1, s2, s3, s4 = st.columns(4)
    s1.metric("Documents indexed", stats["documents"])
    s2.metric("Vocabulary size", stats["vocabulary_size"])
    s3.metric("Total postings", stats["total_postings"])
    s4.metric("TF-IDF matrix density %", stats["tfidf_matrix_density_pct"])

    st.markdown("**Most frequent terms (by document frequency)**")
    top_terms = idx.top_terms_by_document_frequency(15)
    st.bar_chart(top_terms.set_index("term")["document_frequency"])

    st.markdown("---")
    st.markdown("**Postings lookup** -- inspect the inverted index for a term")
    term_query = st.text_input("Term", value="market")
    postings = idx.inverted_index.get(term_query.lower().strip(), set())
    st.write(f"`{term_query}` appears in **{len(postings)}** document(s).")
    if postings:
        sample_ids = [idx.doc_ids[p] for p in list(postings)[:10]]
        st.write(sample_ids)

# ------------------------------------------------------------------
# TAB 5: SEARCH & RANKING
# ------------------------------------------------------------------
with TABS[4]:
    st.subheader("Search & Ranking Visualization")
    idx = get_index()

    if not st.session_state.link_edges:
        st.info("No crawl has been run yet, so a small default crawl is auto-run now to "
                 "populate the link graph used for PageRank/HITS.", icon="ℹ️")
        with timed("auto_crawl_for_ranking"):
            _, _, edges, _ = crawler_mod.crawl(
                [mock_web.HOME_URL], 3, 400, lambda u: mock_web.fetch(u, st.session_state.web_graph))
        st.session_state.link_edges = edges

    with timed("pagerank"):
        pr_scores = ranking_mod.pagerank(st.session_state.link_edges)
    with timed("hits"):
        hub_scores, auth_scores = ranking_mod.hits(st.session_state.link_edges)
    st.session_state.pagerank_scores = pr_scores
    st.session_state.hits_scores = (hub_scores, auth_scores)

    q1, q2, q3 = st.columns(3)
    with q1:
        query = st.text_input("Search query", value="technology company investment")
    with q2:
        top_k = st.slider("Top-K results", 3, 25, 10)
    with q3:
        alpha = st.slider("Relevance weight α (1-α = link-importance weight)", 0.0, 1.0, 0.7)

    o1, o2 = st.columns(2)
    expand_syn = o1.checkbox("Expand synonyms/abbreviations", value=True)
    correct_typo = o2.checkbox("Auto-correct spelling", value=True)

    if query.strip():
        with timed("search"):
            base_results = search_mod.search(query, idx, top_k=max(top_k * 3, 20),
                                               expand_synonyms=expand_syn, correct_typos=correct_typo)
        parsed = search_mod.process_query(query, idx, expand_synonyms=expand_syn, correct_typos=correct_typo)

        st.markdown("**Query processing**")
        qc1, qc2, qc3 = st.columns(3)
        qc1.write(f"Raw tokens: `{parsed['raw_tokens']}`")
        qc2.write(f"Expanded terms: `{parsed['expanded_terms']}`")
        qc3.write(f"Spelling corrections: `{parsed['corrections']}`" if parsed["corrections"] else "No corrections needed")

        pr_doc_scores = doc_id_link_scores(pr_scores)
        auth_doc_scores = doc_id_link_scores(auth_scores)

        rank_tabs = st.tabs(["📄 TF-IDF baseline", "🌐 PageRank-boosted", "🔗 HITS authority-boosted"])
        with rank_tabs[0]:
            base_view = base_results.head(top_k)
            st.dataframe(base_view.merge(
                st.session_state.metadata_df[["id", "title", "category"]], left_on="doc_id", right_on="id",
                how="left")[["doc_id", "title", "category", "relevance_score"]], use_container_width=True)
        with rank_tabs[1]:
            pr_view = search_mod.rerank_with_link_scores(base_results, pr_doc_scores, alpha=alpha).head(top_k)
            st.dataframe(pr_view.merge(
                st.session_state.metadata_df[["id", "title"]], left_on="doc_id", right_on="id",
                how="left")[["doc_id", "title", "relevance_score", "link_score_raw", "blended_score",
                             "old_rank", "new_rank", "rank_change"]], use_container_width=True)
            st.caption("Positive `rank_change` = the document moved UP after folding in PageRank "
                       "importance; negative = it moved down. This is the concrete demonstration "
                       "that ranking strategy (not just relevance) changes what users see first.")
        with rank_tabs[2]:
            hits_view = search_mod.rerank_with_link_scores(base_results, auth_doc_scores, alpha=alpha).head(top_k)
            st.dataframe(hits_view.merge(
                st.session_state.metadata_df[["id", "title"]], left_on="doc_id", right_on="id",
                how="left")[["doc_id", "title", "relevance_score", "link_score_raw", "blended_score",
                             "old_rank", "new_rank", "rank_change"]], use_container_width=True)

        st.markdown("**Relevance vs. blended score (PageRank-boosted view, top-K)**")
        chart_df = pr_view.set_index("doc_id")[["relevance_score", "blended_score"]]
        st.bar_chart(chart_df)
    else:
        st.warning("Enter a query above to search.")

# ------------------------------------------------------------------
# TAB 6: RECOMMENDATIONS
# ------------------------------------------------------------------
with TABS[5]:
    st.subheader("Recommendation Panel")
    idx = get_index()
    meta = st.session_state.metadata_df
    st.info("Content-based recommendations use the real corpus (TF-IDF similarity). "
             "Collaborative/hybrid recommendations use a **reproducible synthetic user-rating "
             "matrix** (seeded, saved to `data/synthetic_ratings.csv`) since this static news "
             "corpus has no real user interaction log -- disclosed in full in the report.", icon="⭐")

    rec_mode = st.radio("Recommendation strategy", ["Content-based", "Collaborative", "Hybrid"], horizontal=True)

    if rec_mode == "Content-based":
        doc_choice = st.selectbox("Document you're currently viewing", idx.doc_ids,
                                   format_func=lambda i: f"{i} — {meta.set_index('id').loc[i, 'title'] if i in meta['id'].values else i}")
        top_k = st.slider("Top-K", 3, 15, 5, key="cb_topk")
        if st.button("Get content-based recommendations"):
            with timed("content_based_recommend"):
                recs = rec_mod.content_based_recommend(doc_choice, idx.doc_ids, idx.tfidf_matrix, top_k=top_k)
            recs = recs.merge(meta[["id", "title", "category"]], left_on="doc_id", right_on="id", how="left")
            st.dataframe(recs[["doc_id", "title", "category", "similarity_score"]], use_container_width=True)
            st.bar_chart(recs.set_index("doc_id")["similarity_score"])

    elif rec_mode == "Collaborative":
        ratings_df = get_ratings_df()
        user_choice = st.selectbox("Synthetic user", sorted(ratings_df["user_id"].unique()))
        top_k = st.slider("Top-K", 3, 15, 5, key="cf_topk")
        st.markdown("**This user's rating history**")
        st.dataframe(ratings_df[ratings_df["user_id"] == user_choice].merge(
            meta[["id", "title", "category"]], left_on="doc_id", right_on="id", how="left"
        )[["doc_id", "title", "category", "rating"]], use_container_width=True, height=180)
        if st.button("Get collaborative recommendations"):
            with timed("collaborative_recommend"):
                recs = rec_mod.collaborative_recommend(user_choice, ratings_df, top_k=top_k)
            recs = recs.merge(meta[["id", "title", "category"]], left_on="doc_id", right_on="id", how="left")
            st.dataframe(recs[["doc_id", "title", "category", "predicted_rating"]], use_container_width=True)
            st.bar_chart(recs.set_index("doc_id")["predicted_rating"])

    else:  # Hybrid
        ratings_df = get_ratings_df()
        user_choice = st.selectbox("Synthetic user", sorted(ratings_df["user_id"].unique()), key="hy_user")
        doc_choice = st.selectbox("Document currently being viewed (content seed)", idx.doc_ids, key="hy_doc")
        alpha = st.slider("Weight α on content-based score (1-α on collaborative)", 0.0, 1.0, 0.5, key="hy_alpha")
        top_k = st.slider("Top-K", 3, 15, 5, key="hy_topk")
        if st.button("Get hybrid recommendations"):
            with timed("hybrid_recommend"):
                recs = rec_mod.hybrid_recommend(user_choice, doc_choice, ratings_df, idx.doc_ids,
                                                 idx.tfidf_matrix, top_k=top_k, alpha=alpha)
            recs = recs.merge(meta[["id", "title", "category"]], left_on="doc_id", right_on="id", how="left")
            st.dataframe(recs[["doc_id", "title", "category", "content_score", "collaborative_score",
                                "hybrid_score"]], use_container_width=True)
            st.bar_chart(recs.set_index("doc_id")[["content_score", "collaborative_score", "hybrid_score"]])

# ------------------------------------------------------------------
# TAB 7: EVALUATION
# ------------------------------------------------------------------
with TABS[6]:
    st.subheader("Evaluation Dashboard")
    idx = get_index()
    queries = get_eval_queries()
    st.caption(f"Using {len(queries)} category-anchored evaluation queries with relevance judgments "
               "derived deterministically from corpus category labels + query-term matching "
               "(see `modules/evaluation.py::DEFAULT_QUERY_SEEDS`).")

    k = st.slider("K (for Precision@K / Recall@K / NDCG@K)", 3, 20, 10)

    pr_doc_scores = doc_id_link_scores(st.session_state.pagerank_scores) if st.session_state.pagerank_scores else {}
    hub_scores, auth_doc_scores_raw = st.session_state.hits_scores
    auth_doc_scores = doc_id_link_scores(auth_doc_scores_raw) if auth_doc_scores_raw else {}

    def _search_fn_baseline(q):
        return search_mod.search(q, idx, top_k=50)["doc_id"].tolist()

    def _search_fn_pagerank(q):
        base = search_mod.search(q, idx, top_k=50)
        return search_mod.rerank_with_link_scores(base, pr_doc_scores, alpha=0.7)["doc_id"].tolist()

    def _search_fn_hits(q):
        base = search_mod.search(q, idx, top_k=50)
        return search_mod.rerank_with_link_scores(base, auth_doc_scores, alpha=0.7)["doc_id"].tolist()

    if st.button("📏 Run evaluation across ranking methods", type="primary"):
        with timed("evaluate_compare_methods"):
            comparison = eval_mod.compare_methods(queries, {
                "TF-IDF baseline": _search_fn_baseline,
                "TF-IDF + PageRank": _search_fn_pagerank,
                "TF-IDF + HITS": _search_fn_hits,
            }, k=k)
        st.markdown("**Mean metrics by ranking method**")
        st.dataframe(comparison, use_container_width=True)
        metric_cols = [c for c in comparison.columns if c != "method"]
        st.markdown("**Comparative visualization**")
        st.bar_chart(comparison.set_index("method")[metric_cols])

        st.markdown("---")
        st.markdown("**Per-query detail — TF-IDF + PageRank**")
        detail = eval_mod.evaluate_system(queries, _search_fn_pagerank, k=k)
        st.dataframe(detail, use_container_width=True)

# ------------------------------------------------------------------
# TAB 8: PERFORMANCE ANALYTICS
# ------------------------------------------------------------------
with TABS[7]:
    st.subheader("Performance Analytics")
    log_df = pd.DataFrame(st.session_state.perf_log)
    if log_df.empty:
        st.info("Interact with the other tabs (crawl, build index, search, evaluate...) to populate "
                "timing data here.")
    else:
        st.markdown("**Operation timing log**")
        st.dataframe(log_df, use_container_width=True, height=250)
        st.markdown("**Total time spent per operation type**")
        st.bar_chart(log_df.groupby("action")["duration_seconds"].sum())

    if st.session_state.corpus_size_history:
        st.markdown("**Corpus growth across crawl runs**")
        growth_df = pd.DataFrame(st.session_state.corpus_size_history)
        st.line_chart(growth_df.set_index("time")["documents"])

    idx = st.session_state.index
    if idx is not None:
        st.markdown("**Current index footprint**")
        st.json(idx.stats())
        st.caption(
            "Scalability note: the inverted index is a Python dict of term→doc-position sets and "
            "the TF-IDF matrix is a sparse SciPy matrix, so memory scales roughly linearly with "
            "vocabulary × non-zero entries, not vocabulary × documents. For corpora much larger "
            "than this one (~1,100 articles), the natural next step would be an on-disk index "
            "(e.g. SQLite-backed postings, as already used for metadata/content) rather than an "
            "in-memory dict.")

# ------------------------------------------------------------------
# TAB 9: INFERENCE & DISCUSSION (Section G -- compulsory)
# ------------------------------------------------------------------
with TABS[8]:
    st.subheader("🧠 Inference & Discussion (compulsory)")

    meta = st.session_state.metadata_df
    n_exact = int(meta["is_exact_duplicate"].sum())
    n_near = int(meta["is_near_duplicate"].sum())

    q1_answer = """
**Symptom:** highly relevant documents are retrieved (they are *in* the candidate set) but land
low in the ranked list.

**Likely causes observed in this system:**
1. *Pure lexical/TF-IDF relevance ignores document authority.* A short, keyword-dense but
   low-authority page can out-score a comprehensive, well-linked article on raw cosine similarity.
2. *Query-term mismatch* (synonyms, abbreviations, morphological variants) lowers the relevance
   score of a genuinely relevant document that uses different wording than the query.
3. *No length normalization / topic drift* -- very long documents can dilute term weights even
   when they cover the topic well.
4. *Static ranking function* -- a single global scoring formula cannot adapt to query intent
   (navigational vs. informational).

**Improvements implemented / proposed here:**
- Blend relevance with **link-based importance** (PageRank/HITS authority) as done in the
  Search & Ranking tab (`search.rerank_with_link_scores`) -- see the `rank_change` column, which
  is direct, observable evidence of poorly-ranked-but-relevant documents moving up.
- **Query expansion** (synonyms/abbreviations) and **spelling correction** before scoring, so
  relevant documents are not missed purely due to vocabulary mismatch.
- Longer-term improvements: learning-to-rank over multiple signals (relevance, authority,
  freshness, click-through in a real system), BM25 instead of raw TF-IDF for better length
  normalization, and query-dependent weighting of the relevance/authority blend (α).
"""

    q2_answer = f"""
**Observed in this corpus:** {n_exact} exact-duplicate and {n_near} near-duplicate pages were
flagged by the crawler (`modules/crawler.py`, content-hash + Jaccard-similarity detection).

**Effects if left unmitigated:**
- **Indexing:** duplicate content inflates document frequency for the same terms, skewing IDF
  weights (terms in duplicated stories look artificially more "common" corpus-wide), and wastes
  index space on redundant postings.
- **Ranking:** duplicate/near-duplicate pages that both link to (or are linked from) the same hub
  artificially inflate PageRank/HITS authority for that story -- our `doc_id_link_scores()`
  helper *sums* link scores across all URL variants of the same `doc_id`, so a document with a
  mirror page visibly gets a higher aggregated authority score purely from duplication, not from
  independent endorsement. This is a real, demonstrable distortion of "importance".
- **Recommendation:** content-based similarity treats each duplicate as a distinct item, so a
  user could receive two near-identical "different" recommendations, hurting perceived diversity.
- **Evaluation:** if duplicates are counted as separate relevant documents, Precision/Recall/MAP
  are inflated relative to genuinely distinct relevant content -- effectiveness looks better than
  it really is.

**Mitigations implemented:**
- Exact duplicates are dropped from the content store outright (kept only in metadata, flagged).
- Near-duplicates are flagged via Jaccard similarity over token sets and can optionally be
  excluded from the content store (the "Drop near-duplicates" toggle in the Crawling tab).
- Link-authority aggregation is deliberately shown *unmitigated by default* (summed across
  duplicates) specifically so its distorting effect is visible for this discussion; an
  alternative "canonicalize before scoring" strategy (dedupe first, then run PageRank on the
  deduplicated graph) removes the distortion entirely.
"""

    q3_answer = """
**Content-based recommendation** (used here via TF-IDF cosine similarity between documents):
- Strength: works from the very first interaction (no cold-start for new items or users, as long
  as item content is available), fully explainable ("recommended because it shares these
  keywords"), unaffected by sparsity of user behaviour data.
- Weakness: limited serendipity -- it recommends more of the same topic/style and cannot capture
  cross-topic taste correlations a user might actually have.

**Collaborative recommendation** (used here via latent-factor SVD on a synthetic user-item
rating matrix):
- Strength: captures behavioural/taste signals content features cannot see (e.g. "users who liked
  X also liked Y" even across categories), and can surface serendipitous recommendations.
- Weakness: cold-start for new users/items with no ratings yet, and needs enough interaction data
  to be reliable (our synthetic 60-user matrix is intentionally small -- in production this needs
  thousands of real interactions to generalize).

**When to prefer which:** content-based is preferable for new/sparse catalogs, cold-start items,
and when explainability matters (e.g. a news reader with no login). Collaborative is preferable
once a system has an established, reasonably dense interaction history and wants to capture taste
beyond topical similarity. The **hybrid** approach implemented here (weighted blend, α-tunable)
is the practical answer in most real systems: it degrades gracefully to content-based when
collaborative signal is weak (new item/user) and leans on collaborative signal once it is
available.
"""

    q4_answer = """
Each stage in this system feeds the next, and weaknesses at one stage propagate downstream:
**crawling** determines *what* exists in the corpus (coverage, freshness, duplication) and *what
link structure* is available for authority scoring; **text mining** turns raw text into the
structured features (TF-IDF vectors, keywords, categories) every later stage depends on --
indexing, search relevance, content-based similarity, and classification all reuse the same
vectorization; **indexing** is what makes search over that structure fast enough to be usable
at all (boolean pre-filter + TF-IDF, rather than scanning every document per query); **search &
ranking** combines indexed relevance with link-derived importance (PageRank/HITS) to decide what
a user actually sees first, which is exactly where crawling's link graph gets reused;
**recommendation** reuses the same TF-IDF space (content-based) and layers on behavioural signal
(collaborative) to personalize beyond a single query; and **evaluation** closes the loop by
measuring whether all of the above choices (feature type, normalization, ranking blend, α) are
actually improving retrieval quality, which is what justifies picking one configuration over
another. No single stage is suffient alone -- e.g. a perfect index over a duplicate-polluted,
poorly-crawled corpus still returns redundant/low-quality results, and the best ranking algorithm
cannot fix features that were poorly engineered upstream. The end-to-end integration is what lets
this system be evaluated and tuned as a whole, not just judged stage-by-stage.
"""

    q5_answer = """
Key learnings from building and running this system end-to-end:
1. **Ranking is not the same as relevance.** The Search & Ranking tab's `rank_change` column
   shows concretely that blending in PageRank/HITS reorders results even when the underlying
   TF-IDF relevance scores don't change -- ranking strategy is a distinct design decision from
   feature engineering.
2. **Duplicates quietly distort every downstream stage**, not just the index -- authority scores,
   recommendation diversity, and evaluation metrics all move when duplicate content is present,
   which is easy to miss if you only inspect the index in isolation.
3. **Preprocessing choices measurably change classification/retrieval quality** -- the
   comparative analysis (BoW vs TF-IDF × raw/stem/lemma) in the Text Mining tab shows accuracy
   shifting by a few points across strategies on the same corpus, reinforcing that "preprocessing"
   is a modeling decision, not a fixed preliminary step.
4. **Recommendation quality depends heavily on what data is actually available** -- content-based
   recommendation was reliable from a static corpus alone, while collaborative filtering needed a
   synthetic rating matrix to demonstrate at all, underscoring how much real collaborative systems
   depend on volume of genuine interaction data.
5. **A dependency-light, offline-safe design was essential for reproducibility** -- because the
   grading environment (BITS virtual lab) has no guaranteed internet access, building the crawler
   over a deterministic local web graph (rather than live URLs) was the difference between a demo
   that always works and one that is flaky by construction.
"""

    for title, answer in [
        ("1. Highly relevant documents ranked poorly — causes & fixes", q1_answer),
        ("2. Impact of duplicate/near-duplicate documents", q2_answer),
        ("3. Content-based vs. Collaborative recommendation", q3_answer),
        ("4. How crawling → mining → indexing → search → ranking → recommendation integrate", q4_answer),
        ("5. Overall learnings from results", q5_answer),
    ]:
        with st.expander(title, expanded=False):
            st.markdown(answer)

    summary_md = "\n\n".join([
        "# Inference & Discussion Summary — SmartIR-2 (Assignment 2, Team 47)",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"Corpus size: {len(meta)} documents | Exact duplicates: {n_exact} | Near-duplicates: {n_near}",
        "## Q1: Highly relevant but poorly ranked documents", q1_answer,
        "## Q2: Impact of duplicate/near-duplicate documents", q2_answer,
        "## Q3: Content-based vs. Collaborative recommendation", q3_answer,
        "## Q4: End-to-end integration", q4_answer,
        "## Q5: Overall learnings", q5_answer,
    ])
    st.download_button("⬇️ Download inference summary (Markdown)", summary_md,
                        file_name="inference_summary.md", mime="text/markdown")
