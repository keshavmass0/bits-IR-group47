import re
import time
import math
import fnmatch
from dataclasses import dataclass
from collections import defaultdict, Counter
from typing import Dict, List, Tuple, Set, Optional

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# -----------------------------
# Page configuration
# -----------------------------
st.set_page_config(
    page_title="SmartIR Lab | Information Retrieval System",
    page_icon="🔎",
    layout="wide",
)

# -----------------------------
# Lightweight NLP utilities
# -----------------------------
STOPWORDS = {
    "a", "an", "the", "and", "or", "but", "if", "while", "with", "without", "of", "to", "in", "on", "for", "from", "by", "as", "at", "is", "are", "was", "were",
    "be", "been", "being", "this", "that", "these", "those", "it", "its", "into", "about", "after", "before", "than", "then", "also", "may", "will", "would", "should",
    "could", "can", "has", "have", "had", "do", "does", "did", "not", "new", "said", "their", "his", "her", "they", "them", "he", "she", "we", "you", "i", "our", "your"
}


def tokenize(text: str, hyphen_mode: str = "split") -> List[str]:
    """Tokenize text with configurable hyphen handling."""
    text = str(text)
    if hyphen_mode == "split":
        text = text.replace("-", " ")
    elif hyphen_mode == "join":
        text = text.replace("-", "")
    elif hyphen_mode == "underscore":
        text = text.replace("-", "_")
    if hyphen_mode == "keep":
        # Preserve intra-word hyphens so "long-term" stays a single token.
        return re.findall(r"[A-Za-z0-9_]+(?:-[A-Za-z0-9_]+)*", text)
    return re.findall(r"[A-Za-z0-9_]+", text)


def simple_stem(word: str) -> str:
    """A deterministic suffix-based stemmer fallback. It is intentionally transparent for classroom demos."""
    w = word.lower()
    rules = [
        ("ization", "ize"), ("ational", "ate"), ("fulness", "ful"), ("ousness", "ous"),
        ("iveness", "ive"), ("tional", "tion"), ("ments", "ment"), ("ingly", ""),
        ("edly", ""), ("ing", ""), ("ied", "y"), ("ies", "y"), ("ed", ""),
        ("ly", ""), ("es", ""), ("s", "")
    ]
    for suffix, repl in rules:
        if len(w) > len(suffix) + 2 and w.endswith(suffix):
            return w[:-len(suffix)] + repl
    return w


LEMMAS = {
    "children": "child", "men": "man", "women": "woman", "people": "person", "mice": "mouse",
    "geese": "goose", "feet": "foot", "teeth": "tooth", "better": "good", "best": "good",
    "worse": "bad", "won": "win", "winning": "win", "ran": "run", "running": "run",
    "companies": "company", "policies": "policy", "prices": "price", "services": "service",
    "families": "family", "technologies": "technology", "queries": "query", "documents": "document"
}


def simple_lemma(word: str) -> str:
    """A lightweight lemmatizer that preserves readable base forms without external downloads."""
    w = word.lower()
    if w in LEMMAS:
        return LEMMAS[w]
    if len(w) > 4 and w.endswith("ies"):
        return w[:-3] + "y"
    if len(w) > 5 and w.endswith("ing"):
        base = w[:-3]
        if len(base) > 2 and base[-1] == base[-2]:
            base = base[:-1]
        return base
    if len(w) > 4 and w.endswith("ed"):
        return w[:-2]
    if len(w) > 4 and w.endswith("es"):
        return w[:-2]
    if len(w) > 3 and w.endswith("s"):
        return w[:-1]
    return w


def preprocess_tokens(
    text: str,
    lower: bool = True,
    remove_stop: bool = True,
    hyphen_mode: str = "split",
    normalization: str = "lemmatization",
) -> List[str]:
    tokens = tokenize(text, hyphen_mode)
    if lower:
        tokens = [t.lower() for t in tokens]
    if remove_stop:
        tokens = [t for t in tokens if t.lower() not in STOPWORDS]
    if normalization == "stemming":
        tokens = [simple_stem(t) for t in tokens]
    elif normalization == "lemmatization":
        tokens = [simple_lemma(t) for t in tokens]
    return [t for t in tokens if t]


# -----------------------------
# Index construction
# -----------------------------
def build_inverted_index(token_docs: Dict[str, List[str]]) -> Dict[str, Dict[str, int]]:
    idx = defaultdict(dict)
    for doc_id, tokens in token_docs.items():
        for term, tf in Counter(tokens).items():
            idx[term][doc_id] = tf
    return dict(idx)


def build_positional_index(token_docs: Dict[str, List[str]]) -> Dict[str, Dict[str, List[int]]]:
    idx = defaultdict(lambda: defaultdict(list))
    for doc_id, tokens in token_docs.items():
        for pos, term in enumerate(tokens):
            idx[term][doc_id].append(pos)
    return {term: dict(posting) for term, posting in idx.items()}


def build_biword_index(token_docs: Dict[str, List[str]]) -> Dict[str, Set[str]]:
    idx = defaultdict(set)
    for doc_id, tokens in token_docs.items():
        for i in range(len(tokens) - 1):
            idx[f"{tokens[i]} {tokens[i+1]}"].add(doc_id)
    return dict(idx)


def build_kgram_index(vocab: List[str], k: int = 3) -> Dict[str, Set[str]]:
    idx = defaultdict(set)
    for term in vocab:
        padded = f"${term}$"
        grams = [padded[i:i+k] for i in range(max(0, len(padded)-k+1))]
        for gram in grams:
            idx[gram].add(term)
    return dict(idx)


# -----------------------------
# Phrase search
# -----------------------------
def phrase_search_biword(query_tokens: List[str], biword_index: Dict[str, Set[str]]) -> Set[str]:
    if len(query_tokens) == 1:
        return set()
    biwords = [f"{query_tokens[i]} {query_tokens[i+1]}" for i in range(len(query_tokens)-1)]
    posting_sets = [biword_index.get(b, set()) for b in biwords]
    if not posting_sets:
        return set()
    return set.intersection(*posting_sets) if posting_sets else set()


def phrase_search_positional(query_tokens: List[str], positional_index: Dict[str, Dict[str, List[int]]]) -> Set[str]:
    if not query_tokens:
        return set()
    candidate_docs = set(positional_index.get(query_tokens[0], {}).keys())
    for term in query_tokens[1:]:
        candidate_docs &= set(positional_index.get(term, {}).keys())
    results = set()
    for doc_id in candidate_docs:
        first_positions = positional_index[query_tokens[0]][doc_id]
        for start in first_positions:
            ok = True
            for offset, term in enumerate(query_tokens[1:], start=1):
                if start + offset not in positional_index.get(term, {}).get(doc_id, []):
                    ok = False
                    break
            if ok:
                results.add(doc_id)
                break
    return results


# -----------------------------
# Dictionary structures
# -----------------------------
@dataclass
class BSTNode:
    key: str
    left: Optional['BSTNode'] = None
    right: Optional['BSTNode'] = None


class BST:
    def __init__(self):
        self.root = None

    def insert(self, key: str):
        if self.root is None:
            self.root = BSTNode(key)
            return
        cur = self.root
        while True:
            if key == cur.key:
                return
            if key < cur.key:
                if cur.left is None:
                    cur.left = BSTNode(key)
                    return
                cur = cur.left
            else:
                if cur.right is None:
                    cur.right = BSTNode(key)
                    return
                cur = cur.right

    def search(self, key: str) -> bool:
        cur = self.root
        while cur:
            if key == cur.key:
                return True
            cur = cur.left if key < cur.key else cur.right
        return False

    def height(self) -> int:
        # Iterative level-order traversal: safe even for skewed trees where recursion would overflow.
        if self.root is None:
            return 0
        depth = 0
        level = [self.root]
        while level:
            depth += 1
            level = [child for node in level for child in (node.left, node.right) if child]
        return depth


class BTreeNode:
    def __init__(self, leaf=True):
        self.leaf = leaf
        self.keys = []
        self.children = []


class BTree:
    def __init__(self, t=3):
        self.root = BTreeNode(True)
        self.t = t

    def search(self, key, node=None):
        node = node or self.root
        i = 0
        while i < len(node.keys) and key > node.keys[i]:
            i += 1
        if i < len(node.keys) and key == node.keys[i]:
            return True
        if node.leaf:
            return False
        return self.search(key, node.children[i])

    def insert(self, key):
        if self.search(key):
            return
        root = self.root
        if len(root.keys) == 2 * self.t - 1:
            new_root = BTreeNode(False)
            new_root.children.append(root)
            self._split_child(new_root, 0)
            self.root = new_root
            self._insert_non_full(new_root, key)
        else:
            self._insert_non_full(root, key)

    def _insert_non_full(self, node, key):
        i = len(node.keys) - 1
        if node.leaf:
            node.keys.append(None)
            while i >= 0 and key < node.keys[i]:
                node.keys[i+1] = node.keys[i]
                i -= 1
            node.keys[i+1] = key
        else:
            while i >= 0 and key < node.keys[i]:
                i -= 1
            i += 1
            if len(node.children[i].keys) == 2 * self.t - 1:
                self._split_child(node, i)
                if key > node.keys[i]:
                    i += 1
            self._insert_non_full(node.children[i], key)

    def _split_child(self, parent, i):
        t = self.t
        y = parent.children[i]
        z = BTreeNode(y.leaf)
        parent.keys.insert(i, y.keys[t-1])
        parent.children.insert(i+1, z)
        z.keys = y.keys[t:]
        y.keys = y.keys[:t-1]
        if not y.leaf:
            z.children = y.children[t:]
            y.children = y.children[:t]

    def height(self):
        h = 1
        node = self.root
        while not node.leaf:
            h += 1
            node = node.children[0]
        return h


# -----------------------------
# Retrieval utilities
# -----------------------------
def make_snippet(text: str, query: str, width: int = 260) -> str:
    text = str(text).strip()
    terms = [re.escape(t) for t in tokenize(query) if t]
    if not terms:
        return text[:width] + ("..." if len(text) > width else "")
    m = re.search("|".join(terms), text, flags=re.IGNORECASE)
    if not m:
        snippet = text[:width]
    else:
        start = max(0, m.start() - 90)
        snippet = text[start:start+width]
        if start > 0:
            snippet = "..." + snippet
    for term in sorted(set(terms), key=len, reverse=True):
        snippet = re.sub(f"({term})", r"**\1**", snippet, flags=re.IGNORECASE)
    return snippet + ("..." if len(snippet) >= width else "")


def boolean_search(query: str, inverted_index: Dict[str, Dict[str, int]], all_docs: Set[str]) -> Set[str]:
    q = query.lower().strip()
    tokens = q.split()
    if not tokens:
        return set()
    def term_docs(term):
        # Try the raw term first (covers normalization="none"), then lemma and stem forms.
        for candidate in (term, simple_lemma(term), simple_stem(term)):
            if candidate in inverted_index:
                return set(inverted_index[candidate].keys())
        return set()
    # Simple AND/OR/NOT parser, evaluated left-to-right for demo transparency.
    current = None
    op = "AND"
    negate_next = False
    for tok in tokens:
        upper = tok.upper()
        if upper in {"AND", "OR"}:
            op = upper
            continue
        if upper == "NOT":
            negate_next = True
            continue
        docs = term_docs(tok)
        if negate_next:
            docs = all_docs - docs
            negate_next = False
        if current is None:
            current = docs
        elif op == "AND":
            current &= docs
        else:
            current |= docs
    return current or set()


def edit_distance(a: str, b: str) -> int:
    dp = np.zeros((len(a)+1, len(b)+1), dtype=int)
    dp[:, 0] = np.arange(len(a)+1)
    dp[0, :] = np.arange(len(b)+1)
    for i in range(1, len(a)+1):
        for j in range(1, len(b)+1):
            cost = 0 if a[i-1] == b[j-1] else 1
            dp[i, j] = min(dp[i-1, j] + 1, dp[i, j-1] + 1, dp[i-1, j-1] + cost)
    return int(dp[len(a), len(b)])


def suggest_correction(term: str, vocab: List[str], max_candidates: int = 5) -> List[Tuple[str, int]]:
    term = term.lower()
    candidates = []
    for v in vocab:
        if abs(len(v) - len(term)) <= 3:
            d = edit_distance(term, v)
            if d <= 3:
                candidates.append((v, d))
    return sorted(candidates, key=lambda x: (x[1], x[0]))[:max_candidates]


def soundex(word: str) -> str:
    """Classic Soundex phonetic code: first letter + 3 digits."""
    word = re.sub(r"[^a-z]", "", word.lower())
    if not word:
        return ""
    codes = {
        "b": "1", "f": "1", "p": "1", "v": "1",
        "c": "2", "g": "2", "j": "2", "k": "2", "q": "2", "s": "2", "x": "2", "z": "2",
        "d": "3", "t": "3", "l": "4", "m": "5", "n": "5", "r": "6",
    }
    first = word[0].upper()
    encoded = []
    prev = codes.get(word[0], "")
    for ch in word[1:]:
        code = codes.get(ch, "")
        if code and code != prev:
            encoded.append(code)
        if ch not in "hw":
            prev = code
    return (first + "".join(encoded) + "000")[:4]


def phonetic_matches(term: str, vocab: List[str], max_candidates: int = 10) -> List[Tuple[str, str, int]]:
    """Vocabulary terms sharing the query term's Soundex code, ranked by edit distance."""
    code = soundex(term)
    matches = []
    for v in vocab:
        if soundex(v) == code:
            matches.append((v, code, edit_distance(term.lower(), v)))
    return sorted(matches, key=lambda x: (x[2], x[0]))[:max_candidates]


@st.cache_data(show_spinner=False)
def load_sample():
    return pd.read_csv("sample_bbc_news.csv")


@st.cache_data(show_spinner=False)
def prepare_indexes(df: pd.DataFrame, text_col: str, lower: bool, stop: bool, hyphen_mode: str, normalization: str):
    token_docs = {
        str(row["doc_id"]): preprocess_tokens(row[text_col], lower, stop, hyphen_mode, normalization)
        for _, row in df.iterrows()
    }
    inv = build_inverted_index(token_docs)
    pos = build_positional_index(token_docs)
    bi = build_biword_index(token_docs)
    vocab = sorted(inv.keys())
    kg = build_kgram_index(vocab)
    return token_docs, inv, pos, bi, vocab, kg


def ranked_tfidf(df: pd.DataFrame, text_col: str, query: str, top_k: int = 10,
                 lower: bool = True, stop: bool = True, hyphen_mode: str = "split",
                 normalization: str = "lemmatization"):
    corpus = df[text_col].fillna("").astype(str).tolist()
    # Use the same preprocessing pipeline selected in the sidebar so the chosen
    # options flow end-to-end into ranked retrieval as well.
    vectorizer = TfidfVectorizer(
        tokenizer=lambda t: preprocess_tokens(t, lower, stop, hyphen_mode, normalization),
        preprocessor=str,
        lowercase=False,
        token_pattern=None,
    )
    matrix = vectorizer.fit_transform(corpus)
    qv = vectorizer.transform([query])
    scores = cosine_similarity(qv, matrix).flatten()
    order = np.argsort(scores)[::-1]
    rows = []
    for rank, idx in enumerate(order[:top_k], start=1):
        if scores[idx] <= 0:
            continue
        row = df.iloc[idx]
        rows.append({
            "Rank": rank,
            "Document ID": row["doc_id"],
            "Category": row.get("category", ""),
            "Title": row.get("title", ""),
            "TF-IDF Score": round(float(scores[idx]), 4),
            "Snippet": make_snippet(row[text_col], query)
        })
    return pd.DataFrame(rows)


# -----------------------------
# User Interface
# -----------------------------
st.title("🔎 SmartIR Lab: Advanced Information Retrieval System")
st.caption("End-to-end Streamlit IR laboratory: preprocessing, inverted index, phrase search, dictionary trees and tolerant retrieval.")

with st.sidebar:
    st.header("1) Dataset")
    uploaded = st.file_uploader("Upload CSV or TXT collection", type=["csv", "txt"])
    use_sample = st.checkbox("Use included BBC-style sample dataset", value=True if uploaded is None else False)

    st.header("2) Preprocessing")
    lower = st.checkbox("Lowercasing", value=True)
    stop = st.checkbox("Stop word removal", value=True)
    hyphen_mode = st.selectbox("Hyphen handling", ["split", "join", "underscore", "keep"], index=0)
    normalization = st.selectbox("Normalization", ["lemmatization", "stemming", "none"], index=0)

    st.header("3) Retrieval")
    model = st.selectbox("Retrieval model", [
        "TF-IDF Ranked Retrieval",
        "Boolean Search",
        "Phrase Search: Biword vs Positional",
        "Dictionary Search: BST vs B-Tree",
        "Tolerant Retrieval"
    ])
    query = st.text_input("Enter query", value="prime minister economic policy")
    top_k = st.slider("Top K results", 3, 20, 10)

if uploaded is not None:
    if uploaded.name.lower().endswith(".csv"):
        df = pd.read_csv(uploaded)
    else:
        raw = uploaded.read().decode("utf-8", errors="ignore")
        docs = [d.strip() for d in re.split(r"\n\s*\n", raw) if d.strip()]
        df = pd.DataFrame({"doc_id": [f"D{i+1:03d}" for i in range(len(docs))], "text": docs})
elif use_sample:
    df = load_sample()
else:
    st.warning("Upload a dataset or enable the sample dataset.")
    st.stop()

if "doc_id" not in df.columns:
    df.insert(0, "doc_id", [f"D{i+1:04d}" for i in range(len(df))])
else:
    df["doc_id"] = df["doc_id"].astype(str)
    if df["doc_id"].duplicated().any():
        st.warning("Duplicate doc_id values detected in the upload — suffixing duplicates so no document is lost.")
        dup_counts = df.groupby("doc_id").cumcount()
        df.loc[dup_counts > 0, "doc_id"] = df.loc[dup_counts > 0, "doc_id"] + "_dup" + dup_counts[dup_counts > 0].astype(str)

candidate_text_cols = [c for c in df.columns if df[c].dtype == object and c != "doc_id"]
if not candidate_text_cols:
    st.error("No text column found. Please upload a CSV with at least one text column.")
    st.stop()
text_col = st.sidebar.selectbox("Text column", candidate_text_cols, index=candidate_text_cols.index("text") if "text" in candidate_text_cols else 0)

# Prepare indexes
token_docs, inverted_index, positional_index, biword_index, vocab, kgram_index = prepare_indexes(
    df, text_col, lower, stop, hyphen_mode, normalization
)
all_doc_ids = set(df["doc_id"].astype(str))

# Main tabs
tabs = st.tabs([
    "Dataset Overview", "Preprocessing", "Inverted Index", "Search Results",
    "Phrase Comparison", "BST vs B-Tree", "Tolerant Retrieval", "Inference Dashboard"
])

with tabs[0]:
    st.subheader("Dataset Overview")
    total_tokens = sum(len(toks) for toks in token_docs.values())
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Documents", len(df))
    c2.metric("Vocabulary Size", len(vocab))
    c3.metric("Average Tokens / Doc", round(total_tokens / max(len(df), 1), 2))
    c4.metric("Missing Text Values", int(df[text_col].isna().sum()))
    st.markdown("**All uploaded documents**")
    st.dataframe(df, use_container_width=True, height=420)
    with st.expander("Read a full document"):
        view_doc = st.selectbox("Document", df["doc_id"].astype(str).tolist(), key="overview_doc_viewer")
        view_row = df[df["doc_id"].astype(str) == view_doc].iloc[0]
        if "title" in df.columns:
            st.markdown(f"**{view_row.get('title', '')}**")
        st.write(str(view_row[text_col]))
    if "category" in df.columns:
        st.bar_chart(df["category"].value_counts())

with tabs[1]:
    st.subheader("Text Preprocessing Explorer")
    doc_choice = st.selectbox("Choose a document", df["doc_id"].astype(str).tolist())
    raw_text = str(df.loc[df["doc_id"].astype(str) == doc_choice, text_col].iloc[0])
    st.markdown("**Original Text**")
    st.write(raw_text)
    pipeline_rows = []
    pipeline_rows.append({"Step": "Tokenization", "Output": tokenize(raw_text, hyphen_mode="keep")[:80]})
    pipeline_rows.append({"Step": "Lowercasing", "Output": [t.lower() for t in tokenize(raw_text, hyphen_mode="keep")][:80]})
    pipeline_rows.append({"Step": f"Hyphen handling = {hyphen_mode}", "Output": tokenize(raw_text, hyphen_mode=hyphen_mode)[:80]})
    tokens_no_stop = [t.lower() for t in tokenize(raw_text, hyphen_mode=hyphen_mode) if t.lower() not in STOPWORDS]
    pipeline_rows.append({"Step": "Stop word removal", "Output": tokens_no_stop[:80]})
    pipeline_rows.append({"Step": "Stemming", "Output": [simple_stem(t) for t in tokens_no_stop][:80]})
    pipeline_rows.append({"Step": "Lemmatization", "Output": [simple_lemma(t) for t in tokens_no_stop][:80]})
    st.dataframe(pd.DataFrame(pipeline_rows), use_container_width=True)

    st.markdown("### Stemming vs Lemmatization Retrieval Quality")
    st.write("For each test query both pipelines are run. Besides the raw match counts, retrieval quality is "
             "measured as the **average TF-IDF cosine similarity** between the query and the documents each "
             "pipeline retrieves — a semantic relevance proxy: higher similarity means the retrieved set is "
             "more on-topic, not just larger.")
    test_queries = st.text_area("Test queries, one per line", "economic growth\nmarket price\nnational service\nfootball match\ntechnology company")

    # Shared TF-IDF space over the raw corpus to score retrieved sets from either pipeline.
    quality_corpus = df[text_col].fillna("").astype(str).tolist()
    quality_vectorizer = TfidfVectorizer(stop_words="english")
    quality_matrix = quality_vectorizer.fit_transform(quality_corpus)
    doc_pos = {str(d): i for i, d in enumerate(df["doc_id"].astype(str))}

    def retrieval_quality(q: str, doc_ids: Set[str]) -> float:
        """Average cosine similarity between the query and the retrieved documents."""
        if not doc_ids:
            return 0.0
        qv = quality_vectorizer.transform([q])
        sims = cosine_similarity(qv, quality_matrix).flatten()
        return float(np.mean([sims[doc_pos[d]] for d in doc_ids if d in doc_pos]))

    stem_docs, stem_inv, *_ = prepare_indexes(df, text_col, True, True, hyphen_mode, "stemming")
    lemma_docs, lemma_inv, *_ = prepare_indexes(df, text_col, True, True, hyphen_mode, "lemmatization")

    comparison = []
    for q in [x.strip() for x in test_queries.splitlines() if x.strip()]:
        q_stem = [simple_stem(t) for t in preprocess_tokens(q, True, True, hyphen_mode, "none")]
        q_lemma = [simple_lemma(t) for t in preprocess_tokens(q, True, True, hyphen_mode, "none")]
        stem_result = set.intersection(*[set(stem_inv.get(t, {}).keys()) for t in q_stem]) if q_stem else set()
        lemma_result = set.intersection(*[set(lemma_inv.get(t, {}).keys()) for t in q_lemma]) if q_lemma else set()
        union = stem_result | lemma_result
        jaccard = len(stem_result & lemma_result) / len(union) if union else 0
        stem_quality = retrieval_quality(q, stem_result)
        lemma_quality = retrieval_quality(q, lemma_result)
        if lemma_quality != stem_quality:
            better = "Lemmatization" if lemma_quality > stem_quality else "Stemming"
        else:
            better = "Lemmatization" if len(lemma_result) >= len(stem_result) else "Stemming"
        comparison.append({
            "Query": q,
            "Stemmed Results": len(stem_result),
            "Lemmatized Results": len(lemma_result),
            "Stem Avg Similarity": round(stem_quality, 4),
            "Lemma Avg Similarity": round(lemma_quality, 4),
            "Jaccard Similarity": round(jaccard, 3),
            "Better": better,
        })
    comparison_df = pd.DataFrame(comparison)
    st.dataframe(comparison_df, use_container_width=True)

    if not comparison_df.empty:
        lemma_wins = int((comparison_df["Better"] == "Lemmatization").sum())
        stem_wins = int((comparison_df["Better"] == "Stemming").sum())
        verdict = "Lemmatization" if lemma_wins >= stem_wins else "Stemming"
        st.success(
            f"**Conclusion for this dataset:** {verdict} is more suitable "
            f"({lemma_wins} queries favoured lemmatization vs {stem_wins} stemming, by average retrieval similarity). "
            f"Lemmatization preserves meaningful dictionary base forms (e.g. *companies → company*), keeping the "
            f"retrieved sets readable and precise, while stemming is faster but over-truncates "
            f"(e.g. *prices → price* vs stem *pric*), which can merge unrelated terms and hurt precision."
        )

with tabs[2]:
    st.subheader("Inverted Index")
    st.write("Posting list format: term → documents with term frequency.")
    index_rows = []
    for term in vocab[:1000]:
        postings = inverted_index[term]
        index_rows.append({
            "Term": term,
            "Document Frequency": len(postings),
            "Posting List with TF": ", ".join([f"{d}:{tf}" for d, tf in list(postings.items())[:15]])
        })
    st.dataframe(pd.DataFrame(index_rows).sort_values("Document Frequency", ascending=False), use_container_width=True)

with tabs[3]:
    st.subheader("Search Results")
    st.write(f"Selected model: **{model}**")
    if model == "TF-IDF Ranked Retrieval":
        res = ranked_tfidf(df, text_col, query, top_k, lower, stop, hyphen_mode, normalization)
        st.dataframe(res, use_container_width=True)
    elif model == "Boolean Search":
        docs = boolean_search(query, inverted_index, all_doc_ids)
        rows = []
        for rank, doc_id in enumerate(sorted(docs)[:top_k], start=1):
            row = df[df["doc_id"].astype(str) == doc_id].iloc[0]
            rows.append({"Rank": rank, "Document ID": doc_id, "Title": row.get("title", ""), "Category": row.get("category", ""), "Snippet": make_snippet(row[text_col], query)})
        st.dataframe(pd.DataFrame(rows), use_container_width=True)
    else:
        st.info("Open the matching specialized tab for this selected model.")

with tabs[4]:
    st.subheader("Phrase Query Processing: Biword Index vs Positional Index")
    st.write("A phrase query matches only documents where the words occur **contiguously and in order** — "
             "unlike ranked/boolean retrieval, where words may appear anywhere in the document.")
    phrase_query = st.text_input("Phrase query (exact phrase to search)", value="stock market prices")
    # Query tokens must be preprocessed exactly like the indexed documents, otherwise positions will not align.
    q_tokens = preprocess_tokens(phrase_query, lower=lower, remove_stop=stop, hyphen_mode=hyphen_mode, normalization=normalization)
    bi_docs = phrase_search_biword(q_tokens, biword_index)
    pos_docs = phrase_search_positional(q_tokens, positional_index)
    false_positive = bi_docs - pos_docs
    m1, m2, m3 = st.columns(3)
    m1.metric("Biword Matches", len(bi_docs))
    m2.metric("Positional Matches", len(pos_docs))
    m3.metric("Biword False Positives", len(false_positive))

    if len(q_tokens) == 1:
        st.info(
            "A single-term query has no biwords, so the biword index cannot be used (matches show 0 by design). "
            "The positional index still retrieves every document containing the term. Enter two or more words "
            "to compare the two phrase indexes."
        )
    elif not bi_docs and not pos_docs and len(q_tokens) >= 2:
        st.info(
            f"**0 matches is a correct result here:** the exact phrase *\"{phrase_query}\"* does not occur "
            f"contiguously in any document, even though the individual words may appear separately "
            f"(check the positional postings below — the positions are never consecutive). "
            f"Try a phrase that actually occurs, e.g. *prime minister*, *economic growth* or *stock market prices*."
        )

    st.markdown("**Biword representation for query**")
    st.code(" | ".join([f"{q_tokens[i]} {q_tokens[i+1]}" for i in range(max(0, len(q_tokens)-1))]) or "Need at least two query terms")

    rep_col1, rep_col2 = st.columns(2)
    with rep_col1:
        st.markdown("**Biword index representation** (pairs containing the query terms)")
        related_biwords = [bw for bw in biword_index if any(t in bw.split() for t in q_tokens)]
        bi_rep_rows = [{"Biword": bw, "Posting List": ", ".join(sorted(biword_index[bw]))}
                       for bw in sorted(related_biwords)[:30]]
        if not bi_rep_rows:
            bi_rep_rows = [{"Biword": bw, "Posting List": ", ".join(sorted(docs))}
                           for bw, docs in list(biword_index.items())[:30]]
        st.dataframe(pd.DataFrame(bi_rep_rows), use_container_width=True)
    with rep_col2:
        st.markdown("**Positional index representation** (query terms with positions)")
        pos_rep_rows = []
        for t in q_tokens:
            postings = positional_index.get(t, {})
            pos_rep_rows.append({
                "Term": t,
                "Postings (doc → positions)": "; ".join(f"{d}: {p}" for d, p in list(postings.items())[:8]) or "—"
            })
        st.dataframe(pd.DataFrame(pos_rep_rows), use_container_width=True)

    st.markdown("**Comparison table**")
    phrase_rows = []
    queries = [phrase_query, "prime minister", "economic growth", "stock market prices", "government policy reform"]
    for q in queries:
        toks = preprocess_tokens(q, lower, stop, hyphen_mode, normalization)
        b = phrase_search_biword(toks, biword_index)
        p = phrase_search_positional(toks, positional_index)
        phrase_rows.append({"Phrase Query": q, "Biword Results": len(b), "Positional Results": len(p), "Biword False Positives": len(b-p), "More Accurate": "Positional Index" if len(b-p) else "Tie / Both Exact"})
    st.dataframe(pd.DataFrame(phrase_rows), use_container_width=True)

    res_col1, res_col2 = st.columns(2)
    with res_col1:
        st.markdown("**Query result using biword index**")
        rows = []
        for doc_id in sorted(bi_docs)[:top_k]:
            row = df[df["doc_id"].astype(str) == doc_id].iloc[0]
            rows.append({"Document ID": doc_id, "Title": row.get("title", ""), "False Positive?": "Yes" if doc_id in false_positive else "No"})
        st.dataframe(pd.DataFrame(rows), use_container_width=True)
    with res_col2:
        st.markdown("**Query result using positional index**")
        rows = []
        for doc_id in sorted(pos_docs)[:top_k]:
            row = df[df["doc_id"].astype(str) == doc_id].iloc[0]
            rows.append({"Document ID": doc_id, "Title": row.get("title", ""), "Snippet": make_snippet(row[text_col], phrase_query)})
        st.dataframe(pd.DataFrame(rows), use_container_width=True)

    st.markdown("### Inference: false positives and accuracy")
    if false_positive:
        fp_example = sorted(false_positive)[0]
        st.warning(
            f"**False positive case observed:** document `{fp_example}` matched the biword index but not the "
            f"positional index. Every consecutive word pair of the query appears somewhere in the document, "
            f"but never as one continuous phrase in the correct order."
        )
    st.markdown(
        """
        - **Why biword index gives false positives:** for a query of 3+ words it only verifies that each
          *pair* of adjacent words exists somewhere in the document. The pairs may come from different
          sentences, so the full phrase may never actually occur (e.g. query *"stock market prices"* matches a
          document containing *"stock market crash"* and *"food market prices"*).
        - **Why positional index is more accurate:** it stores the exact position of every term occurrence and
          verifies that positions are strictly consecutive (`pos`, `pos+1`, `pos+2`, …), guaranteeing the exact
          phrase occurs in order. The trade-off is a larger index and a slower merge of position lists.
        """
    )

with tabs[5]:
    st.subheader("Dictionary Search: Binary Search Tree vs B-Tree")
    st.write("The dictionary of terms is built from the vocabulary of the document collection. "
             "**Query search time** = time to locate the term in the tree dictionary. "
             "**Retrieval time** = search time + time to fetch the posting list and the matching documents.")

    insertion_order = st.radio(
        "Dictionary insertion order — applied to BOTH trees (experiment with tree skew)",
        ["Median-first (balanced)", "Sorted (worst case: skewed BST)"],
        horizontal=True,
    )
    repeats = st.slider("Timing repetitions per query (averaged)", 50, 1000, 200, step=50)

    def median_first(sorted_terms: List[str]) -> List[str]:
        """Recursively pick medians so the BST stays balanced."""
        order = []
        stack = [(0, len(sorted_terms) - 1)]
        while stack:
            lo, hi = stack.pop()
            if lo > hi:
                continue
            mid = (lo + hi) // 2
            order.append(sorted_terms[mid])
            stack.append((mid + 1, hi))
            stack.append((lo, mid - 1))
        return order

    bst_terms = vocab if insertion_order.startswith("Sorted") else median_first(vocab)

    t0 = time.perf_counter()
    bst = BST()
    for term in bst_terms:
        bst.insert(term)
    bst_build_ms = (time.perf_counter() - t0) * 1000

    # Same insertion order for both trees so the comparison is apples-to-apples;
    # the B-Tree stays balanced either way, which is exactly the point being demonstrated.
    t0 = time.perf_counter()
    btree = BTree(t=4)
    for term in bst_terms:
        btree.insert(term)
    btree_build_ms = (time.perf_counter() - t0) * 1000

    terms_to_test = st.text_area("Dictionary query terms, one per line", "economy\nfootball\nelection\ntechnology\ncryptocurrency\nminister").splitlines()

    def fetch_docs(term: str) -> List[str]:
        return list(inverted_index.get(term, {}).keys())

    rows = []
    for term in [t.strip().lower() for t in terms_to_test if t.strip()]:
        # Query search time: dictionary lookup only, averaged over many repetitions for stable numbers.
        t0 = time.perf_counter()
        for _ in range(repeats):
            found_bst = bst.search(term)
        bst_search = (time.perf_counter() - t0) * 1000 / repeats

        t0 = time.perf_counter()
        for _ in range(repeats):
            found_bt = btree.search(term)
        bt_search = (time.perf_counter() - t0) * 1000 / repeats

        # Retrieval time: dictionary lookup + posting list fetch + document fetch.
        t0 = time.perf_counter()
        for _ in range(repeats):
            docs = fetch_docs(term) if bst.search(term) else []
        bst_retrieval = (time.perf_counter() - t0) * 1000 / repeats

        t0 = time.perf_counter()
        for _ in range(repeats):
            docs = fetch_docs(term) if btree.search(term) else []
        bt_retrieval = (time.perf_counter() - t0) * 1000 / repeats

        rows.append({
            "Query Term": term,
            "Found?": found_bst and found_bt,
            "Matched Docs": len(fetch_docs(term)),
            "BST Search Time (ms)": round(bst_search, 6),
            "B-Tree Search Time (ms)": round(bt_search, 6),
            "BST Retrieval Time (ms)": round(bst_retrieval, 6),
            "B-Tree Retrieval Time (ms)": round(bt_retrieval, 6),
            "Faster (Search)": "BST" if bst_search < bt_search else "B-Tree",
            "Faster (Retrieval)": "BST" if bst_retrieval < bt_retrieval else "B-Tree",
        })

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Dictionary Terms", len(vocab))
    c2.metric("BST Height", bst.height())
    c3.metric("B-Tree Height", btree.height())
    c4.metric("BST Build (ms)", round(bst_build_ms, 2))
    c5.metric("B-Tree Build (ms)", round(btree_build_ms, 2))

    results_df = pd.DataFrame(rows)
    st.dataframe(results_df, use_container_width=True)

    if not results_df.empty:
        avg_bst = results_df["BST Search Time (ms)"].mean()
        avg_bt = results_df["B-Tree Search Time (ms)"].mean()
        winner = "B-Tree" if avg_bt < avg_bst else "BST"
        st.success(
            f"**Experimental inference:** average search time — BST: {avg_bst:.6f} ms, B-Tree: {avg_bt:.6f} ms → "
            f"**{winner}** was faster on average for this run. "
            f"BST height is {bst.height()} vs B-Tree height {btree.height()}: the B-Tree stays shallow and balanced "
            f"regardless of insertion order, while BST performance depends strongly on insertion order — "
            f"switch the insertion order above to 'Sorted' to see the BST degenerate into a linked list."
        )

with tabs[6]:
    st.subheader("Tolerant Retrieval")
    tol_method = st.selectbox("Tolerant retrieval method", ["Wildcard", "Spelling Correction / Edit Distance", "K-Gram Suggestions", "Phonetic Correction (Soundex)"])
    term = st.text_input("Imperfect query term", value="goverment" if tol_method != "Wildcard" else "govern*", key=f"tol_term_{tol_method}")

    if tol_method == "Wildcard":
        expanded = sorted([v for v in vocab if fnmatch.fnmatch(v, term.lower())])[:50]
        matched_docs = set()
        for e in expanded:
            matched_docs |= set(inverted_index.get(e, {}).keys())
        st.write("Expanded terms:", expanded)
        st.metric("Matching documents", len(matched_docs))
    elif tol_method == "Spelling Correction / Edit Distance":
        suggestions = suggest_correction(term, vocab)
        st.dataframe(pd.DataFrame(suggestions, columns=["Suggested Correction", "Edit Distance"]), use_container_width=True)
        if suggestions:
            corrected = suggestions[0][0]
            st.success(f"Best correction: {corrected}")
            res = ranked_tfidf(df, text_col, corrected, top_k, lower, stop, hyphen_mode, normalization)
            st.dataframe(res, use_container_width=True)
    elif tol_method == "K-Gram Suggestions":
        k = 3
        padded = f"${term.lower()}$"
        grams = [padded[i:i+k] for i in range(max(0, len(padded)-k+1))]
        candidate_counter = Counter()
        for gram in grams:
            for candidate in kgram_index.get(gram, set()):
                candidate_counter[candidate] += 1
        candidates = candidate_counter.most_common(10)
        st.write("Query k-grams:", grams)
        st.dataframe(pd.DataFrame(candidates, columns=["Candidate Term", "Shared K-Grams"]), use_container_width=True)
    else:
        code = soundex(term)
        st.write(f"Soundex code for `{term}`: **{code}**")
        matches = phonetic_matches(term, vocab)
        st.dataframe(
            pd.DataFrame(matches, columns=["Phonetically Similar Term", "Soundex Code", "Edit Distance"]),
            use_container_width=True,
        )
        if matches:
            corrected = matches[0][0]
            st.success(f"Best phonetic correction: {corrected}")
            res = ranked_tfidf(df, text_col, corrected, top_k, lower, stop, hyphen_mode, normalization)
            st.dataframe(res, use_container_width=True)
        else:
            st.warning("No vocabulary term shares this Soundex code.")

with tabs[7]:
    st.subheader("Inference Dashboard")
    st.markdown(
        """
        ### Key experimental inferences
        1. **Preprocessing:** Lowercasing and stopword removal reduce vocabulary noise and improve matching consistency.
        2. **Stemming vs Lemmatization:** Measured by average TF-IDF cosine similarity of retrieved documents, lemmatization is recommended for news documents because it preserves readable base forms, while stemming is faster but more aggressive.
        3. **Phrase Query Accuracy:** Positional index is more accurate for multi-word phrase queries because it checks exact word positions and order; the biword index produces measurable false positives for 3+ word phrases.
        4. **Dictionary Structure:** B-Tree is more scalable because it stays shallow regardless of insertion order; BST search and retrieval time degrade badly when terms arrive sorted (skewed tree).
        5. **Tolerant Retrieval:** Wildcard, edit distance, k-gram and phonetic (Soundex) suggestions make the system robust for partial words and misspellings.
        6. **Limitations:** The current system is lexical, not deeply semantic. Very large collections should use persistent disk-based indexes.
        7. **Future Improvements:** Add BM25, vector embeddings, relevance feedback, PDF/DOCX parsing and persistent index storage.
        """
    )
    summary = pd.DataFrame([
        {"Component": "Streamlit workflow", "Status": "Implemented", "Evidence in App": "Upload, view, query, choose options, display outputs"},
        {"Component": "Preprocessing", "Status": "Implemented", "Evidence in App": "Tokenization, lowercasing, stopword removal, hyphen handling, stemming, lemmatization"},
        {"Component": "Inverted index", "Status": "Implemented", "Evidence in App": "Term, document frequency, posting list with TF"},
        {"Component": "Stemming vs lemmatization", "Status": "Implemented", "Evidence in App": "Query-wise comparison table with avg TF-IDF cosine similarity and conclusion"},
        {"Component": "Phrase query", "Status": "Implemented", "Evidence in App": "Biword/positional index representations, query results from both, false-positive analysis"},
        {"Component": "BST vs B-Tree", "Status": "Implemented", "Evidence in App": "Search time, retrieval time, build time, height; sorted vs balanced insertion experiment"},
        {"Component": "Tolerant retrieval", "Status": "Implemented", "Evidence in App": "Wildcard, edit distance, k-gram and phonetic (Soundex) suggestions"},
    ])
    st.dataframe(summary, use_container_width=True)
    st.download_button("Download inference summary as CSV", data=summary.to_csv(index=False), file_name="smartir_inference_summary.csv", mime="text/csv")
