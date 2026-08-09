"""
text_mining.py
--------------
Section C: "scalable text mining framework that transforms raw textual data
into structured feature representations through preprocessing, feature
engineering, and statistical analysis" -- keyword extraction, document
profiling, document classification, and comparative analysis of preprocessing
/ feature-extraction strategies.
"""
from __future__ import annotations

import re
from collections import Counter
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

from modules.preprocessing import STOPWORDS, preprocess_to_string, tokenize


# ------------------------------------------------------------------
# Feature engineering
# ------------------------------------------------------------------
def build_bow(texts: List[str], max_features: int = 5000, stop_words: str = "english"
              ) -> Tuple[CountVectorizer, np.ndarray]:
    vec = CountVectorizer(max_features=max_features, stop_words=stop_words)
    matrix = vec.fit_transform(texts)
    return vec, matrix


def build_tfidf(texts: List[str], max_features: int = 5000, stop_words: str = "english"
                 ) -> Tuple[TfidfVectorizer, np.ndarray]:
    # stop_words="english" uses sklearn's bundled stopword list (no download
    # needed) so keyword extraction below doesn't surface function words like
    # "and"/"the" as top "keywords".
    vec = TfidfVectorizer(max_features=max_features, stop_words=stop_words)
    matrix = vec.fit_transform(texts)
    return vec, matrix


# ------------------------------------------------------------------
# Keyword extraction
# ------------------------------------------------------------------
def top_keywords_tfidf(tfidf_vec: TfidfVectorizer, tfidf_matrix, doc_idx: int, top_n: int = 8
                        ) -> List[Tuple[str, float]]:
    row = tfidf_matrix[doc_idx].toarray().ravel()
    if row.sum() == 0:
        return []
    top_ids = row.argsort()[::-1][:top_n]
    features = tfidf_vec.get_feature_names_out()
    return [(features[i], round(float(row[i]), 4)) for i in top_ids if row[i] > 0]


def rake_keywords(text: str, top_n: int = 8) -> List[Tuple[str, float]]:
    """A compact RAKE (Rapid Automatic Keyword Extraction) implementation.

    Splits text into candidate phrases at stopword/punctuation boundaries,
    scores each word by degree(co-occurrence)/frequency, and scores each
    phrase as the sum of its word scores -- the standard RAKE formulation,
    written from scratch so no extra dependency (e.g. `rake-nltk`) is needed.
    """
    words = re.findall(r"[a-z0-9]+", text.lower())
    phrases: List[List[str]] = []
    current: List[str] = []
    for w in words:
        if w in STOPWORDS:
            if current:
                phrases.append(current)
                current = []
        else:
            current.append(w)
    if current:
        phrases.append(current)

    freq: Counter = Counter()
    degree: Counter = Counter()
    for phrase in phrases:
        deg = len(phrase) - 1
        for w in phrase:
            freq[w] += 1
            degree[w] += deg

    word_score = {w: (degree[w] + freq[w]) / freq[w] for w in freq}
    phrase_scores = [(" ".join(p), round(sum(word_score[w] for w in p), 3)) for p in phrases if p]
    phrase_scores.sort(key=lambda x: x[1], reverse=True)

    # de-duplicate phrases
    seen = set()
    result = []
    for phrase, score in phrase_scores:
        if phrase not in seen:
            seen.add(phrase)
            result.append((phrase, score))
        if len(result) >= top_n:
            break
    return result


# ------------------------------------------------------------------
# Document profiling
# ------------------------------------------------------------------
def _count_syllables(word: str) -> int:
    word = word.lower()
    vowels = "aeiouy"
    count, prev_vowel = 0, False
    for ch in word:
        is_vowel = ch in vowels
        if is_vowel and not prev_vowel:
            count += 1
        prev_vowel = is_vowel
    if word.endswith("e") and count > 1:
        count -= 1
    return max(count, 1)


def flesch_reading_ease(text: str) -> float:
    """Approximate Flesch Reading Ease score (0-100, higher = easier to read)."""
    sentences = max(len(re.findall(r"[.!?]+", text)), 1)
    words = tokenize(text)
    if not words:
        return 0.0
    syllables = sum(_count_syllables(w) for w in words)
    score = 206.835 - 1.015 * (len(words) / sentences) - 84.6 * (syllables / len(words))
    return round(max(0.0, min(100.0, score)), 1)


def document_profile(doc_id: str, text: str, category: str,
                      tfidf_vec: TfidfVectorizer, tfidf_matrix, doc_idx: int) -> Dict:
    tokens = tokenize(text)
    return {
        "id": doc_id,
        "category": category,
        "word_count": len(tokens),
        "unique_words": len(set(tokens)),
        "lexical_diversity": round(len(set(tokens)) / len(tokens), 3) if tokens else 0.0,
        "readability_flesch": flesch_reading_ease(text),
        "top_keywords_tfidf": top_keywords_tfidf(tfidf_vec, tfidf_matrix, doc_idx, top_n=6),
        "top_keywords_rake": rake_keywords(text, top_n=6),
    }


# ------------------------------------------------------------------
# Document classification
# ------------------------------------------------------------------
def classify_documents(texts: List[str], labels: List[str], feature_type: str = "tfidf",
                        model_type: str = "nb", test_size: float = 0.25, random_state: int = 42,
                        min_class_size: int = 2) -> Dict:
    """Trains a classifier and returns accuracy/F1/confusion matrix.

    feature_type: "bow" | "tfidf"
    model_type:   "nb" (MultinomialNB) | "logreg" (LogisticRegression)

    Classes with fewer than `min_class_size` documents are dropped before the
    stratified split (rather than crashing): a stratified train_test_split
    needs every class represented in both splits, which is impossible for a
    singleton class. This matters in practice once the working corpus
    includes crawled navigational pages (e.g. the crawler's "home" page is
    a single document with its own pseudo-category) mixed in with real news
    categories that have hundreds of documents each.
    """
    counts = Counter(labels)
    dropped_classes = sorted(c for c, n in counts.items() if n < min_class_size)
    if dropped_classes:
        keep = [i for i, lab in enumerate(labels) if counts[lab] >= min_class_size]
        texts = [texts[i] for i in keep]
        labels = [labels[i] for i in keep]

    vec = (CountVectorizer(max_features=5000, stop_words="english") if feature_type == "bow"
           else TfidfVectorizer(max_features=5000, stop_words="english"))
    X = vec.fit_transform(texts)

    X_train, X_test, y_train, y_test = train_test_split(
        X, labels, test_size=test_size, random_state=random_state, stratify=labels
    )
    model = MultinomialNB() if model_type == "nb" else LogisticRegression(max_iter=500)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    classes = sorted(set(labels))
    cm = confusion_matrix(y_test, preds, labels=classes)
    return {
        "accuracy": round(accuracy_score(y_test, preds), 4),
        "f1_macro": round(f1_score(y_test, preds, average="macro"), 4),
        "confusion_matrix": pd.DataFrame(cm, index=classes, columns=classes),
        "n_train": X_train.shape[0],
        "n_test": X_test.shape[0],
        "classes": classes,
        "dropped_classes": dropped_classes,
    }


def comparative_analysis(df: pd.DataFrame, text_col: str = "text", label_col: str = "category"
                          ) -> pd.DataFrame:
    """Compares {BoW, TF-IDF} x {raw, stemmed, lemmatized} on classification
    accuracy/F1 -- the "comparative analysis of different preprocessing and
    feature extraction strategies" required by Section C.
    """
    raw_texts = df[text_col].astype(str).tolist()
    labels = df[label_col].astype(str).tolist()

    rows = []
    for normalize in ("none", "stem", "lemma"):
        processed = [preprocess_to_string(t, use_stopwords=True, normalize=normalize) for t in raw_texts]
        for feature_type in ("bow", "tfidf"):
            result = classify_documents(processed, labels, feature_type=feature_type)
            rows.append({
                "normalization": normalize,
                "feature_type": feature_type.upper(),
                "accuracy": result["accuracy"],
                "f1_macro": result["f1_macro"],
                "n_train": result["n_train"],
                "n_test": result["n_test"],
            })
    return pd.DataFrame(rows)


__all__ = ["build_bow", "build_tfidf", "top_keywords_tfidf", "rake_keywords",
           "flesch_reading_ease", "document_profile", "classify_documents", "comparative_analysis"]
