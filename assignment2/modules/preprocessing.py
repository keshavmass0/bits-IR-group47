"""
preprocessing.py
-----------------
Dependency-free text preprocessing (Section C). Deliberately avoids nltk/spacy
downloads (which need internet the first time they run) so it works
unmodified on the BITS virtual lab, consistent with Assignment 1's approach.
"""
from __future__ import annotations

import re
from typing import List

STOPWORDS = {
    "a", "an", "the", "and", "or", "but", "if", "while", "with", "without", "of", "to", "in",
    "on", "for", "from", "by", "as", "at", "is", "are", "was", "were", "be", "been", "being",
    "this", "that", "these", "those", "it", "its", "into", "about", "after", "before", "than",
    "then", "also", "may", "will", "would", "should", "could", "can", "has", "have", "had",
    "do", "does", "did", "not", "said", "their", "his", "her", "they", "them", "he", "she",
    "we", "you", "i", "our", "your", "there", "which", "who", "whom", "what", "when", "where",
}

_SUFFIX_RULES = [
    ("ization", "ize"), ("ational", "ate"), ("fulness", "ful"), ("ousness", "ous"),
    ("iveness", "ive"), ("tional", "tion"), ("ments", "ment"), ("ingly", ""),
    ("edly", ""), ("ing", ""), ("ied", "y"), ("ies", "y"), ("ed", ""),
    ("ly", ""), ("es", ""), ("s", ""),
]

_LEMMAS = {
    "children": "child", "men": "man", "women": "woman", "people": "person", "mice": "mouse",
    "geese": "goose", "feet": "foot", "teeth": "tooth", "better": "good", "best": "good",
    "worse": "bad", "won": "win", "winning": "win", "ran": "run", "running": "run",
    "companies": "company", "policies": "policy", "prices": "price", "services": "service",
    "families": "family", "technologies": "technology", "queries": "query", "documents": "document",
    "matches": "match", "players": "player", "leaders": "leader", "reports": "report",
}


def clean_text(text: str) -> str:
    """Lowercase, strip markup/punctuation noise, collapse whitespace."""
    text = str(text)
    text = re.sub(r"<[^>]+>", " ", text)          # strip any HTML remnants from crawled pages
    text = re.sub(r"http\S+|www\.\S+", " ", text)  # strip URLs
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s-]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def tokenize(text: str) -> List[str]:
    return re.findall(r"[a-z0-9]+", clean_text(text))


def remove_stopwords(tokens: List[str]) -> List[str]:
    return [t for t in tokens if t not in STOPWORDS and len(t) > 1]


def stem(word: str) -> str:
    """Transparent suffix-stripping Porter-style stemmer (no external download)."""
    w = word.lower()
    for suffix, repl in _SUFFIX_RULES:
        if len(w) > len(suffix) + 2 and w.endswith(suffix):
            return w[: -len(suffix)] + repl
    return w


def lemmatize(word: str) -> str:
    """Lightweight lemmatizer using an irregular-form lookup + simple suffix rules."""
    w = word.lower()
    if w in _LEMMAS:
        return _LEMMAS[w]
    if len(w) > 4 and w.endswith("ies"):
        return w[:-3] + "y"
    if len(w) > 5 and w.endswith("ing"):
        return w[:-3]
    if len(w) > 3 and w.endswith("s") and not w.endswith("ss"):
        return w[:-1]
    return w


def preprocess(text: str, use_stopwords: bool = True, normalize: str = "none") -> List[str]:
    """Full pipeline: tokenize -> (stopword removal) -> (stem|lemma|none)."""
    tokens = tokenize(text)
    if use_stopwords:
        tokens = remove_stopwords(tokens)
    if normalize == "stem":
        tokens = [stem(t) for t in tokens]
    elif normalize == "lemma":
        tokens = [lemmatize(t) for t in tokens]
    return tokens


def preprocess_to_string(text: str, use_stopwords: bool = True, normalize: str = "none") -> str:
    return " ".join(preprocess(text, use_stopwords, normalize))


def ngrams(tokens: List[str], n: int = 2) -> List[str]:
    return [" ".join(tokens[i:i + n]) for i in range(len(tokens) - n + 1)]


__all__ = ["clean_text", "tokenize", "remove_stopwords", "stem", "lemmatize",
           "preprocess", "preprocess_to_string", "ngrams", "STOPWORDS"]
