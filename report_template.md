# SmartIR Lab: Streamlit-Based Advanced Information Retrieval System

## 1. Introduction

This project implements an end-to-end Information Retrieval system using Streamlit. The system allows users to upload a document collection, inspect preprocessing, build indexes, execute different retrieval models, compare phrase query techniques, compare dictionary structures, and test tolerant retrieval.

## 2. Objective

The objective is to design and implement an interactive IR system where the complete workflow is executable from the Streamlit front end. The system demonstrates preprocessing, indexing, querying, phrase search, dictionary lookup and tolerant retrieval.

## 3. Dataset Description

Dataset used: BBC News Articles / BBC-style news document collection.

The selected dataset is suitable because news documents contain meaningful domain vocabulary and common phrase queries such as “prime minister”, “economic growth”, “stock market”, “football match” and “technology company”. This makes the collection appropriate for testing preprocessing, phrase search and tolerant retrieval.

Fields used:

| Field | Description |
|---|---|
| doc_id | Unique document identifier |
| category | News category |
| title | Document title |
| text | Main document content |

## 4. System Architecture

```text
Dataset Upload
→ Document Viewer
→ Text Preprocessing
→ Inverted Index Construction
→ Query Processing
→ Retrieval Model Selection
→ Ranked / Boolean / Phrase Retrieval
→ Tolerant Retrieval
→ Experimental Inference Dashboard
```

## 5. Streamlit Workflow

The Streamlit app contains the following tabs:

1. Dataset Overview
2. Preprocessing
3. Inverted Index
4. Search Results
5. Phrase Comparison
6. BST vs B-Tree
7. Tolerant Retrieval
8. Inference Dashboard

Add screenshots here from Virtual Lab:

- Screenshot 1: Dataset upload and overview
- Screenshot 2: Preprocessing output
- Screenshot 3: Inverted index table
- Screenshot 4: Phrase query comparison
- Screenshot 5: BST vs B-Tree comparison
- Screenshot 6: Tolerant retrieval output

## 6. Text Preprocessing

The following preprocessing steps were implemented and displayed on the front end:

| Step | Purpose |
|---|---|
| Tokenization | Splits documents into individual terms |
| Lowercasing | Normalizes terms and avoids case mismatch |
| Stop word removal | Removes frequent non-informative terms |
| Hyphen handling | Processes terms such as prime-minister or long-term |
| Stemming | Reduces words to crude root forms |
| Lemmatization | Reduces words to meaningful dictionary base forms |

## 7. Inverted Index

The inverted index maps every term to a posting list of documents. The implementation also stores term frequency.

Example format:

```text
economy → {D001: 2, D006: 1, D010: 1}
```

Displayed fields:

| Term | Document Frequency | Posting List with TF |
|---|---:|---|

## 8. Stemming vs Lemmatization Comparison

The system runs test queries using both stemmed and lemmatized document representations. Retrieval quality is measured as the **average TF-IDF cosine similarity** between the query and the documents retrieved by each pipeline (a semantic relevance proxy — higher means the retrieved set is more on-topic).

Experimental results (sample dataset, default preprocessing: lowercase + stopword removal + hyphen split):

| Query | Stemmed Results | Lemmatized Results | Stem Avg Similarity | Lemma Avg Similarity | Jaccard Similarity | Better |
|---|---:|---:|---:|---:|---:|---|
| economic growth | 2 | 2 | 0.2725 | 0.2725 | 1.000 | Lemmatization |
| market price | 0 | 2 | 0.0000 | 0.2053 | 0.000 | Lemmatization |
| national service | 0 | 1 | 0.0000 | 0.1905 | 0.000 | Lemmatization |
| football match | 2 | 2 | 0.3468 | 0.3468 | 1.000 | Lemmatization |
| technology company | 2 | 2 | 0.1174 | 0.1174 | 1.000 | Lemmatization |

Inference:

Based on the average retrieval similarity scores, lemmatization is more suitable for the selected news dataset. The queries *market price* and *national service* show the failure mode of stemming concretely: the stemmer maps the document term *prices → pric* and *services → servic*, while the query terms *price* and *service* stem to themselves — so the stemmed query never matches the stemmed documents and **recall drops to zero**. Lemmatization maps both query and document forms to the same dictionary base word (*prices → price*, *services → service*), retrieving the relevant documents (avg similarity 0.2053 and 0.1905 vs 0.0000 for stemming). Stemming is faster but over-truncates, which both reduces interpretability and, as measured, can destroy recall.

## 9. Phrase Query Processing

Two phrase query techniques were implemented.

### 9.1 Biword Index

The biword index stores adjacent term pairs.

Example:

```text
prime minister → D002, D011, D021
stock market → D001, D010
```

### 9.2 Positional Index

The positional index stores term positions in each document.

Example:

```text
prime → {D002: [0], D011: [8], D021: [0]}
minister → {D002: [1], D011: [9], D021: [1]}
```

### 9.3 Comparison

Experimental results (sample dataset, default preprocessing):

| Phrase Query | Biword Results | Positional Results | Biword False Positives | More Accurate |
|---|---|---|---|---|
| prime minister | 3 (D002, D011, D021) | 3 (D002, D011, D021) | 0 | Tie / Both Exact |
| economic growth | 2 (D001, D006) | 2 (D001, D006) | 0 | Tie / Both Exact |
| stock market prices | 2 (D010, D026) | 1 (D010) | 1 (D026) | Positional Index |
| government policy reform | 0 | 0 | 0 | Tie / Both Exact |

**False positive case observed:** for the query *"stock market prices"*, the biword index returns D026 in addition to the correct D010. D026 contains *"followed the **stock market** all week while global food **market prices** rose"* — both biwords *stock market* and *market prices* exist in the document, but the full three-word phrase never occurs contiguously. The positional index correctly rejects D026 because the positions of *stock*, *market*, *prices* are never consecutive.

Inference:

For two-word phrases the biword index is exact (a biword *is* the phrase), which is why *prime minister* and *economic growth* tie. For phrases of three or more words the biword index only checks that each adjacent pair occurs *somewhere* in the document, so pairs from different sentences can combine into a false positive (D026). The positional index verifies positions are strictly consecutive (`pos, pos+1, pos+2`), guaranteeing the exact phrase in order — at the cost of a larger index and a slower position-list merge.

## 10. Dictionary Search using BST and B-Tree

A vocabulary dictionary was created from the document collection. Two dictionary structures were compared:

1. Binary Search Tree
2. B-Tree

Two timings are reported for multiple queries, each averaged over many repetitions:

- **Query search time** — time to locate the term in the tree dictionary.
- **Retrieval time** — search time plus fetching the posting list and matching documents.

Experimental table — fill the timing columns from your own Virtual Lab run (timings are machine-dependent; the Found/Docs columns below are fixed for the sample dataset, vocabulary = 307 terms):

| Query Term | BST Search (ms) | B-Tree Search (ms) | BST Retrieval (ms) | B-Tree Retrieval (ms) | Found? | Matched Docs | Faster |
|---|---:|---:|---:|---:|---|---:|---|
| economy |  |  |  |  | True | 1 |  |
| football |  |  |  |  | True | 2 |  |
| election |  |  |  |  | True | 1 |  |
| technology |  |  |  |  | True | 5 |  |
| cryptocurrency |  |  |  |  | False | 0 |  |
| minister |  |  |  |  | True | 3 |  |

Also record: BST height, B-Tree height, BST build time, B-Tree build time — under both **median-first (balanced)** and **sorted (skewed)** insertion order (the app applies the chosen order to both trees; measured heights for 307 terms: balanced BST ≈ 9, skewed BST = 307, B-Tree = 3–4 in both cases).

Inference:

B-Tree is more scalable for large dictionaries because it remains shallow and balanced regardless of insertion order. BST performance depends strongly on insertion order: with sorted insertion the BST degenerates into a linked list (height = number of terms) and search time grows linearly, while the B-Tree height stays logarithmic.

## 11. Tolerant Retrieval

The system demonstrates tolerant retrieval using the following methods:

Experimental results (sample dataset):

| Method | Example | Output |
|---|---|---|
| Wildcard query | govern* | government |
| Edit distance correction | goverment | government (distance 1) |
| K-gram suggestions | econmy | economy |
| Phonetic correction (Soundex) | goverment (G165) | government (G165, distance 1) |

Inference:

The system can handle common spelling errors, prefix queries and partial words. It may still struggle with very short terms, highly ambiguous misspellings and semantic mismatches.

## 12. Limitations

1. The current retrieval model is mainly lexical.
2. It does not deeply understand semantic meaning.
3. Very large datasets may require disk-based persistent indexing.
4. Biword index can generate false positives for long phrase queries.
5. Spelling correction may fail for highly ambiguous terms.
6. The system currently focuses on English text.

## 13. Future Improvements

1. Add BM25 ranking.
2. Add semantic search using sentence transformers.
3. Add hybrid ranking using TF-IDF and embeddings.
4. Add relevance feedback.
5. Add PDF and DOCX parsing.
6. Store indexes persistently using SQLite or disk files.
7. Add multilingual retrieval support.

## 14. Conclusion

SmartIR Lab successfully implements an end-to-end Information Retrieval workflow in Streamlit. It demonstrates document upload, preprocessing, inverted indexing, ranked retrieval, phrase query processing, dictionary search and tolerant retrieval. The system provides experimental tables and inferences, making it suitable for academic evaluation and practical IR learning.
