# SmartIR Lab: Advanced Information Retrieval System

SmartIR Lab is a Streamlit-based end-to-end Information Retrieval system designed for the BITS Information Retrieval Assignment 1. It allows the evaluator to upload a document collection, inspect preprocessing outputs, build indexes, run queries, compare retrieval techniques, and view experimental inferences directly from the front end.

## Features

- Upload CSV or TXT document collections
- View uploaded documents and dataset statistics
- Tokenization and inverted index creation
- Lowercasing
- Stop word removal
- Hyphen handling
- Stemming and lemmatization
- Stemming vs lemmatization retrieval comparison with average TF-IDF cosine similarity (retrieval quality measure) and an automatic conclusion
- TF-IDF ranked retrieval
- Boolean retrieval with AND, OR and NOT
- Phrase query processing using:
  - Biword index (index representation + query results displayed)
  - Positional index (index representation + query results displayed)
- False-positive analysis for biword phrase search with worked example
- Dictionary search using:
  - Binary Search Tree (with balanced vs sorted/skewed insertion experiment)
  - B-Tree
- Query search time AND retrieval time comparison for BST vs B-Tree (averaged over configurable repetitions), plus build time and tree height
- Tolerant retrieval using:
  - Wildcard queries
  - Edit distance spelling correction
  - K-gram suggestions
  - Phonetic correction (Soundex)
- Inference dashboard with downloadable summary

## Recommended Dataset

Use the BBC News Summary / BBC News Articles dataset. It is easily available on Kaggle and contains real-world news articles from categories such as business, entertainment, politics, sport and technology.

A small BBC-style sample dataset is already included as `sample_bbc_news.csv` so the project runs immediately even without downloading an external dataset.

## Project Structure

```text
SmartIR_Lab/
├── app.py
├── requirements.txt
├── sample_bbc_news.csv
├── README.md
└── report_template.md
```

## Installation

Create and activate a virtual environment if required:

```bash
python -m venv venv
venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Run the Application

```bash
streamlit run app.py
```

## How to Use

1. Start the Streamlit app.
2. Use the included sample dataset or upload your own CSV/TXT collection.
3. Select the text column if using CSV.
4. Choose preprocessing options from the sidebar.
5. Select a retrieval model.
6. Enter a query.
7. Explore each tab:
   - Dataset Overview
   - Preprocessing
   - Inverted Index
   - Search Results
   - Phrase Comparison
   - BST vs B-Tree
   - Tolerant Retrieval
   - Inference Dashboard

## Suggested Demo Queries

```text
prime minister economic policy
economic growth
stock market prices
football match
government policy reform
technology company
```

Queries where stemming and lemmatization visibly differ (stemming retrieves 0 documents, lemmatization retrieves the relevant ones):

```text
market price
national service
```

Phrase query that demonstrates a biword false positive (D026 matches biword but not positional):

```text
stock market prices
```

## Suggested Tolerant Retrieval Queries

Wildcard:

```text
govern*
econom*
tech*
```

Spelling correction:

```text
goverment
economi
futball
tecnology
```

K-gram suggestions:

```text
econmy
govermnt
technlogy
```

Phonetic correction (Soundex):

```text
goverment
ekonomy
fonetic
```

## Notes for Assignment Report

Use `report_template.md` to prepare the final written report. Add screenshots from your own Streamlit run in the Virtual Lab portal.

## Important Inference Summary

- Lemmatization is generally better for news documents because it preserves meaningful base words.
- Positional index is more accurate than biword index for long phrase queries.
- B-Tree is more scalable than BST for large dictionary lookup.
- Tolerant retrieval improves usability for spelling errors, wildcard queries and partial terms.
