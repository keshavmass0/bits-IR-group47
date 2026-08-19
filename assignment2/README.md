# SmartIR-2: End-to-End Information Retrieval System

**BITS WILP — Information Retrieval (Merged AIMLCZG537/DSECLZG537), S2-25 — Assignment 2 — Team 47**

SmartIR-2 is a single Streamlit application covering the complete IR lifecycle end-to-end:
web crawling → text preprocessing & mining → indexing → search & ranking (TF-IDF +
PageRank/HITS) → recommendation (content-based, collaborative, hybrid) → evaluation
(Precision/Recall/F1/P@K/R@K/MAP/MRR/NDCG) → inference & discussion. Every step is driven
from the Streamlit front end only — there is no separate notebook or backend script the
evaluator needs to run.

> **🔗 Live demo — test it right now, no installation needed:**
> **https://assignment2-irgroup47.streamlit.app/**
> Deployed on Streamlit Community Cloud directly from this repo's `assignment2/` folder
> (main file path `assignment2/app.py`), redeploying automatically on every push to `main`.
> All 9 tabs are fully interactive — crawl, build the index, search, get recommendations,
> and run the evaluation comparison directly in the browser.

## Project structure

```text
assignment2/
├── app.py                      # Streamlit front end (9 tabs, see below)
├── requirements.txt
├── README.md
├── modules/
│   ├── mock_web.py             # Deterministic local "web" graph + optional live-internet fetch
│   ├── mock_api.py             # Simulated news-API source
│   ├── crawler.py              # BFS crawler: depth control, multi-seed, dedup
│   ├── corpus.py                # Merges dataset + API + crawl sources into one corpus
│   ├── storage.py               # SQLite persistence, metadata table separate from content table
│   ├── preprocessing.py         # Tokenization, stopwords, stemming, lemmatization
│   ├── text_mining.py           # Keyword extraction (TF-IDF/RAKE), profiling, classification
│   ├── indexing.py              # Inverted index + TF-IDF index management
│   ├── ranking.py               # Manual PageRank + HITS (numpy, no networkx dependency)
│   ├── search.py                # Query optimization, two-stage retrieval, ranking blends
│   ├── recommender.py           # Content-based / collaborative (synthetic ratings) / hybrid
│   └── evaluation.py            # IR metrics + query-set builder + method comparison
├── data/
│   ├── news_corpus.csv          # Base dataset (BBC-style news, 1,100 articles, 5 categories)
│   ├── ir_store.db              # Generated at runtime: SQLite metadata + content tables
│   └── synthetic_ratings.csv    # Generated at runtime: synthetic user-item ratings for CF
├── report/
│   ├── REPORT.md                       # Full assignment report source (see submission checklist below)
│   ├── SmartIR-2_Report_Team47.pdf     # Submission-ready PDF rendering of the above
│   └── generate_pdf.py                 # Regenerates the PDF after editing REPORT.md
├── screenshots/                 # App screenshots embedded throughout REPORT.md §2-8
└── bits_lab_screenshots/        # 17 real BITS Virtual Lab captures embedded in REPORT.md §9
```

## Installation

```bash
python -m venv venv
source venv/bin/activate        # venv\Scripts\activate on Windows
pip install -r requirements.txt
```

Dependencies are intentionally minimal (`streamlit`, `pandas`, `numpy`, `scikit-learn` — the
same four packages Assignment 1 used) so installation succeeds reliably on the BITS Virtual
Lab. No `nltk`/`spacy` model downloads, no `requests`/`beautifulsoup4`, no internet access is
required to run every feature of this app.

## Run the application

```bash
streamlit run app.py
```

Then open the printed local URL in a browser. All 9 sections are tabs across the top:

1. 🏠 **Dashboard** — corpus overview, source/category breakdown, pipeline map.
2. 🕷️ **Crawling** — configurable seeds/depth/max-pages, duplicate handling, metadata vs.
   content storage, crawl history.
3. 🧪 **Text Mining** — corpus statistics, keyword extraction (TF-IDF & RAKE), document
   profiling, classification, comparative analysis of preprocessing/feature strategies.
4. 🗂️ **Index Management** — build/rebuild the inverted index + TF-IDF matrix, inspect
   index statistics and postings lists.
5. 🔍 **Search & Ranking** — query processing (synonym expansion, spelling correction),
   ranked retrieval, and a side-by-side comparison of TF-IDF-only vs. PageRank-boosted vs.
   HITS-boosted ranking, showing exactly how much re-ranking changes result order.
6. ⭐ **Recommendations** — content-based, collaborative, and hybrid Top-K recommendations
   with scores.
7. 📊 **Evaluation** — Precision, Recall, F1, Precision@K, Recall@K, MAP, MRR, NDCG,
   compared across ranking methods, with tables and charts.
8. 📈 **Performance Analytics** — timing log for every operation, corpus growth, index
   footprint and a scalability note.
9. 🧠 **Inference & Discussion** — written answers to all five compulsory discussion
   questions, plus a one-click downloadable inference summary.

## Design notes (read before grading)

- **Why crawling isn't against live URLs by default:** the BITS Virtual Lab has no
  guaranteed outbound internet access. To make Section B's crawling requirements
  (configurable depth, multiple seeds, duplicate handling, metadata/content separation)
  demonstrable and reliably reproducible, the crawler runs a genuine breadth-first
  traversal (`modules/crawler.py`) over a deterministic **local web graph**
  (`modules/mock_web.py`) built from the same news corpus — home page → category hubs →
  articles → related-story links — with intentionally injected duplicate and
  near-duplicate mirror pages. The crawling *logic* is identical to what would run
  against real URLs (it is plain BFS over whatever `fetch(url)` returns); a "attempt live
  internet crawl" checkbox in the Crawling tab additionally lets you crawl real URLs with
  only the Python standard library (`urllib`) when internet access *is* available (e.g.
  running locally). This is discussed further in the report.
- **Why collaborative filtering uses a synthetic rating matrix:** the underlying corpus is
  a static news dataset with no real user-interaction log. A reproducible (fixed-seed)
  synthetic user-item rating matrix is generated once (`modules/recommender.py`) and saved
  to `data/synthetic_ratings.csv` purely to demonstrate the collaborative-filtering
  mechanics (latent-factor SVD) — this is disclosed clearly in the app and in the report.
  Content-based recommendation uses only the real corpus.
- **Heterogeneous sources (Section B):** the working corpus combines (1) the publicly
  available BBC-style news dataset, (2) a simulated news API feed (`modules/mock_api.py`),
  and (3) crawled pages from the local web graph — all merged into one schema with
  metadata stored in a separate SQLite table from content (`modules/storage.py`,
  `data/ir_store.db`).

## Suggested demo flow for screenshots

1. **Dashboard** — screenshot as-is.
2. **Crawling** — pick seeds `local-news://home`, depth 2, max pages 200 → Run Crawl →
   screenshot the stats + duplicate inspector.
3. **Text Mining** — pick any document → screenshot keyword extraction/profile → click
   "Train classifier" → click "Run comparative analysis" → screenshot both results.
4. **Index Management** — click "Build / Rebuild Index" → screenshot stats + top terms.
5. **Search & Ranking** — try query `technology company investment` → screenshot all three
   ranking tabs (baseline / PageRank / HITS) to show `rank_change`.
6. **Recommendations** — try all three modes (content-based / collaborative / hybrid).
7. **Evaluation** — click "Run evaluation across ranking methods" → screenshot the
   comparison table and chart.
8. **Performance Analytics** — screenshot the timing log after doing the above.
9. **Inference & Discussion** — expand all five answers, then download the inference
   summary.

## Submission checklist (Assignment 2 requirements)

- [x] Streamlit application code (`app.py` + `modules/*.py`)
- [x] Dataset used (`data/news_corpus.csv`, plus generated `ir_store.db` /
  `synthetic_ratings.csv`)
- [x] `README.md` with install + run instructions (this file)
- [x] Report with implementation explanation, screenshots, experimental results/inferences
  (`report/REPORT.md` / `SmartIR-2_Report_Team47.pdf`) — including §9's 17 real BITS
  Virtual Lab screenshots (`bits_lab_screenshots/`)
- [x] Demo evidence: §9's 17-screenshot Virtual Lab walkthrough covers this; a screen
  recording (see the demo flow above) is optional on top of it, not required

Everything is complete and verified — including execution on the BITS Virtual Lab
portal, evidenced in `report/REPORT.md` §9.
