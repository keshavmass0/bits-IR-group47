# BITS WILP — Information Retrieval (S2-25) — Team 47

This repository holds both Information Retrieval assignments for Team 47, each in its own
self-contained folder so they can be run, graded, and deployed independently — including
as two separate apps on Streamlit Community Cloud from this single repo (see below).

## Contents

| Folder | Assignment | App | Live URL |
|---|---|---|---|
| [`assignment1/`](assignment1/) | Assignment 1 — SmartIR Lab (preprocessing, indexing, phrase/boolean/tolerant retrieval) | `assignment1/app.py` | _fill in after deploying — see below_ |
| [`assignment2/`](assignment2/) | Assignment 2 — SmartIR-2 (crawling, text mining, search & ranking, recommenders, evaluation) | `assignment2/app.py` | **[assignment2-irgroup47.streamlit.app](https://assignment2-irgroup47.streamlit.app/)** ✅ live — test it directly |

Each folder has its own `README.md` with install/run instructions specific to that
assignment — start there for details.

## Running locally

```bash
# Assignment 1
cd assignment1 && pip install -r requirements.txt && streamlit run app.py

# Assignment 2
cd assignment2 && pip install -r requirements.txt && streamlit run app.py
```

## Deploying both apps to Streamlit Community Cloud from this one repo

Streamlit Community Cloud deploys **one app per "main file path"**, not one app per repo —
so a single GitHub repo (this one) can back multiple independently-URLed apps, each
pointed at a different file. That's exactly how this repo is laid out: two folders, two
`requirements.txt` files (Streamlit Cloud looks for `requirements.txt` next to the main
file first), two apps.

For each assignment, on [share.streamlit.io](https://share.streamlit.io):

1. **New app** → pick this GitHub repo and the `main` branch (or whichever branch you
   deploy from).
2. **Main file path**: `assignment1/app.py` for the first app, `assignment2/app.py` for
   the second.
3. **App URL**: choose a distinct subdomain for each, e.g. `bits-ir47-a1` and
   `bits-ir47-a2` → `https://bits-ir47-a1.streamlit.app` / `https://bits-ir47-a2.streamlit.app`.
4. Deploy. Repeat for the second app with its own main file path/URL.

Both apps redeploy automatically on every push to the branch they're linked to, and are
otherwise fully independent (separate logs, separate restart/reboot controls, separate
resource limits) despite sharing one repo.

## Team

Team 47 — BITS WILP Information Retrieval, Merged AIMLCZG537/DSECLZG537, S2-25.
