"""Streamlit Community Cloud entrypoint.

Kept for consistency with assignment1/streamlit_app.py. Streamlit Cloud lets
you specify any "Main file path" (this repo's deploys point directly at
app.py — see README.md), but some setups still default to looking for
`streamlit_app.py`, so this thin wrapper re-executes app.py on every rerun (a
plain `import app` would only run once because Python caches modules across
Streamlit reruns).
"""
import sys
from pathlib import Path
import runpy

APP_DIR = Path(__file__).parent
# app.py does `from modules import ...`; make sure this directory is on
# sys.path so that import resolves regardless of how the runner invoked us
# (runpy.run_path doesn't add the target script's directory itself).
sys.path.insert(0, str(APP_DIR))

runpy.run_path(str(APP_DIR / "app.py"), run_name="__main__")
