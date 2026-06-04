"""Streamlit Community Cloud entrypoint.

Streamlit Cloud looks for `streamlit_app.py` by default. This thin wrapper
re-executes `app.py` on every rerun (a plain `import app` would only run once
because Python caches modules across Streamlit reruns).
"""
from pathlib import Path
import runpy

APP_PATH = Path(__file__).parent / "app.py"
runpy.run_path(str(APP_PATH), run_name="__main__")
