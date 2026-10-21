"""Streamlit dashboard over the research and paper tables. It only reads; the pipeline writes."""

import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import latest_run  # noqa: E402

st.set_page_config(page_title="Binary edge research", page_icon="📈", layout="wide")

views = Path(__file__).resolve().parent / "views"
pages = [
    st.Page(str(views / "overview.py"), title="Overview", default=True),
    st.Page(str(views / "calibration.py"), title="Calibration", url_path="calibration"),
    st.Page(str(views / "hypotheses.py"), title="Hypotheses", url_path="hypotheses"),
    st.Page(str(views / "conditions.py"), title="When not to trade", url_path="conditions"),
    st.Page(str(views / "paper.py"), title="Paper log", url_path="paper"),
]
nav = st.navigation(pages)

with st.sidebar:
    run = latest_run()
    if run:
        meta = run["meta"]
        st.markdown(f"**Research run {run['run_id']}**")
        st.caption(
            f"History {meta['start'][:10]} to {meta['research_cutoff'][:10]}, paper period to {meta['end'][:10]}. "
            f"Hypothesis registry {run['summary'].get('registrations', 1)} registration(s)."
        )
    st.caption("Synthetic data. Every figure here is computed by the pipeline from the synthetic market.")

nav.run()
