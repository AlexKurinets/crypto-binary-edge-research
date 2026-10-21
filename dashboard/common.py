"""Shared helpers for the dashboard pages: database reads, the latest run, formatting, colours."""

from __future__ import annotations

import os

import pandas as pd
import psycopg
import streamlit as st

from edgelab.db import query_df

BLUE = "#2a78d6"  # model, walk-forward
ORANGE = "#eb6834"  # market, paper
POSITIVE = "#2a78d6"
NEGATIVE = "#e34948"
MUTED = "#8a8984"
VERDICT_ICON = {"survives": "✅", "edge erased by costs": "➖", "no edge": "❌", "too few to tell": "❔"}
CONDITION_ICON = {"keep": "✅", "avoid": "⛔", "within noise": "➖", "too few": "❔"}


def dsn() -> str:
    return os.environ.get("EDGELAB_DSN", "postgresql://edgelab:edgelab@localhost:5432/edgelab")


@st.cache_data(ttl=30, show_spinner=False)
def q(sql: str, params: tuple | None = None) -> pd.DataFrame:
    with psycopg.connect(dsn(), autocommit=True) as conn:
        return query_df(conn, sql, params)


def latest_run() -> dict | None:
    df = q("select run_id, created_at, research_start, cutoff, summary from research.runs order by run_id desc limit 1")
    if df.empty:
        return None
    row = df.iloc[0].to_dict()
    row["meta"] = {r["key"]: r["value"] for r in q("select key, value from market.meta").to_dict("records")}
    return row


def require_run() -> dict:
    run = latest_run()
    if run is None:
        st.info("No research run yet. Run `edgelab research` (the Compose pipeline does this on startup), then refresh.")
        st.stop()
    return run


def cents(x, signed: bool = True) -> str:
    if x is None or pd.isna(x):
        return "n/a"
    digits = 2 if abs(x) < 0.001 else 1  # keep a CI bound like +0.05c from printing as +0.0c
    return f"{x * 100:+.{digits}f}¢" if signed else f"{x * 100:.{digits}f}¢"


def usd(x) -> str:
    return "n/a" if x is None or pd.isna(x) else f"{'-' if x < 0 else ''}${abs(x):,.0f}"


def md(text: str) -> str:
    """Escape dollar signs so Streamlit markdown does not read them as LaTeX."""
    return text.replace("$", "\\$")
