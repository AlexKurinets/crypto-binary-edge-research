"""Postgres access: migrations, bulk COPY in and out, and small queries as DataFrames."""

from __future__ import annotations

import io
from datetime import datetime

import numpy as np
import pandas as pd
import psycopg

from edgelab.config import ROOT, dsn


def connect(url: str | None = None, autocommit: bool = False) -> psycopg.Connection:
    return psycopg.connect(url or dsn(), autocommit=autocommit)


def migrate(conn: psycopg.Connection) -> list[str]:
    applied = []
    for path in sorted((ROOT / "sql").glob("*.sql")):
        conn.execute(path.read_text(encoding="utf-8"))
        applied.append(path.name)
    conn.commit()
    return applied


def copy_frame(conn: psycopg.Connection, table: str, df: pd.DataFrame, columns: list[str]) -> int:
    """Bulk insert with COPY. Empty strings are NULL."""
    if df.empty:
        return 0
    out = df[columns].copy()
    for col in columns:  # numpy formats UTC timestamps far faster than to_csv's date_format
        if isinstance(out[col].dtype, pd.DatetimeTZDtype):
            values = out[col].dt.tz_convert("UTC").dt.tz_localize(None).to_numpy()
            text = np.char.add(np.datetime_as_string(values, unit="s"), "+00")
            out[col] = np.where(pd.isna(values), "", text)
    buf = io.StringIO()
    out.to_csv(buf, index=False, header=False, na_rep="")
    with conn.cursor() as cur, cur.copy(f"COPY {table} ({', '.join(columns)}) FROM STDIN WITH (FORMAT csv, NULL '')") as cp:
        cp.write(buf.getvalue())
    return len(df)


def read_frame(conn: psycopg.Connection, query: str, parse_dates: list[str] | None = None) -> pd.DataFrame:
    """Bulk read with COPY ... TO STDOUT. The query takes no parameters."""
    buf = io.BytesIO()
    with conn.cursor() as cur, cur.copy(f"COPY ({query}) TO STDOUT WITH (FORMAT csv, HEADER)") as cp:
        for chunk in cp:
            buf.write(chunk)
    buf.seek(0)
    df = pd.read_csv(buf)
    for col in parse_dates or []:
        df[col] = pd.to_datetime(df[col], utc=True, format="ISO8601")
    for col in df.columns:  # Postgres booleans arrive as t/f
        if pd.api.types.is_string_dtype(df[col]) and df[col].notna().any() and set(df[col].dropna().unique()) <= {"t", "f"}:
            df[col] = df[col].map({"t": True, "f": False}).astype("boolean" if df[col].isna().any() else bool)
    return df


def query_df(conn: psycopg.Connection, query: str, params: tuple | dict | None = None) -> pd.DataFrame:
    with conn.cursor() as cur:
        cur.execute(query, params)
        cols = [d.name for d in cur.description]
        rows = cur.fetchall()
    df = pd.DataFrame(rows, columns=cols)
    for col in df.columns:
        first = df[col].dropna()
        if len(first) and isinstance(first.iloc[0], datetime):
            df[col] = pd.to_datetime(df[col], utc=True)
    return df


def meta(conn: psycopg.Connection) -> dict:
    rows = conn.execute("select key, value from market.meta").fetchall()
    return {k: v for k, v in rows}
