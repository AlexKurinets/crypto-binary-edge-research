"""Load the synthetic market into the same tables the recorder fills in production."""

from __future__ import annotations

import json

import psycopg

from edgelab.db import copy_frame
from edgelab.synthetic import generate


def is_empty(conn: psycopg.Connection) -> bool:
    return conn.execute("select not exists (select 1 from market.bars)").fetchone()[0]


def seed(conn: psycopg.Connection, research_days: int = 112, paper_days: int = 42, seed_value: int = 20260518) -> dict:
    market = generate(research_days=research_days, paper_days=paper_days, seed=seed_value)
    conn.execute("truncate market.quotes, market.contracts, market.bars, market.releases, market.meta, research.runs, research.registrations restart identity cascade")
    copy_frame(conn, "market.bars", market.bars, ["asset", "ts", "open", "high", "low", "close"])
    copy_frame(conn, "market.releases", market.releases, ["ts", "name"])
    copy_frame(conn, "market.contracts", market.contracts, ["ticker", "asset", "open_ts", "close_ts", "strike", "settle_avg", "result"])
    copy_frame(
        conn,
        "market.quotes",
        market.quotes,
        ["ticker", "ts", "yes_bid", "yes_ask", "bid_size", "ask_size", "fill_ts", "fill_yes_bid", "fill_yes_ask"],
    )
    with conn.cursor() as cur:
        cur.executemany(
            "insert into market.meta (key, value) values (%s, %s::jsonb)",
            [(k, json.dumps(v)) for k, v in market.meta.items()],
        )
    conn.commit()
    return {
        "bars": len(market.bars),
        "contracts": len(market.contracts),
        "quote_snapshots": len(market.quotes),
        "releases": len(market.releases),
        **{k: market.meta[k] for k in ("start", "research_cutoff", "end")},
    }
