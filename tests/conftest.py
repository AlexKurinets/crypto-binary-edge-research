"""One small synthetic market, researched and paper-traded once per session against real Postgres.

The suite drops and rebuilds the three schemas in the database at EDGELAB_TEST_DSN (or
EDGELAB_DSN), so point it at a scratch database.
"""

import json
import os

import psycopg
import pytest

from edgelab import db
from edgelab.config import load_registry
from edgelab.paper import run_paper
from edgelab.research import run_research
from edgelab.seed import seed

TEST_DSN = os.environ.get("EDGELAB_TEST_DSN") or os.environ.get("EDGELAB_DSN", "postgresql://edgelab:edgelab@localhost:5432/edgelab")
RESEARCH_DAYS, PAPER_DAYS = 36, 3  # one walk-forward week after the warm-up and training window


@pytest.fixture(scope="session")
def dsn():
    os.environ["EDGELAB_DSN"] = TEST_DSN
    return TEST_DSN


@pytest.fixture(scope="session")
def pipeline(dsn):
    with psycopg.connect(dsn) as conn:
        conn.execute("drop schema if exists paper cascade; drop schema if exists research cascade; drop schema if exists market cascade")
        conn.commit()
        db.migrate(conn)
        seeded = seed(conn, RESEARCH_DAYS, PAPER_DAYS)
        research = run_research(conn)
        if research["frozen_hypothesis"] is None:
            # One test week is too little for anything to survive, so freeze H02 by hand to exercise the paper path.
            h02 = next(h for h in load_registry()["hypotheses"] if h["id"] == "H02")
            avoid = [{"dimension": d, "bucket": b, "source": "registered"} for d, b in h02["no_trade"]]
            conn.execute(
                "update research.frozen_rules set hypothesis_id = 'H02', params = %s, avoid = %s where run_id = %s",
                (json.dumps(h02), json.dumps(avoid), research["run_id"]),
            )
            conn.commit()
        paper = run_paper(conn)
    return {"seed": seeded, "research": research, "paper": paper}


@pytest.fixture
def conn(dsn, pipeline):
    with psycopg.connect(dsn) as c:
        yield c
        c.rollback()
