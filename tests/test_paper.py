import pandas as pd
import psycopg
import pytest

from edgelab.config import load_research_config
from edgelab.db import meta, query_df
from edgelab.features import bar_features
from edgelab.hypotheses import add_edges, signals
from edgelab.model import FittedModel
from edgelab.paper import ReplayFeed
from edgelab.research import build_snapshots, load_market


def test_decisions_are_logged_before_they_settle(conn, pipeline):
    d = query_df(conn, "select * from paper.decisions where run_id = %s", (pipeline["research"]["run_id"],))
    assert len(d) > 0
    assert (d["logged_at"] == d["decided_at"]).all()
    settled = d.dropna(subset=["settled_at"])
    assert len(settled) > 0 and (settled["settled_at"] > settled["logged_at"]).all()


def test_a_logged_decision_cannot_be_edited(conn, pipeline):
    decision_id = conn.execute("select min(decision_id) from paper.decisions where settled_at is not null").fetchone()[0]
    with pytest.raises(psycopg.errors.RaiseException, match="cannot be changed"):
        conn.execute("update paper.decisions set p_model = 0.99 where decision_id = %s", (decision_id,))
    conn.rollback()
    with pytest.raises(psycopg.errors.RaiseException, match="already settled"):
        conn.execute("update paper.decisions set outcome = not outcome where decision_id = %s", (decision_id,))


def test_feed_never_hands_out_the_future(conn, pipeline):
    info = meta(conn)
    start = pd.Timestamp(info["research_cutoff"])
    feed = ReplayFeed(conn, start, pd.Timestamp(info["end"]), latency_s=2)
    now = int((start + pd.Timedelta(hours=5, minutes=10)).timestamp())
    assert feed.history("BTC", now, 1441)["ts"].max() == now
    assert feed.history("BTC", now - 30, 1441)["ts"].max() == now - 60
    snap = feed.snapshot(now)
    with pytest.raises(RuntimeError, match="before the order"):
        feed.send(snap["ticker"][0], now, now + 1)
    assert set(feed.send(snap["ticker"][0], now, now + 2)) == {"fill_yes_bid", "fill_yes_ask"}
    settled = feed.settled(now)
    assert (settled["close_ts"] <= pd.Timestamp(now, unit="s", tz="UTC")).all() and len(settled) == 5 * 2 * 5


def test_paper_trader_sends_the_orders_the_backtest_would(conn, pipeline):
    """The live path (point-in-time features, hourly decisions) and the research path agree on every order."""
    cfg = load_research_config()
    run = pipeline["research"]["run_id"]
    info = meta(conn)
    params, model_json = conn.execute("select params, model from research.frozen_rules where run_id = %s", (run,)).fetchone()
    start, end = pd.Timestamp(info["research_cutoff"]), pd.Timestamp(info["end"])
    m = load_market(conn)
    feats = bar_features(m["bars"])
    snaps = build_snapshots(m["quotes"][(m["quotes"]["ts"] >= start) & (m["quotes"]["ts"] < end)], m["contracts"], feats, m["releases"])
    model = FittedModel.from_json(model_json)
    snaps["p_model"] = model.prob(snaps["close"], snaps["strike"], snaps["tau_s"], snaps["rv15"], snaps["rv60"], snaps["rv1440"])
    ex = cfg["execution"]
    expected = signals(add_edges(snaps, ex["order_contracts"], ex["fee_coefficient"]), params)
    got = query_df(conn, "select ticker, decided_at, side from paper.decisions where run_id = %s", (run,))
    assert set(zip(expected["ticker"], expected["ts"], expected["side"], strict=True)) == set(
        zip(got["ticker"], got["decided_at"], got["side"], strict=True)
    )


def test_skipped_rows_name_their_filter_and_never_fill(conn, pipeline):
    d = query_df(conn, "select * from paper.decisions where run_id = %s and action = 'skipped'", (pipeline["research"]["run_id"],))
    assert d["skip_reason"].str.startswith("Scheduled release").all()
    assert (d["contracts"] == 0).all() and d["fill_price"].isna().all()


def test_paper_summary_is_stored_with_the_run(conn, pipeline):
    p = conn.execute("select summary -> 'paper' from research.runs where run_id = %s", (pipeline["research"]["run_id"],)).fetchone()[0]
    assert p["signals"] == p["filled"] + p["missed"] + p["skipped"]
    assert p["predictions"] == 3 * 24 * 10 * 2 * 5
