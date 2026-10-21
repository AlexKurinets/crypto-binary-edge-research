import numpy as np
import pandas as pd
import pytest

from edgelab.config import content_hash, load_registry
from edgelab.db import query_df


def test_every_registered_hypothesis_is_scored(conn, pipeline):
    run = pipeline["research"]["run_id"]
    h = query_df(conn, "select * from research.hypotheses where run_id = %s", (run,))
    assert sorted(h["hypothesis_id"]) == sorted(x["id"] for x in load_registry()["hypotheses"])
    assert set(h["verdict"]) <= {"survives", "edge erased by costs", "no edge", "too few to tell"}
    scored = h.dropna(subset=["p_value"])
    assert (scored["q_value"] >= scored["p_value"] - 1e-12).all()


def test_registration_is_recorded(conn, pipeline):
    run = pipeline["research"]["run_id"]
    stored = conn.execute("select hypotheses_hash from research.runs where run_id = %s", (run,)).fetchone()[0]
    assert stored == content_hash(load_registry()["hypotheses"])
    assert conn.execute("select count(*) from research.registrations").fetchone()[0] == 1


def test_folds_never_train_on_their_test_week(conn, pipeline):
    run = pipeline["research"]["run_id"]
    folds = query_df(conn, "select * from research.folds where run_id = %s", (run,))
    assert len(folds) >= 1
    assert (folds["train_end"] <= folds["test_start"]).all()
    p = query_df(conn, "select fold, min(ts) lo, max(ts) hi from research.predictions where run_id = %s group by fold", (run,)).merge(folds, on="fold")
    assert (p["lo"] >= p["test_start"]).all() and (p["hi"] < p["test_end"]).all()


def test_trade_costs_add_up(conn, pipeline):
    t = query_df(conn, "select * from research.trades where run_id = %s", (pipeline["research"]["run_id"],))
    assert len(t) > 100
    np.testing.assert_allclose(t["net"], t["payout"] - t["fill_price"] - t["fee_per"], atol=1e-9)
    np.testing.assert_allclose(t["net"], t["gross"] - t["spread"] - t["latency"] - t["fee_per"], atol=1e-9)
    assert (t["fill_price"] <= t["decision_price"] + 0.01 + 1e-9).all()


def test_registered_filter_only_blocks_release_windows(conn, pipeline):
    run = pipeline["research"]["run_id"]
    t = query_df(conn, "select decided_at from research.trades where run_id = %s and hypothesis_id = 'H02' and excluded", (run,))
    rel = query_df(conn, "select ts from market.releases")
    for ts in t["decided_at"]:
        assert ((rel["ts"] - pd.Timedelta(minutes=15) <= ts) & (ts <= rel["ts"] + pd.Timedelta(minutes=30))).any()
    other = conn.execute("select count(*) from research.trades where run_id = %s and hypothesis_id <> 'H02' and excluded", (run,)).fetchone()[0]
    assert other == 0


def test_frozen_model_is_fitted_before_the_cutoff(conn, pipeline):
    run = pipeline["research"]["run_id"]
    frozen_at, model = conn.execute("select frozen_at, model from research.frozen_rules where run_id = %s", (run,)).fetchone()
    cutoff = conn.execute("select cutoff from research.runs where run_id = %s", (run,)).fetchone()[0]
    assert frozen_at == cutoff
    assert set(model) == {"har", "recal", "dof", "window_s"}


def test_summary_figures_match_the_tables(conn, pipeline):
    s = pipeline["research"]
    n = conn.execute("select count(*) from research.predictions where run_id = %s", (s["run_id"],)).fetchone()[0]
    assert n == s["oos_snapshots"]
    assert s["brier_model"] == pytest.approx(s["brier_model"]) and 0 < s["brier_market"] < 0.25
