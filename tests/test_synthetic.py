import numpy as np
import pandas as pd
import pytest

from edgelab.synthetic import DECISION_OFFSETS_MIN, generate


@pytest.fixture(scope="module")
def market():
    return generate(research_days=3, paper_days=1, seed=21)


def test_same_seed_same_market(market):
    again = generate(research_days=3, paper_days=1, seed=21)
    pd.testing.assert_frame_equal(market.bars, again.bars)
    pd.testing.assert_frame_equal(market.quotes, again.quotes)


def test_contracts_settle_on_the_last_minute_average(market):
    c = market.contracts
    assert (c["result"] == (c["settle_avg"] >= c["strike"])).all()
    last_bar = market.bars.set_index(["asset", "ts"]).loc[list(zip(c["asset"], c["close_ts"], strict=True))]
    assert (c["settle_avg"].to_numpy() <= last_bar["high"].to_numpy() + 1e-9).all()
    assert (c["settle_avg"].to_numpy() >= last_bar["low"].to_numpy() - 1e-9).all()


def test_quotes_are_valid_and_on_schedule(market):
    q = market.quotes
    assert q["yes_bid"].between(1, 98).all() and (q["yes_ask"] > q["yes_bid"]).all() and (q["yes_ask"] <= 99).all()
    minutes = sorted(set(q["ts"].dt.minute))
    assert minutes == sorted({60 - m for m in DECISION_OFFSETS_MIN})
    assert ((q["fill_ts"] - q["ts"]) == pd.Timedelta(seconds=2)).all()


def test_releases_tuesday_and_thursday(market):
    r = market.releases
    assert set(r["ts"].dt.dayofweek) <= {1, 3}
    assert set(r["ts"].dt.strftime("%H:%M")) == {"12:30"}


def test_index_has_no_drift_to_exploit():
    m = generate(research_days=20, paper_days=0, seed=5)
    r = np.log(m.bars[m.bars["asset"] == "BTC"]["close"]).diff().dropna().to_numpy()
    autocorr = np.corrcoef(r[:-15], r[15:])[0, 1]
    assert abs(autocorr) < 0.03
