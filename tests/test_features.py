import numpy as np
import pandas as pd
import pytest

from edgelab.features import FEATURE_COLUMNS, LOOKBACK_BARS, RANGE_BARS, bar_features, classify_regime, features_at
from edgelab.synthetic import generate


@pytest.fixture(scope="module")
def bars():
    return generate(research_days=3, paper_days=0, seed=11).bars


def test_changing_the_future_cannot_change_the_past(bars):
    btc = bars[bars["asset"] == "BTC"].reset_index(drop=True)
    cut = btc["ts"].iloc[2_000]
    future = btc["ts"] > cut
    shocked = btc.copy()
    shocked.loc[future, ["open", "high", "low", "close"]] *= 1.07
    a = bar_features(btc).set_index("ts")
    b = bar_features(shocked).set_index("ts")
    past = a.index[a.index <= cut]
    pd.testing.assert_frame_equal(a.loc[past, FEATURE_COLUMNS], b.loc[past, FEATURE_COLUMNS])
    assert not np.allclose(a.loc[a.index > cut, "close"], b.loc[b.index > cut, "close"])


def test_live_features_match_research_features(bars):
    eth = bars[bars["asset"] == "ETH"].reset_index(drop=True)
    vec = bar_features(eth).set_index("ts")
    rng = np.random.default_rng(3)
    for i in rng.integers(LOOKBACK_BARS, len(eth), 25):
        hist = eth.iloc[: i + 1]
        live = features_at(hist["close"].to_numpy(), hist["high"].to_numpy()[-RANGE_BARS:], hist["low"].to_numpy()[-RANGE_BARS:])
        row = vec.loc[eth["ts"].iloc[i]]
        for k in FEATURE_COLUMNS:
            assert live[k] == pytest.approx(row[k], rel=1e-9, abs=1e-15), k
        assert live["regime"] == row["regime"]


def test_live_features_need_a_full_day(bars):
    with pytest.raises(ValueError):
        features_at(np.ones(100), np.ones(300), np.ones(300))


def test_regime_priority():
    assert list(classify_regime([5, 5, 1, 1], [1, 0, 1, 0], [0.9, 0.9, 0.9, 0.1])) == ["shock", "shock", "breakout", "range"]
    assert str(classify_regime(1.0, 0, 0.5)) == "trend"
