import numpy as np
import pandas as pd
import pytest

from edgelab.costs import breakeven_win_rate, fill, side_price, taker_fee


@pytest.mark.parametrize(
    ("price", "contracts", "fee"),
    [(0.50, 25, 0.44), (0.50, 1, 0.02), (0.92, 25, 0.13), (0.99, 25, 0.02), (0.10, 100, 0.63)],
)
def test_taker_fee_rounds_up_per_order(price, contracts, fee):
    assert taker_fee(price, contracts) == pytest.approx(fee)


def test_fee_peaks_at_fifty_cents():
    prices = np.linspace(0.01, 0.99, 99)
    assert taker_fee(prices, 100).max() == pytest.approx(taker_fee(0.50, 100))  # rounding flattens the top, but 50c is on it
    assert taker_fee(0.05, 100) < taker_fee(0.25, 100) < taker_fee(0.50, 100)
    assert taker_fee(0.50, 100) / 100 == pytest.approx(0.0175)


def test_breakeven_at_fifty_one_cents():
    # 51 cents plus 1.76 cents of fee: the side has to win about 52.8% of the time
    assert breakeven_win_rate(0.51, 25) == pytest.approx(0.5276)


def test_no_costs_one_minus_the_yes_bid():
    assert side_price(np.array(["no"]), 37, 40)[0] == pytest.approx(0.63)
    assert side_price(np.array(["yes"]), 37, 40)[0] == pytest.approx(0.40)


def _order(side, bid, ask, fill_bid, fill_ask, result, ask_size=100, bid_size=100):
    return pd.DataFrame(
        [{"side": side, "yes_bid": bid, "yes_ask": ask, "fill_yes_bid": fill_bid, "fill_yes_ask": fill_ask, "ask_size": ask_size, "bid_size": bid_size, "result": result}]
    )


def test_cost_components_add_up_to_net():
    t = fill(_order("yes", 48, 50, 49, 51, True), 25, 1, 0.07).iloc[0]
    assert t["filled"]
    assert t["net"] == pytest.approx(t["gross"] - t["spread"] - t["latency"] - t["fee_per"])
    assert t["net"] == pytest.approx(1 - 0.51 - 0.44 / 25)


def test_order_misses_when_price_runs_past_the_limit():
    t = fill(_order("yes", 48, 50, 50, 52, True), 25, 1, 0.07).iloc[0]
    assert not t["filled"] and t["contracts"] == 0 and t["pnl_usd"] == 0


def test_fill_is_capped_at_displayed_size():
    t = fill(_order("no", 60, 62, 60, 62, False, bid_size=7), 25, 1, 0.07).iloc[0]
    assert t["contracts"] == 7
    assert t["fee"] == pytest.approx(taker_fee(0.40, 7))


def test_fill_without_result_has_no_outcome_columns():
    t = fill(_order("yes", 48, 50, 48, 50, True).drop(columns="result"), 25, 1, 0.07)
    assert "net" not in t and t["filled"].iloc[0]
