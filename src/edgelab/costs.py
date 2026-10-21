"""What a trade costs: the fee, the spread and the price move during latency.

The taker fee follows the exchange's published quadratic schedule, round_up(0.07 x C x P x (1 - P))
dollars per order, where C is the contract count and P the price in dollars. The fee is largest
at 50 cents (1.75 cents a contract) and small near 0 or 100, so a rule that trades
near-the-money contracts has to clear the most expensive part of the schedule.

Buying NO costs 100 minus the YES bid, so every price here is the price of the side bought.
"""

from __future__ import annotations

import numpy as np


def taker_fee(price, contracts, coefficient: float = 0.07):
    """Fee in dollars for one order, rounded up to the next cent. Vectorized."""
    raw = coefficient * np.asarray(contracts) * np.asarray(price) * (1 - np.asarray(price))
    return np.ceil(np.round(raw * 100, 9)) / 100


def side_price(side, yes_bid, yes_ask):
    """Price in dollars of buying the given side at the touch."""
    side = np.asarray(side)
    return np.where(side == "yes", np.asarray(yes_ask), 100 - np.asarray(yes_bid)) / 100


def side_mid(side, yes_bid, yes_ask):
    mid_yes = (np.asarray(yes_bid) + np.asarray(yes_ask)) / 200
    return np.where(np.asarray(side) == "yes", mid_yes, 1 - mid_yes)


def breakeven_win_rate(price: float, contracts: int = 25, coefficient: float = 0.07) -> float:
    """Win rate a side bought at `price` needs to break even after the fee."""
    return float(price + taker_fee(price, contracts, coefficient) / contracts)


def fill(frame, order_contracts: int, slip_cents: int, coefficient: float):
    """Simulate an immediate-or-cancel order priced at the decision quote plus slip_cents.

    The order reaches the book latency_s after the decision, so it is matched against the fill
    quote. If the side's price has moved beyond the limit the order misses; otherwise it fills
    up to the displayed size. Per-contract cost components add up exactly:
    net = gross - spread - latency - fee. Outcome columns are added only when the frame carries
    a settled `result`, so the paper trader can fill an order before anything is known.
    """
    side = frame["side"].to_numpy()
    decision_price = side_price(side, frame["yes_bid"], frame["yes_ask"])
    fill_price = side_price(side, frame["fill_yes_bid"], frame["fill_yes_ask"])
    filled = fill_price <= decision_price + slip_cents / 100 + 1e-9
    available = np.where(side == "yes", frame["ask_size"], frame["bid_size"])
    contracts = np.where(filled, np.minimum(order_contracts, available), 0)
    fee = np.where(filled, taker_fee(fill_price, contracts, coefficient), 0.0)
    fee_per = np.where(contracts > 0, fee / np.maximum(contracts, 1), 0.0)
    mid = side_mid(side, frame["yes_bid"], frame["yes_ask"])
    out = frame.copy()
    out["decision_price"] = decision_price
    out["fill_price"] = np.where(filled, fill_price, np.nan)
    out["filled"] = filled
    out["contracts"] = contracts
    out["fee"] = fee
    out["spread"] = decision_price - mid
    out["latency"] = fill_price - decision_price
    out["fee_per"] = fee_per
    if "result" in frame:
        result = frame["result"].astype(bool).to_numpy()
        payout = np.where(side == "yes", result, ~result).astype(int)
        out["payout"] = payout
        out["gross"] = payout - mid
        out["net"] = payout - fill_price - fee_per
        out["pnl_usd"] = np.where(filled, contracts * out["net"], 0.0)
    return out
